#!/usr/bin/env python3
"""Open the local point-and-click experiment launcher (Python 3.9+)."""
import argparse
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
from urllib.parse import unquote, urlsplit
import webbrowser

ROOT = Path(__file__).resolve().parent
RUNS = ROOT/'results'/'click-runs'
PRESETS = ('quick', 'reproduce', 'paper')
EXPERIMENTS = ('all', 'skew-t-200d-nuts', 'skew-t-200d-baselines', 'skew-t-100d-fixed',
               'skew-t-50d-nu-0.5', 'eight-schools', 'garch11', 'logistic-regression')


def environment_python():
    return ROOT/'.venv'/('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def python_version(executable):
    try:
        p = subprocess.run([str(executable), '-c', 'import sys; print("%s.%s" % sys.version_info[:2])'],
                           capture_output=True, text=True, timeout=10)
        return p.stdout.strip() if p.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def find_python312():
    candidates = [environment_python(), sys.executable, shutil.which('python3.12'),
        '/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12', '/opt/homebrew/bin/python3.12',
        '/usr/local/bin/python3.12', '/usr/bin/python3.12']
    if os.name == 'nt':
        candidates.append(str(Path(os.environ.get('LOCALAPPDATA', ''))/'Programs/Python/Python312/python.exe'))
        try:
            p = subprocess.run(['py', '-3.12', '-c', 'import sys; print(sys.executable)'], capture_output=True, text=True, timeout=15)
            if p.returncode == 0:
                candidates.insert(0, p.stdout.strip())
        except (OSError, subprocess.TimeoutExpired):
            pass
    for candidate in candidates:
        if candidate and python_version(candidate) == '3.12':
            return str(candidate)
    return None


def environment_ready():
    exe = environment_python()
    if python_version(exe) != '3.12':
        return False
    code = ('import importlib.metadata as m, pathlib; '
            'lines=pathlib.Path("requirements-cpu.txt").read_text(encoding="utf-8").splitlines(); '
            'assert all(m.version(k)==v for k,v in (line.split("==") for line in lines if line and not line.startswith("#")))')
    try:
        p = subprocess.run([str(exe), '-c', code], cwd=ROOT, capture_output=True, text=True, timeout=20)
        return p.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def commands_for_run(preset, experiment, directory, resume=False, plot_only=False):
    if preset not in PRESETS or experiment not in EXPERIMENTS:
        raise ValueError('Choose a listed preset and experiment')
    py = str(environment_python())
    directory = Path(directory)
    if experiment == 'all':
        command = [py, '-u', str(ROOT/'scripts/reproduce_all.py'), '--preset', preset, '--output-root', str(directory)]
        if plot_only:
            command.append('--plot-only')
        elif resume:
            command.append('--resume')
        return [command]
    result = directory/experiment
    if plot_only:
        return [[py, '-u', str(ROOT/'run_experiment.py'), '--plot-only', str(result)]]
    commands = []
    reference = directory/'logistic-reference'
    if experiment == 'logistic-regression':
        command = [py, '-u', str(ROOT/'prepare_reference.py'), '--preset', preset, '--output-dir', str(reference)]
        if resume and reference.exists():
            command.append('--resume')
        commands.append(command)
    command = [py, '-u', str(ROOT/'run_experiment.py'), experiment, '--preset', preset,
               '--output-dir', str(result), '--reference-dir', str(reference)]
    if resume and (result/'manifest.json').exists():
        command.append('--resume')
    commands.append(command)
    return commands


class Controller:
    def __init__(self):
        self.lock = threading.RLock()
        self.messages = deque(maxlen=600)
        self.active = False
        self.process = None
        self.cancelled = False
        self.ready = environment_ready()
        self.status = 'Ready to run' if self.ready else 'Set up the experiment environment first'
        self.current_run = None

    def log(self, value):
        with self.lock:
            self.messages.append(value.rstrip())

    def saved_runs(self):
        values = []
        for p in sorted(RUNS.glob('*/launcher-run.json'), reverse=True):
            try:
                value = json.loads(p.read_text())
                value['id'] = p.parent.name
                values.append(value)
            except (OSError, ValueError):
                pass
        return values

    def snapshot(self):
        with self.lock:
            return {'active': self.active, 'ready': self.ready, 'status': self.status,
                    'log': '\n'.join(self.messages), 'runs': self.saved_runs(), 'current_run': self.current_run,
                    'folder': str(ROOT)}

    def start_job(self, work, status):
        with self.lock:
            if self.active:
                raise ValueError('A job is already running')
            self.active, self.cancelled, self.status = True, False, status
            self.messages.clear()
        def execute():
            try:
                work()
                with self.lock:
                    self.status = 'Stopped — completed chains can be resumed' if self.cancelled else 'Finished successfully'
            except Exception as error:
                with self.lock:
                    self.status = 'Stopped — completed chains can be resumed' if self.cancelled else 'Needs attention — see the log'
                self.log(str(error))
            finally:
                with self.lock:
                    self.active, self.process = False, None
        threading.Thread(target=execute, daemon=True).start()

    def run_command(self, command, logfile=None):
        with self.lock:
            if self.cancelled:
                raise RuntimeError('Run stopped')
            options = {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == 'nt' else {'start_new_session': True}
            self.process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace', bufsize=1, env=dict(os.environ, PYTHONUNBUFFERED='1'), **options)
            process = self.process
        for line in process.stdout:
            self.log(line)
            if logfile:
                logfile.write(line)
                logfile.flush()
        code = process.wait()
        with self.lock:
            self.process = None
        if code:
            raise RuntimeError(f'The process ended with code {code}. Completed chain files are retained.')

    def setup(self):
        def work():
            base = find_python312()
            if not base:
                raise RuntimeError('Python 3.12 was not found. Install its 64-bit version using the link on this page, then click Set up again.')
            exe = environment_python()
            if python_version(exe) != '3.12':
                folder = ROOT/'.venv'
                if folder.exists() or folder.is_symlink():
                    folder.rename(ROOT/('.venv-before-'+datetime.now().strftime('%Y%m%d-%H%M%S')))
                self.log('Creating the Python 3.12 environment…')
                self.run_command([base, '-m', 'venv', str(folder)])
            self.log('Installing the pinned scientific packages. This requires internet access…')
            self.run_command([str(exe), '-m', 'pip', 'install', '--disable-pip-version-check', '-r', str(ROOT/'requirements-cpu.txt')])
            self.run_command([str(exe), '-m', 'pip', 'check'])
            self.ready = environment_ready()
            if not self.ready:
                raise RuntimeError('Environment validation failed; see installation output above')
        self.start_job(work, 'Setting up the environment…')

    def run(self, preset='quick', experiment='all', run_id=None, plot_only=False):
        if not self.ready:
            raise ValueError('Click Set up environment first')
        if run_id:
            if run_id in ('.', '..') or not re.fullmatch(r'[A-Za-z0-9_.-]+', run_id):
                raise ValueError('Invalid saved run')
            directory = RUNS/run_id
            saved = json.loads((directory/'launcher-run.json').read_text())
            preset, experiment = saved['preset'], saved['experiment']
        else:
            run_id = datetime.now().strftime('%Y%m%d-%H%M%S-%f')+'-'+preset
            directory = RUNS/run_id
        commands = commands_for_run(preset, experiment, directory, resume=bool(run_id and directory.exists()), plot_only=plot_only)
        def work():
            directory.mkdir(parents=True, exist_ok=True)
            (directory/'launcher-run.json').write_text(json.dumps({'preset': preset, 'experiment': experiment}, indent=2)+'\n')
            self.current_run = run_id
            with (directory/'launcher.log').open('a', encoding='utf-8') as logfile:
                for command in commands:
                    self.run_command(command, logfile)
        self.start_job(work, 'Rebuilding figures…' if plot_only else f'Running {experiment} ({preset})…')

    def stop(self):
        with self.lock:
            self.cancelled = True
            process = self.process
            if process is None or process.poll() is not None:
                return
            if os.name == 'nt':
                try:
                    process.send_signal(signal.CTRL_BREAK_EVENT)
                except OSError:
                    subprocess.run(['taskkill', '/PID', str(process.pid), '/T'], capture_output=True)
            else:
                os.killpg(process.pid, signal.SIGINT)


def make_handler(controller, token):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, payload, status=200, content_type='application/json'):
            body = json.dumps(payload).encode() if content_type == 'application/json' else payload
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(body)

        def authorized(self):
            host = f'127.0.0.1:{self.server.server_port}'
            return (self.headers.get('Host') == host and self.headers.get('X-Launcher-Token') == token
                    and self.headers.get('Origin', f'http://{host}') == f'http://{host}')

        def do_GET(self):
            host = f'127.0.0.1:{self.server.server_port}'
            if self.headers.get('Host') != host:
                return self.respond({'error': 'Invalid host'}, 403)
            path = unquote(urlsplit(self.path).path)
            if path == '/':
                page = (ROOT/'launcher/index.html').read_text(encoding='utf-8').replace('__TOKEN__', json.dumps(token))
                return self.respond(page.encode(), content_type='text/html; charset=utf-8')
            if path == '/state':
                if not self.authorized(): return self.respond({'error': 'Unauthorized'}, 403)
                return self.respond(controller.snapshot())
            if path == '/artifacts':
                if not self.authorized(): return self.respond({'error': 'Unauthorized'}, 403)
                links = []
                if controller.current_run:
                    folder = RUNS/controller.current_run
                    for p in sorted(folder.rglob('*')):
                        if p.is_file() and (p.suffix in ('.png', '.pdf', '.tex') or p.name == 'posteriordb_rmse.csv'):
                            links.append({'name': str(p.relative_to(folder)), 'url': '/artifact/'+str(p.relative_to(RUNS))})
                return self.respond(links)
            if path.startswith('/artifact/'):
                p = (RUNS/path[len('/artifact/'):]).resolve()
                if not p.is_relative_to(RUNS.resolve()) or p.suffix not in ('.png', '.pdf', '.csv', '.tex') or not p.is_file():
                    return self.respond({'error': 'File not available'}, 404)
                return self.respond(p.read_bytes(), content_type=mimetypes.guess_type(str(p))[0] or 'application/octet-stream')
            return self.respond({'error': 'Not found'}, 404)

        def do_POST(self):
            if not self.authorized(): return self.respond({'error': 'Unauthorized'}, 403)
            try:
                size = int(self.headers.get('Content-Length', 0))
                if not 0 <= size <= 4096: raise ValueError('Invalid request size')
                data = json.loads(self.rfile.read(size) or b'{}')
                if self.path == '/setup': controller.setup()
                elif self.path == '/run': controller.run(data.get('preset', 'quick'), data.get('experiment', 'all'))
                elif self.path == '/resume': controller.run(run_id=data['id'])
                elif self.path == '/replot': controller.run(run_id=data['id'], plot_only=True)
                elif self.path == '/show':
                    if data['id'] not in {r['id'] for r in controller.saved_runs()}: raise ValueError('Unknown run')
                    controller.current_run = data['id']
                elif self.path == '/stop': controller.stop()
                elif self.path == '/quit':
                    controller.stop()
                    threading.Thread(target=self.server.shutdown, daemon=True).start()
                else: return self.respond({'error': 'Unknown action'}, 404)
                return self.respond({'ok': True})
            except Exception as error:
                return self.respond({'error': str(error)}, 400)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    controller = Controller()
    token = secrets.token_urlsafe(32)
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(controller, token))
    url = f'http://127.0.0.1:{server.server_port}'
    print(f'Experiment launcher: {url}', flush=True)
    print('Keep this window open while experiments run. Use the buttons in your browser.', flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop()
        server.server_close()


if __name__ == '__main__':
    main()

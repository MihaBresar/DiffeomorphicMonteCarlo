"""Fetch immutable PosteriorDB reference draws, models, and metadata.

No credentials are used. Archives are retained and parsed without extracting
paths from the archive. Re-running verifies existing artifacts against the
checksums written on the first download.
"""
import hashlib
import json
from pathlib import Path
import urllib.request

REVISION = '5545a1dd07ae297c36edecbcd82aa49097b4c385'
BASE = f'https://raw.githubusercontent.com/stan-dev/posteriordb/{REVISION}/posterior_database/'
ROOT = Path(__file__).resolve().parents[1] / 'data' / 'posteriordb'
FILES = [
    'LICENCE.md', 'models/info/garch11.info.json',
    'models/info/eight_schools_centered.info.json', 'models/info/eight_schools_noncentered.info.json',
    'models/stan/garch11.stan', 'models/stan/eight_schools_centered.stan',
    'models/stan/eight_schools_noncentered.stan',
    'posteriors/garch-garch11.json', 'posteriors/eight_schools-eight_schools_centered.json',
    'data/data/garch.json.zip', 'data/data/eight_schools.json.zip',
    'data/info/garch.info.json', 'data/info/eight_schools.info.json',
    'reference_posteriors/draws/draws/garch-garch11.json.zip',
    'reference_posteriors/draws/draws/eight_schools-eight_schools_noncentered.json.zip',
    'reference_posteriors/draws/info/garch-garch11.info.json',
    'reference_posteriors/draws/info/eight_schools-eight_schools_noncentered.info.json',
]


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        'repository': 'https://github.com/stan-dev/posteriordb', 'revision': REVISION, 'files': {}}
    if manifest['revision'] != REVISION:
        raise ValueError('Existing manifest has another revision')
    for relative in FILES:
        path = ROOT / relative
        url = f'https://raw.githubusercontent.com/stan-dev/posteriordb/{REVISION}/LICENCE.md' if relative == 'LICENCE.md' else BASE + relative
        if not path.exists():
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = response.read()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        previous = manifest['files'].get(relative)
        if previous and previous['sha256'] != digest:
            raise ValueError(f'Checksum mismatch: {relative}')
        manifest['files'][relative] = {'url': url, 'sha256': digest}
        print(relative, digest)
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()

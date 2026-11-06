"""Reproduce fitting and matched final-map radius/step calibration.

Both optimized families use fixed zero centres and zero deformation before every
pilot transition. Original SCS remains an unchanged author-fitted baseline.
Selection never reads production estimates or the independent holdout.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
ASSETS = SOURCE/'studies'/'zero_centre'/'maps'
DIMENSIONS = (100, 500)
R_FACTORS = (.5, 1., 2., 4.)
DCS_ELL = (.25, .5, 1., 2.)
SCS_H = (.025, .05, .1, .2)
METHODS = ('DCS-BW', 'SCS')
PILOT_N, HOLDOUT_N, LAG, BATCH = 4096, 20000, 8, 256
SEED = 842901174
FIT_SEEDS = {'ball': 42, 'sphere': 0}
KERNEL_FILES = ('dcs_bench/timed_radial.py', 'dcs_bench/targets.py',
    'dcs_bench/transforms.py', 'dcs_bench/references.py', 'dcs_bench/scs_paper.py',
    'dcs_bench/tuning.py', 'dcs_bench/ball_boundary.py', 'dcs_bench/billiard.py',
    'third_party/scs_paper/src/scp_core.py', 'third_party/scs_paper/src/train.py',
    'third_party/scs_paper/LICENSE')
BUDGET = dict(candidates=16, shared_exact_target_starts=4096,
    transitions_per_start=8, transitions_total=524288,
    same_physical_starts_reused_across_families_and_candidates=True)
FINAL_MAP_POLICY = {'DCS-BW': 'zero_mu_zero_delta', 'SCS': 'zero_mu_zero_observer'}

def require(condition, message):
    if not condition:
        raise ValueError(message)

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def exact_states(spec, stage, count):
    import numpy as np
    from dcs_bench.targets import Target
    from dcs_bench.references import skew_t_iid
    start = time.perf_counter()
    entropy = [SEED, spec['dimension'], stage, 0]
    rng = np.random.default_rng(np.random.SeedSequence(entropy))
    x = np.empty((count, spec['dimension']))
    for lo in range(0, count, BATCH):
        x[lo:lo+BATCH] = skew_t_iid(Target(spec), rng, min(BATCH, count-lo))
    return x, dict(exact_target_draws=count, target_seed_entropy=entropy,
        physical_state_sha256=hashlib.sha256(x.tobytes()).hexdigest(), generation_seconds=time.perf_counter()-start)

class Evaluator:
    def __init__(self, spec, threshold, physical, stage):
        import numpy as np
        from scipy import stats
        self.spec, self.d, self.stage = spec, spec['dimension'], stage
        self.physical, self.threshold = physical, threshold
        self.initial_norm = np.linalg.norm(physical, axis=1)
        self.u0 = stats.f.cdf(self.initial_norm**2/self.d, self.d, spec['nu'])
        self.i0 = self.initial_norm >= threshold
        self.p = float(stats.f.sf(threshold**2/self.d, self.d, spec['nu']))
        self.cache, self.programs, self.inputs, self.keys = {}, {}, {}, {}
        self.arrays = dict(initial_norm=self.initial_norm, initial_radial_cdf=self.u0, initial_tail=self.i0.astype(np.uint8))
        self.costs = dict(inverse_seconds={}, key_generation_seconds={}, compile_seconds={})

    def evaluate(self, candidate):
        import jax
        import jax.numpy as jnp
        import numpy as np
        from scipy import stats
        from dcs_bench import scs_paper, timed_radial
        from dcs_bench.transforms import BallMap
        method, mapping, h = candidate['method'], candidate['mapping'], candidate['h']
        key = candidate['id']
        if key in self.cache:
            return self.cache[key]
        map_key = method+'-'+json.dumps(mapping, sort_keys=True)
        if map_key not in self.programs:
            block, projection, _, _ = timed_radial._components(mapping, self.spec, method)
            def one(state, random_key, step, expected):
                states, accepted = block(state, random_key, LAG, step)
                before, after = projection(state), projection(states[-1])
                roundtrip = jnp.max(jnp.abs(before-expected))/(1+jnp.max(jnp.abs(expected)))
                previous = jnp.concatenate((state[None, :], states[:-1]), axis=0)
                rejected_change = jnp.max(jnp.where(accepted[:, None] == 0., jnp.abs(states-previous), 0.))
                return jnp.linalg.norm(after), jnp.sum(accepted), roundtrip, rejected_change
            self.programs[map_key] = jax.jit(jax.vmap(one, in_axes=(0, 0, None, 0)))
            start = time.perf_counter()
            if method == 'DCS-BW':
                active = BallMap(**mapping).inverse(self.physical)
            else:
                model = scs_paper._author().SCP(self.d, mapping['latitude'])
                params = scs_paper._parameters(model, mapping)
                inverse = jax.jit(jax.vmap(model.inverse_projection, in_axes=(None, 0)))
                active = np.concatenate([np.asarray(jax.block_until_ready(inverse(params, jnp.asarray(self.physical[lo:lo+BATCH]))))
                    for lo in range(0, len(self.physical), BATCH)])
            require(np.isfinite(active).all(), 'Nonfinite inverse-map state')
            self.inputs[map_key] = active
            self.costs['inverse_seconds'][candidate['method']+'-R-'+str(candidate['R'])] = time.perf_counter()-start
        if method not in self.keys:
            start = time.perf_counter()
            random_key = jax.random.key(SEED)
            for v in (self.d, self.stage, 1 if method == 'DCS-BW' else 2):
                random_key = jax.random.fold_in(random_key, v)
            self.keys[method] = jax.block_until_ready(jax.vmap(lambda i:jax.random.fold_in(random_key, i))(jnp.arange(len(self.physical), dtype=jnp.uint32)))
            self.costs['key_generation_seconds'][method] = time.perf_counter()-start
        active, keys = self.inputs[map_key], self.keys[method]
        program = self.programs[map_key]
        sizes = sorted({min(BATCH, len(active)-lo) for lo in range(0, len(active), BATCH)})
        compiled = {}
        start = time.perf_counter()
        for size in sizes:
            compiled[size] = program.lower(jnp.asarray(active[:size]), keys[:size], jnp.asarray(h), jnp.asarray(self.physical[:size])).compile()
        compilation_seconds = time.perf_counter()-start
        self.costs['compile_seconds'][key] = compilation_seconds
        start = time.perf_counter()
        columns = [[] for _ in range(4)]
        for lo in range(0, len(active), BATCH):
            size = min(BATCH, len(active)-lo)
            values = jax.block_until_ready(compiled[size](jnp.asarray(active[lo:lo+size]), keys[lo:lo+size], jnp.asarray(h), jnp.asarray(self.physical[lo:lo+size])))
            for dest, value in zip(columns, values):
                dest.append(np.asarray(value))
        evaluation_seconds = time.perf_counter()-start
        norm, accepted, roundtrip, rejected = [np.concatenate(c) for c in columns]
        require(all(np.isfinite(a).all() for a in (norm, accepted, roundtrip, rejected)), 'Nonfinite transition output')
        require(np.max(roundtrip) < 1e-8 and np.max(rejected) == 0., 'Map or rejection inconsistency')
        require(np.all((accepted >= 0) & (accepted <= LAG) & (accepted == np.floor(accepted))), 'Invalid acceptance counts')
        start = time.perf_counter()
        u8 = stats.f.cdf(norm**2/self.d, self.d, self.spec['nu'])
        i8 = norm >= self.threshold
        sq_u, sq_i = (u8-self.u0)**2, (i8.astype(float)-self.i0.astype(float))**2
        normalized = {'radial_cdf':sq_u/(2/12), 'tail_indicator':sq_i/(2*self.p*(1-self.p))}
        metrics = {name:dict(score=float(v.mean()), empirical_standard_error=float(v.std(ddof=1)/math.sqrt(len(v))),
            sum=float(v.sum()), n=len(v)) for name, v in normalized.items()}
        result = dict(candidate, starts=len(norm), transitions=len(norm)*LAG,
            accepted_transitions=int(accepted.sum()), acceptance=float(accepted.sum()/(len(norm)*LAG)),
            score=min(m['score'] for m in metrics.values()), components=metrics,
            initial_tail_count=int(self.i0.sum()), final_tail_count=int(i8.sum()),
            tail_to_core=int(np.sum(self.i0 & ~i8)), core_to_tail=int(np.sum(~self.i0 & i8)),
            tail_flips=int(sq_i.sum()), analytic_tail_probability=self.p,
            max_relative_roundtrip_error=float(roundtrip.max()), max_rejected_state_change=float(rejected.max()),
            evaluation_seconds=evaluation_seconds, compilation_seconds=compilation_seconds,
            diagnostic_transitions_per_second=len(norm)*LAG/evaluation_seconds,
            scoring_seconds=time.perf_counter()-start,
            timing_note='Batched diagnostic transition throughput with terminal radial projection; not production sampling throughput. Density calls are unchanged and not separately counted.',
            uncertainty_note='Finite bounded component scores; empirical component standard errors only. The minimum and selected maximum have selection bias; no confidence guarantee for the winning score.')
        prefix = key.replace('-', '_')
        self.arrays[prefix+'_final_norm'] = norm
        self.arrays[prefix+'_accepted_steps'] = accepted.astype(np.uint8)
        self.arrays[prefix+'_radial_cdf_squared_change'] = sq_u
        self.arrays[prefix+'_tail_squared_change'] = sq_i.astype(np.uint8)
        result['array_key'] = prefix
        self.cache[key] = result
        return result

def select(rows):
    def ordering(row):
        anchor = row['ell'] if row['method'] == 'DCS-BW' else row['h']/.1
        return (-row['score'], abs(math.log(row['R_factor'])), abs(math.log(anchor)), row['R_factor'], row['h'])
    winner = min(rows, key=ordering)
    return dict(candidate_id=winner['id'], R=winner['R'], R_factor=winner['R_factor'], h=winner['h'],
        score=winner['score'], status='positive_pilot_score' if winner['score'] > 0 else 'no_positive_pilot_score',
        R_grid_edge=winner['R_factor'] in (R_FACTORS[0], R_FACTORS[-1]),
        step_grid_edge=(winner['ell'] in (DCS_ELL[0], DCS_ELL[-1]) if winner['method']=='DCS-BW' else winner['h'] in (SCS_H[0], SCS_H[-1])))


def target_spec(dimension):
    require(dimension in DIMENSIONS, 'Choose dimension 100 or 500')
    return dict(kind='skew_t', dimension=dimension, nu=3., alpha={'0':100., '1':-100.})


def fit_configuration(dimension):
    """Fully explicit fit recipe, independent of production preset defaults."""
    return dict(target=target_spec(dimension), dimension=dimension, beta=2., latitude=1.1,
        optimizer_steps=1000, optimizer_samples=2000, fit_seed=FIT_SEEDS['ball'],
        vi='on', vi_batch_policy='fixed', ball_map_policy='mobius',
        map_scale_initialization='unit', sphere_map_policy='author-paper-vi',
        vi_budget_policy='equal_updates_and_fixed_reference_clouds_v1',
        scs_vi=dict(optimizer_steps=1000, optimizer_samples=2000))


def runtime_manifest():
    versions = {}
    for package in ('numpy', 'scipy', 'jax', 'jaxlib', 'optax'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return dict(python=platform.python_version(), platform=platform.platform(), packages=versions)


def initialize_runtime():
    from dcs_bench.cpu import worker_environment
    os.environ.update(worker_environment())
    from dcs_bench import scs_paper
    scs_paper._author()  # Establish the same x64 author runtime before fitting or pilots.


def validate_fit(saved, dimension):
    require(saved['config']['target'] == target_spec(dimension), 'Unexpected fitting target')
    require(saved['maps']['ball']['beta'] == 2., 'Student-t beta must remain 2')
    require(saved['maps']['sphere']['latitude'] == 1.1, 'SCS latitude changed')
    expected = dict(steps=1000, gradient_updates=1000, training_draws=2000,
        total_training_draws=2000, unique_reference_draws=2000,
        objective_point_evaluations=2000000, batch_policy='fixed')
    for family in ('ball', 'sphere'):
        fit = saved['fitting'][family]
        require(all(fit.get(k) == v for k,v in expected.items()), 'VI budget/history mismatch')
        require(fit['seed'] == FIT_SEEDS[family], 'Fitting seed changed')
        require(len(fit['loss']) == 1000 and all(math.isfinite(v) for v in fit['loss']),
                'Invalid VI losses')
    for name,digest in saved['fitting']['sphere']['source_sha256'].items():
        require(sha(SOURCE/'third_party/scs_paper'/name) == digest, 'Pinned author source changed')
    return saved


def compare_fits(actual, expected, *, rtol=1e-8, atol=1e-10):
    """Check numerical maps/losses, excluding timing and descriptive metadata.

    Floating point and compiled libraries need not give byte-identical JSON.
    Every map field and every optimization loss is checked; tolerances do not
    permit choosing different fit seeds or a different recipe.
    """
    import numpy as np
    checks = {}
    for family in ('ball','sphere'):
        require(set(actual['maps'][family]) == set(expected['maps'][family]), 'Map fields differ')
        fields = dict(actual['maps'][family], loss=actual['fitting'][family]['loss'])
        baseline = dict(expected['maps'][family], loss=expected['fitting'][family]['loss'])
        checks[family] = {}
        for name, value in fields.items():
            a,b = np.asarray(value,dtype=np.float64),np.asarray(baseline[name],dtype=np.float64)
            require(a.shape == b.shape and np.isfinite(a).all(), 'Nonfinite or mismatched fit output')
            difference = float(np.max(np.abs(a-b)))
            close = bool(np.allclose(a,b,rtol=rtol,atol=atol))
            checks[family][name] = dict(exact_equal=bool(np.array_equal(a,b)),
                within_tolerance=close, max_absolute_difference=difference)
    return dict(passed=all(v['within_tolerance'] for f in checks.values() for v in f.values()),
        rtol=rtol,atol=atol,checked_fields=checks,
        scope='Numerical map fields and full loss histories; elapsed times and JSON hashes may differ.')


def reproduce_fits(output, *, verify_bundled=False):
    """Fit both families from their registered seeds, without production samples."""
    from scripts.reproduce_scs_100d import prepare
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    require(not any(output.iterdir()), 'Preserve existing fit outputs; choose an empty directory')
    plan = dict(schema='matched-comparison-fitting-plan-v1',seed_by_family=FIT_SEEDS,
        configurations={str(d):fit_configuration(d) for d in DIMENSIONS},
        budget_per_family=dict(updates=1000,fixed_reference_points=2000,objective_point_evaluations=2000000),
        original_scs='Pinned author reverse KL optimizer, seed 0, Adam .01, coordinate clip 1000.',
        dcs='Fixed uniform-ball draws, seed 42, Adam .01, gradient norm cap 10.',
        source_sha256=sha(__file__),kernel_source_hashes={n:sha(SOURCE/n) for n in KERNEL_FILES},
        runtime=runtime_manifest())
    write(output/'fitting_plan.json',plan)
    summaries = {}
    for d in DIMENSIONS:
        config = fit_configuration(d)
        print(f'{d}D: reproducing both full-budget VI fits',flush=True)
        maps,fitting = prepare(config)
        saved = validate_fit(dict(config=config,maps=maps,fitting=fitting),d)
        numerical_check = compare_fits(saved,validate_fit(read(ASSETS/f'full_budget_{d}d.json'),d))
        saved['bundled_numerical_check'] = numerical_check
        write(output/f'full_budget_{d}d.json',saved)
        summaries[str(d)] = dict(file=f'full_budget_{d}d.json',sha256=sha(output/f'full_budget_{d}d.json'),
            seconds={f:fitting[f]['seconds'] for f in ('ball','sphere')},
            bundled_numerical_check=numerical_check)
        write(output/'fitting_results.json',dict(plan_sha256=sha(output/'fitting_plan.json'),dimensions=summaries))
        if verify_bundled:
            require(numerical_check['passed'], f'{d}D fit differs from bundled numerical values; inspect fitting_results.json')
    return output


def candidates(dimension, saved):
    """Final zero coordinates are imposed before evaluating either family."""
    rows=[]
    for method in METHODS:
        family = 'ball' if method=='DCS-BW' else 'sphere'
        for ri,factor in enumerate(R_FACTORS):
            for hi,value in enumerate(DCS_ELL if method=='DCS-BW' else SCS_H):
                mapping=copy.deepcopy(saved['maps'][family])
                mapping.update(scale=mapping['scale']*factor,mu=[0.]*dimension)
                mapping['delta' if method=='DCS-BW' else 'observer']=[0.]*dimension
                h=value/math.sqrt(dimension) if method=='DCS-BW' else value
                rows.append(dict(id=f'{method}-r{ri}-h{hi}',method=method,R_factor=factor,
                    R=mapping['scale'],h=h,ell=value if method=='DCS-BW' else None,
                    mapping=mapping,baseline_author_scs=False))
    return rows


def build_plan(fit_dir=ASSETS):
    fit_dir=Path(fit_dir)
    dimensions={}
    for d in DIMENSIONS:
        filename=fit_dir/f'full_budget_{d}d.json'
        saved=validate_fit(read(filename),d)
        rows=candidates(d,saved)
        original=dict(id='SCS-original-author',method='SCS',R_factor=1.,
            R=saved['maps']['sphere']['scale'],h=.1,ell=None,
            mapping=copy.deepcopy(saved['maps']['sphere']),baseline_author_scs=True)
        dimensions[str(d)]=dict(input_map_file=filename.name,original_map_sha256=sha(filename),
            target=target_spec(d),threshold=50.*math.sqrt(d/100.),
            candidates=rows,original_author_candidate=original)
    policy=dict(coordinate_policy='zero_mu_zero_deformation_before_pilot',
        criterion='Maximize min(E[(U8-U0)^2]/(2/12), E[(I8-I0)^2]/(2*p*(1-p))). U is the analytic radial CDF; I is the fixed radial-tail indicator. Rejected/repeated states remain included.',
        exact_tie_rule='Prefer R factor closest to 1 in absolute log distance, then ell closest to 1 for BW / h closest to .1 for SCS in absolute log distance, then lower R factor, then lower h.',
        no_positive_score_rule='Retain the exact-tie winner and disclose no_positive_pilot_score; do not enlarge grids or substitute a different criterion.',
        holdout_policy='Freeze both selections before drawing 20,000 stationary starts from the separate stage-1 stream, lag 8. Assess both selections and original author SCS without retuning. This reproducible stream is not claimed never previously examined.',
        dcs_use='One selected BW radius and step are shared by RWM, BW, reflected BW and billiard. Gaussian RWM coordinate sigma is h/sqrt(d+2). No separate tuning of DCS variants.')
    return dict(schema='reported-matched-zero-radius-step-plan-v1',
        source_pilot_code='scripts/prepare_comparison.py',source_pilot_code_sha256=sha(__file__),
        kernel_source_hashes={n:sha(SOURCE/n) for n in KERNEL_FILES},
        budget_per_family_per_dimension=BUDGET,final_map_policy=FINAL_MAP_POLICY,
        seed=SEED,seed_policy='NumPy exact starts: SeedSequence([seed,dimension,stage,0]); JAX key folds dimension,stage,family(1=BW,2=SCS), then start index. Stage 0 is pilot, stage 1 is holdout. Starts are shared across all candidates and families; within-family transition keys are shared across candidates.',
        fit_seed_by_family=FIT_SEEDS,R_factors=list(R_FACTORS),DCS_ell=list(DCS_ELL),SCS_h=list(SCS_H),
        holdout_starts=HOLDOUT_N,batch_size=BATCH,selection_policy=policy,dimensions=dimensions,
        scope='Equal candidate and transition budgets on final zero maps, not equal elapsed time. Exact target draws and analytic references provide equal oracle assistance for these synthetic targets. Original SCS remains an additional unmodified baseline.',
        interpretation='The lag-8 score is a bounded radial proxy, not integrated ESS or MSE and not an optimality guarantee. Equal current budgets do not equate cumulative exploratory development.',
        timing_scope='VI, exact draws, inverse maps, compilation, synchronized pilot evaluation and scoring are recorded separately; production sampling excludes this preparation.')


def run_pilot(output, *, fit_dir=ASSETS):
    import numpy as np
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    require(not any(output.iterdir()),'Preserve existing pilot outputs; choose an empty directory')
    plan=build_plan(fit_dir)
    write(output/'pilot_plan.json',plan)
    write(output/'started.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),
        plan_sha256=sha(output/'pilot_plan.json'),runtime=runtime_manifest()))
    asset={k:copy.deepcopy(plan[k]) for k in ('source_pilot_code','source_pilot_code_sha256',
        'kernel_source_hashes','budget_per_family_per_dimension','final_map_policy','seed',
        'seed_policy','fit_seed_by_family','R_factors','DCS_ell','SCS_h','selection_policy','scope','interpretation','timing_scope')}
    asset.update(schema='reported-matched-zero-radius-step-calibration-v1',
        source_pilot_plan_sha256=sha(output/'pilot_plan.json'),frozen_before_assessment=True,
        holdout_starts=HOLDOUT_N,dimensions={})
    for d in DIMENSIONS:
        cell=plan['dimensions'][str(d)]
        physical,meta=exact_states(cell['target'],0,PILOT_N)
        evaluator=Evaluator(cell['target'],cell['threshold'],physical,0)
        rows=[]
        for candidate in cell['candidates']:
            row=evaluator.evaluate(candidate);rows.append(row)
            print(f'{d}D {row["id"]}: R={row["R"]:.9g}, h={row["h"]:.9g}; score={row["score"]:.8g}; flips={row["tail_flips"]}',flush=True)
        winners={m:select([r for r in rows if r['method']==m]) for m in METHODS}
        write(output/f'{d}d-selection.json',dict(frozen_before_holdout=True,selections=winners,
            plan_sha256=sha(output/'pilot_plan.json')))
        print(f'{d}D SELECTED: {winners}',flush=True)
        np.savez_compressed(output/f'{d}d-pilot-arrays.npz',**evaluator.arrays)
        write(output/f'{d}d-pilot.json',dict(metadata=meta,candidates=rows,costs=evaluator.costs))
        selected=[next(r for r in cell['candidates'] if r['id']==winners[m]['candidate_id']) for m in METHODS]
        holdout_physical,holdout_meta=exact_states(cell['target'],1,HOLDOUT_N)
        holdout=Evaluator(cell['target'],cell['threshold'],holdout_physical,1)
        checks=[holdout.evaluate(r) for r in selected+[cell['original_author_candidate']]]
        np.savez_compressed(output/f'{d}d-holdout-arrays.npz',**holdout.arrays)
        cost_summary = {m: dict(transition_evaluation_seconds=sum(r['evaluation_seconds'] for r in rows if r['method']==m),
            compilation_seconds=sum(r['compilation_seconds'] for r in rows if r['method']==m),
            scoring_seconds=sum(r['scoring_seconds'] for r in rows if r['method']==m),
            inverse_seconds=sum(v for k,v in evaluator.costs['inverse_seconds'].items() if k.startswith(m+'-')),
            key_generation_seconds=evaluator.costs['key_generation_seconds'][m]) for m in METHODS}
        entry=dict(original_map_sha256=cell['original_map_sha256'],selections=winners,
            historical_pilot_costs=cost_summary,shared_exact_state_generation_seconds=meta['generation_seconds'],
            cost_scope='Matched final-map pilot preparation; not production sampling. Each family uses 524288 transitions. Holdout costs are separate.',
            selection_sha256=sha(output/f'{d}d-selection.json'),frozen_before_holdout=True,
            pilot_transitions_per_family={m:sum(r['transitions'] for r in rows if r['method']==m) for m in METHODS},
            pilot_seed=SEED,physical_state_sha256=meta['physical_state_sha256'],pilot_exact_state_metadata=meta,
            candidates=[{k:v for k,v in r.items() if k!='mapping'} for r in rows],pilot_costs=evaluator.costs,
            pilot_array_sha256=sha(output/f'{d}d-pilot-arrays.npz'),
            holdout=dict(metadata=holdout_meta,candidates=[{k:v for k,v in r.items() if k!='mapping'} for r in checks],
                costs=holdout.costs,no_retuning=True,unique_transitions=sum(r['transitions'] for r in checks),
                arrays_sha256=sha(output/f'{d}d-holdout-arrays.npz')))
        require(entry['pilot_transitions_per_family']=={m:524288 for m in METHODS},'Pilot budget mismatch')
        require(build_plan(fit_dir)==plan,'Sources or inputs changed during pilot')
        write(output/f'{d}d-result.json',entry)
        asset['dimensions'][str(d)]=entry
    write(output/'matched_calibration.json',asset)
    return asset


def compare_calibrations(actual, expected, *, rtol=1e-8, atol=1e-10):
    """Verify frozen decisions and every candidate score, without reselection."""
    checks = {}
    for d in DIMENSIONS:
        observed,baseline = actual['dimensions'][str(d)],expected['dimensions'][str(d)]
        rows = {r['id']:r for r in observed['candidates']}
        prior = {r['id']:r for r in baseline['candidates']}
        require(set(rows)==set(prior),'Calibration candidate sets differ')
        scores = {k: math.isclose(rows[k]['score'],prior[k]['score'],rel_tol=rtol,abs_tol=atol) for k in rows}
        choices = {}
        for method in METHODS:
            a,b = observed['selections'][method],baseline['selections'][method]
            choices[method] = (a['candidate_id']==b['candidate_id'] and
                all(math.isclose(a[k],b[k],rel_tol=rtol,abs_tol=atol) for k in ('R','h')))
        checks[str(d)] = dict(selected_parameters_match=choices,candidate_scores_match=scores,
            maximum_candidate_score_difference=max(abs(rows[k]['score']-prior[k]['score']) for k in rows))
    return dict(passed=all(all(c['selected_parameters_match'].values()) and all(c['candidate_scores_match'].values()) for c in checks.values()),
        rtol=rtol,atol=atol,dimensions=checks,
        scope='Both selections and all pilot scores; timings and JSON hashes are excluded.')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('fit','pilot','all'),default='all',
        help='all reproduces both VI fits and the matched pilot; pilot reuses supplied fits')
    parser.add_argument('--output-dir',type=Path,required=True,help='Fresh directory for recorded preparation')
    parser.add_argument('--fit-dir',type=Path,help='Existing fits for --stage pilot; defaults to bundled fits')
    parser.add_argument('--verify-bundled',action='store_true',
        help='Require freshly fitted maps and loss histories to agree with bundled values within numerical tolerance')
    args=parser.parse_args(argv)
    require(args.fit_dir is None or args.stage=='pilot','--fit-dir is only available with --stage pilot')
    require(not args.verify_bundled or args.stage!='pilot','--verify-bundled requires a fresh fitting stage')
    require(not args.output_dir.exists() or not any(args.output_dir.iterdir()),
        'Choose a fresh output directory; existing preparation evidence is never overwritten')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    initialize_runtime()
    fit_dir=args.fit_dir or ASSETS
    if args.stage in ('fit','all'):
        fit_dir=reproduce_fits(args.output_dir/'fits',verify_bundled=args.verify_bundled)
    if args.stage in ('pilot','all'):
        result=run_pilot(args.output_dir/'calibration',fit_dir=fit_dir)
        bundled=ASSETS/'matched_calibration.json'
        if bundled.exists():
            verification=compare_calibrations(result,read(bundled))
            write(args.output_dir/'calibration'/'bundled_calibration_check.json',verification)
            if args.verify_bundled:
                require(verification['passed'],'Calibration differs from bundled selections/scores; inspect bundled_calibration_check.json')
        else:
            write(args.output_dir/'calibration'/'bundled_calibration_check.json',dict(status='no_bundled_calibration_yet',
                scope='Preparation completed; no frozen matched calibration asset was available to compare.'))
    print(f'Preparation saved to {args.output_dir.resolve()}',flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())

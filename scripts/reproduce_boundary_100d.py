"""Separate equal-time test of direct rejection versus reflected-chord BW."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import reproduce_scs_100d as comparison

METHODS = ('DCS-BW', 'DCS-BW-reflect', 'SCS')


def configuration(preset='quick', workers=1, overrides=None):
    config = comparison.configuration(preset, workers, overrides)
    config.update(experiment='skew-t-100d-boundary', protocol='timed-boundary-dcs-scs-v3',
        implementation_revision='jax-dcs-chord-reflection-v1', algorithms=list(METHODS),
        step_policy='separate_method_specific_pretuning_then_frozen',
        step_tuning_policies={'DCS-BW-reflect': 'ball_esjd_pilot_v1'},
        boundary_comparison='Standard direct-rejection BW, reflected-chord BW, and unmodified author SCS',
        reflection_note='Reflection along a ball chord preserves proposal reversibility; '
                        'this is a new kernel, not the literal spherical SCS step-out.',
        tuning_note='Ordinary BW and SCS retain their main-comparison acceptance-target 0.234 calibration. '
                    'Reflected BW uses its existing warmup transition budget for up to eight rounds of ball-movement '
                    'pilots at h/2, h and 2h, rotating candidate order. Each round selects the largest observed '
                    'mean squared displacement inside the ball; rejected moves contribute zero. '
                    'The selected step is frozen before sampling. There is no reflected-only ceiling of 2; '
                    'the shared numerical log-step guard of 15 remains. This finite pilot search is not '
                    'a claim of optimal tuning or the same tuning objective for all methods.')
    return config


def main(argv=None):
    return comparison.main(argv, configuration_factory=configuration)


if __name__ == '__main__':
    if os.name == 'nt':
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, 'reconfigure'):
                stream.reconfigure(encoding='utf-8', errors='backslashreplace')
    from scripts.reproduce_reported_study import main as current_main
    raise SystemExit(current_main())

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
    config.update(experiment='skew-t-100d-boundary', protocol='timed-boundary-dcs-scs-v2',
        implementation_revision='jax-dcs-chord-reflection-v1', algorithms=list(METHODS),
        step_scale_ceilings={'DCS-BW-reflect': 2.},
        boundary_comparison='Standard direct-rejection BW, reflected-chord BW, and unmodified author SCS',
        reflection_note='Reflection along a ball chord preserves proposal reversibility; '
                        'this is a new kernel, not the literal spherical SCS step-out.',
        tuning_note='All methods use separate acceptance-target 0.234 calibration. Ordinary BW and SCS '
                    'retain the main comparison tuning, including its original log-step ceiling of 15. '
                    'Only reflected BW has an explicit step ceiling of 2. '
                    'Reflection can make that target unattainable; scale-ceiling saturation is recorded.')
    return config


def main(argv=None):
    return comparison.main(argv, configuration_factory=configuration)


if __name__ == '__main__':
    if os.name == 'nt':
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, 'reconfigure'):
                stream.reconfigure(encoding='utf-8', errors='backslashreplace')
    raise SystemExit(main())

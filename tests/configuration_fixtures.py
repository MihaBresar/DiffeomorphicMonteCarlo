"""Fixed synthetic inputs for numerical regression tests, outside the catalogue."""
from pathlib import Path
from unittest.mock import patch

from dcs_bench import engine


def load_config(name, preset='quick', overrides=None):
    if name == 'skew-t-100d-fixed':
        with patch.object(engine, 'ROOT', Path(__file__).parent / 'fixtures'):
            return engine.load_config(name, preset, overrides)
    return engine.load_config(name, preset, overrides)

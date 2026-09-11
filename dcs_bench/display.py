"""Readable numerical labels without modifying saved numerical results."""
from numbers import Integral, Real
import math


def number(value, *, scientific=False, integer=False):
    """Two decimals for estimates; exact integer counts; retain tiny nonzeros."""
    if value is None:
        return 'unavailable'
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if not isinstance(value, Real):
        return str(value)
    if not math.isfinite(value):
        return 'unavailable'
    if integer and float(value).is_integer():
        return str(int(value))
    if isinstance(value, Integral) and not scientific:
        return str(value)
    if scientific or (0 < abs(value) < .01):
        return f'{value:.2e}'
    return f'{value:.2f}'


def cell(value, key=''):
    """Format an HTML cell, including nested values, without rounding data."""
    if isinstance(value, dict):
        return '{'+', '.join(str(k)+': '+cell(v, str(k)) for k, v in value.items())+'}'
    if isinstance(value, (list, tuple)) or hasattr(value, 'tolist') and getattr(value, 'ndim', 0) > 0:
        return '['+', '.join(cell(v, key) for v in value)+']'
    if hasattr(value, 'item'):
        value = value.item()
    scientific = any(part in key.lower() for part in ('mse', 'rmse', 'mcse', 'bias', 'variance', 'ci95', 'bootstrap', 'difference'))
    count = any(part in key.lower() for part in ('iterations', 'retained', 'samples', 'chains', 'draws', 'count'))
    return number(value, scientific=scientific, integer=count)


def figure_ticks(fig):
    """Format ordinary numeric axes; preserve custom and logarithmic ticks."""
    from matplotlib.ticker import ScalarFormatter, FuncFormatter
    for ax in fig.axes:
        for axis, label in ((ax.xaxis, ax.get_xlabel()), (ax.yaxis, ax.get_ylabel())):
            if isinstance(axis.get_major_formatter(), ScalarFormatter):
                count = any(word in label.lower() for word in ('iterations', 'chain number', 'samples', 'retained values', 'draws'))
                axis.set_major_formatter(FuncFormatter(lambda value, position, count=count: number(value, integer=count)))

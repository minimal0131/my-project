from .base import Strategy
from .breakout import BreakoutStrategy
from .mean_reversion import MeanReversionStrategy
from .pullback import PullbackStrategy

REGISTRY: dict[str, type[Strategy]] = {
    "pullback": PullbackStrategy,
    "breakout": BreakoutStrategy,
    "mean_reversion": MeanReversionStrategy,
}


def build(name: str, params: dict) -> Strategy:
    return REGISTRY[name](params)

"""Green AI: carbon footprint estimator for LLMs, with prompt processing and per task accounting."""

from ._version import __version__
from .engine import DEFAULT_HARDWARE, ElectricityMix, Hardware, Impacts
from .estimator import Estimate, Estimator, default_estimator, estimate, register_model
from .prefill import CENTRAL, LOW, NONE, PRICING, PrefillModel
from .ranges import Range
from .registry import Match, MatchStatus, ModelNotFoundError, ModelSpec, Registry, default_registry
from .tracker import Tracker

__all__ = [
    "CENTRAL",
    "DEFAULT_HARDWARE",
    "ElectricityMix",
    "Estimate",
    "Estimator",
    "Hardware",
    "Impacts",
    "LOW",
    "Match",
    "MatchStatus",
    "ModelNotFoundError",
    "ModelSpec",
    "NONE",
    "PRICING",
    "PrefillModel",
    "Range",
    "Registry",
    "Tracker",
    "default_estimator",
    "default_registry",
    "estimate",
    "register_model",
    "__version__",
]

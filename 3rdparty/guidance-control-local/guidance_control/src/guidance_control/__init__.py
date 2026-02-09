"""Guidance Control - Portable LLM decoding controllers.

AICI-style callback-based protocol for portable LLM decoding algorithms.
Controllers determine decoding strategy while inference engines handle execution.
"""

# Controller implementations
from .controllers import (
    BeamCandidate,
    BeamSearchController,
    BestOfNController,
    Candidate,
    CustomController,
    Particle,
    SMCController,
)

# Core protocol
from .protocol import Controller

# Type definitions
from .types import PreProcessResult, SequenceId, TokenConstraint

__version__ = "0.0.1"

__all__ = [
    "BeamCandidate",
    "BeamSearchController",
    "BestOfNController",
    "CustomController",
    "Candidate",
    "Controller",
    "Particle",
    "PreProcessResult",
    "SMCController",
    "SequenceId",
    "TokenConstraint",
]

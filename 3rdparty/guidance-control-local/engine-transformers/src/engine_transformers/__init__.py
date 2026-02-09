"""Engine-Transformers: Reference engine using HuggingFace transformers.

A reference implementation showing how inference engines integrate with the
guidance_control controller protocol.
"""

from .engine import TransformersEngine
from .sequence import SequenceManager, SequenceState

__version__ = "0.0.1"

__all__ = [
    "SequenceManager",
    "SequenceState",
    "TransformersEngine",
]

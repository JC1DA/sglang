"""Core type definitions for the AICI-style controller protocol.

This module defines the callback result types and constraints used throughout
the guidance_control package. All types are immutable dataclasses for type safety
without runtime overhead.
"""

from dataclasses import dataclass
from typing import Literal, TypeAlias

TokenId: TypeAlias = int
SequenceId: TypeAlias = str | int


@dataclass(frozen=True)
class PreProcessResult:
    """Result from pre_process callback indicating what to do with a sequence.

    Attributes:
        action: What action to take - "continue", "fork", or "stop"
        num_forks: Number of child sequences to create (only used when action="fork")
    """

    action: Literal["continue", "fork", "stop"]
    num_forks: int = 1
    extra_tokens: list[TokenId] | None = None

    def __post_init__(self) -> None:
        """Validate that num_forks is reasonable when forking."""
        if self.action == "fork" and self.num_forks < 1:
            raise ValueError(
                f"num_forks must be >= 1 when forking, got {self.num_forks}"
            )


# Token constraint types


@dataclass(frozen=True)
class TokenConstraint:
    """Constraints to apply to token selection before sampling.

    Attributes:
        allowed_tokens: Set of allowed token IDs. None means all tokens allowed.
        logit_bias: Additional bias to add to specific token logits.
        temperature: Sampling temperature to use. Default 1.0.
    """

    allowed_tokens: set[int] | None = None
    logit_bias: dict[int, float] | None = None
    temperature: float = 1.0

    @staticmethod
    def unconstrained() -> "TokenConstraint":
        """Create unconstrained token constraint (allows all tokens).

        Returns:
            TokenConstraint with no restrictions
        """
        return TokenConstraint()

    def __post_init__(self) -> None:
        """Validate temperature is positive."""
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")


@dataclass(frozen=True)
class TokenSplice:
    """
    Represents a splice operation on a token sequence, allowing backtracking and token replacement.

    Attributes:
        backtrack: Number of tokens to remove from the end of the current sequence.
        ff_tokens: Tokens to append after backtracking.
    """

    backtrack: int
    ff_tokens: tuple[TokenId, ...]


__all__ = [
    "PreProcessResult",
    "SequenceId",
    "TokenConstraint",
]

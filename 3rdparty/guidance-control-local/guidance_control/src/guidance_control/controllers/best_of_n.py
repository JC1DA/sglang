"""Best-of-N controller implementation.

This module implements best-of-N sampling, which generates N independent
candidate completions in parallel and selects the best according to a
scoring function.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ..types import PreProcessResult, SequenceId, TokenConstraint, TokenId, TokenSplice


@dataclass
class Candidate:
    """Internal representation of a best-of-N candidate.

    Attributes:
        sequence_id: Unique identifier for this sequence
        tokens: Generated token IDs
        score: Score assigned by scoring function (None until finished)
    """

    sequence_id: SequenceId
    tokens: list[int]
    score: float | None = None


class BestOfNController:
    """Generate N independent candidates and select the best.

    This controller forks the initial sequence into N parallel sequences,
    allows them all to complete, scores each one, and selects the best.

    Useful for scenarios where you want diversity in generation followed
    by selection (e.g., best-of-N sampling for improved quality).

    Example:
        >>> def score_fn(tokens):
        ...     # Custom scoring (e.g., reward model, length, etc.)
        ...     return -len(tokens)  # Prefer shorter sequences
        >>> controller = BestOfNController(
        ...     n=5,
        ...     scorer=score_fn,
        ...     eos_token_id=tokenizer.eos_token_id
        ... )
        >>> # Engine will call: init(), pre_process(), mid_process(), post_process()
        >>> # After generation:
        >>> best_seq_id = controller.get_best_sequence_id()
        >>> best_candidate = controller.get_best()
    """

    def __init__(
        self,
        n: int,
        scorer: Callable[[list[int]], float] | None = None,
        max_length: int = 1024,
        eos_token_id: int | None = None,
    ) -> None:
        """Initialize best-of-N controller.

        Args:
            n: Number of independent candidates to generate
            scorer: Function that takes token list and returns score (higher is better).
                   Defaults to sequence length (longer is better).
            max_length: Maximum sequence length before forced termination
            eos_token_id: Token ID that marks end of sequence. If None, sequences
                         only finish at max_length.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if max_length < 1:
            raise ValueError(f"max_length must be >= 1, got {max_length}")

        self.n = n
        self.scorer = scorer or self._default_scorer
        self.max_length = max_length
        self.eos_token_id = eos_token_id

        # Internal state - maintained by controller
        self._prompt_tokens: list[int] = []
        self._candidates: dict[SequenceId, Candidate] = {}
        self._finished: list[Candidate] = []
        self._initialized = False
        self._best: Candidate | None = None

    def init(
        self, seq_id: SequenceId, prompt_tokens: Sequence[TokenId]
    ) -> Sequence[TokenId]:
        """Initialize controller with initial sequence ID and prompt tokens.

        Args:
            seq_id: The sequence ID assigned by the engine for the initial sequence
            prompt_tokens: Token IDs from the initial prompt

        Returns:
            The potentially modified sequence of prompt tokens
        """
        self._prompt_tokens = list(prompt_tokens)
        self._candidates[seq_id] = Candidate(
            sequence_id=seq_id,
            tokens=self._prompt_tokens.copy(),
            score=None,
        )
        self._initialized = False
        return self._prompt_tokens

    def pre_process(self, seq_id: SequenceId) -> PreProcessResult:
        """Decide what to do with a sequence before token generation.

        Args:
            seq_id: Sequence identifier

        Returns:
            PreProcessResult indicating fork/continue/stop action
        """
        # First call: fork into N candidates
        if not self._initialized:
            self._initialized = True
            return PreProcessResult(action="fork", num_forks=self.n)

        # Check if this candidate exists
        if seq_id not in self._candidates:
            # Unknown sequence
            return PreProcessResult(action="stop")

        candidate = self._candidates[seq_id]

        # Stop if we've scored this candidate (it's finished)
        if candidate.score is not None:
            return PreProcessResult(action="stop")

        # Stop if we've reached max length
        if len(candidate.tokens) >= self.max_length:
            # Will be scored in post_process
            return PreProcessResult(action="stop")

        # Continue generating
        return PreProcessResult(action="continue")

    def mid_process(self, seq_id: SequenceId) -> TokenConstraint:  # noqa: ARG002
        """Apply token constraints before sampling.

        Best-of-N uses unconstrained sampling for diversity.

        Args:
            seq_id: Sequence identifier

        Returns:
            TokenConstraint (unconstrained for diversity)
        """
        # No constraints - we want diverse candidates
        return TokenConstraint.unconstrained()

    def post_fork(self, parent_id: SequenceId, child_ids: Sequence[SequenceId]) -> None:
        """Track forked child sequences.

        Args:
            parent_id: The sequence ID that was forked
            child_ids: The new sequence IDs created from the fork
        """
        if parent_id not in self._candidates:
            return

        parent = self._candidates[parent_id]

        for child_id in child_ids:
            if child_id not in self._candidates:
                self._candidates[child_id] = Candidate(
                    sequence_id=child_id,
                    tokens=parent.tokens.copy(),
                    score=None,
                )

    def post_process(
        self,
        seq_id: SequenceId,
        token: int,
        logprob: float,  # noqa: ARG002
        correction: float = 0.0,  # noqa: ARG002
    ) -> TokenSplice:
        """Update internal state after a token is sampled.

        Args:
            seq_id: Sequence identifier
            token: Token ID that was sampled
            logprob: Log probability of the sampled token (unused in best-of-N)
            correction: Log correction factor for constraint perturbation (unused in best-of-N)
        Returns:
            TokenSplice indicating how to splice the new token into the sequence
        """
        candidate = self._candidates[seq_id]

        # Append new token
        candidate.tokens.append(token)

        # Check if this sequence should finish
        should_finish = False

        # Check for EOS token
        if self.eos_token_id is not None and token == self.eos_token_id:
            should_finish = True

        # Check max length
        if len(candidate.tokens) >= self.max_length:
            should_finish = True

        # If finished, score it
        if should_finish and candidate.score is None:
            candidate.score = self.scorer(candidate.tokens)
            self._finished.append(candidate)

            # Select best if all candidates are done
            if len(self._finished) == self.n and self._best is None:
                self._best = max(
                    self._finished,
                    key=lambda c: c.score if c.score is not None else float("-inf"),
                )
        return TokenSplice(backtrack=0, ff_tokens=(token,))

    def is_complete(self) -> bool:
        """Check if all N candidates are finished.

        Returns:
            True if all N sequences have completed
        """
        return len(self._finished) == self.n

    def get_best_sequence_id(self) -> SequenceId | None:
        """Get the sequence ID of the best result.

        Returns:
            The sequence ID with highest score, or None if not yet complete
        """
        if self._best is None:
            return None
        return self._best.sequence_id

    def get_best(self) -> Candidate | None:
        """Get the best candidate after completion.

        Returns:
            The highest-scoring candidate, or None if not yet complete
        """
        return self._best

    def get_all_candidates(self) -> list[Candidate]:
        """Get all finished candidates sorted by score.

        Returns:
            List of candidates sorted by score (descending)
        """
        return sorted(
            self._finished,
            key=lambda c: c.score if c.score is not None else float("-inf"),
            reverse=True,
        )

    def _default_scorer(self, tokens: list[int]) -> float:
        """Default scorer: prefer longer sequences.

        Args:
            tokens: Token IDs for the sequence

        Returns:
            Score based on sequence length
        """
        return float(len(tokens))


__all__ = ["BestOfNController", "Candidate"]

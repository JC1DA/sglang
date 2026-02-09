"""Beam search controller implementation.

This module implements classic beam search decoding, maintaining the top-k
sequences by cumulative log probability and pruning lower-scoring branches.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from ..types import PreProcessResult, SequenceId, TokenConstraint, TokenId, TokenSplice


@dataclass
class BeamCandidate:
    """Internal representation of a beam search candidate.

    Attributes:
        sequence_id: Unique identifier for this sequence
        score: Cumulative log probability of the sequence
        tokens: Generated token IDs so far
        finished: Whether this sequence has reached EOS
    """

    sequence_id: SequenceId
    score: float
    tokens: list[int]
    finished: bool = False


class BeamSearchController:
    """Classic beam search with configurable width and length penalty.

    Maintains top-k sequences by cumulative log probability, pruning lower-scoring
    sequences at each step. Supports length normalization and early stopping.

    Example:
        >>> controller = BeamSearchController(
        ...     beam_width=4,
        ...     length_penalty=0.6,
        ...     eos_token_id=tokenizer.eos_token_id
        ... )
        >>> # Engine will call: init(), pre_process(), mid_process(), post_process()
        >>> # After generation:
        >>> best_seq_id = controller.get_best_sequence_id()
        >>> all_results = controller.get_results()
    """

    def __init__(
        self,
        beam_width: int,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
        eos_token_id: int | None = None,
    ) -> None:
        """Initialize beam search controller.

        Args:
            beam_width: Number of top sequences to maintain (k in top-k)
            length_penalty: Exponent for length normalization. Default 1.0 (no penalty).
                          Values < 1.0 favor longer sequences, > 1.0 favor shorter.
            early_stopping: If True, stop when beam_width sequences are finished.
                          If False, continue until all active sequences finish.
            eos_token_id: Token ID that marks end of sequence. If None, sequences
                         won't finish early.
        """
        if beam_width < 1:
            raise ValueError(f"beam_width must be >= 1, got {beam_width}")

        self.beam_width = beam_width
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.eos_token_id = eos_token_id

        # Internal state - maintained by controller
        self._prompt_tokens: list[int] = []
        self._candidates: dict[SequenceId, BeamCandidate] = {}
        self._finished: list[BeamCandidate] = []
        self._initialized = False

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
        self._candidates[seq_id] = BeamCandidate(
            sequence_id=seq_id,
            tokens=self._prompt_tokens.copy(),
            score=0.0,
            finished=False,
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
        # First call: fork initial sequence into beam_width candidates
        if not self._initialized:
            self._initialized = True
            return PreProcessResult(action="fork", num_forks=self.beam_width)

        # Check if this sequence should be stopped
        if seq_id not in self._candidates:
            # Unknown sequence - should not happen
            return PreProcessResult(action="stop")

        candidate = self._candidates[seq_id]

        # Stop finished sequences
        if candidate.finished:
            return PreProcessResult(action="stop")

        # Continue active sequences
        return PreProcessResult(action="continue")

    def mid_process(self, seq_id: SequenceId) -> TokenConstraint:  # noqa: ARG002
        """Apply token constraints before sampling.

        Beam search uses greedy/sampling without hard constraints.

        Args:
            seq_id: Sequence identifier

        Returns:
            TokenConstraint (unconstrained for basic beam search)
        """
        # Beam search doesn't constrain tokens - just samples and prunes later
        return TokenConstraint.unconstrained()

    def post_fork(self, parent_id: SequenceId, child_ids: Sequence[SequenceId]) -> None:
        """Track forked child sequences.

        Args:
            parent_id: The sequence ID that was forked
            child_ids: The new sequence IDs created from the fork
        """
        if parent_id not in self._candidates:
            return  # Defensive - shouldn't happen

        parent = self._candidates[parent_id]

        # Create child candidates
        for child_id in child_ids:
            if child_id not in self._candidates:
                self._candidates[child_id] = BeamCandidate(
                    sequence_id=child_id,
                    tokens=parent.tokens.copy(),
                    score=parent.score,
                    finished=False,
                )

    def post_process(
        self,
        seq_id: SequenceId,
        token: int,
        logprob: float,
        correction: float = 0.0,  # noqa: ARG002
    ) -> TokenSplice:
        """Update internal state after a token is sampled.

        Args:
            seq_id: Sequence identifier
            token: Token ID that was sampled
            logprob: Log probability of the sampled token
            correction: Log correction factor for constraint perturbation (unused in beam search)
        Returns:
            TokenSplice indicating how to splice the new token into the sequence
        """
        candidate = self._candidates[seq_id]

        # Incremental update - much more efficient than recalculation
        candidate.tokens.append(token)
        candidate.score += logprob

        # Check for EOS token
        if self.eos_token_id is not None and token == self.eos_token_id:
            candidate.finished = True
            self._finished.append(candidate)

        # Prune worst candidates if we have too many
        self._prune_if_needed()
        return TokenSplice(backtrack=0, ff_tokens=(token,))

    def _prune_if_needed(self) -> None:
        """Prune lowest-scoring candidates to maintain beam_width."""
        # Only prune active (non-finished) candidates
        active_candidates = {
            seq_id: cand
            for seq_id, cand in self._candidates.items()
            if not cand.finished
        }

        if len(active_candidates) > self.beam_width:
            # Calculate normalized scores for ranking
            scored = [
                (
                    seq_id,
                    (
                        candidate.score / (len(candidate.tokens) ** self.length_penalty)
                        if len(candidate.tokens) > 0
                        else candidate.score
                    ),
                )
                for seq_id, candidate in active_candidates.items()
            ]

            # Sort by normalized score (descending)
            scored.sort(key=lambda x: x[1], reverse=True)

            # Keep only top beam_width candidates
            to_keep = {seq_id for seq_id, _ in scored[: self.beam_width]}

            # Remove pruned candidates
            for seq_id in list(self._candidates.keys()):
                if seq_id not in to_keep and not self._candidates[seq_id].finished:
                    del self._candidates[seq_id]

    def is_complete(self) -> bool:
        """Check if beam search is complete.

        Returns:
            True if done (beam_width finished sequences or no active sequences)
        """
        if self.early_stopping:
            # Stop when we have enough finished sequences
            return len(self._finished) >= self.beam_width

        # Otherwise, continue until no active sequences remain
        active_count = sum(1 for c in self._candidates.values() if not c.finished)
        return active_count == 0

    def get_best_sequence_id(self) -> SequenceId | None:
        """Get the sequence ID of the best result.

        Returns:
            The sequence ID with highest normalized score, or None if no sequences finished
        """
        if not self._finished:
            return None

        # Find candidate with highest normalized score
        best = max(
            self._finished,
            key=lambda c: (
                c.score / (len(c.tokens) ** self.length_penalty)
                if len(c.tokens) > 0
                else c.score
            ),
        )
        return best.sequence_id

    def get_results(self) -> list[BeamCandidate]:
        """Get finished sequences sorted by normalized score.

        Returns:
            List of finished beam candidates, best-scoring first
        """
        # Sort by normalized score (descending)
        return sorted(
            self._finished,
            key=lambda c: (
                c.score / (len(c.tokens) ** self.length_penalty)
                if len(c.tokens) > 0
                else c.score
            ),
            reverse=True,
        )


__all__ = ["BeamCandidate", "BeamSearchController"]

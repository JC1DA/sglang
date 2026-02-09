"""AICI-style callback-based controller protocol definition.

This module defines the core Controller protocol using a three-phase callback
system: pre_process, mid_process, and post_process. Controllers are event-driven
and maintain internal state rather than receiving full state from the engine.
"""

from collections.abc import Sequence
from typing import Protocol

from .types import PreProcessResult, SequenceId, TokenConstraint, TokenId, TokenSplice


class Controller(Protocol):
    """Protocol for AICI-style callback-based controllers.

    Controllers are stateful objects that respond to events during token generation.
    The engine calls controller methods at three points in the generation cycle:

    1. **pre_process**: Before token generation - decide whether to continue, fork, or stop
    2. **mid_process**: During token generation - apply constraints to token selection
    3. **post_process**: After token sampling - update internal state

    Controllers maintain their own internal state and only receive incremental
    updates (new tokens, logprobs) rather than full sequence history.

    Example:
        >>> class MyController:
        ...     def init(self, seq_id, prompt_tokens):
        ...         self.tokens = prompt_tokens.copy()
        ...         self.seq_id = seq_id
        ...         self.finished_seq_id = None
        ...         return self.tokens
        ...
        ...     def pre_process(self, seq_id):
        ...         return PreProcessResult(action="continue")
        ...
        ...     def mid_process(self, seq_id):
        ...         return TokenConstraint.unconstrained()
        ...
        ...     def post_process(self, seq_id, token, logprob, correction=0.0):
        ...         self.tokens.append(token)
        ...         if len(self.tokens) >= 100:
        ...             self.finished_seq_id = seq_id
        ...         return TokenSplice(backtrack=0, ff_tokens=(token,))
        ...
        ...     def post_fork(self, parent_id, child_ids):
        ...         # Track forked sequences
        ...         pass
        ...
        ...     def is_complete(self):
        ...         return len(self.tokens) >= 100
        ...
        ...     def get_best_sequence_id(self):
        ...         return self.finished_seq_id
    """

    def init(
        self, seq_id: SequenceId, prompt_tokens: Sequence[TokenId]
    ) -> Sequence[TokenId]:
        """Initialize controller with initial sequence ID and prompt tokens.

        Called once at the start of generation before any callbacks.
        The controller can modify the prompt tokens before generation begins.

        Args:
            seq_id: The sequence ID assigned by the engine for the initial sequence
            prompt_tokens: The tokenized input prompt

        Returns:
            The potentially modified sequence of prompt tokens
        """
        ...

    def pre_process(self, seq_id: SequenceId) -> PreProcessResult:
        """Called before token generation for each sequence.

        The controller decides what action to take with this sequence:
        - "continue": Generate next token normally
        - "fork": Duplicate into multiple child sequences
        - "stop": Terminate this sequence

        This is the main control point for beam search, tree search, and
        early stopping logic.

        Args:
            seq_id: ID of the sequence to process

        Returns:
            PreProcessResult indicating action to take
        """
        ...

    def mid_process(self, seq_id: SequenceId) -> TokenConstraint:
        """Called during token generation to apply constraints.

        The controller can constrain which tokens are allowed before sampling.
        This enables:
        - Format constraints (JSON, XML, regex)
        - Grammar constraints (context-free grammars)
        - Vocabulary restrictions
        - Logit biasing for guidance

        Runs in parallel with GPU generation (on CPU) for zero overhead.

        Args:
            seq_id: ID of the sequence being generated

        Returns:
            TokenConstraint specifying allowed tokens, biases, and temperature
        """
        ...

    def post_process(
        self, seq_id: SequenceId, token: int, logprob: float, correction: float = 0.0
    ) -> TokenSplice:
        """Called after token is sampled.

        The controller updates its internal state with the newly generated token.
        This is where controllers:
        - Append tokens to internal history
        - Update cumulative scores (correcting for constraint bias if needed)
        - Check for completion conditions
        - Prune candidates

        Args:
            seq_id: ID of the sequence
            token: The sampled token ID
            logprob: Log probability of the sampled token from the perturbed distribution
            correction: Log correction factor to account for constraint perturbation.
                       When constraints are applied (masking, logit bias), we sample from
                       a perturbed distribution p'(x) instead of the original p(x).
                       The correction factor is: log(Z'/Z) where Z and Z' are the
                       partition functions before/after constraints. This equals
                       log(sum_x p(x) * e^bias(x)). Algorithms like SMC use this to
                       maintain proper importance weights. Default 0.0 when no constraints.
        Returns:
            TokenSplice specifying what to do with after accepting the token.
                - Attributes:
                    backtrack:  number of tokens to backtrack (0 means no backtrack, 1 means remove last token, etc.)
                    ff_tokens: tuple of tokens to append after backtracking (empty tuple means no additional tokens)
                - Typically, expect backtrack=0 and ff_tokens=(sampled_token,) for most use cases.
        """
        ...

    def post_fork(self, parent_id: SequenceId, child_ids: Sequence[SequenceId]) -> None:
        """Called after sequences are forked to communicate new child IDs.

        The engine calls this immediately after creating child sequences from a fork
        operation, allowing the controller to track genealogy and maintain proper
        lineage information.

        Args:
            parent_id: The sequence ID that was forked
            child_ids: The new sequence IDs created from the fork (in order)
        """
        ...

    def is_complete(self) -> bool:
        """Check if controller has finished all work.

        Returns:
            True if all sequences are finished/stopped and no more work needed.
            False if there are still active sequences or pending operations.
        """
        ...

    def get_best_sequence_id(self) -> SequenceId | None:
        """Get the sequence ID of the best/main result.

        Called after generation completes to retrieve the primary result.
        Different controllers use different criteria:
        - Beam search: highest normalized score
        - Best-of-N: highest scorer result
        - SMC: highest weight particle

        Returns:
            The sequence ID of the best result, or None if no sequences finished
        """
        ...


__all__ = ["Controller"]

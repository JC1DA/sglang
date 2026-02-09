"""Sequence state management for the transformers engine.

This module provides classes for managing sequence states, including tokens,
log probabilities, and KV caches during generation.
"""

import copy
from dataclasses import dataclass

from guidance_control import SequenceId


@dataclass
class SequenceState:
    """State for a single sequence during generation.

    Attributes:
        seq_id: Unique identifier for this sequence
        tokens: List of generated token IDs
        past_key_values: KV cache from transformers (tuple of tensors)
        finished: Whether this sequence has reached a stopping condition
    """

    seq_id: SequenceId
    tokens: list[int]
    past_key_values: tuple | None
    finished: bool


class SequenceManager:
    """Manages all active sequences and their states.

    Handles creation, forking, and pruning of sequences. Maintains KV caches
    and provides state information to controllers.
    """

    def __init__(self) -> None:
        """Initialize an empty sequence manager."""
        self.sequences: dict[SequenceId, SequenceState] = {}
        self._next_id = 0

    def create_sequence(
        self, tokens: list[int], past_key_values: tuple | None = None
    ) -> SequenceId:
        """Create a new sequence.

        Args:
            tokens: Initial token IDs (typically the prompt)
            past_key_values: Optional KV cache from previous generation

        Returns:
            The new sequence ID
        """
        seq_id = self._next_id
        self._next_id += 1

        self.sequences[seq_id] = SequenceState(
            seq_id=seq_id,
            tokens=tokens,
            past_key_values=past_key_values,
            finished=False,
        )

        return seq_id

    def fork_sequence(self, parent_id: SequenceId, num_forks: int) -> list[SequenceId]:
        """Fork a sequence into multiple children.

        Creates num_forks copies of the parent sequence, each with an
        independent copy of the KV cache.

        Args:
            parent_id: The sequence to fork
            num_forks: Number of child sequences to create

        Returns:
            List of new child sequence IDs
        """
        if parent_id not in self.sequences:
            raise ValueError(f"Sequence {parent_id} not found")

        parent = self.sequences[parent_id]
        child_ids: list[SequenceId] = []

        for _ in range(num_forks):
            child_id = self._next_id
            self._next_id += 1

            # Deep copy the KV cache to ensure independence
            past_key_values_copy = (
                copy.deepcopy(parent.past_key_values)
                if parent.past_key_values is not None
                else None
            )

            self.sequences[child_id] = SequenceState(
                seq_id=child_id,
                tokens=parent.tokens.copy(),
                past_key_values=past_key_values_copy,
                finished=False,
            )

            child_ids.append(child_id)

        return child_ids

    def prune_sequence(self, seq_id: SequenceId) -> None:
        """Remove a sequence and free its resources.

        Args:
            seq_id: The sequence to remove
        """
        if seq_id in self.sequences:
            # Delete KV cache to free memory
            seq = self.sequences[seq_id]
            seq.past_key_values = None
            del self.sequences[seq_id]

    def finish_sequence(self, seq_id: SequenceId) -> None:
        """Mark a sequence as finished.

        Args:
            seq_id: The sequence to finish
        """
        if seq_id in self.sequences:
            self.sequences[seq_id].finished = True

    def get_finished_sequences(self) -> list[SequenceState]:
        """Get all finished sequences.

        Returns:
            List of finished sequence states
        """
        return [seq for seq in self.sequences.values() if seq.finished]


__all__ = ["SequenceManager", "SequenceState"]

"""Sequential Monte Carlo (SMC) controller implementation.

This module implements Sequential Monte Carlo (particle filter) decoding,
which maintains a population of weighted particles and performs resampling
when the effective sample size drops below a threshold.
"""

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ..types import PreProcessResult, SequenceId, TokenConstraint, TokenId, TokenSplice


@dataclass
class Particle:
    """Internal representation of an SMC particle.

    Attributes:
        sequence_id: Unique identifier for this sequence
        tokens: Generated token IDs
        log_weight: Log of importance weight
        ancestor_id: Parent particle ID (for tracking lineage)
        finished: Whether this particle has reached a stopping condition
    """

    sequence_id: SequenceId
    tokens: list[int]
    log_weight: float
    ancestor_id: SequenceId | None = None
    finished: bool = False


class SMCController:
    """Sequential Monte Carlo (particle filter) controller.

    Maintains a population of particles with importance weights. Performs
    resampling when the effective sample size (ESS) drops below a threshold,
    pruning low-weight particles and forking high-weight ones.

    SMC is useful for maintaining diversity while focusing computational
    resources on promising sequences, particularly in constrained generation
    or guided decoding scenarios.

    Example:
        >>> def weight_fn(tokens):
        ...     # Custom weighting (e.g., constraint satisfaction)
        ...     return 0.0  # Uniform weights
        >>> controller = SMCController(
        ...     num_particles=10,
        ...     resample_threshold=0.5,
        ...     weight_function=weight_fn,
        ...     eos_token_id=tokenizer.eos_token_id
        ... )
        >>> # Engine will call: init(), pre_process(), mid_process(), post_process()
        >>> # After generation:
        >>> best_seq_id = controller.get_best_sequence_id()
        >>> all_particles = controller.get_particles()
    """

    def __init__(
        self,
        num_particles: int,
        resample_threshold: float = 0.5,
        weight_function: Callable[[list[int]], float] | None = None,
        max_length: int = 1024,
        eos_token_id: int | None = None,
    ) -> None:
        """Initialize SMC controller.

        Args:
            num_particles: Number of particles to maintain
            resample_threshold: Resample when ESS < threshold * num_particles.
                              Range: [0, 1]. Lower values resample more aggressively.
            weight_function: Function that takes tokens and returns log weight.
                           Defaults to uniform weighting (log weight = 0).
            max_length: Maximum sequence length before termination
            eos_token_id: Token ID that marks end of sequence. If None, particles
                         only finish at max_length.
        """
        if num_particles < 1:
            raise ValueError(f"num_particles must be >= 1, got {num_particles}")
        if not 0 <= resample_threshold <= 1:
            raise ValueError(
                f"resample_threshold must be in [0, 1], got {resample_threshold}"
            )
        if max_length < 1:
            raise ValueError(f"max_length must be >= 1, got {max_length}")

        self.num_particles = num_particles
        self.resample_threshold = resample_threshold
        self.weight_function = weight_function or self._default_weight
        self.max_length = max_length
        self.eos_token_id = eos_token_id

        # Internal state - maintained by controller
        self._prompt_tokens: list[int] = []
        self._particles: dict[SequenceId, Particle] = {}
        self._generation_step = 0
        self._initialized = False
        self._should_resample = False
        self._resample_plan: dict[SequenceId, PreProcessResult] = {}

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
        self._particles[seq_id] = Particle(
            sequence_id=seq_id,
            tokens=self._prompt_tokens.copy(),
            log_weight=0.0,
            ancestor_id=None,  # No parent for initial sequence
            finished=False,
        )
        self._initialized = False
        self._generation_step = 0
        return self._prompt_tokens

    def pre_process(self, seq_id: SequenceId) -> PreProcessResult:
        """Decide what to do with a particle before token generation.

        Args:
            seq_id: Sequence identifier

        Returns:
            PreProcessResult indicating fork/continue/stop action
        """
        # First call: fork into num_particles
        if not self._initialized:
            self._initialized = True
            return PreProcessResult(action="fork", num_forks=self.num_particles)

        # If we have a resample plan, execute it
        if self._resample_plan and seq_id in self._resample_plan:
            result = self._resample_plan[seq_id]
            # Clear this entry from the plan
            del self._resample_plan[seq_id]
            return result

        # Check if this particle exists
        if seq_id not in self._particles:
            return PreProcessResult(action="stop")

        particle = self._particles[seq_id]

        # Stop finished particles
        if particle.finished:
            return PreProcessResult(action="stop")

        # Check if we should resample (computed in post_process)
        if self._should_resample:
            # Prepare resample plan for all particles
            self._prepare_resample_plan()
            # Execute plan for this sequence
            if seq_id in self._resample_plan:
                result = self._resample_plan[seq_id]
                del self._resample_plan[seq_id]
                return result

        # Default: continue
        return PreProcessResult(action="continue")

    def mid_process(self, seq_id: SequenceId) -> TokenConstraint:  # noqa: ARG002
        """Apply token constraints before sampling.

        SMC can use weights to adjust sampling temperature.

        Args:
            seq_id: Sequence identifier

        Returns:
            TokenConstraint with adjusted temperature based on particle weight
        """
        # Could adjust temperature based on particle weight
        # For now, use unconstrained sampling
        return TokenConstraint.unconstrained()

    def post_fork(self, parent_id: SequenceId, child_ids: Sequence[SequenceId]) -> None:
        """Track forked child particles with proper genealogy.

        Args:
            parent_id: The sequence ID that was forked
            child_ids: The new sequence IDs created from the fork
        """
        if parent_id not in self._particles:
            return

        parent = self._particles[parent_id]

        for child_id in child_ids:
            if child_id not in self._particles:
                self._particles[child_id] = Particle(
                    sequence_id=child_id,
                    tokens=parent.tokens.copy(),
                    log_weight=parent.log_weight,
                    ancestor_id=parent_id,  # Track lineage!
                    finished=False,
                )

    def post_process(
        self,
        seq_id: SequenceId,
        token: int,
        logprob: float,  # noqa: ARG002
        correction: float = 0.0,
    ) -> TokenSplice:
        """Update internal state after a token is sampled.

        Args:
            seq_id: Sequence identifier
            token: Token ID that was sampled
            logprob: Log probability of the sampled token from perturbed distribution
            correction: Log correction factor to account for constraint perturbation.
                       Added to particle log weight to maintain proper importance sampling.
        Returns:
            TokenSplice indicating how to splice the new token into the sequence
        """
        particle = self._particles[seq_id]

        # Append new token
        particle.tokens.append(token)

        # Update weight using provided function and apply correction for constraint bias
        particle.log_weight = self.weight_function(particle.tokens) + correction

        # Check for stopping conditions
        should_finish = False

        # Check for EOS token
        if self.eos_token_id is not None and token == self.eos_token_id:
            should_finish = True

        # Check max length
        if len(particle.tokens) >= self.max_length:
            should_finish = True

        if should_finish:
            particle.finished = True

        # After processing all particles, check if we should resample
        # (Note: This is called once per particle per step)
        self._check_resample_needed()

        self._generation_step += 1

        return TokenSplice(
            backtrack=0,
            ff_tokens=(token,),
        )

    def _check_resample_needed(self) -> None:
        """Check if resampling is needed based on ESS."""
        # Only check if we have all particles active
        active_count = sum(1 for p in self._particles.values() if not p.finished)

        if active_count < self.num_particles:
            return

        # Calculate effective sample size
        ess = self._effective_sample_size()
        ess_threshold = self.resample_threshold * self.num_particles

        self._should_resample = ess < ess_threshold

    def _prepare_resample_plan(self) -> None:
        """Prepare resampling plan for all particles."""
        if not self._should_resample:
            return

        # Compute normalized weights
        active_particles = [
            (seq_id, p) for seq_id, p in self._particles.items() if not p.finished
        ]

        if not active_particles:
            return

        log_weights = [p.log_weight for _, p in active_particles]
        max_log_weight = max(log_weights)
        normalized_weights = [math.exp(w - max_log_weight) for w in log_weights]
        total = sum(normalized_weights)

        if total == 0:
            # All weights are zero - uniform resampling
            probs = [1.0 / len(active_particles)] * len(active_particles)
        else:
            probs = [w / total for w in normalized_weights]

        # Perform systematic resampling
        resampled_indices = self._systematic_resample(probs, self.num_particles)

        # Count how many times each particle was selected
        resampled_counts: dict[int, int] = {}
        for idx in resampled_indices:
            resampled_counts[idx] = resampled_counts.get(idx, 0) + 1

        # Generate resampling plan
        new_plan: dict[SequenceId, PreProcessResult] = {}
        particles_to_remove: list[SequenceId] = []

        for i, (seq_id, _particle) in enumerate(active_particles):
            count = resampled_counts.get(i, 0)

            if count == 0:
                # Particle not selected - stop it
                new_plan[seq_id] = PreProcessResult(action="stop")
                particles_to_remove.append(seq_id)
            elif count == 1:
                # Particle selected once - continue
                new_plan[seq_id] = PreProcessResult(action="continue")
            else:
                # Particle selected multiple times - fork it
                new_plan[seq_id] = PreProcessResult(action="fork", num_forks=count)

        # Remove pruned particles
        for seq_id in particles_to_remove:
            del self._particles[seq_id]

        self._resample_plan = new_plan
        self._should_resample = False

    def is_complete(self) -> bool:
        """Check if SMC generation is complete.

        Returns:
            True if any particle has finished
        """
        return any(p.finished for p in self._particles.values())

    def get_best_sequence_id(self) -> SequenceId | None:
        """Get the sequence ID of the best result.

        Returns:
            The sequence ID of the particle with highest weight, or None if no particles finished
        """
        finished_particles = [p for p in self._particles.values() if p.finished]
        if not finished_particles:
            return None

        # Return particle with highest log weight
        best = max(finished_particles, key=lambda p: p.log_weight)
        return best.sequence_id

    def get_particles(self) -> list[Particle]:
        """Get current particles sorted by weight.

        Returns:
            List of particles sorted by log weight (descending)
        """
        return sorted(
            self._particles.values(), key=lambda p: p.log_weight, reverse=True
        )

    def _effective_sample_size(self) -> float:
        """Compute effective sample size from particle weights.

        ESS measures how well the particle weights are distributed.
        ESS = (sum weights)^2 / (sum weights^2), normalized to [0, num_particles].

        Returns:
            Effective sample size in range [0, num_particles]
        """
        active_particles = [p for p in self._particles.values() if not p.finished]

        if not active_particles:
            return 0.0

        log_weights = [p.log_weight for p in active_particles]
        max_log_weight = max(log_weights)

        # Normalize in log space for numerical stability
        normalized_weights = [math.exp(w - max_log_weight) for w in log_weights]
        sum_weights = sum(normalized_weights)
        sum_sq_weights = sum(w**2 for w in normalized_weights)

        if sum_sq_weights == 0:
            return 0.0

        return (sum_weights**2) / sum_sq_weights

    def _systematic_resample(self, weights: list[float], n: int) -> list[int]:
        """Systematic resampling algorithm.

        Deterministic low-variance resampling method that samples n particles
        from the weighted distribution.

        Args:
            weights: Normalized probability weights (sum to 1)
            n: Number of particles to resample

        Returns:
            List of n indices sampled according to weights
        """
        if not weights:
            return []

        # Generate n evenly-spaced positions with random offset
        u = random.random()
        positions = [(u + i) / n for i in range(n)]

        indices = []
        cumsum = 0.0
        i = 0

        for pos in positions:
            while cumsum < pos and i < len(weights):
                cumsum += weights[i]
                i += 1
            # i-1 is the selected index (or last if we've exhausted weights)
            indices.append(max(0, i - 1))

        return indices

    def _default_weight(self, tokens: list[int]) -> float:  # noqa: ARG002
        """Default weight function: uniform weighting.

        Args:
            tokens: Token IDs for the particle

        Returns:
            Log weight (0.0 for uniform)
        """
        return 0.0


__all__ = ["Particle", "SMCController"]

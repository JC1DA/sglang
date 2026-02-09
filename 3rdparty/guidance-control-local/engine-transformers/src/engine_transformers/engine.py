"""Transformers engine implementation integrating with guidance_control.

This module provides a reference implementation of how inference engines can
integrate with the guidance_control controller protocol.
"""

import torch
from guidance_control import Controller, SequenceId, TokenConstraint
from guidance_control.types import TokenSplice
from transformers import PreTrainedModel, PreTrainedTokenizer

from .sequence import SequenceManager, SequenceState


class TransformersEngine:
    """Reference engine using HuggingFace transformers.

    This engine demonstrates how to integrate the guidance_control protocol
    with a real inference backend. It manages KV caches, executes controller
    callbacks, and handles token generation.

    Example:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> from guidance_control import BeamSearchController
        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        >>> engine = TransformersEngine(model, tokenizer, device="cpu")
        >>> controller = BeamSearchController(
        ...     beam_width=4,
        ...     eos_token_id=tokenizer.eos_token_id
        ... )
        >>> finished = engine.generate("The quick brown fox", controller)
        >>> best_seq_id = controller.get_best_sequence_id()
        >>> best_seq = engine.seq_manager.sequences[best_seq_id]
        >>> print(tokenizer.decode(best_seq.tokens))
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "cpu",
    ) -> None:
        """Initialize the transformers engine.

        Args:
            model: HuggingFace transformers model
            tokenizer: Corresponding tokenizer
            device: Device to run on ("cpu" or "cuda")
        """
        self.model = model.to(device)  # type: ignore[arg-type]
        self.tokenizer = tokenizer
        self.device = device
        self.seq_manager = SequenceManager()

        # Set model to eval mode
        self.model.eval()

    def generate(
        self, prompt: str, controller: Controller, max_steps: int = 100
    ) -> list[SequenceState]:
        """Generate text using the AICI-style controller protocol.

        This is the main generation loop that integrates with controllers using
        the three-phase callback system: pre_process, mid_process, post_process.

        Args:
            prompt: Input text to complete
            controller: Controller implementing the generation strategy
            max_steps: Maximum generation steps to prevent infinite loops

        Returns:
            List of finished sequence states
        """
        # Tokenize prompt
        prompt_tokens = self.tokenizer.encode(prompt, return_tensors="pt")[0].tolist()

        # Create initial sequence first to get the sequence ID
        initial_id = self.seq_manager.create_sequence(list(prompt_tokens))

        # Initialize controller with sequence ID and get potentially modified tokens
        modified_prompt_tokens = controller.init(initial_id, prompt_tokens)

        # Update the sequence if tokens were modified
        if list(modified_prompt_tokens) != list(prompt_tokens):
            self.seq_manager.sequences[initial_id].tokens = list(modified_prompt_tokens)

        active_sequences = {initial_id}

        step = 0

        # Main generation loop using AICI-style callbacks
        while not controller.is_complete() and step < max_steps:
            step += 1

            # Phase 1: Pre-process - decide actions for each sequence
            new_sequences = set()
            sequences_to_process = list(active_sequences)

            for seq_id in sequences_to_process:
                result = controller.pre_process(seq_id)

                if result.action == "fork":
                    # Fork sequence into multiple children
                    children = self.seq_manager.fork_sequence(seq_id, result.num_forks)

                    # Notify controller of the fork results
                    controller.post_fork(seq_id, children)

                    new_sequences.update(children)
                    # Parent continues as one of the forks
                    new_sequences.add(seq_id)

                elif result.action == "stop":
                    # Remove from active sequences
                    active_sequences.discard(seq_id)
                    # Mark as finished if not already
                    if seq_id in self.seq_manager.sequences:
                        self.seq_manager.finish_sequence(seq_id)

                elif result.action == "continue":
                    # Keep in active set
                    new_sequences.add(seq_id)

            active_sequences = new_sequences

            # Phase 2 & 3: Generate tokens for active sequences
            for seq_id in list(active_sequences):
                if seq_id not in self.seq_manager.sequences:
                    continue

                seq = self.seq_manager.sequences[seq_id]

                # Don't generate for finished sequences
                if seq.finished:
                    active_sequences.discard(seq_id)
                    continue

                # Get logits from model
                logits = self._get_logits(seq_id)

                # Phase 2: Mid-process - get token constraints
                constraint = controller.mid_process(seq_id)
                # Apply constraints to logits (note, could be done in parallel with mid_process)
                constrained_logits = self._apply_constraint(logits, constraint)

                # Sample token and compute correction factor
                token, logprob, correction = self._sample_token(
                    logits, constrained_logits, constraint.temperature
                )

                # Phase 3: Post-process - notify controller of new token
                splice = controller.post_process(seq_id, token, logprob, correction)

                # Update sequence
                self._update_sequence(seq_id, splice)

        # Return finished sequences
        return self.seq_manager.get_finished_sequences()

    def _get_logits(self, seq_id: SequenceId) -> torch.Tensor:
        """Get logits for next token prediction.

        Args:
            seq_id: Sequence identifier

        Returns:
            Logits tensor for next token
        """
        seq = self.seq_manager.sequences[seq_id]

        # If we have KV cache, only need the last token
        # Otherwise, need full sequence
        if seq.past_key_values is not None:
            input_ids = torch.tensor([[seq.tokens[-1]]], device=self.device)
        else:
            input_ids = torch.tensor([seq.tokens], device=self.device)

        # Forward pass with no gradient
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=seq.past_key_values,
                use_cache=True,
            )

        # Store updated KV cache
        seq.past_key_values = outputs.past_key_values

        # Return logits for next token
        return outputs.logits[0, -1, :]

    def _apply_constraint(
        self, logits: torch.Tensor, constraint: TokenConstraint
    ) -> torch.Tensor:
        """Apply token constraints to logits.

        Args:
            logits: Raw logits from model
            constraint: Token constraint to apply

        Returns:
            Constrained logits
        """
        constrained = logits.clone()

        # Apply allowed tokens mask
        if constraint.allowed_tokens is not None:
            # Mask out disallowed tokens with -inf
            mask = torch.ones_like(logits) * float("-inf")
            for token_id in constraint.allowed_tokens:
                mask[token_id] = 0.0
            constrained = constrained + mask

        # Apply logit bias
        if constraint.logit_bias is not None:
            for token_id, bias in constraint.logit_bias.items():
                constrained[token_id] += bias

        return constrained

    def _sample_token(
        self,
        original_logits: torch.Tensor,
        constrained_logits: torch.Tensor,
        temperature: float,
    ) -> tuple[int, float, float]:
        """Sample token from constrained logits and compute correction factor.

        Uses greedy sampling (argmax) for deterministic generation.

        The correction factor accounts for the change in normalization when
        constraints are applied:
            log_L = log_Z' - log_Z
        where Z and Z' are the partition functions before/after constraints.

        Args:
            original_logits: Original logits before constraints
            constrained_logits: Logits after applying constraints
            temperature: Sampling temperature

        Returns:
            Tuple of (token_id, log_probability, correction_factor)
            - token_id: Sampled token ID
            - log_probability: Log prob from perturbed distribution p'(x)
            - correction_factor: log(Z'/Z) = log(sum_x p(x) * e^bias(x))
        """
        # Apply temperature
        original_logits = original_logits / temperature
        constrained_logits = constrained_logits / temperature

        # Compute log partition functions (log sum exp)
        log_Z = torch.logsumexp(original_logits, dim=-1)
        log_Z_prime = torch.logsumexp(constrained_logits, dim=-1)

        # Correction factor: log(Z'/Z)
        correction = float((log_Z_prime - log_Z).item())

        # Sample with temperature
        dist = torch.distributions.Categorical(logits=constrained_logits)
        token_id = dist.sample().item()
        logprob = dist.probs[token_id].log().item()

        return token_id, logprob, correction

    def _update_sequence(self, seq_id: SequenceId, splice: TokenSplice) -> None:
        """Update sequence with new token.

        Args:
            seq_id: Sequence identifier
            splice: TokenSplice specifying backtrack and tokens to append
        """
        seq = self.seq_manager.sequences[seq_id]

        # Append token and logprob
        if splice.backtrack > 0:
            seq.tokens = seq.tokens[: -splice.backtrack]
        seq.tokens.extend(splice.ff_tokens)

        # Check if we hit EOS token
        if self.tokenizer.eos_token_id is not None and seq.tokens[-1:] == [
            self.tokenizer.eos_token_id
        ]:
            seq.finished = True


__all__ = ["TransformersEngine"]

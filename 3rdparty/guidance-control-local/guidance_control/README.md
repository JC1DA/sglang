# Guidance Control

Core library for portable LLM decoding controllers.

## Overview

This library defines an AICI-style callback protocol that enables portable decoding algorithms across different inference engines. Controllers implement three callbacks that engines invoke during generation to determine decoding strategy while the engine handles execution details.

### Why Callbacks?

Controllers are event-driven and stateful. Instead of examining full sequence state and returning actions, they respond to specific events:

- **pre_process**: Called before token generation — decide to continue, fork, or stop
- **mid_process**: Called during generation — apply token constraints
- **post_process**: Called after sampling — update internal state

This design integrates naturally with engine generation loops, enables better performance (no context switching), and simplifies state management (controllers track what they need).

## Installation

```bash
uv pip install -e guidance_control
```

## Quick Example

```python
from guidance_control import BeamSearchController

# Create controller
controller = BeamSearchController(
    beam_width=4,
    length_penalty=0.6,
    eos_token_id=2  # Use tokenizer.eos_token_id
)

# Engine integration (pseudo-code):
controller.init(prompt_tokens)

while not controller.is_complete():
    for seq_id in active_sequences:
        # 1. Pre-process: decide action
        result = controller.pre_process(seq_id)
        if result.action == "fork":
            create_forks(seq_id, result.num_forks)
        elif result.action == "stop":
            remove_sequence(seq_id)
            continue

        # 2. Mid-process: get constraints
        logits = model.forward(...) # Can run in parallel!
        constraint = controller.mid_process(seq_id)

        # 3. Apply constraints and sample
        token, logprob, correction = sample_with_constraint(logits, constraint)

        # 4. Post-process: update controller
        controller.post_process(seq_id, token, logprob, correction)

# Get final results
results = controller.get_results()
```

See [../engine-transformers/](../engine-transformers/) for a complete working implementation.

## Available Controllers

### BeamSearchController

Classic beam search with length normalization:

```python
from guidance_control import BeamSearchController

controller = BeamSearchController(
    beam_width=4,              # Keep top-4 sequences
    length_penalty=0.6,        # < 1.0 favors longer sequences
    early_stopping=True,       # Stop when beam_width sequences finish
    eos_token_id=2             # End-of-sequence token
)

# After generation
results = controller.get_results()  # Returns list[BeamCandidate]
for result in results:
    print(result.tokens)   # Generated tokens
    print(result.score)    # Length-normalized score
```

### BestOfNController

Generate N candidates and select the best:

```python
from guidance_control import BestOfNController

def scorer(tokens: list[int]) -> float:
    """Higher scores are better."""
    return -len(tokens)  # Prefer shorter sequences

controller = BestOfNController(
    n=5,                # Generate 5 independent candidates
    scorer=scorer,      # Custom scoring function
    max_length=1024
)

# After generation
results = controller.get_results()  # Returns list[Candidate] sorted by score
best = results[0]  # Highest scoring candidate
```

### SMCController

Sequential Monte Carlo with importance resampling:

```python
from guidance_control import SMCController

def weight_function(tokens: list[int]) -> float:
    """Return log weight for this token sequence."""
    # Example: favor sequences containing specific patterns
    return 0.0  # Uniform weighting

controller = SMCController(
    num_particles=10,
    resample_threshold=0.5,  # Resample when ESS < 50% of particles
    weight_function=weight_function,
    max_length=1024
)

# After generation
results = controller.get_results()  # Returns list[Particle] with weights
```

## Writing Custom Controllers

Create a class implementing the callback protocol. No registration or inheritance needed—structural typing via `Protocol`!

```python
from guidance_control import PreProcessResult, TokenConstraint, SequenceId

class SimpleController:
    """Basic controller that generates up to max_length tokens."""

    def __init__(self, max_length: int = 100, eos_token_id: int | None = None):
        self.max_length = max_length
        self.eos_token_id = eos_token_id
        self.sequences: dict[SequenceId, list[int]] = {}
        self.finished: set[SequenceId] = set()

    def init(self, prompt_tokens: list[int]) -> None:
        """Store prompt length for reference."""
        self.prompt_length = len(prompt_tokens)

    def pre_process(self, seq_id: SequenceId) -> PreProcessResult:
        """Continue until max length reached."""
        if seq_id not in self.sequences:
            self.sequences[seq_id] = []

        if seq_id in self.finished or len(self.sequences[seq_id]) >= self.max_length:
            return PreProcessResult(action="stop")

        return PreProcessResult(action="continue")

    def mid_process(self, seq_id: SequenceId) -> TokenConstraint:
        """No constraints - allow all tokens."""
        return TokenConstraint.unconstrained()

    def post_process(
        self,
        seq_id: SequenceId,
        token: int,
        logprob: float,
        correction: float = 0.0
    ) -> None:
        """Update sequence and check for EOS."""
        self.sequences[seq_id].append(token)

        if self.eos_token_id is not None and token == self.eos_token_id:
            self.finished.add(seq_id)

    def is_complete(self) -> bool:
        """Done when all sequences stopped or finished."""
        return all(
            seq_id in self.finished or len(tokens) >= self.max_length
            for seq_id, tokens in self.sequences.items()
        )

    def get_results(self):
        """Return generated sequences."""
        return [{"seq_id": sid, "tokens": tokens}
                for sid, tokens in self.sequences.items()]
```

## Core Types

### PreProcessResult

Result from `pre_process` callback:

```python
from guidance_control import PreProcessResult

# Continue generating
PreProcessResult(action="continue")

# Fork into N children
PreProcessResult(action="fork", num_forks=4)

# Stop this sequence
PreProcessResult(action="stop")
```

### TokenConstraint

Constraints for `mid_process` callback:

```python
from guidance_control import TokenConstraint

# Unconstrained
TokenConstraint.unconstrained()

# Full specification
TokenConstraint(
    allowed_tokens=set[int] | None,  # Restrict to specific tokens
    logit_bias=dict[int, float] | None,  # Bias specific token logits
    temperature=float  # Sampling temperature (default 1.0)
)
```

### SequenceId

Flexible sequence identifier:

```python
SequenceId = str | int  # Engines can use strings or integers
```

## Architecture Notes

**Zero runtime overhead**: Uses `Protocol` for structural typing (duck typing at type-check time), `dataclass` for data structures. No metaclasses, no registration, no runtime checks.

**Stateful design**: Controllers maintain internal state across callbacks. Engines don't need to serialize/deserialize or pass full sequence history—just incremental updates (new tokens, logprobs).

**Engine-agnostic**: Same controller works on any engine that implements the callback protocol. Write once, run everywhere.

**Synchronous callbacks**: Callbacks block and return immediately. For async behavior, controllers can return `action="continue"` and maintain readiness state internally.

## Integration Guide for Engine Authors

To integrate this protocol into your inference engine:

1. **Initialize**: Call `controller.init(prompt_tokens)` before generation
2. **Pre-process loop**: For each active sequence, call `controller.pre_process(seq_id)`
   - Handle `fork` action by duplicating sequence state (including KV cache)
   - Handle `stop` action by removing sequence from batch
3. **Mid-process**: Concurrently with logits computation, call `controller.mid_process(seq_id)`
   - Apply returned constraints to logits before sampling
   - Track correction factor: `correction = log(sum(p(x) * exp(bias(x))))`
4. **Post-process**: After sampling, call `controller.post_process(seq_id, token, logprob, correction)` to get a TokenSplice (backtrack, ff_tokens)
5. **Check completion**: After each step, check `controller.is_complete()`

See [../engine-transformers/src/engine_transformers/engine.py](../engine-transformers/src/engine_transformers/engine.py) for reference implementation.

## API Reference

### Exported from `guidance_control`

**Protocol**:
- `Controller` — Protocol definition

**Types**:
- `PreProcessResult(action, num_forks)` — Pre-process result
- `TokenConstraint(allowed_tokens, logit_bias, temperature)` — Token constraints
- `SequenceId` — Sequence identifier type alias

**Controllers**:
- `BeamSearchController` — Beam search with length normalization
- `BestOfNController` — Best-of-N with custom scorer
- `SMCController` — Sequential Monte Carlo

**Result types**:
- `BeamCandidate(sequence_id, score, tokens, finished)` — Beam search result
- `Candidate(sequence_id, score, tokens)` — Best-of-N result
- `Particle(sequence_id, weight, tokens)` — SMC result

## License

MIT License

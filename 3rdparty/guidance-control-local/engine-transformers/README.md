# Engine-Transformers

A reference implementation of an inference engine using HuggingFace transformers that integrates with the [guidance_control](../guidance_control) controller protocol.

## Overview

This package demonstrates how inference engines can integrate with the guidance_control protocol to support portable decoding algorithms. The engine handles model execution, KV cache management, and token generation while controllers determine the decoding strategy.

## Installation

```bash
# From the workspace root
uv pip install -e engine-transformers
```

This will automatically install dependencies including `guidance-control`, `torch`, and `transformers`.

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from engine_transformers import TransformersEngine
from guidance_control import BeamSearchController

# Load model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

# Create engine
engine = TransformersEngine(model, tokenizer, device="cpu")

# Create controller (beam search, best-of-N, SMC, etc.)
controller = BeamSearchController(beam_width=4)

# Generate
results = engine.generate("The quick brown fox", controller, max_steps=20)

# Get results from controller
for result in controller.get_results():
    print(tokenizer.decode(result.tokens))
```

## Examples

The [examples/](examples/) directory contains complete demonstrations:

- [beam_search.py](examples/beam_search.py) - Beam search with Llama
- [best_of_n.py](examples/best_of_n.py) - Best-of-N sampling with custom scorer
- [smc.py](examples/smc.py) - Sequential Monte Carlo with resampling

Run an example:
```bash
uv run --directory engine-transformers python examples/beam_search.py
```

## How It Works

### Architecture

```
┌─────────────────────────┐
│   TransformersEngine    │
├─────────────────────────┤
│ - Load model/tokenizer  │
│ - Manage KV caches      │
│ - Execute actions       │
│ - Generate tokens       │
└───────────┬─────────────┘
            │ calls
            ▼
┌─────────────────────────┐
│      Controller         │
├─────────────────────────┤
│ controller.step(states) │
│   → returns actions     │
└─────────────────────────┘
```

### Generation Loop

1. **Engine tokenizes prompt** and creates initial sequence
2. **Engine builds state dict** with tokens, logprobs, metadata
3. **Controller examines state** and returns list of actions
4. **Engine executes actions**:
   - `ForkAction` → Deep copy KV cache for beam/tree search
   - `ContinueAction` → Generate next token
   - `PruneAction` → Remove sequence from batch
   - `FinishAction` → Mark sequence as complete
5. **Repeat** until `controller.is_complete()`

### KV Cache Management

The engine maintains independent KV caches for each sequence:

- **Fork**: Deep copies `past_key_values` tuple to create independent branches
- **Continue**: Reuses and extends existing KV cache
- **Prune**: Deletes KV cache to free memory

This allows efficient branching for beam search and tree search algorithms.

## API Reference

### TransformersEngine

```python
class TransformersEngine:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "cpu"
    ):
        """Initialize engine with model and tokenizer."""

    def generate(
        self,
        prompt: str,
        controller: Controller,
        max_steps: int = 100
    ) -> list[SequenceState]:
        """Generate text using controller protocol."""
```

### SequenceManager

```python
class SequenceManager:
    def create_sequence(self, tokens, past_key_values=None) -> SequenceId
    def fork_sequence(self, parent_id, num_forks) -> list[SequenceId]
    def prune_sequence(self, seq_id) -> None
    def finish_sequence(self, seq_id) -> None
    def get_states_for_controller(self) -> dict[SequenceId, dict]
```

### SequenceState

```python
@dataclass
class SequenceState:
    seq_id: SequenceId
    tokens: list[int]
    logprobs: list[float]
    past_key_values: tuple | None
    finished: bool

    def to_controller_state(self) -> dict
```

## Integration Guide

This reference implementation shows how other engines can integrate the protocol:

### 1. State Building

Build state dicts matching the protocol:

```python
{
    "tokens": list[int],
    "logprobs": list[float],
    "metadata": {"finished": bool}
}
```

### 2. Action Execution

Handle all action types from controllers:

```python
match action:
    case ForkAction(parent_id, num_forks):
        # Duplicate sequence and KV cache
    case ContinueAction(seq_id):
        # Generate next token
    case PruneAction(seq_id):
        # Remove from batch
    case FinishAction(seq_id):
        # Mark complete
    case NotReadyAction():
        # Continue gathering state
```

### 3. Generation Loop

```python
while not controller.is_complete():
    states = build_states()
    actions = controller.step(states)
    execute_actions(actions)
```

## Limitations

This reference implementation prioritizes clarity over performance:

- **No batching**: Sequences processed one at a time
- **Greedy sampling only**: No temperature/top-k/top-p
- **CPU optimized**: Not tuned for GPU
- **Simple memory**: No advanced cache management

For production use, consider:
- Batching sequences with compatible KV cache states
- GPU-optimized tensor operations
- Memory pooling for KV caches
- Advanced sampling strategies

## Performance

Expected generation speed on CPU (Llama 3.2 1B):
- Single sequence: ~2-3 tokens/sec
- Beam search (width=4): ~0.5-1 token/sec per beam

GPU inference is much faster - use `device="cuda"` if available.

## Supported Models

Works with any HuggingFace model that supports:
- `use_cache=True` for KV caching
- Standard `past_key_values` format

Tested with:
- Llama 3.2 1B
- Llama 3.1 8B
- GPT-2 variants
- GPT-Neo variants

## Contributing

This is a reference implementation. For production engines, consider:
- Integration with vLLM, SGLang, TensorRT-LLM
- Batched decoding across sequences
- Speculative decoding support
- Quantization (int8, int4)

## License

MIT License - see root workspace LICENSE file.

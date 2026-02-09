# guidance-control

Portable decoding controllers for LLM inference engines.

## The Problem

Advanced decoding algorithms—beam search, best-of-N sampling, sequential Monte Carlo, constrained generation—are typically either hard-coded into specific inference engines or implemented externally with poor integration. This creates duplication across engines and makes new algorithms inaccessible to most users.

## The Solution

An AICI-style callback protocol that inverts control flow. Instead of your algorithm driving the engine, the engine drives your algorithm by calling controller methods at key points during generation. This allows:

- **Portable algorithms**: Write once, run on any compatible engine
- **Engine control**: Engines manage batching, memory, and scheduling
- **Zero overhead**: Callbacks integrate naturally into existing generation loops
- **Incremental adoption**: Engines integrate once, all controllers become available

## Architecture

```
┌─────────────────────────────────────────┐
│        Inference Engine                 │
│    (vLLM, SGLang, transformers...)      │
│  ┌───────────────────────────────────┐  │
│  │   Generation Loop                 │  │
│  │   • pre_process(seq_id)          │──┼──► Controller
│  │   • mid_process(seq_id)          │◄─┼─┐  decides strategy
│  │   • post_process(seq_id, token)  │──┼─┘
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

Controllers implement three callbacks:
1. **pre_process**: Decide to continue, fork, or stop sequences
2. **mid_process**: Apply token constraints (format, grammar, bias)
3. **post_process**: Update state with sampled tokens

## Workspace Organization

This is a monorepo containing:

### `guidance_control/`
Core library defining the controller protocol and built-in controllers.

- **Protocol**: Callback interface (`init`, `pre_process`, `mid_process`, `post_process`, `is_complete`)
- **Types**: `PreProcessResult`, `TokenConstraint`, `SequenceId`
- **Controllers**: `BeamSearchController`, `BestOfNController`, `SMCController`
- **Utilities**: Sequence tree management

See [guidance_control/README.md](guidance_control/README.md) for detailed API documentation.

### `engine-transformers/`
Reference engine implementation using HuggingFace transformers.

- Demonstrates protocol integration
- Manages KV caches for forked sequences
- Includes complete examples with Llama models

See [engine-transformers/README.md](engine-transformers/README.md) for engine integration guide.

## Quick Start

```bash
# Install dependencies
uv pip install -e guidance_control
uv pip install -e engine-transformers

# Run beam search example
uv run --directory engine-transformers python examples/beam_search.py
```

Example code:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from engine_transformers import TransformersEngine
from guidance_control import BeamSearchController

# Load model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

# Create engine and controller
engine = TransformersEngine(model, tokenizer, device="cpu")
controller = BeamSearchController(beam_width=4, eos_token_id=tokenizer.eos_token_id)

# Generate with beam search
engine.generate("The capital of France is", controller, max_steps=50)

# Get results
for result in controller.get_results():
    print(tokenizer.decode(result.tokens))
```

## Available Controllers

- **BeamSearchController**: Classic beam search with length normalization
- **BestOfNController**: Generate N candidates, select best by custom scorer
- **SMCController**: Sequential Monte Carlo with importance resampling

See controller documentation in [guidance_control/README.md](guidance_control/README.md).

## Status

Early exploration. The protocol is stabilizing based on the reference implementation. Next steps include integration proposals for production engines (vLLM, SGLang, TensorRT-LLM) and additional controllers (MCTS, speculative sampling, advanced constraints).

## Why Callbacks?

This callback-based approach offers advantages over action-based protocols:

1. **Natural integration**: Engines already have generation loops—callbacks fit naturally
2. **Better performance**: No context switching, engines can optimize callback integration
3. **Simpler state**: Controllers maintain their own state, no serialization overhead
4. **Proven design**: AICI demonstrated this pattern works at scale for complex constraints

## Development

```bash
# Install with dev dependencies
uv pip install -e guidance_control -e engine-transformers
uv pip sync --group dev

# Run linting
ruff check .
ruff format .

# Run type checking
mypy guidance_control/src
```

## License

MIT License

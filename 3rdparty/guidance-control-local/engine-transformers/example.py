"""Unified examples for guidance_control controllers using Llama with the TransformersEngine.

This script demonstrates three different generation strategies:
- beam_search: Beam search decoding with multiple beams
- best_of_n: Generate N candidates and select the best according to a scoring function
- smc: Sequential Monte Carlo particle filtering with importance weighting
"""

import argparse
from collections.abc import Callable

from engine_transformers import TransformersEngine
from guidance_control import BeamSearchController, BestOfNController, SMCController
from transformers import AutoModelForCausalLM, AutoTokenizer


# Scoring functions for different methods
def length_scorer(tokens: list[int]) -> float:
    """Score by preferring longer sequences (for best-of-N).

    Args:
        tokens: Token IDs

    Returns:
        Score (higher is better)
    """
    return float(len(tokens))


def uniform_weight(tokens: list[int]) -> float:  # noqa: ARG001
    """Uniform weight function (all particles equal) for SMC.

    In a real application, you might use:
    - Constraint satisfaction probability
    - Reward model scores
    - Grammar validity checks

    Args:
        tokens: Token IDs

    Returns:
        Log weight (0.0 for uniform)
    """
    return 0.0


def run_beam_search(
    engine: TransformersEngine,
    tokenizer: AutoTokenizer,
    prompt: str,
    beam_width: int,
    max_steps: int,
) -> None:
    """Run beam search example.

    Args:
        engine: TransformersEngine instance
        tokenizer: HuggingFace tokenizer
        prompt: Input prompt
        beam_width: Number of beams to maintain
        max_steps: Maximum generation steps
    """
    controller = BeamSearchController(
        beam_width=beam_width,
        length_penalty=0.6,
        early_stopping=True,
        eos_token_id=tokenizer.eos_token_id,
    )

    print(f"Prompt: {prompt}")
    print(f"Generating with beam search (width={beam_width})...\n")

    _ = engine.generate(prompt=prompt, controller=controller, max_steps=max_steps)

    # Display results (finished sequences)
    finished = controller.get_results()
    if finished:
        print(f"Generated {len(finished)} finished sequences:\n")
        for i, result in enumerate(finished, 1):
            text = tokenizer.decode(result.tokens, skip_special_tokens=True)
            print(f"{i}. [score={result.score:.2f}, length={len(result.tokens)}]")
            print(f"   {text}\n")
    else:
        # Show active sequences if none finished
        print("No sequences finished (no EOS token). Showing active sequences:\n")
        for seq_id, seq in engine.seq_manager.sequences.items():
            if not seq.finished:
                text = tokenizer.decode(seq.tokens, skip_special_tokens=True)
                print(f"{seq_id}. [length={len(seq.tokens)}]")
                print(f"   {text}\n")


def run_best_of_n(
    engine: TransformersEngine,
    tokenizer: AutoTokenizer,
    prompt: str,
    n: int,
    max_steps: int,
    scorer: Callable[[list[int]], float],
) -> None:
    """Run best-of-N sampling example.

    Args:
        engine: TransformersEngine instance
        tokenizer: HuggingFace tokenizer
        prompt: Input prompt
        n: Number of candidates to generate
        max_steps: Maximum generation steps
        scorer: Scoring function for candidates
    """
    controller = BestOfNController(
        n=n,
        scorer=scorer,
        max_length=50,
        eos_token_id=tokenizer.eos_token_id,
    )

    print(f"Prompt: {prompt}")
    print(f"Generating {n} candidates and selecting the best...\n")

    _ = engine.generate(prompt=prompt, controller=controller, max_steps=max_steps)

    # Display all candidates
    print(f"All {n} candidates:\n")
    for i, candidate in enumerate(controller.get_all_candidates(), 1):
        text = tokenizer.decode(candidate.tokens, skip_special_tokens=True)
        print(f"{i}. [score={candidate.score:.0f}]")
        print(f"   {text}\n")

    # Display best
    best = controller.get_best()
    if best:
        text = tokenizer.decode(best.tokens, skip_special_tokens=True)
        print("Best candidate (by length):")
        print(f"[score={best.score:.0f}] {text}")


def run_smc(
    engine: TransformersEngine,
    tokenizer: AutoTokenizer,
    prompt: str,
    num_particles: int,
    max_steps: int,
    weight_function: Callable[[list[int]], float],
) -> None:
    """Run Sequential Monte Carlo example.

    Args:
        engine: TransformersEngine instance
        tokenizer: HuggingFace tokenizer
        prompt: Input prompt
        num_particles: Number of particles to maintain
        max_steps: Maximum generation steps
        weight_function: Function to compute particle weights
    """
    controller = SMCController(
        num_particles=num_particles,
        resample_threshold=0.5,
        weight_function=weight_function,
        max_length=50,
        eos_token_id=tokenizer.eos_token_id,
    )

    print(f"Prompt: {prompt}")
    print(f"Running SMC with {num_particles} particles...\n")

    _ = engine.generate(prompt=prompt, controller=controller, max_steps=max_steps)

    # Display particles
    particles = controller.get_particles()
    print(f"Final {len(particles)} particles:\n")

    for i, particle in enumerate(particles, 1):
        text = tokenizer.decode(particle.tokens, skip_special_tokens=True)
        print(f"{i}. [weight={particle.log_weight:.2f}]")
        print(f"   {text}\n")


def main() -> None:
    """Parse arguments and run the selected example."""
    parser = argparse.ArgumentParser(
        description="Run guidance_control examples with different generation strategies"
    )
    parser.add_argument(
        "method",
        choices=["beam_search", "best_of_n", "smc"],
        help="Generation method to use",
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.2-1B",
        help="HuggingFace model name (default: meta-llama/Llama-3.2-1B)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run on (default: cpu)",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Custom prompt (default: method-specific prompt)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Maximum generation steps (default: 50)",
    )

    # Method-specific arguments
    parser.add_argument(
        "--beam-width",
        type=int,
        default=4,
        help="Beam width for beam_search (default: 4)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of candidates for best_of_n (default: 5)",
    )
    parser.add_argument(
        "--num-particles",
        type=int,
        default=6,
        help="Number of particles for smc (default: 6)",
    )

    args = parser.parse_args()

    # Load model and tokenizer
    print(f"Loading {args.model}...")
    if args.model == "meta-llama/Llama-3.2-1B":
        print("(First run will download the model - this may take a few minutes)")

    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"Model loaded: {args.model}\n")

    # Create engine
    engine = TransformersEngine(model, tokenizer, device=args.device)

    # Set default prompts if not provided
    if args.prompt is None:
        default_prompts = {
            "beam_search": "The capital of France is",
            "best_of_n": "The capital of France is",
            "smc": "In the future, artificial intelligence will",
        }
        prompt = default_prompts[args.method]
    else:
        prompt = args.prompt

    # Run the selected method
    if args.method == "beam_search":
        run_beam_search(engine, tokenizer, prompt, args.beam_width, args.max_steps)
    elif args.method == "best_of_n":
        run_best_of_n(engine, tokenizer, prompt, args.n, args.max_steps, length_scorer)
    elif args.method == "smc":
        run_smc(
            engine,
            tokenizer,
            prompt,
            args.num_particles,
            args.max_steps,
            uniform_weight,
        )


if __name__ == "__main__":
    main()

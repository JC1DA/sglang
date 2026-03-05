# LlguidanceWorker: Grammar-Guided Fast-Forward Decoding

This document explains what `LlguidanceWorker` does, how it integrates with
the SGLang scheduler, and how it accelerates constrained generation by
fast-forwarding through deterministic tokens.

**File:** `python/sglang/srt/speculative/llguidance_worker.py`

---

## 1. What Problem Does It Solve?

When a user requests **constrained generation** -- such as generating valid
JSON matching a schema, or following a grammar-based template -- many tokens in
the output are *deterministic*. For example, given the JSON schema:

```json
{ "type": "object", "properties": { "name": { "type": "string" } } }
```

After the model produces `{"name": "Alice`, the closing tokens `"}` are forced
by the grammar. There is no need to run the full sampling pipeline for each of
these tokens one at a time.

`LlguidanceWorker` solves this by **fast-forwarding** through deterministic
tokens. Instead of running a separate decode step for each forced token, it:

1. Detects which tokens are deterministic (from the grammar or a controller).
2. Bundles them together with the last sampled token.
3. Converts what would have been single-token decode steps into one multi-token
   **extend** (prefill-like) pass.

This can dramatically reduce the number of forward passes needed for
grammar-constrained generation.

---

## 2. Where It Fits in the Architecture

`LlguidanceWorker` is registered as a **speculative algorithm** in SGLang's
speculative decoding framework, but it works quite differently from other
speculative workers like EAGLE or NGRAM.

### 2.1 Registration

**File:** `python/sglang/srt/speculative/spec_info.py:15-23`

```python
class SpeculativeAlgorithm(Enum):
    EAGLE = auto()
    EAGLE3 = auto()
    STANDALONE = auto()
    NGRAM = auto()
    NONE = auto()
    LLGUIDANCE = auto()     # <-- registered here
```

The `create_worker` method (line 108-116) returns the `LlguidanceWorker` class
when this algorithm is selected:

```python
elif self.is_llguidance():
    if enable_overlap:
        raise ValueError(...)  # overlap scheduling NOT supported
    from sglang.srt.speculative.llguidance_worker import LlguidanceWorker
    return LlguidanceWorker
```

### 2.2 Initialization in the Scheduler

**File:** `python/sglang/srt/managers/scheduler.py:513-551`

When the server is started with `--speculative-algorithm LLGUIDANCE`, the
scheduler creates the worker during initialization:

```python
def maybe_init_draft_worker(self):
    if self.spec_algorithm.is_none():
        self.draft_worker = None
        return
    DraftWorkerClass = self.spec_algorithm.create_worker(self.server_args)
    self.draft_worker = DraftWorkerClass(**draft_worker_kwargs)

def init_model_worker(self):
    self.init_tp_model_worker()
    self.maybe_init_draft_worker()
    if self.spec_algorithm.is_none():
        self.model_worker = self.tp_worker        # normal path
    else:
        self.model_worker = self.draft_worker      # LlguidanceWorker wraps the real worker
```

The key insight: `LlguidanceWorker` **replaces** `self.model_worker`. All
subsequent `forward_batch_generation` calls go through it, and it delegates to
the real `TpModelWorker` (stored as `self.target_worker`) after preparing the
batch.

### 2.3 Server Args Validation

**File:** `python/sglang/srt/server_args.py:2419-2424`

Using LLGUIDANCE requires the `llguidance` grammar backend:

```python
if self.speculative_algorithm == "LLGUIDANCE":
    assert self.grammar_backend == "llguidance", (
        "When using LLGuidance fast-forward tokens decoding, "
        "please set grammar_backend to 'llguidance'."
    )
```

---

## 3. Class Structure

Unlike EAGLE or Standalone workers, `LlguidanceWorker` does **not** inherit
from `BaseSpecWorker`. It is a lightweight wrapper with a simple interface:

```
┌─────────────────────────────────────────────────────────────┐
│  LlguidanceWorker                                           │
│                                                             │
│  target_worker: TpModelWorker    # the real GPU worker      │
│                                                             │
│  + clear_cache_pool()            # no-op                    │
│  + forward_batch_generation()    # main entry point         │
│  - _prepare_for_decode()         # ff_token injection logic │
└─────────────────────────────────────────────────────────────┘
              │
              │  delegates to
              ▼
┌─────────────────────────────────────────────────────────────┐
│  TpModelWorker                                              │
│  + forward_batch_generation()    # actual model forward     │
└─────────────────────────────────────────────────────────────┘
```

### Constructor

**File:** `llguidance_worker.py:21-31`

```python
class LlguidanceWorker:
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.target_worker = target_worker
```

The constructor signature matches the kwargs passed by `maybe_init_draft_worker`
in the scheduler. The only state it keeps is a reference to the underlying
`TpModelWorker`.

---

## 4. Two Sources of Fast-Forward Tokens

`LlguidanceWorker` collects deterministic "fast-forward" (ff) tokens from
**two independent sources** before each decode step:

### 4.1 Grammar-Based FF Tokens (llguidance)

When a request has an active grammar constraint (e.g., JSON schema, regex),
the `req.grammar` object is a `GuidanceGrammar` instance that wraps an
`LLMatcher` from the `llguidance` Python package.

**File:** `python/sglang/srt/constrained/llguidance_backend.py:42-52`

```python
class GuidanceGrammar(BaseGrammarObject):
    def __init__(self, ...):
        self.ll_matcher = LLMatcher(
            self.llguidance_tokenizer,
            self.serialized_grammar,
            log_level=int(os.environ.get("LLGUIDANCE_LOG_LEVEL", "1")),
        )
```

The `LLMatcher` tracks the grammar state as tokens are consumed. At each
decode step, `compute_ff_tokens()` returns a list of token IDs that the
grammar **deterministically requires** -- there is only one valid continuation.

**Called in:** `llguidance_worker.py:82`

```python
ff_tokens = req.grammar.ll_matcher.compute_ff_tokens()
```

**Example:** If the grammar requires the string `"true"` at this point and
that string tokenizes to `[2087, 1091]`, then `compute_ff_tokens()` returns
`[2087, 1091]`. These tokens skip sampling entirely.

### 4.2 Controller-Based Extra Tokens (Guidance Controllers)

Requests may also carry a `guidance_controller` -- an external controller that
can inject tokens programmatically. Controllers implement the protocol defined
in:

**File:** `3rdparty/guidance-control-local/guidance_control/src/guidance_control/protocol.py:75`

```python
class Controller(Protocol):
    def pre_process(self, seq_id: str) -> PreProcessResult:
        """Called before each forward pass. May return extra tokens."""
        ...

    def post_process(self, seq_id: str, token: TokenId,
                     logprob: ..., correction: ...) -> TokenSplice:
        """Called after each forward pass with the sampled token."""
        ...
```

The `PreProcessResult` type has these fields:

**File:** `3rdparty/guidance-control-local/guidance_control/src/guidance_control/types.py:16`

```python
@dataclass
class PreProcessResult:
    action: str              # "continue", "fork", or "stop"
    num_forks: int           # number of forks (for "fork" action)
    extra_tokens: list[TokenId] | None   # fast-forward tokens to inject
```

**Called in:** `llguidance_worker.py:86-88`

```python
controller = req.guidance_controller
if controller:
    preproc_result = controller.pre_process(req.rid)
    if preproc_result.extra_tokens:
        ff_tokens.extend(preproc_result.extra_tokens)
```

**Example use case:** The `CustomController` detects tool calls
(e.g., `<calling_tool>get_weather("NYC")</calling_tool>`) in the model output.
It executes the tool, tokenizes the result, and returns it as `extra_tokens`
in the next `pre_process` call. These tokens are injected into the sequence
without running the model to generate them.

### How the Two Sources Combine

Both sources contribute to a single `ff_tokens` list per request. Grammar
tokens come first, then controller tokens are appended:

```
ff_tokens = grammar.ll_matcher.compute_ff_tokens()   # e.g., [2087, 1091]
                    +
controller.pre_process(rid).extra_tokens              # e.g., [508, 42, 99]
                    =
ff_tokens = [2087, 1091, 508, 42, 99]                # combined
```

---

## 5. The Core Logic: `forward_batch_generation`

**File:** `llguidance_worker.py:36-73`

This is the main method called by the scheduler's `run_batch`. Here is the
complete flow:

```
Scheduler.run_batch(batch)
    │
    │  passes full ScheduleBatch (not ModelWorkerBatch)
    ▼
LlguidanceWorker.forward_batch_generation(batch)
    │
    ├─ 1. _prepare_for_decode(batch)
    │     ├─ Compute ff_tokens for each request
    │     ├─ If ff_tokens exist: inject them, switch to EXTEND mode
    │     └─ If no ff_tokens: standard single-token decode setup
    │
    ├─ 2. batch.get_model_worker_batch()
    │     └─ Convert ScheduleBatch → ModelWorkerBatch
    │
    ├─ 3. target_worker.forward_batch_generation(model_worker_batch)
    │     └─ Actual GPU model forward pass
    │
    ├─ 4. controller.post_process(rid, next_token_id, ...)
    │     └─ Notify each controller of the sampled token
    │
    └─ 5. If was decode mode: accept token in grammar, append to output_ids
          └─ Reset forward_mode back to DECODE
```

### Step 1: Determine Processing Mode

```python
to_process_output_ids = not batch.forward_mode.is_extend()
```

If the batch is in **extend** (prefill) mode, ff_token processing is skipped
entirely -- it only applies to decode steps. The flag `to_process_output_ids`
controls whether post-forward token processing runs.

### Step 2: Prepare the Batch

`_prepare_for_decode(batch)` is called (covered in detail in Section 6).

### Step 3: Run the Model

```python
model_worker_batch = batch.get_model_worker_batch()
batch_result = self.target_worker.forward_batch_generation(model_worker_batch)
```

The actual model forward pass runs on the GPU. If ff_tokens were injected in
step 2, the batch was switched to EXTEND mode, so the model processes all
tokens (the last sampled token + ff_tokens) in a single forward pass.

### Step 4: Post-Process with Controllers

```python
for i, req in enumerate(batch.reqs):
    controller = req.guidance_controller
    if controller:
        next_token_id = batch_result.next_token_ids[i].cpu().item()
        controller.post_process(req.rid, next_token_id, None, None)
```

Each controller is notified of the newly sampled token so it can update its
internal state (e.g., detect tool calls, track output).

### Step 5: Update Request State

Only runs when the original mode was decode (not extend/prefill):

```python
if to_process_output_ids:
    for i, req in enumerate(batch.reqs):
        if req.grammar:
            req.grammar.accept_token(next_token_ids[i].item())
        req.output_ids.append(next_token_id)
    batch.forward_mode = ForwardMode.DECODE
```

The grammar consumes the token via `accept_token`, which internally calls
`ll_matcher.consume_token(token)` to advance the grammar state. The token is
appended to the request's output. The forward mode is reset to DECODE for the
next iteration.

---

## 6. The Core Logic: `_prepare_for_decode`

**File:** `llguidance_worker.py:75-197`

This method does the heavy lifting. It only runs when the batch is NOT in
extend mode (i.e., during decode steps).

### 6.1 Collect FF Tokens

```python
ff_tokens_list = []
max_ff_tokens_len = 0
for i, req in enumerate(batch.reqs):
    ff_tokens = []
    if req.grammar:
        ff_tokens = req.grammar.ll_matcher.compute_ff_tokens()

    controller = req.guidance_controller
    if controller:
        preproc_result = controller.pre_process(req.rid)
        if preproc_result.extra_tokens:
            ff_tokens.extend(preproc_result.extra_tokens)

    ff_tokens_list.append(ff_tokens)
    max_ff_tokens_len = max(max_ff_tokens_len, len(ff_tokens))
```

For each request in the batch, it collects grammar ff_tokens and controller
extra_tokens. It also tracks the maximum ff_token length across all requests
to determine which code path to take.

### 6.2 Path A: FF Tokens Exist (`max_ff_tokens_len > 0`)

This is the interesting case. When at least one request has fast-forward
tokens, the entire batch is promoted from DECODE to EXTEND mode.

#### Why DECODE Cannot Handle FF Tokens

In SGLang's attention implementation, **DECODE mode requires every request in
the batch to contribute exactly one new input token**. The attention kernels
are optimized for this invariant: each request has a single query token that
attends to all its cached KV entries. This allows the kernel to use a uniform
memory layout and efficient batched computation.

When ff_tokens are appended, different requests may have different numbers of
new tokens. For example:

```
Request A: 1 decode token + 3 ff_tokens = 4 new tokens
Request B: 1 decode token + 0 ff_tokens = 1 new token
Request C: 1 decode token + 5 ff_tokens = 6 new tokens
```

This **variable-length input per request breaks the DECODE invariant**. The
batch can no longer be processed as a uniform single-token-per-request decode.

**EXTEND mode** (prefill-like) is designed exactly for this: it handles
variable-length input sequences per request using `prefix_lens` and
`extend_lens` to describe each request's cached vs. new portions. By switching
to EXTEND, the worker can process 4 tokens for request A, 1 for request B,
and 6 for request C in a single batched forward pass.

Note that even when *all* requests happen to have the same number of ff_tokens,
the switch to EXTEND mode is still required because the total new tokens per
request is greater than 1 (decode token + ff_tokens).

#### Step 1: Accept FF Tokens in Grammar and Output

```python
for i, req in enumerate(batch.reqs):
    ff_tokens = ff_tokens_list[i]
    for token in ff_tokens:
        if req.grammar:
            req.grammar.accept_token(token)
        req.output_ids.append(token)
```

Each ff_token is consumed by the grammar (advancing its state) and appended to
the request's output. These tokens are "pre-accepted" before the model even
sees them.

#### Step 2: Build New Input IDs

```python
new_input_ids = []
for i, req in enumerate(batch.reqs):
    ff_tokens = ff_tokens_list[i]
    new_input_ids.append(batch.input_ids[i].item())   # last sampled token
    new_input_ids.extend(ff_tokens)                     # + ff_tokens
```

For each request, the new input consists of the token that was sampled in
the previous step (the normal decode token) followed by all ff_tokens. For
example, if the last sampled token was `42` and ff_tokens are `[10, 20, 30]`,
the input becomes `[42, 10, 20, 30]`.

#### Step 3: Update Sequence Metadata

```python
req.kv_committed_len += len(ff_tokens) + 1
req.kv_allocated_len = req.kv_committed_len

batch.prefix_lens[i] = batch.seq_lens_cpu[i].item()
batch.extend_lens[i] = len(ff_tokens) + 1
batch.extend_num_tokens += len(ff_tokens) + 1

batch.seq_lens_cpu[i] += len(ff_tokens) + 1
batch.orig_seq_lens[i] += len(ff_tokens) + 1
```

- `kv_committed_len` grows by `ff_tokens + 1` (the ff_tokens plus the decode
  token).
- `prefix_lens` is set to the current sequence length (what's already cached).
- `extend_lens` is the number of new tokens to process in this extend pass.
- Sequence length counters are updated accordingly.

#### Step 4: Allocate KV Cache Slots

```python
batch.out_cache_loc = alloc_token_slots(batch.tree_cache, len(new_input_ids))
```

**File:** `python/sglang/srt/mem_cache/common.py:201`

KV cache slots are allocated for all the new tokens across all requests. This
is a single bulk allocation.

#### Step 5: Assign Slots to the Request-Token Pool

```python
assign_req_to_token_pool[(bs,)](
    batch.req_pool_indices,
    batch.req_to_token_pool.req_to_token,
    batch.seq_lens,           # start offsets
    end_seq_lens,             # end offsets
    batch.out_cache_loc,      # allocated cache locations
    batch.req_to_token_pool.req_to_token.shape[1],
    triton.next_power_of_2(bs),
)
```

**File:** `python/sglang/srt/speculative/spec_utils.py:87-119`

This is a **Triton kernel** that maps the newly allocated KV cache locations
into the `req_to_token` pool. Each thread block handles one request and copies
cache locations into the correct positions:

```
req_to_token[req_idx]:  [ ... existing ... | new_slot_0 | new_slot_1 | ... ]
                                             ▲                         ▲
                                         seq_lens[i]            end_seq_lens[i]
```

#### Step 6: Switch to EXTEND Mode

```python
batch.seq_lens = end_seq_lens
batch.seq_lens_sum += len(new_input_ids)
batch.input_ids = torch.asarray(new_input_ids, ...)
batch.forward_mode = ForwardMode.EXTEND    # <-- the critical switch
```

This is the critical mode switch explained above. Since the batch now contains
variable-length input per request (1 decode token + N ff_tokens, where N
differs across requests), it cannot remain in DECODE mode. EXTEND mode tells
the attention backend to use `prefix_lens[i]` / `extend_lens[i]` per request,
correctly handling the ragged input lengths in a single forward pass.

### 6.3 Path B: No FF Tokens (`max_ff_tokens_len == 0`)

When no request has ff_tokens, the method falls back to standard single-token
decode behavior:

```python
batch.out_cache_loc = alloc_token_slots(batch.tree_cache, len(batch.input_ids))
assign_req_to_token_pool[(bs,)](...)

for req in batch.reqs:
    req.kv_committed_len += 1
    req.kv_allocated_len += 1

batch.seq_lens.add_(1)         # or non-inplace if overlap mode
batch.seq_lens_cpu.add_(1)
batch.orig_seq_lens.add_(1)
batch.seq_lens_sum += bs
batch.forward_mode = ForwardMode.DECODE
```

This is essentially the same as what a normal decode step does: allocate one
slot per request, update lengths by 1, and keep DECODE mode.

### 6.4 Mamba Support

At the end of `_prepare_for_decode` (lines 181-197), there is optional handling
for Mamba-style models that require extra buffer tracking. This is gated behind
`get_global_server_args().enable_mamba_extra_buffer()`.

---

## 7. Visual: Decode Step With and Without FF Tokens

### Without LlguidanceWorker (Normal Decode)

```
Step 1: decode token A    → model forward → sample token B
Step 2: decode token B    → model forward → sample token C  (deterministic)
Step 3: decode token C    → model forward → sample token D  (deterministic)
Step 4: decode token D    → model forward → sample token E  (deterministic)
Step 5: decode token E    → model forward → sample token F

Total: 5 forward passes
```

### With LlguidanceWorker (FF Tokens)

```
Step 1: decode token A    → model forward → sample token B
        grammar says C, D, E are deterministic (ff_tokens = [C, D, E])
Step 2: extend [B, C, D, E] → model forward → sample token F

Total: 2 forward passes  (3 forward passes saved)
```

The ff_tokens `[C, D, E]` are pre-accepted by the grammar, bundled with `B`,
and processed in a single extend pass. The model produces the attention
context for all of them at once, then samples the next non-deterministic
token (`F`).

---

## 8. How Controllers Integrate

Controllers add a layer of programmability on top of grammar-based
fast-forwarding. They are set up in the scheduler's `event_loop_normal`:

**File:** `python/sglang/srt/managers/scheduler.py:1085-1166`

A `GuidanceControllScheduler` manages controllers. When a request includes a
`guidance_controller` field in its sampling params, the scheduler:

1. Creates a controller instance via `_init_controller` (line 1111-1128).
   Supported types: `"SMC"` (Sequential Monte Carlo) and `"CustomController"`.
2. Calls `controller.init(req_id, input_tokens)` to initialize it.
3. Before each batch runs (lines 1208-1214), attaches the controller to the
   request:

```python
for req in batch.reqs:
    if self.guidance_controll_scheduler.is_guidance_request(req.rid):
        controller = self.guidance_controll_scheduler.get_controller(req.rid)
        req.guidance_controller = controller
```

### Controller Lifecycle Per Token

```
         ┌───────────────────────────────────────────┐
         │         _prepare_for_decode                │
         │                                            │
         │  controller.pre_process(rid)               │
         │    → returns PreProcessResult              │
         │    → extra_tokens injected as ff_tokens    │
         └─────────────────┬─────────────────────────┘
                           │
                    model forward pass
                           │
         ┌─────────────────▼─────────────────────────┐
         │       forward_batch_generation             │
         │                                            │
         │  controller.post_process(rid, token, ...)  │
         │    → returns TokenSplice                   │
         │    → controller updates internal state     │
         └───────────────────────────────────────────┘
```

### Example: Tool Calling with CustomController

The `CustomController` (in `3rdparty/guidance-control-local/.../custom.py`)
demonstrates a practical use case:

1. **post_process:** As tokens are generated, the controller accumulates text.
   When it detects a tool call pattern like
   `<calling_tool>get_weather("NYC")</calling_tool>`, it executes the tool,
   tokenizes the result (e.g., `"sunny, 72F"`), and stores those tokens.

2. **pre_process:** On the next call, it returns the stored result tokens as
   `extra_tokens`. These are injected into the sequence via the ff_token
   mechanism, avoiding the need for the model to "generate" tool results.

---

## 9. How It Differs from Other Speculative Workers

| Aspect               | LlguidanceWorker           | EAGLE / Standalone         | NGRAM                  |
|----------------------|----------------------------|----------------------------|------------------------|
| Inheritance          | Standalone (no base class) | `BaseSpecWorker`           | Standalone             |
| Draft source         | Grammar + controllers      | Learned draft model        | N-gram cache matching  |
| Verification needed  | No (tokens are certain)    | Yes (accept/reject tree)   | Yes (verify candidates)|
| Forward mode switch  | DECODE → EXTEND            | DECODE → DRAFT + VERIFY    | DECODE → TARGET_VERIFY |
| Overlap scheduling   | Not supported              | Supported (v2 variants)    | Not supported          |
| Extra GPU memory     | None                       | Draft model weights        | None                   |
| `clear_cache_pool`   | No-op                      | Clears draft KV cache      | Has cache cleanup      |

The key distinction: **LlguidanceWorker does not speculate**. It does not
guess tokens that might be wrong. The ff_tokens are *deterministic* -- they
are guaranteed to be correct by the grammar or injected by the controller.
There is no verification step, no rejection, and no wasted computation.

This is why it uses EXTEND mode rather than a special VERIFY mode: the tokens
are unconditionally correct, so they can be processed the same way as a
prefill pass.

---

## 10. Key Data Structures

### On Each Request (`Req`)

**File:** `python/sglang/srt/managers/schedule_batch.py`

| Field                | Type                     | Role in LlguidanceWorker                    |
|----------------------|--------------------------|----------------------------------------------|
| `grammar`            | `BaseGrammarObject`      | Source of grammar ff_tokens via `ll_matcher`  |
| `guidance_controller`| `Controller`             | Source of controller extra_tokens              |
| `output_ids`         | `list[int]`              | FF tokens appended here before model forward  |
| `kv_committed_len`   | `int`                    | Incremented by `len(ff_tokens) + 1`           |
| `kv_allocated_len`   | `int`                    | Set equal to `kv_committed_len`               |

### On the Batch (`ScheduleBatch`)

| Field                | Modification in LlguidanceWorker                           |
|----------------------|-------------------------------------------------------------|
| `input_ids`          | Rebuilt as `[decode_token] + [ff_tokens]` per request       |
| `forward_mode`       | Switched from DECODE to EXTEND when ff_tokens exist         |
| `prefix_lens`        | Set to current `seq_lens_cpu` (existing cached portion)     |
| `extend_lens`        | Set to `len(ff_tokens) + 1` per request                     |
| `extend_num_tokens`  | Total tokens across all requests in the extend pass         |
| `seq_lens`           | Increased by `len(ff_tokens) + 1`                           |
| `out_cache_loc`      | Newly allocated KV cache slots for all new tokens           |

---

## 11. Activation

To use `LlguidanceWorker`, start the server with:

```bash
python -m sglang.launch_server \
    --model-path <model> \
    --speculative-algorithm LLGUIDANCE \
    --grammar-backend llguidance
```

Both flags are required. The `--grammar-backend llguidance` flag ensures the
`GuidanceGrammar` backend is used (which provides `ll_matcher` and
`compute_ff_tokens`), and `--speculative-algorithm LLGUIDANCE` activates the
worker as the model's forward pass wrapper.

---

## 12. Summary

`LlguidanceWorker` is a lightweight wrapper around `TpModelWorker` that
accelerates grammar-constrained and controller-guided generation. Its core
mechanism is simple:

1. **Before each decode step**, check if the grammar or controller can
   determine upcoming tokens without sampling.
2. **If yes**, bundle those deterministic tokens with the decode token,
   allocate KV cache, and run an EXTEND pass instead of a DECODE pass.
3. **If no**, fall back to normal single-token decode.
4. **After the model forward**, notify controllers and update grammar state.

This reduces the number of GPU forward passes proportional to the number of
deterministic tokens in the constrained output, with zero risk of rejection
since the tokens are guaranteed correct.

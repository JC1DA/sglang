# Request Lifecycle in SGLang: From HTTP to Response

This document traces the complete journey of a request in SGLang -- from the
moment an OpenAI-compatible HTTP request arrives, through tokenization,
scheduling, model execution, and all the way back to the user as a response.

---

## High-Level Architecture

SGLang runs as **three separate processes** connected by ZMQ IPC channels:

```
┌──────────────────┐       ZMQ        ┌──────────────┐       ZMQ        ┌────────────────────┐
│  TokenizerManager│  ──────────────>  │   Scheduler   │  ──────────────>  │ DetokenizerManager │
│  (+ HTTP Server) │                   │  (+ GPU Worker)│                   │                    │
│                  │  <────────────────────────────────────────────────────  │                    │
└──────────────────┘       ZMQ (results back)           └────────────────────┘
       ▲    │
       │    │  HTTP
       │    ▼
     Client
```

1. **TokenizerManager** -- Runs the FastAPI HTTP server, tokenizes input text,
   sends tokenized requests to the Scheduler, and receives detokenized results.
2. **Scheduler** -- Manages the GPU worker, schedules batches, runs model
   forward passes, and sends raw token IDs to the DetokenizerManager.
3. **DetokenizerManager** -- Converts token IDs back to text strings and
   forwards them to the TokenizerManager.

---

## 1. HTTP Request Arrives

### 1.1 FastAPI Route

When a client sends `POST /v1/chat/completions`, FastAPI receives the request
and parses the JSON body into a Pydantic model.

**File:** `python/sglang/srt/entrypoints/http_server.py:1324`

```python
@app.post("/v1/chat/completions", dependencies=[Depends(validate_json_request)])
async def openai_v1_chat_completions(
    request: ChatCompletionRequest, raw_request: Request
):
    return await raw_request.app.state.openai_serving_chat.handle_request(
        request, raw_request
    )
```

Similarly, `/v1/completions` routes to `openai_serving_completion.handle_request()`.

The request body is automatically validated by Pydantic into a
`ChatCompletionRequest` (defined in
`python/sglang/srt/entrypoints/openai/protocol.py:491`), which includes
fields like `messages`, `model`, `temperature`, `top_p`, `max_tokens`,
`stream`, `tools`, `response_format`, and SGLang-specific extensions.

### 1.2 OpenAI Serving Handler

The handler follows a **template method pattern** defined in the base class.

**File:** `python/sglang/srt/entrypoints/openai/serving_base.py:86`

```python
async def handle_request(self, request, raw_request):
    # 1. Validate the request
    error_msg = self._validate_request(request)
    if error_msg:
        return self.create_error_response(error_msg)

    # 2. Convert to internal format
    adapted_request, processed_request = self._convert_to_internal_request(
        request, raw_request
    )

    # 3. Route to streaming or non-streaming handler
    if hasattr(request, "stream") and request.stream:
        return await self._handle_streaming_request(adapted_request, processed_request, raw_request)
    else:
        return await self._handle_non_streaming_request(adapted_request, processed_request, raw_request)
```

For chat completions, `OpenAIServingChat` (in
`python/sglang/srt/entrypoints/openai/serving_chat.py:86`) implements:

- **`_validate_request()`** (line 185): Validates messages, tools, tool_choice,
  max_completion_tokens, response_format.
- **`_convert_to_internal_request()`** (line 233):
  1. Calls `_process_messages()` to apply the chat template (Jinja2 or
     conversation-based) and extract multimodal data (images/video/audio).
  2. Calls `request.to_sampling_params()` to build the sampling parameters dict.
  3. Constructs a `GenerateReqInput` -- the unified internal request object.

### 1.3 GenerateReqInput

**File:** `python/sglang/srt/managers/io_struct.py:166`

`GenerateReqInput` is the dataclass that bridges the API layer to the backend.
Key fields include:

- `text` / `input_ids` -- the input to generate from
- `sampling_params` -- temperature, top_p, top_k, max_new_tokens, stop, etc.
- `stream` -- whether to stream partial results
- `image_data`, `video_data`, `audio_data` -- multimodal inputs
- `rid` -- unique request ID
- `lora_path` -- optional LoRA adapter

---

## 2. TokenizerManager: Tokenize and Dispatch

The handler calls `tokenizer_manager.generate_request()`, which is an **async
generator** that tokenizes the input, sends it to the Scheduler, and yields
results back.

**File:** `python/sglang/srt/managers/tokenizer_manager.py:490`

```python
async def generate_request(self, obj: GenerateReqInput, request=None):
    obj.normalize_batch_and_arguments()

    # Wait if generation is paused (e.g., during weight updates)
    async with self.is_pause_cond:
        await self.is_pause_cond.wait_for(lambda: not self.is_pause)

    async with self.model_update_lock.reader_lock:
        if obj.is_single:
            tokenized_obj = await self._tokenize_one_request(obj)
            state = self._send_one_request(obj, tokenized_obj, created_time)
            async for response in self._wait_one_response(obj, state, request):
                yield response
```

### 2.1 Tokenization

**`_tokenize_one_request()`** (line 667) handles three input modes:

- **Raw text**: Calls the HuggingFace tokenizer to convert text to token IDs.
- **Pre-tokenized** (`input_ids` provided): Uses them directly.
- **Input embeddings** (`input_embeds` provided): Uses raw embeddings.

For multimodal inputs, it also processes images/video/audio through the model's
processor. The result is a `TokenizedGenerateReqInput` object (defined in
`io_struct.py:689`) containing `input_ids`, `sampling_params`, `rid`, and other
fields.

### 2.2 Send to Scheduler via ZMQ

**`_send_one_request()`** (line 1057):

```python
def _send_one_request(self, obj, tokenized_obj, created_time=None):
    tokenized_obj = wrap_shm_features(tokenized_obj)   # shared memory for large tensors
    self.send_to_scheduler.send_pyobj(tokenized_obj)    # ZMQ PUSH (pickle serialized)
    state = ReqState([], False, asyncio.Event(), obj, created_time=created_time)
    self.rid_to_state[obj.rid] = state
    return state
```

The tokenized request is pickle-serialized and sent over a ZMQ PUSH socket
to the Scheduler process. A `ReqState` object is created and stored in
`rid_to_state` to track the request's progress. The `ReqState` contains:

- `out_list` -- accumulated partial outputs
- `finished` -- whether generation is complete
- `event` -- an `asyncio.Event` used to signal new data

### 2.3 Wait for Results

**`_wait_one_response()`** (line 1100) is the async loop that waits for the
Scheduler to produce results:

```python
async def _wait_one_response(self, obj, state, request=None):
    is_stream = getattr(obj, "stream", False)
    while True:
        await asyncio.wait_for(state.event.wait(), timeout=...)

        out = state.out_list[-1]
        state.out_list = []

        if state.finished:
            yield out       # Final result
            break

        state.event.clear()
        if is_stream:
            yield out       # Intermediate streaming chunk
```

Meanwhile, a background task (`handle_loop`, line 1473) continuously receives
results from the DetokenizerManager via ZMQ and signals the appropriate
`ReqState.event`.

---

## 3. Scheduler: Receive and Enqueue

The Scheduler runs in its own process with a synchronous event loop.

**File:** `python/sglang/srt/managers/scheduler.py:248`

### 3.1 The Main Event Loop

**`event_loop_normal()`** (line 1071):

```python
while True:
    # 1. Receive requests from TokenizerManager
    recv_reqs = self.recv_requests()

    # 2. Process and enqueue them
    self.process_input_requests(recv_reqs)

    # 3. Select the next batch to run
    batch = self.get_next_batch_to_run()

    # 4. Run the batch through the model
    if batch:
        result = self.run_batch(batch)
        self.process_batch_result(batch, result)

    # 5. Update state for next iteration
    self.last_batch = batch
```

### 3.2 Receiving Requests

**`recv_requests()`** (line 1312) polls the ZMQ PULL socket in a non-blocking
loop:

```python
def recv_requests(self):
    recv_reqs = []
    while True:
        try:
            recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
            recv_req = unwrap_shm_features(recv_req)
        except zmq.ZMQError:
            break
        recv_reqs.append(recv_req)
    return recv_reqs
```

For tensor-parallel setups, the rank-0 worker receives and broadcasts to all
other TP ranks.

### 3.3 Processing Input Requests

**`process_input_requests()`** (line 1450) uses a type-based dispatcher to
route each received message to the appropriate handler. For generation requests,
this calls `handle_generate_request()`.

**`handle_generate_request()`** (line 1563) creates the internal `Req` object:

```python
def handle_generate_request(self, recv_req: TokenizedGenerateReqInput):
    req = Req(
        recv_req.rid,
        recv_req.input_text,
        recv_req.input_ids,
        recv_req.sampling_params,
        return_logprob=recv_req.return_logprob,
        stream=recv_req.stream,
        lora_id=recv_req.lora_id,
        ...
    )
    # Validate, set max_new_tokens, process grammar constraints
    self.init_req_max_new_tokens(req)
    # Add to waiting queue
    self._add_request_to_queue(req)
```

### 3.4 Adding to the Waiting Queue

**`_add_request_to_queue()`** (line 1748) appends the request to the waiting
queue after performing priority validation and queue limit checks:

```python
def _add_request_to_queue(self, req, is_retracted=False):
    if not self._set_or_validate_priority(req):
        return
    if self._abort_on_queued_limit(req):
        return
    self._prefetch_kvcache(req)
    self.waiting_queue.append(req)
```

---

## 4. Scheduler: Select and Build a Batch

### 4.1 get_next_batch_to_run

**`get_next_batch_to_run()`** (line 1937) decides what to run next. **Prefill
is always prioritized over decode**:

```python
def get_next_batch_to_run(self):
    # Merge the previous prefill batch into the running (decode) batch
    if self.last_batch and self.last_batch.forward_mode.is_extend():
        self.running_batch.merge_batch(self.last_batch)

    # Try to build a new prefill batch from waiting queue
    new_batch = self.get_new_batch_prefill()

    if new_batch is not None:
        return new_batch                # Prefill first
    else:
        if not self.running_batch.is_empty():
            self.running_batch = self.update_running_batch(self.running_batch)
            return self.running_batch   # Decode batch
        return None                     # Idle
```

### 4.2 Building a Prefill Batch

**`get_new_batch_prefill()`** (line 2027) uses the **scheduling policy** and a
**PrefillAdder** to select requests from the waiting queue:

1. The `SchedulePolicy` (defined in `schedule_policy.py:93`) sorts the waiting
   queue by priority. Supported policies include `lpm` (longest prefix match),
   `fcfs` (first-come-first-served), `lof` (longest first), `random`, etc.

2. The `PrefillAdder` (schedule_policy.py:372) manages token budgets:
   - `max_prefill_tokens` -- the cap on total prefill tokens per batch.
   - `chunked_prefill_size` -- if a request exceeds this, it is chunked
     (partially prefilled and continued later).
   - Available KV cache slots.
   - Maximum batch size (`max_running_requests`).

3. For each request, `init_next_round_input()` is called to match prefixes in
   the RadixCache and determine how many tokens actually need computation.

4. Requests are added via `adder.add_one_req()` until the batch is full or the
   waiting queue is exhausted.

### 4.3 Preparing the Decode Batch

**`update_running_batch()`** (line 2312) prepares the running batch for the
next decode step:

1. Filters out finished requests.
2. Checks if there is enough KV cache memory for one more decode step.
3. If memory is insufficient, **retracts** (preempts) the longest requests,
   freeing their KV cache and re-queuing them.

---

## 5. Scheduler: Run the Batch

### 5.1 Batch Preparation

Before the forward pass, the batch prepares itself depending on the mode:

- **Prefill** (`prepare_for_extend()` in `schedule_batch.py:1496`):
  Allocates KV cache for all new tokens, writes cache indices, builds input
  tensors. See the [KV Cache Guide](kv_cache_guide.md) for details.

- **Decode** (`prepare_for_decode()` in `schedule_batch.py:1969`):
  Allocates one KV slot per request, increments sequence lengths.

### 5.2 Model Forward Pass

**`run_batch()`** (scheduler.py:2387):

```python
def run_batch(self, batch):
    # Convert ScheduleBatch -> ModelWorkerBatch
    worker_batch = batch.get_model_worker_batch()

    # Run the forward pass on the GPU worker
    result = self.model_worker.forward_batch_generation(worker_batch)
    return result
```

The `TpModelWorker.forward_batch_generation()` (in
`python/sglang/srt/managers/tp_worker.py:426`):

1. Creates a `ForwardBatch` from the `ModelWorkerBatch`.
2. Calls `model_runner.forward()` -- runs the transformer model, computing
   attention with the KV cache and producing logits.
3. Calls `model_runner.sample()` -- samples next token IDs from the logits
   using the sampling parameters (temperature, top_p, top_k, etc.).
4. Returns a `GenerationBatchResult` containing `next_token_ids`, logprobs,
   and other metadata.

### 5.3 Process Batch Results

**`process_batch_result()`** (scheduler.py:2697) routes to the appropriate
handler based on the forward mode:

```python
def process_batch_result(self, batch, result):
    if batch.forward_mode.is_decode():
        self.process_batch_result_decode(batch, result)
    elif batch.forward_mode.is_extend():
        self.process_batch_result_prefill(batch, result)
```

**`process_batch_result_prefill()`** (`scheduler_output_processor_mixin.py:87`):

For each request in the batch:
1. Appends the sampled `next_token_id` to `req.output_ids`.
2. Calls `req.check_finished()` -- checks for EOS token, max length, stop
   strings, or other stopping conditions.
3. If finished, releases the request's KV cache (inserting it into the
   RadixCache for future prefix reuse).
4. If not finished, caches the unfinished state for future prefix matching.
5. Processes logprobs if requested.

**`process_batch_result_decode()`** follows the same pattern but for decode
steps.

### 5.4 Stream Output

After processing results, the scheduler sends output to the Detokenizer.

**`stream_output()`** (`scheduler_output_processor_mixin.py:846`):

This method collects all requests that need to produce output and builds a
`BatchTokenIDOutput` containing:

- `rids` -- request IDs
- `output_ids` -- new token IDs to detokenize
- `finished_reasons` -- whether each request is done (and why)
- Logprobs, hidden states, timing metadata, etc.

The `BatchTokenIDOutput` is sent to the DetokenizerManager via ZMQ.

**When does a request produce output?**

- **Finished requests**: Always output immediately.
- **Streaming requests**: Output every `stream_interval` tokens.
- **Non-streaming requests**: Force output periodically (every ~50 tokens) so
  the TokenizerManager can detect client disconnections.

---

## 6. DetokenizerManager: Token IDs to Text

The DetokenizerManager runs in its own process with a simple event loop.

**File:** `python/sglang/srt/managers/detokenizer_manager.py:74`

```python
def event_loop(self):
    while True:
        recv_obj = self.recv_from_scheduler.recv_pyobj()    # ZMQ PULL from Scheduler
        output = self._request_dispatcher(recv_obj)          # Detokenize
        if output is not None:
            self.send_to_tokenizer.send_pyobj(output)        # ZMQ PUSH to TokenizerManager
```

### 6.1 Incremental Detokenization

**`handle_batch_token_id_out()`** (line 360) processes `BatchTokenIDOutput`:

1. For each request, it maintains a `DecodeStatus` that tracks the accumulated
   `decode_ids`, a `surr_offset` (surrogate offset for handling partial
   Unicode codepoints), and a `read_offset`.

2. The token IDs are decoded to text using `tokenizer.batch_decode()`.

3. **Incremental text** is computed by comparing the newly decoded text against
   what was already sent. Only the new portion is included in the output.

4. The result is a `BatchStrOutput` -- the same structure but with `output_strs`
   (text) instead of raw token IDs.

The `BatchStrOutput` is sent to the TokenizerManager via ZMQ.

---

## 7. Back to TokenizerManager: Assemble the Response

### 7.1 Receiving Results

The TokenizerManager's background task `handle_loop` (line 1473) receives the
`BatchStrOutput` and calls `_handle_batch_output()` (line 1482):

```python
async def handle_loop(self):
    while True:
        recv_obj = await self.recv_from_detokenizer.recv_pyobj()
        self._result_dispatcher(recv_obj)
```

**`_handle_batch_output()`** processes each request in the batch:

1. Looks up the `ReqState` from `rid_to_state`.
2. Builds an output dict with `text`, `meta_info` (finish_reason, token counts,
   timing), and optionally `logprobs`.
3. For **streaming**: sends only the new text since the last chunk.
4. For **non-streaming**: sends the full accumulated text.
5. Appends the output dict to `state.out_list` and **signals `state.event`** --
   this unblocks the `_wait_one_response()` coroutine that the API handler is
   awaiting.

### 7.2 API Response Formatting

Back in the API handler, the yielded output dict is formatted into the
OpenAI-compatible response format.

**Non-streaming** (`serving_chat.py`, `_handle_non_streaming_request`):

The handler calls `generate_request().__anext__()` once to get the final result,
then builds a `ChatCompletionResponse` with the full text, usage stats, and
finish reason.

**Streaming** (`serving_chat.py`, `_handle_streaming_request`):

Returns a `StreamingResponse` that wraps `_generate_chat_stream()`. This async
generator iterates over `generate_request()`, yielding SSE (Server-Sent Events)
chunks:

```
data: {"id":"chatcmpl-xxx","choices":[{"delta":{"content":"Hello"},...}]}\n\n
data: {"id":"chatcmpl-xxx","choices":[{"delta":{"content":" world"},...}]}\n\n
data: {"id":"chatcmpl-xxx","choices":[{"finish_reason":"stop",...}]}\n\n
data: [DONE]\n\n
```

Each chunk is a `ChatCompletionStreamResponse` containing the incremental
`delta` text.

---

## 8. Complete Flow Diagram

```
CLIENT
  │
  │  POST /v1/chat/completions  { messages: [...], stream: true, ... }
  ▼
┌─────────────────────────────────────────────────────────────────┐
│                        HTTP SERVER (FastAPI)                     │
│                                                                 │
│  1. Parse JSON -> ChatCompletionRequest (Pydantic validation)   │
│  2. openai_v1_chat_completions()                                │
│     └─> OpenAIServingChat.handle_request()                      │
│         ├─ _validate_request()                                  │
│         ├─ _convert_to_internal_request()                        │
│         │   ├─ Apply chat template -> prompt text               │
│         │   ├─ Build sampling_params                            │
│         │   └─ Create GenerateReqInput                          │
│         └─ _handle_streaming_request()                           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      TOKENIZER MANAGER                          │
│                                                                 │
│  3. generate_request()                                          │
│     ├─ _tokenize_one_request()                                  │
│     │   ├─ Tokenize text -> input_ids (HuggingFace tokenizer)   │
│     │   ├─ Process multimodal inputs (if any)                   │
│     │   └─ Create TokenizedGenerateReqInput                     │
│     ├─ _send_one_request()                                      │
│     │   ├─ Send via ZMQ PUSH to Scheduler ─────────────────┐    │
│     │   └─ Create ReqState (asyncio.Event for signaling)   │    │
│     └─ _wait_one_response()                                │    │
│         └─ await state.event ◄─── (signaled by handle_loop)│    │
└─────────────────────────────────────────────────────────────┼────┘
                                                              │
                      ZMQ (tokenized request)                 │
                                                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         SCHEDULER                               │
│                                                                 │
│  4. recv_requests()           ◄── ZMQ PULL (non-blocking poll)  │
│                                                                 │
│  5. process_input_requests()                                    │
│     └─ handle_generate_request()                                │
│        ├─ Create Req object                                     │
│        └─ _add_request_to_queue() -> waiting_queue.append(req)  │
│                                                                 │
│  6. get_next_batch_to_run()                                     │
│     ├─ Prefill prioritized over decode                          │
│     ├─ get_new_batch_prefill()                                  │
│     │   ├─ SchedulePolicy sorts waiting queue                  │
│     │   ├─ PrefillAdder selects requests within token budget    │
│     │   └─ init_next_round_input() matches RadixCache prefix    │
│     └─ update_running_batch() (for decode)                      │
│         └─ Retract requests if KV cache is full                 │
│                                                                 │
│  7. run_batch(batch)                                            │
│     ├─ batch.prepare_for_extend() / prepare_for_decode()        │
│     │   └─ Allocate KV cache (see kv_cache_guide.md)            │
│     ├─ model_worker.forward_batch_generation()                  │
│     │   ├─ model_runner.forward() -- transformer forward pass   │
│     │   └─ model_runner.sample()  -- sample next tokens         │
│     └─ Returns GenerationBatchResult (next_token_ids, logprobs) │
│                                                                 │
│  8. process_batch_result(batch, result)                         │
│     ├─ Append next_token_id to req.output_ids                   │
│     ├─ req.check_finished() (EOS, max_len, stop strings)        │
│     ├─ Release KV cache for finished requests                   │
│     └─ stream_output() -> Build BatchTokenIDOutput              │
│                           └─ Send via ZMQ PUSH ─────────┐      │
└─────────────────────────────────────────────────────────┼──────┘
                                                          │
                     ZMQ (token IDs)                      │
                                                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DETOKENIZER MANAGER                           │
│                                                                 │
│  9. recv_from_scheduler.recv_pyobj()  ◄── ZMQ PULL              │
│                                                                 │
│ 10. handle_batch_token_id_out()                                 │
│     ├─ Maintain DecodeStatus per request (incremental decode)   │
│     ├─ tokenizer.batch_decode() -- convert IDs to text          │
│     ├─ Compute incremental text (new portion only)              │
│     └─ Build BatchStrOutput (text + metadata)                   │
│         └─ Send via ZMQ PUSH to TokenizerManager ──────┐       │
└────────────────────────────────────────────────────────┼───────┘
                                                         │
                     ZMQ (text strings)                  │
                                                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TOKENIZER MANAGER (handle_loop)               │
│                                                                 │
│ 11. recv_from_detokenizer.recv_pyobj()  ◄── ZMQ PULL            │
│                                                                 │
│ 12. _handle_batch_output()                                      │
│     ├─ Look up ReqState from rid_to_state                       │
│     ├─ Build output dict (text, meta_info, logprobs)            │
│     ├─ state.out_list.append(output)                            │
│     └─ state.event.set()  ──► unblocks _wait_one_response()    │
│                                                                 │
│ 13. _wait_one_response() yields the output dict                 │
└────────────────────────────────────────────────────────┬────────┘
                                                         │
                                                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                        HTTP SERVER                              │
│                                                                 │
│ 14. Format response:                                            │
│     ├─ Streaming: yield SSE chunk (ChatCompletionStreamResponse)│
│     └─ Non-streaming: return ChatCompletionResponse (JSON)      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
  │
  │  HTTP Response (JSON or SSE stream)
  ▼
CLIENT
```

---

## 9. Key Data Structures Along the Path

| Stage | Data Structure | File |
|-------|---------------|------|
| HTTP input | `ChatCompletionRequest` | `entrypoints/openai/protocol.py:491` |
| API -> TokenizerManager | `GenerateReqInput` | `managers/io_struct.py:166` |
| TokenizerManager -> Scheduler | `TokenizedGenerateReqInput` | `managers/io_struct.py:689` |
| Scheduler internal | `Req` | `managers/schedule_batch.py` |
| Scheduler batching | `ScheduleBatch` | `managers/schedule_batch.py` |
| GPU forward | `ForwardBatch` | via `ModelWorkerBatch` |
| Forward result | `GenerationBatchResult` | `managers/tp_worker.py` |
| Scheduler -> Detokenizer | `BatchTokenIDOutput` | `managers/io_struct.py` |
| Detokenizer -> TokenizerManager | `BatchStrOutput` | `managers/io_struct.py` |
| TokenizerManager tracking | `ReqState` | `managers/tokenizer_manager.py:128` |
| HTTP output (streaming) | `ChatCompletionStreamResponse` | `entrypoints/openai/protocol.py` |
| HTTP output (non-streaming) | `ChatCompletionResponse` | `entrypoints/openai/protocol.py` |

---

## 10. IPC Channels

All inter-process communication uses **ZMQ sockets**:

| Direction | Socket Type | Transport |
|-----------|------------|-----------|
| TokenizerManager -> Scheduler | ZMQ PUSH/PULL | `ipc://` (unix) or `tcp://` (DP) |
| Scheduler -> DetokenizerManager | ZMQ PUSH/PULL | `ipc://` |
| DetokenizerManager -> TokenizerManager | ZMQ PUSH/PULL | `ipc://` |

On a single node, sockets use Unix domain sockets (`ipc://`) for low latency.
With data-parallel attention or multi-node setups, TCP sockets are used instead.

Large tensors (e.g., multimodal features) are passed via **shared memory**
(`wrap_shm_features` / `unwrap_shm_features`) to avoid serialization overhead.

---

## 11. Streaming vs Non-Streaming

### Non-Streaming

1. The request flows through the entire pipeline.
2. The Scheduler may send periodic intermediate outputs (every ~50 tokens) so
   the TokenizerManager can detect client disconnections, but these are not
   yielded to the API layer.
3. `_wait_one_response()` loops until `state.finished` is True, then yields
   the final result once.
4. The API handler returns a single `ChatCompletionResponse` JSON object.

### Streaming

1. The Scheduler sends output every `stream_interval` tokens (configurable).
2. `_wait_one_response()` yields each intermediate result as it arrives.
3. The API handler wraps the async generator in a `StreamingResponse`, which
   sends each chunk as an SSE `data:` line.
4. The final chunk includes `finish_reason` and is followed by `data: [DONE]`.

---

## 12. Error Handling and Cancellation

- **Client disconnects**: The TokenizerManager periodically checks
  `request.is_disconnected()` during `_wait_one_response()`. If disconnected,
  it calls `abort_request()` to notify the Scheduler.

- **Queue limits**: If the waiting queue exceeds `max_queued_requests`, new
  requests are rejected (or lower-priority requests are evicted with priority
  scheduling).

- **Memory pressure**: If the KV cache is full during decode, the Scheduler
  retracts (preempts) the longest-running requests, frees their KV cache, and
  re-queues them to be re-prefilled later.

- **Invalid requests**: Validation errors at any stage produce an error
  response sent back through the pipeline as an `AbortReq`.

# KV Cache in SGLang: How It Works

This document explains how the KV cache system works in SGLang, covering
the overall architecture, memory allocation for new requests, appending new
tokens during decoding, and releasing/removing tokens when a request finishes
or needs to be preempted.

---

## 1. Overview: What Is a KV Cache?

In transformer-based language models, every attention layer computes **Key** (K)
and **Value** (V) projections for each input token. During autoregressive
generation, each new token attends to all previous tokens. Without caching,
this means recomputing K and V for the entire sequence at every step -- an
O(n^2) cost per generated token.

A **KV cache** stores the K and V tensors from previous forward passes so they
can be reused. When generating the next token, only the new token's K/V needs
to be computed and appended to the cache. The attention operation then reads
from the cache rather than recomputing everything.

### The Challenge for Serving Systems

In a serving system like SGLang, many requests run concurrently, each with a
different sequence length, and memory must be managed dynamically. The KV cache
system needs to:

- Allocate memory for incoming requests.
- Grow the cache as new tokens are generated.
- Free memory when requests complete.
- Share cached prefixes across requests (e.g., system prompts).
- Evict cached data under memory pressure.

SGLang addresses all of this with a three-layer architecture.

---

## 2. Architecture: Three Layers

The KV cache system is organized into three layers, from high-level mapping
down to physical GPU memory:

```
┌──────────────────────────────────────────────────────┐
│  Layer 1: ReqToTokenPool                             │
│  Maps each request to its token locations (indices)  │
│  Shape: [max_num_reqs, max_context_len]              │
│  Each cell stores an index into the KV pool          │
├──────────────────────────────────────────────────────┤
│  Layer 2: TokenToKVPoolAllocator                     │
│  Manages which KV pool indices are free/allocated    │
│  Tracks free_pages and release_pages tensors         │
├──────────────────────────────────────────────────────┤
│  Layer 3: KVCache (MHATokenToKVPool, etc.)           │
│  Holds the actual GPU tensors for K and V data       │
│  Per-layer buffers: [num_slots, num_heads, head_dim] │
└──────────────────────────────────────────────────────┘
```

An optional **RadixCache** (a radix tree) sits above this stack to enable
prefix sharing across requests.

### 2.1 ReqToTokenPool

**File:** `python/sglang/srt/mem_cache/memory_pool.py:126`

```python
class ReqToTokenPool:
    """A memory pool that maps a request to its token locations."""

    def __init__(self, size, max_context_len, device, enable_memory_saver):
        self.req_to_token = torch.zeros(
            (size, max_context_len), dtype=torch.int32, device=device
        )
        self.free_slots = list(range(size))
```

- `req_to_token` is a 2D tensor of shape `(max_num_reqs, max_context_len)`.
- Each **row** is a request slot. Each column stores the **KV pool index** for
  that token position.
- `free_slots` is a Python list tracking available request slots.

When a request gets slot `idx`, `req_to_token[idx, 0:seq_len]` contains the
KV pool indices for all of that request's tokens. These indices point into the
physical K/V buffers in the KVCache layer.

### 2.2 TokenToKVPoolAllocator

**File:** `python/sglang/srt/mem_cache/allocator.py:117`

This allocator manages which indices into the physical KV buffers are free.
It operates on two tensors:

- `free_pages`: indices available for immediate allocation.
- `release_pages`: recently freed indices (merged back into `free_pages` when
  needed, to amortize sorting costs).

```python
class TokenToKVPoolAllocator:
    def clear(self):
        # Index 0 is reserved as a padding slot for dummy writes.
        self.free_pages = torch.arange(1, self.size + 1, dtype=torch.int64, device=self.device)
        self.release_pages = torch.empty((0,), dtype=torch.int64, device=self.device)
```

Allocation takes indices from the front of `free_pages`; freeing appends them
back (or to `release_pages` if deferred sorting is enabled).

For **paged allocation** (page_size > 1), a `PagedTokenToKVPoolAllocator`
manages page-granularity indices instead, converting page indices to token
indices via `page_idx * page_size + offset`.

### 2.3 KVCache (Physical Storage)

**File:** `python/sglang/srt/mem_cache/memory_pool.py:601`

The abstract `KVCache` class defines the interface. Concrete implementations:

- **`MHATokenToKVPool`** (line 697): Standard multi-head attention. Stores
  separate `k_buffer` and `v_buffer` lists, one tensor per layer, each of
  shape `[num_slots + page_size, num_kv_heads, head_dim]`.

- **`MLATokenToKVPool`** (line 1377): Multi-Latent Attention (DeepSeek-style).
  Uses a single `kv_buffer` per layer because MLA compresses K and V into a
  joint low-rank representation.

The extra `page_size` slots at the beginning serve as padding for dummy writes
from padded tokens.

### 2.4 RadixCache (Prefix Sharing)

**File:** `python/sglang/srt/mem_cache/radix_cache.py`

The RadixCache is a radix tree where:

- Each **node** stores a segment of token IDs (`key`) and the corresponding
  KV pool indices (`value`).
- **Children** are keyed by the first token(s) of their segment.
- A **lock_ref** counter on each node prevents eviction while any active
  request is using that prefix.

When a new request arrives, its token IDs are matched against the tree.
Matched KV indices are reused directly (no recomputation needed), and only
the unmatched suffix requires a new forward pass.

---

## 3. Allocating Memory for a New Request

When a new request is scheduled for its first forward pass (the "extend" or
"prefill" phase), the system performs several allocation steps.

**Entry point:** `alloc_for_extend()` in `python/sglang/srt/mem_cache/common.py:328`

### Step 1: Prefix Matching

Before allocation, the request checks the RadixCache for a shared prefix:

```python
# python/sglang/srt/managers/schedule_batch.py:877
def init_next_round_input(self, tree_cache):
    self.fill_ids = self.origin_input_ids + self.output_ids
    match_result = tree_cache.match_prefix(...)
    self.prefix_indices = match_result.device_indices   # KV indices already cached
    self.extend_input_len = len(self.fill_ids) - len(self.prefix_indices)
```

The matched `prefix_indices` are KV pool indices that already exist in the
cache from a previous request. The request only needs to compute and allocate
KV for the remaining `extend_input_len` tokens.

### Step 2: Allocate a Request Slot

```python
# common.py:351
req_pool_indices = alloc_req_slots(batch.req_to_token_pool, batch.reqs, batch.tree_cache)
```

This pops a row index from `ReqToTokenPool.free_slots` and assigns it to the
request via `req.req_pool_idx`. Chunked requests (partially processed in a
prior step) reuse their existing slot.

### Step 3: Allocate KV Token Slots

```python
# common.py:358
if batch.tree_cache.page_size == 1:
    out_cache_loc = alloc_token_slots(batch.tree_cache, batch.extend_num_tokens)
else:
    out_cache_loc = alloc_paged_token_slots_extend(...)
```

**Token-based (page_size=1):**

- `alloc_token_slots()` (common.py:201) first calls `evict_from_tree_cache()`
  to evict cached prefixes if free space is insufficient.
- Then calls `allocator.alloc(num_tokens)`, which slices the first
  `num_tokens` elements from `free_pages`.

**Page-based (page_size>1):**

- Uses a Triton kernel (`alloc_extend_kernel`, allocator.py:174) that handles
  three cases per request:
  1. Fill remaining space in the last partial page (from a prefix).
  2. Fill new full pages.
  3. Fill a final partial page.

Both return `out_cache_loc`, a 1D tensor of KV pool indices where the new
K/V data will be written.

### Step 4: Write Mappings into ReqToTokenPool

```python
# common.py:377
write_cache_indices(out_cache_loc, req_pool_indices_device, ..., req_to_token_pool)
```

A Triton kernel writes two things into each request's row in `req_to_token`:

1. **Prefix region** (positions `0` to `prefix_len`): Copies the
   `prefix_indices` (shared KV indices from the RadixCache).
2. **Extend region** (positions `prefix_len` to `seq_len`): Copies the
   newly allocated `out_cache_loc` indices.

After this step, `req_to_token[req_idx, 0:seq_len]` fully maps every token
position to its KV pool location.

### Step 5: Compute and Store KV

During the model forward pass, each attention layer computes K and V for the
new tokens and writes them into the physical buffers:

```python
# memory_pool.py:951
def set_kv_buffer(self, layer, loc, cache_k, cache_v, ...):
    # Writes cache_k and cache_v into k_buffer[layer][loc] and v_buffer[layer][loc]
```

The `loc` parameter is exactly the `out_cache_loc` tensor from allocation.

### Summary Diagram

```
Request arrives with tokens [A, B, C, D, E]

1. Prefix match: [A, B, C] found in RadixCache
   -> prefix_indices = [42, 17, 91]  (existing KV pool indices)
   -> extend_input_len = 2  (tokens D and E need computation)

2. Allocate request slot: req_pool_idx = 5

3. Allocate KV slots for 2 new tokens:
   -> out_cache_loc = [103, 88]

4. Write into req_to_token[5]:
   Position:  0    1    2    3    4
   KV Index: [42] [17] [91] [103] [88]
              ↑ from prefix  ↑ newly allocated

5. Forward pass computes K/V for tokens D, E
   -> Writes into k_buffer[layer][103], k_buffer[layer][88], etc.
```

---

## 4. Appending New Tokens (Decode Phase)

After prefill, the model enters the decode phase: generating one token at a
time. Each decode step needs to allocate space for the new token's KV and
append it to the cache.

**Entry point:** `alloc_for_decode()` in `python/sglang/srt/mem_cache/common.py:423`

### Step 1: Allocate One KV Slot Per Request

```python
# common.py:435
if batch.tree_cache.page_size == 1:
    out_cache_loc = alloc_token_slots(batch.tree_cache, bs * token_per_req)
else:
    # Paged: check if current page has room, or allocate a new page
    last_loc = batch.req_to_token_pool.req_to_token[batch.req_pool_indices, batch.seq_lens - 1]
    out_cache_loc = alloc_paged_token_slots_decode(...)
```

For token-based allocation, this allocates `batch_size` new indices (one per
request). For paged allocation, the allocator checks whether the current page
has room; if the new token crosses a page boundary, a new page is allocated.

### Step 2: Write the New Index into ReqToTokenPool

```python
# common.py:458
batch.req_to_token_pool.write(
    (batch.req_pool_indices, locs), out_cache_loc.to(torch.int32)
)
```

Where `locs` is `batch.seq_lens` (the position of the new token). This
writes the new KV index at the next position in each request's row.

### Step 3: Update Sequence Lengths

```python
# schedule_batch.py:1969 (in prepare_for_decode)
for req in self.reqs:
    req.kv_committed_len += 1
    req.kv_allocated_len += 1

self.seq_lens.add_(1)
```

### Step 4: Compute and Store KV

Just like in prefill, the model forward pass calls `set_kv_buffer()` to write
the new token's K/V into the physical buffers at the allocated location.

### How It Looks in Memory

```
Before decode step (seq_len = 5):
  req_to_token[5]: [42, 17, 91, 103, 88, ?, ?, ...]

Allocate 1 new slot: out_cache_loc = [201]
Write at position 5:
  req_to_token[5]: [42, 17, 91, 103, 88, 201, ?, ...]

seq_len becomes 6.
Forward pass writes K/V for the new token into k_buffer[layer][201].
```

---

## 5. Removing Tokens from a Request

Tokens are removed from the KV cache in several scenarios: when a request
finishes, when the system runs out of memory and needs to preempt a request,
or when cached prefixes are evicted.

### 5.1 When a Request Finishes

**Function:** `release_kv_cache()` in `python/sglang/srt/mem_cache/common.py:465`

```python
def release_kv_cache(req, tree_cache, is_insert=True):
    # 1. Insert KV indices into the RadixCache for future reuse
    tree_cache.cache_finished_req(req, is_insert=is_insert)

    # 2. Free any over-allocated KV cache (from speculative decoding)
    start_p, end_p = req.pop_overallocated_kv_cache()
    if start_p < end_p:
        indices_to_free = tree_cache.req_to_token_pool.req_to_token[req.req_pool_idx][start_p:end_p]
        tree_cache.token_to_kv_pool_allocator.free(indices_to_free)

    # 3. Free the request slot
    tree_cache.req_to_token_pool.free(req)
```

**What `cache_finished_req` does:**

The RadixCache's `cache_finished_req()` method inserts the request's token
sequence and KV indices into the radix tree. This makes the KV data available
for future requests that share the same prefix.

- If a portion of the tree already exists (from another request), the
  **duplicate KV indices are freed** immediately -- the tree already owns
  equivalent data.
- The tree takes ownership of new (non-duplicate) KV indices.
- The lock reference on the matched nodes is decremented, making them eligible
  for eviction if no other request uses them.

If `is_insert=False` (e.g., during retraction), the KV indices are simply
freed without inserting into the tree.

### 5.2 Evicting Cached Prefixes (RadixCache Eviction)

**Function:** `evict()` in `python/sglang/srt/mem_cache/radix_cache.py:564`

When allocation cannot find enough free slots, the system evicts cached
prefixes from the radix tree:

```python
def evict(self, params):
    leaves = list(self.evictable_leaves)
    eviction_heap = [(self.eviction_strategy.get_priority(node), node) for node in leaves]
    heapq.heapify(eviction_heap)

    while num_evicted < num_tokens and len(eviction_heap):
        _priority, x = heapq.heappop(eviction_heap)

        # Free the KV indices stored in this leaf node
        self.token_to_kv_pool_allocator.free(x.value)
        num_evicted += len(x.value)

        # Remove the leaf from the tree
        self._delete_leaf(x)

        # If the parent becomes a childless unlocked node, it can be evicted too
        if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
            heapq.heappush(eviction_heap, (..., x.parent))
```

Key points:

- Only **leaf nodes with `lock_ref == 0`** are evictable. Nodes used by active
  requests are protected.
- The default eviction strategy is **LRU** (least recently used). Other
  strategies (LFU, FIFO, etc.) are available.
- Eviction is triggered automatically before each allocation by
  `evict_from_tree_cache()` (common.py:229).

### 5.3 Retracting Decode Requests (Preemption)

When memory runs out during the decode phase (not enough slots for even one
token per request), the scheduler **retracts** requests -- removing them from
the running batch entirely:

**Function:** `retract_decode()` in `python/sglang/srt/managers/schedule_batch.py:1868`

```python
def retract_decode(self, server_args):
    # Sort by output length (longest first) to free the most memory
    sorted_indices.sort(key=lambda i: (len(self.reqs[i].output_ids), ...), reverse=True)

    while not self.check_decode_mem(selected_indices=sorted_indices):
        idx = sorted_indices.pop()
        retracted_reqs.append(self.reqs[idx])
        self.release_req(idx, ...)  # Frees KV cache
```

`release_req()` calls `release_kv_cache(req, tree_cache, is_insert=False)`,
which frees all KV indices **without** inserting into the radix tree. The
retracted request is re-queued and will be re-prefilled later when memory is
available.

### 5.4 How Freeing Works

**Token-based** (allocator.py:155):

```python
def free(self, free_index):
    if self.need_sort:
        # Deferred: append to release_pages, merge later
        self.release_pages = torch.cat((self.release_pages, free_index))
    else:
        # Immediate: append back to free_pages
        self.free_pages = torch.cat((self.free_pages, free_index))
```

**Page-based** (allocator.py:430):

```python
def free(self, free_index):
    # Convert token indices to page indices
    free_page_indices = torch.unique(free_index // self.page_size)
    # Then append to free_pages or release_pages
```

The `free_group` mechanism batches multiple free operations together for
efficiency, collecting indices during a group and issuing a single
concatenation at the end.

### 5.5 Lock References: Preventing Premature Eviction

The RadixCache uses a lock reference mechanism to prevent evicting nodes that
are actively in use:

```python
# radix_cache.py:593
def inc_lock_ref(self, node):
    while node != self.root_node:
        if node.lock_ref == 0:
            self.evictable_size_ -= len(node.key)
        node.lock_ref += 1
        node = node.parent
```

When a request matches a prefix, `inc_lock_ref()` walks from the matched node
up to the root, incrementing the counter on every ancestor. This makes the
entire path non-evictable. When the request finishes, `dec_lock_ref()` reverses
the process. Only when `lock_ref` drops to 0 does a node become evictable.

---

## 6. Complete Request Lifecycle

```
1. REQUEST ARRIVES
   ├── match_prefix() against RadixCache
   ├── prefix_indices = reusable KV indices
   └── extend_input_len = tokens needing computation

2. PREFILL (extend)
   ├── alloc_req_slots()     → get a row in req_to_token_pool
   ├── alloc_token_slots()   → get KV pool indices for new tokens
   ├── write_cache_indices() → write prefix + new indices into the row
   ├── inc_lock_ref()        → protect prefix nodes from eviction
   └── model forward         → set_kv_buffer() writes K/V into GPU buffers

3. DECODE (repeated for each new token)
   ├── alloc_for_decode()    → allocate 1 KV slot per request
   ├── write to req_to_token → append new index at next position
   └── model forward         → set_kv_buffer() writes new K/V

4. REQUEST FINISHES
   ├── cache_finished_req()  → insert sequence into RadixCache
   │   ├── tree takes ownership of KV indices
   │   └── duplicates freed immediately
   ├── pop_overallocated_kv_cache() → free speculative extras
   ├── dec_lock_ref()        → allow prefix eviction
   └── free request slot     → return row to req_to_token_pool

5. MEMORY PRESSURE
   ├── evict_from_tree_cache() → evict LRU leaves from RadixCache
   └── retract_decode()        → preempt running requests, free their KV
```

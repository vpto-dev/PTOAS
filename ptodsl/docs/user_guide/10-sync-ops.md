# 10. Synchronization Operations

Chapters 7 and 8 covered data movement and computation. This chapter covers the synchronization primitives that keep those operations correctly ordered across the NPU's concurrent hardware pipelines.

The Ascend NPU executes work across multiple independent pipelines — MTE (DMA), Vector, and Cube — each with its own instruction stream. Synchronization operations coordinate these pipelines: a DMA must finish loading data before the vector unit starts computing on it; a matrix multiply must complete before the result is stored. Without explicit synchronization, pipelines race, and results are undefined.

## 10.1 Enum types for synchronization

PTODSL provides three enum types for type-safe specification of synchronization parameters.

### `BarrierType`

Memory barrier types used with `pto.mem_bar`. Each value specifies which category of prior instruction must complete before which category of subsequent instruction may proceed.

| Member | Meaning |
|--------|---------|
| `VV_ALL` | All vector ops before → all vector ops after |
| `VST_VLD` | Vector stores before → vector loads after |
| `VLD_VST` | Vector loads before → vector stores after |
| `VST_VST` | Vector stores before → vector stores after |
| `VS_ALL` | All vector ops before → all scalar ops after |
| `VST_LD` | Vector stores before → scalar loads after |
| `VLD_ST` | Vector loads before → scalar stores after |
| `VST_ST` | Vector stores before → scalar stores after |
| `SV_ALL` | All scalar ops before → all vector ops after |
| `ST_VLD` | Scalar stores before → vector loads after |
| `LD_VST` | Scalar loads before → vector stores after |
| `ST_VST` | Scalar stores before → vector stores after |
| `SYNC` | Full ordering — all prior memory operations (all pipes) complete before any subsequent operation |

`SYNC` is a convenience value equivalent to a full pipeline barrier. It is the idiomatic choice for separating compute phases inside a ukernel when fine-grained barrier types are not needed.

The naming convention: `V` = vector, `S` = scalar, `ST` = store, `LD` = load. `VST_VLD` reads "Vector STore before Vector LoaD."

### `Pipe`

Hardware pipeline identifiers used with `pto.set_flag`, `pto.wait_flag`, and `pto.pipe_barrier`.

| Member | Pipeline |
|--------|----------|
| `S` | Scalar / control pipeline |
| `V` | Vector pipeline (SIMD) |
| `M` | Matrix / Cube pipeline |
| `MTE1` | Memory Transfer Engine 1 |
| `MTE2` | Memory Transfer Engine 2 |
| `MTE3` | Memory Transfer Engine 3 |
| `MTE4` | Memory Transfer Engine 4 |
| `ALL` | All pipelines (for barrier operations) |

The most commonly used pipes in synchronization are `MTE2` (GM ↔ UB DMA), `MTE3` (UB ↔ UB DMA), `V` (vector compute), and `M` (matrix compute).

### `Event`

Event identifiers for pipeline synchronization flags. The hardware provides 8 event IDs (0–7) per pipeline pair, supporting up to 8 concurrent in-flight DMA/compute sequences.

| Member | Value |
|--------|-------|
| `ID0` | Event 0 |
| `ID1` | Event 1 |
| `ID2` | Event 2 |
| `ID3` | Event 3 |
| `ID4` | Event 4 |
| `ID5` | Event 5 |
| `ID6` | Event 6 |
| `ID7` | Event 7 |

Events are per-pipeline-pair: the same `ID0` used between `MTE2 → V` is independent from `ID0` used between `MTE3 → V`.

---

## 10.2 Pipeline synchronization: `set_flag`, `wait_flag`, `pipe_barrier`

Pipeline synchronization is the primary mechanism for ordering work across pipelines. The pattern is always **signal then wait**: the producer pipeline sets a flag when its work is done; the consumer pipeline waits on that flag before proceeding.

### `pto.set_flag(pipe_from, pipe_to, event_id)`

**Description**: Sets a synchronization flag between two hardware pipelines. The producing pipeline signals that work up to this point is complete.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `pipe_from` | `Pipe` | Source pipeline — the pipeline that has completed its work |
| `pipe_to` | `Pipe` | Destination pipeline — the pipeline being notified |
| `event_id` | `Event` | Event identifier for this specific synchronization point |

**Returns**: None (side-effect operation).

**Example**:

```python
from pto import Pipe, Event

# MTE2 has finished loading tile data — signal Vector pipeline
pto.set_flag(Pipe.MTE2, Pipe.V, Event.ID0)
```

### `pto.wait_flag(pipe_from, pipe_to, event_id)`

**Description**: Waits for a synchronization flag. The consuming pipeline blocks until the flag is set by the producing pipeline.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `pipe_from` | `Pipe` | Source pipeline that set the flag |
| `pipe_to` | `Pipe` | Destination pipeline — the pipeline that is waiting |
| `event_id` | `Event` | Event identifier matching the corresponding `set_flag` |

**Returns**: None (side-effect operation).

**Example**:

```python
from pto import Pipe, Event

# Vector pipeline waits for MTE2 to finish loading
pto.wait_flag(Pipe.MTE2, Pipe.V, Event.ID0)
```

### `pto.pipe_barrier(pipes)`

**Description**: Executes a barrier across the specified pipelines. All work before the barrier in the named pipelines must complete before any work after the barrier may begin.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `pipes` | `Pipe` | Pipeline specification — typically `Pipe.ALL` for a full barrier |

**Returns**: None (side-effect operation).

**Example**:

```python
from pto import Pipe

# Full hardware barrier — all pipelines synchronize
pto.pipe_barrier(Pipe.ALL)
```

### Typical usage pattern

A common ukernel pattern interleaves DMA and compute with `set_flag` / `wait_flag` pairs:

```python
@pto.ukernel
def gemm_block(q_tile, k_tile, v_tile, o_tile, ...):
    # DMA: load K and V tiles from GM to UB
    # mte_load derives strides, burst sizes, etc. from k_part / k_tile types
    pto.mte_load(k_part, k_tile)
    pto.mte_load(v_part, v_tile)

    # Signal: DMA done, UB data ready
    pto.set_flag(Pipe.MTE2, Pipe.V, Event.ID0)

    # Wait: vector pipeline stalls until data arrives
    pto.wait_flag(Pipe.MTE2, Pipe.V, Event.ID0)

    # Compute: now safe to use k_tile and v_tile
    qk_matmul(q_tile, k_tile, ...)
    pv_matmul(p_tile, v_tile, ...)

    # Signal: compute done, results ready for store
    pto.set_flag(Pipe.V, Pipe.MTE3, Event.ID1)
    pto.wait_flag(Pipe.V, Pipe.MTE3, Event.ID1)

    # DMA: store results back to GM
    pto.mte_store(o_tile, o_part)
```

---

## 10.3 Buffer management: `get_buf`, `rls_buf`

Double-buffering is a common optimization in NPU kernels: while one buffer is being computed on, the other is being loaded with the next block of data. The `get_buf` / `rls_buf` pair coordinates buffer ownership between pipelines.

### `pto.get_buf(pipe, buf_id, mode=0)`

**Description**: Acquire a buffer slot for inter-pipeline double-buffering coordination. The calling pipeline claims ownership of the buffer, blocking if the buffer is still in use by another pipeline.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `pipe` | `Pipe` | Pipeline identifier of the acquiring pipeline |
| `buf_id` | `pto.i64` | Buffer identifier (0-based index into the buffer pool) |
| `mode` | `pto.i64` | Acquisition mode (default 0) |

**Returns**: None (side-effect operation).

### `pto.rls_buf(pipe, buf_id, mode=0)`

**Description**: Release a buffer slot, allowing another pipeline to acquire it.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `pipe` | `Pipe` | Pipeline identifier of the releasing pipeline |
| `buf_id` | `pto.i64` | Buffer identifier matching the corresponding `get_buf` |
| `mode` | `pto.i64` | Release mode (default 0) |

**Returns**: None (side-effect operation).

### Double-buffering example

```python
from pto import Pipe

# Pipeline V acquires buffer 0 for compute
pto.get_buf(Pipe.V, 0, 0)

# ... compute into buffer 0 ...

# Release buffer 0 — DMA can now refill it
pto.rls_buf(Pipe.V, 0, 0)

# Pipeline MTE2 acquires buffer 0 for reload
pto.get_buf(Pipe.MTE2, 0, 0)

# ... DMA loads next block into buffer 0 ...

pto.rls_buf(Pipe.MTE2, 0, 0)
```

---

## 10.4 Memory barriers: `mem_bar`

Within a single pipeline, load and store instructions may be reordered by the hardware. `mem_bar` enforces ordering when UB addresses alias between operations — for example, when a store to a region must be visible to a subsequent load from the same region.

### `pto.mem_bar(barrier_type)`

**Description**: Inserts a memory barrier that enforces ordering of prior and subsequent instructions within the same pipeline.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `barrier_type` | `BarrierType` | Barrier type controlling which categories of prior instructions must complete before which categories of subsequent instructions may proceed |

**Returns**: None (side-effect operation).

**Example**:

```python
from pto import BarrierType

# Ensure all prior vector stores are visible before any subsequent vector loads
pto.mem_bar(BarrierType.VST_VLD)
```

The most commonly used barrier types in practice:

| Use case | Barrier type |
|----------|--------------|
| General vector ordering | `BarrierType.VV_ALL` |
| Store-then-load to same UB region | `BarrierType.VST_VLD` |
| Vector → scalar handoff | `BarrierType.VS_ALL` |
| Scalar → vector handoff | `BarrierType.SV_ALL` |

### Usage in ukernel blocks

In flash attention, `mem_bar` separates logically independent computation phases within the same ukernel:

```python
@pto.ukernel
def flash_attention_block(q_tile, k_tile, v_tile, ...):
    # Phase 1: load K/V
    pto.mte_load(k_part, k_tile)
    pto.mte_load(v_part, v_tile)
    pto.mem_bar(BarrierType.SYNC)

    # Phase 2: S = Q @ K^T
    qk_matmul(q_tile, k_tile, ...)
    pto.mem_bar(BarrierType.SYNC)

    # Phase 3: softmax(S)
    online_softmax(s_tile, ...)
    pto.mem_bar(BarrierType.SYNC)

    # Phase 4: PV = P @ V
    pv_matmul(p_tile, v_tile, ...)
    pto.mem_bar(BarrierType.SYNC)

    # Phase 5: blend output
    blend_output(o_prev_tile, pv_tile, ...)
    pto.mem_bar(BarrierType.SYNC)
```

---

## 10.5 Cross-core and intra-block synchronization

Section 10.2 covers the general pipe-to-pipe sync mechanism (`set_flag`/`wait_flag`). This section covers two additional sync domains that the pipe-flag mechanism does not address: **cross-core** communication between separate NPU cores, and **intra-block** synchronization between the Cube and Vector units within a block.

### 10.5.1 Cross-core sync: `set_cross_core`, `wait_flag_dev`

When a kernel spans multiple cores, cores need to coordinate through shared resources. `set_cross_core` sends a signal to another core; `wait_flag_dev` blocks the calling core until the expected signal arrives.

These are core-level (SU) operations — `wait_flag_dev` stalls the entire core, not just a single pipeline. Use them sparingly: splitting work so that each core operates independently for as long as possible minimises cross-core sync overhead.

#### `pto.set_cross_core(core_id, event_id)`

**Description**: Signal an event to another core, indicating that shared data or a pipeline stage is ready.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `core_id` | `pto.i64` | Target core identifier (platform-specific mapping) |
| `event_id` | `Event` | Cross-core event identifier |

**Returns**: None (side-effect operation).

**Example**:

```python
from pto import Event

# Signal core 0 that our computation is complete
pto.set_cross_core(0, Event.ID0)
```

#### `pto.wait_flag_dev(core_id, event_id)`

**Description**: Wait for an event from another core. Core-level (SU) blocking — the entire core stalls until the event is received.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `core_id` | `pto.i64` | Source core identifier |
| `event_id` | `Event` | Event identifier to wait on |

**Returns**: None (side-effect operation).

**Example**:

```python
from pto import Event

# Core 1 waits for core 0 to signal event ID0
pto.wait_flag_dev(0, Event.ID0)
```

### 10.5.2 Intra-block sync: `set_intra_block`, `wait_intra_core`

The Cube unit (matrix pipeline) has a dedicated synchronization channel separate from the standard pipe-flag mechanism used by MTE and Vector pipelines. `set_intra_block` and `wait_intra_core` synchronize Cube and Vector within the same block, ensuring that shared UB tile data is not accessed before the producer finishes.

Unlike `wait_flag_dev`, `wait_intra_core` only stalls the specified pipeline — the SU and other pipelines continue executing.

#### `pto.set_intra_block(block_id, event_id)`

**Description**: Signal a synchronization event within a block. Specifies which trigger pipe fires the event.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `block_id` | `pto.i64` | Block or pipeline identifier for the trigger source |
| `event_id` | `Event` | Event identifier |

**Returns**: None (side-effect operation).

**Example**:

```python
from pto import Event

# Signal event ID0 on block/pipeline 0
pto.set_intra_block(0, Event.ID0)
```

#### `pto.wait_intra_core(block_id, event_id)`

**Description**: Wait for an intra-block event. Only the specified pipeline stalls — the SU and other pipelines continue executing independently.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `block_id` | `pto.i64` | Block or pipeline identifier specifying which pipeline waits |
| `event_id` | `Event` | Event identifier to wait on |

**Returns**: None (side-effect operation).

**Example**:

```python
from pto import Event

# Pipeline 1 waits for event ID0 from pipeline 0 within the same block
pto.wait_intra_core(1, Event.ID0)
```

### 10.5.3 Intra-core configuration: `set_intra_core`

#### `pto.set_intra_core(config)`

**Description**: Configures intra-core synchronization parameters. The meaning of `config` is hardware-specific.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `pto.i32` | Hardware-specific configuration value |

**Returns**: None (side-effect operation).

**Example**:

```python
pto.set_intra_core(3)
```

---

## 10.6 Synchronization in the abstraction hierarchy

Where do sync operations belong in PTODSL's layered model?

| Layer | Sync responsibility |
|-------|---------------------|
| L1 `@pto.jit` | Tile ops require sync, but PTOAS **auto-inserts** `set_flag`/`wait_flag` pairs based on op-to-pipe mapping — the user does not write sync explicitly |
| L2 `@pto.ukernel` | User writes micro-instructions directly and takes full responsibility for sync: `set_flag`/`wait_flag` between DMA and compute, `mem_bar` between compute phases, `pipe_barrier` at block boundaries |
| L3 `@pto.cube` / `@pto.simd` | Cross-pipeline sync (`set_flag`/`wait_flag`) is managed by the calling ukernel. Sub-kernels may still use `mem_bar` for intra-pipeline ordering (e.g., store-then-load to the same UB region) |

**Rule of thumb**: at L1, sync can be manual or auto-inserted (`--enable-insert-sync`). At L2, sync is always explicit.

### Auto-sync at the tile level

When writing `@pto.jit` code with tile ops (`tload`, `tstore`, `tadd`, etc.), each op carries a pipe assignment (e.g., `tload` → `PIPE_MTE2`, `tadd` → `PIPE_V`). PTOAS's sync-insertion pass analyzes the op sequence, infers the necessary `set_flag`/`wait_flag` pairs from the pipe transitions, and injects them into the lowered code. The tile ops themselves still require synchronization — the difference is that the compiler, not the user, writes it.

### Quick reference: which sync for which scenario

| Scenario | Sync primitive |
|----------|----------------|
| DMA load must finish before compute | `set_flag(MTE2, V, id)` + `wait_flag(MTE2, V, id)` |
| Compute must finish before DMA store | `set_flag(V, MTE3, id)` + `wait_flag(V, MTE3, id)` |
| Two compute phases must not overlap | `mem_bar(BarrierType.VV_ALL)` |
| Store must be visible to later load (same UB) | `mem_bar(BarrierType.VST_VLD)` |
| Full pipeline sync point | `pipe_barrier(Pipe.ALL)` |
| Double-buffer handoff (compute → DMA) | `rls_buf(V, id)` + `get_buf(MTE2, id)` |
| Double-buffer handoff (DMA → compute) | `rls_buf(MTE2, id)` + `get_buf(V, id)` |
| Core A notifies core B | `set_cross_core(B, id)` + `wait_flag_dev(A, id)` |

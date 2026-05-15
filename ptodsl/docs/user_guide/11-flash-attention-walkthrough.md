# 11. Flash Attention Complete Walkthrough

This chapter walks through `demos/flash_attention_sketch.py` layer by layer, tracing a complete flash attention implementation from the user-facing Python wrapper down to hardware-bound sub-kernels. Every API discussed in Chapters 1–10 appears in context here.

The sketch computes **online-softmax flash attention** for one `(batch, head)` slice per launch instance. It partitions Q into blocks along the sequence dimension, iterates over KV blocks for each Q block, and maintains rolling softmax state across KV iterations.

## 11.1 Architecture overview

```
flash_attention(...)           L0  user-facing wrapper
  └─ @pto.jit flash_attention_kernel
       ├─ Tile Ops                 tload / tstore at the GM↔UB boundary
       └─ @pto.ukernel  kv_block_process
            ├─ @pto.simt   materialize_tile_bounds
            ├─ @pto.cube   qk_matmul
            ├─ @pto.simd   online_softmax_rows
            ├─ @pto.cube   pv_matmul
            └─ @pto.simt   blend_output_rows
```

The dataflow for one KV block:

```
ukernel loads K/V block and sequences the pipeline
       │
       ├─ cube:  Q + K  ───────────────► S
       ├─ simd:  S + (m_prev, l_prev) ─► P, (m_next, l_next), alpha, beta
       ├─ cube:  P + V  ───────────────► PV
       └─ simt:  (o_prev, PV, alpha, beta) ─► o_next

After each KV block:
  (m_prev, l_prev, o_prev) := (m_next, l_next, o_next)
```

## 11.2 L0 — Python wrapper

```python
def flash_attention(Q, K, V, *, O=None, causal=False,
                    block_q=128, block_kv=128, stream=None):
    if O is None:
        O = pto.empty_like(Q)

    batch, seq_q, heads, dim = Q.shape
    _, seq_k, _, _ = K.shape

    compiled = flash_attention_kernel.compile(
        BLOCK_Q=block_q, BLOCK_KV=block_kv, CAUSAL=causal,
    )
    compiled[batch * heads, stream](Q, K, V, O)
    return O
```

This is plain Python — no PTO types, no IR. It handles ergonomic runtime concerns:

- **Output allocation**: `pto.empty_like(Q)` when the caller doesn't provide one.
- **Shape extraction**: reads `batch`, `seq_q`, `heads`, `dim` from the framework tensors.
- **Compile + launch**: `flash_attention_kernel.compile(...)` JIT-compiles the kernel with the given constexpr parameters, then launches it with a `[batch * heads]` grid — one block per `(batch, head)` slice.

L0 knows nothing about tiles, UB, or pipelines. It is the boundary between the user's tensor world and the PTO device world.

## 11.3 L1 — `@pto.jit` kernel entry

```python
@pto.jit(target="a5")
def flash_attention_kernel(
    Q, K, V, O, *,
    BLOCK_Q: pto.constexpr = 128,
    BLOCK_KV: pto.constexpr = 128,
    CAUSAL: pto.constexpr = False,
    NUM_STAGES: pto.constexpr = 2,
):
```

The `@pto.jit` decorator marks the compile + launch boundary. Inputs are Python-native tensors; outputs are written in-place to `O`. Keyword-only `constexpr` parameters (`BLOCK_Q`, `BLOCK_KV`, `CAUSAL`) are baked at compile time.

### 11.3.1 TensorView construction

```python
q_view = pto.make_tensor_view(Q, shape=[batch, seq_q, heads, dim],
                              strides=Q.strides)
k_view = pto.make_tensor_view(K, shape=[batch, seq_k, heads, dim],
                              strides=K.strides)
v_view = pto.make_tensor_view(V, shape=[batch, seq_k, heads, dim],
                              strides=V.strides)
o_view = pto.make_tensor_view(O, shape=[batch, seq_q, heads, dim],
                              strides=O.strides)
```

`make_tensor_view` wraps each framework tensor with a PTO TensorView descriptor — a GM pointer paired with shape and stride metadata. These descriptors are what the rest of the kernel uses to address global memory. No data moves yet.

### 11.3.2 SPMD launch contract

```python
block_idx = pto.get_block_idx()
block_num = pto.get_block_num()
subblock_idx = pto.get_subblock_idx()
subblock_num = pto.get_subblock_num()

batch_idx = block_idx // heads
head_idx = block_idx % heads
```

The launch grid is `[batch * heads]`. Each block computes one `(batch, head)` slice. `get_block_idx()` returns the current block's linear index; dividing by `heads` recovers the batch and head indices.

### 11.3.3 Per-head view selection

```python
q_head = pto.select_head_view(q_view, batch=batch_idx, head=head_idx,
                              shape=[seq_q, dim])
k_head = pto.select_head_view(k_view, batch=batch_idx, head=head_idx,
                              shape=[seq_k, dim])
v_head = pto.select_head_view(v_view, batch=batch_idx, head=head_idx,
                              shape=[seq_k, dim])
o_head = pto.select_head_view(o_view, batch=batch_idx, head=head_idx,
                              shape=[seq_q, dim])
```

`select_head_view` extracts a 2D slice `[seq, dim]` from the 4D tensor view for the current head. The resulting views are the working set for this block's entire computation.

### 11.3.4 Tile allocation

Two categories of tiles are allocated:

**UB-resident tiles** — data tiles that live in the Unified Buffer:

```python
q_tile  = pto.alloc_tile(shape=[Br, dim], dtype=pto.f32)
k_tile  = pto.alloc_tile(shape=[Bc, dim], dtype=pto.f32)
v_tile  = pto.alloc_tile(shape=[Bc, dim], dtype=pto.f32)

o_prev_tile = pto.alloc_tile(shape=[Br, dim], dtype=pto.f32)
o_next_tile = pto.alloc_tile(shape=[Br, dim], dtype=pto.f32)
m_prev_tile = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32)
m_next_tile = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32)
l_prev_tile = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32)
l_next_tile = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32)

s_tile   = pto.alloc_tile(shape=[Br, Bc], dtype=pto.f32)
p_tile   = pto.alloc_tile(shape=[Br, Bc], dtype=pto.f32)
pv_tile  = pto.alloc_tile(shape=[Br, dim], dtype=pto.f32)
alpha_tile = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32)
beta_tile  = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32)
```

The online-softmax algorithm requires **ping-pong state tiles**: `m_prev`/`m_next`, `l_prev`/`l_next`, `o_prev`/`o_next`. After each KV block, `next` becomes `prev` for the following iteration.

**Cube-local scratch tiles** — allocated in specific memory spaces:

```python
q_l0a  = pto.alloc_tile(shape=[Br, dim], dtype=pto.f16,
                        memory_space=pto.MemorySpace.LEFT)
p_l0a  = pto.alloc_tile(shape=[Br, Bc], dtype=pto.f16,
                        memory_space=pto.MemorySpace.LEFT)
rhs_l0b = pto.alloc_tile(shape=[Bc, dim], dtype=pto.f16,
                         memory_space=pto.MemorySpace.RIGHT)
qk_acc_tile = pto.alloc_tile(shape=[Br, Bc], dtype=pto.f32,
                             memory_space=pto.MemorySpace.ACC)
pv_acc_tile = pto.alloc_tile(shape=[Br, dim], dtype=pto.f32,
                             memory_space=pto.MemorySpace.ACC)
```

Cube scratch tiles are NOT UB buffers. `LEFT`, `RIGHT`, and `ACC` are distinct hardware memory spaces inside the Cube unit. They serve as staging for matrix operands and accumulators.

### 11.3.5 SIMT metadata buffer

```python
meta_tile = pto.alloc_tile(shape=[3, 1], dtype=pto.i32)
meta_ptr = pto.tile_buf_addr(meta_tile)
```

A small UB tile stores three scalar loop bounds (`row_start`, `row_stop`, `valid_cols`). `tile_buf_addr` materializes a typed UB pointer into it, which is passed to the ukernel as scalar control metadata.

### 11.3.6 Outer Q loop + inner KV loop

```python
with pto.for_(0, q_blocks, step=1) as qi:
    q_part = pto.partition_view(q_head, offsets=[qi * Br, 0],
                                sizes=[Br, dim])
    o_part = pto.partition_view(o_head, offsets=[qi * Br, 0],
                                sizes=[Br, dim])

    pto.tload(q_part, q_tile)

    m_prev_tile.fill(float("-inf"))
    l_prev_tile.fill(0.0)
    o_prev_tile.fill(0.0)

    kv_loop = pto.for_(0, kv_blocks, step=1).carry(
        m=m_prev_tile, l=l_prev_tile, o=o_prev_tile,
    )
    with kv_loop:
        kj = kv_loop.iv
        m_cur = kv_loop.m
        l_cur = kv_loop.l
        o_cur = kv_loop.o
        k_part = pto.partition_view(k_head,
                    offsets=[kj * Bc, 0], sizes=[Bc, dim])
        v_part = pto.partition_view(v_head,
                    offsets=[kj * Bc, 0], sizes=[Bc, dim])

        kv_block_process(
            q_tile, k_part, v_part, k_tile, v_tile,
            o_cur, o_next_tile,
            m_cur, l_cur, m_next_tile, l_next_tile,
            s_tile, p_tile, pv_tile,
            alpha_tile, beta_tile,
            q_l0a, p_l0a, rhs_l0b,
            qk_acc_tile, pv_acc_tile,
            meta_ptr,
        )

        kv_loop.update(m=m_next_tile, l=l_next_tile, o=o_next_tile)

    o_final_tile = kv_loop.final("o")
    pto.tstore(o_final_tile, o_part)
```

Key points:

- **`tload` at the L1 boundary**: Q is loaded once per Q block using a tile op. The compiler auto-inserts the necessary `set_flag`/`wait_flag` pairs.
- **State initialization**: `fill(float("-inf"))` and `fill(0.0)` initialize the online-softmax accumulators before the first KV block.
- **Carry state**: the inner `kv_loop` carries three ping-pong tiles (`m`, `l`, `o`) across iterations using `.carry(...)` / `.update(...)` / `.final(...)`. After each KV block, the loop updates the carried values to the `_next` tiles. After the loop, `.final("o")` extracts the final output accumulator.
- **`tstore` at the L1 boundary**: writes the final result for this Q block back to GM.

## 11.4 L2 — `@pto.ukernel`

```python
@pto.ukernel
def kv_block_process(
    q_tile, k_part, v_part, k_tile, v_tile,
    o_prev_tile, o_next_tile,
    m_prev_tile, l_prev_tile, m_next_tile, l_next_tile,
    s_tile, p_tile, pv_tile,
    alpha_tile, beta_tile,
    q_l0a, p_l0a, rhs_l0b,
    qk_acc_tile, pv_acc_tile,
    meta_ptr,
):
```

The ukernel processes one KV block against an already-loaded Q tile. It owns the execution sandwich:

### Phase 0 — Stage K/V data

```python
pto.mte_load(k_part, k_tile)
pto.mte_load(v_part, v_tile)
pto.mem_bar(pto.BarrierType.SYNC)
```

`mte_load` copies the current K and V block from GM to UB. `mem_bar` ensures the DMA stores are visible before the cube unit reads `k_tile`/`v_tile`.

### Phase 0b — Materialize loop bounds

```python
materialize_tile_bounds(meta_ptr,
    pto.tile_valid_rows(q_tile),
    pto.tile_valid_rows(k_tile))
row_start = scalar.load(meta_ptr + 0)
row_stop  = scalar.load(meta_ptr + 4)
valid_cols = scalar.load(meta_ptr + 8)
```

The SIMT sub-kernel `materialize_tile_bounds` writes `{0, valid_rows, valid_cols}` into the metadata buffer. The ukernel then loads these scalars. They control the row iteration range in subsequent sub-kernels, handling partial tail blocks.

### Phase 1 — `S = Q @ K^T`

```python
qk_matmul(q_tile, k_tile, q_l0a, rhs_l0b, qk_acc_tile, s_tile)
pto.mem_bar(pto.BarrierType.SYNC)
```

Dispatches the cube sub-kernel. `mem_bar` separates the matrix multiply from the subsequent softmax.

### Phase 2 — Online softmax

```python
online_softmax_rows(
    s_tile, p_tile,
    m_prev_tile, l_prev_tile,
    m_next_tile, l_next_tile,
    alpha_tile, beta_tile,
    row_start, row_stop, valid_cols,
)
pto.mem_bar(pto.BarrierType.SYNC)
```

The simd sub-kernel computes per-row softmax on `S`, updates the running `m`/`l` state, and writes `P`, `alpha`, and `beta`.

### Phase 3 — `PV = P @ V`

```python
pv_matmul(p_tile, v_tile, p_l0a, rhs_l0b, pv_acc_tile, pv_tile)
pto.mem_bar(pto.BarrierType.SYNC)
```

Second cube dispatch. `rhs_l0b` is reused for `V` (it previously held `K`). `pv_acc_tile` is reused from the QK^T accumulator.

### Phase 4 — Blend output

```python
blend_output_rows(
    o_prev_tile, pv_tile, alpha_tile, beta_tile,
    o_next_tile, row_start, row_stop,
    pto.tile_valid_cols(v_tile),
)
pto.mem_bar(pto.BarrierType.SYNC)
```

The simt sub-kernel blends the old output accumulator with the new PV contribution, weighted by `alpha` and `beta`.

### Why the ukernel owns sync

Each `mem_bar` between phases is explicit in the ukernel body. This is intentional: at the L2 micro-instruction level, the user controls pipeline ordering. There is no auto-sync insertion — the ukernel is the single place where the hardware execution sequence is spelled out.

## 11.5 L3a — `@pto.cube` sub-kernels

### `qk_matmul` — `S = Q @ K^T`

```python
@pto.cube
def qk_matmul(q_tile, k_tile, q_l0a, k_l0b, s_acc, s_tile):
    m = pto.tile_valid_rows(q_tile)
    k = pto.tile_valid_cols(q_tile)
    n = pto.tile_valid_rows(k_tile)

    pto.mte_l1_l0a(q_tile, q_l0a, m, k)
    pto.mte_l1_l0b(k_tile, k_l0b, k, n, transpose=True)
    pto.mad(q_l0a, k_l0b, s_acc)
    pto.mte_l0c_ub(s_acc, s_tile, m, n)
```

Four cube ops:

1. **`mte_l1_l0a`**: load Q tile from UB into LEFT scratch (`q_l0a`).
2. **`mte_l1_l0b`**: load K tile from UB into RIGHT scratch (`k_l0b`), with `transpose=True` for K^T.
3. **`mad`**: matrix multiply-accumulate — `s_acc = q_l0a @ k_l0b`.
4. **`mte_l0c_ub`**: write the accumulator result to the UB output tile `s_tile`.

The cube kernel does not allocate scratch — the caller (L1) owns scratch lifetime. The cube kernel only expresses dataflow.

### `pv_matmul` — `PV = P @ V`

```python
@pto.cube
def pv_matmul(p_tile, v_tile, p_l0a, v_l0b, pv_acc, pv_tile):
    m = pto.tile_valid_rows(p_tile)
    k = pto.tile_valid_cols(p_tile)
    n = pto.tile_valid_cols(v_tile)

    pto.mte_l1_l0a(p_tile, p_l0a, m, k)
    pto.mte_l1_l0b(v_tile, v_l0b, k, n)
    pto.mad(p_l0a, v_l0b, pv_acc)
    pto.mte_l0c_ub(pv_acc, pv_tile, m, n)
```

Structurally identical to `qk_matmul`, but without transposition and with different input/output tiles. The scratch tiles `p_l0a`, `v_l0b`, and `pv_acc` are reused across KV blocks — the caller (L1) allocates them once.

## 11.6 L3b — `@pto.simd` online softmax

```python
@pto.simd
def online_softmax_rows(
    s_tile, p_tile,
    m_prev_tile, l_prev_tile,
    m_next_tile, l_next_tile,
    alpha_tile, beta_tile,
    row_start, row_stop, valid_cols,
):
```

The simd kernel iterates over rows with `pto.for_`, processing one row per iteration:

```python
with pto.for_(row_start, row_stop, step=1) as row:
    col_mask = pto.make_mask(pto.f32, valid_cols)

    s_row   = pto.vlds(s_tile[row, 0:])
    m_prev  = scalar.load(m_prev_tile[row, 0])
    l_prev  = scalar.load(l_prev_tile[row, 0])
```

- **Mask creation**: `make_mask(pto.f32, valid_cols)` generates a tail mask for the column dimension. On the last KV block, `valid_cols` may be less than the full block width.
- **Vector load**: `vlds(s_tile[row, 0:])` loads one entire row of `S` from UB into a vector register. The slice syntax `[row, 0:]` selects the full row.
- **Scalar load**: `lds` reads per-row scalars (`m_prev`, `l_prev`) from the state tiles.

### Softmax computation

```python
    row_max   = pto.vcgmax(s_row, col_mask)
    m_next    = scalar.max(m_prev, row_max)

    s_shifted = pto.vsubs(s_row, m_next, col_mask)
    p_row     = pto.vexp(s_shifted, col_mask)

    row_sum   = pto.vcgadd(p_row, col_mask)
    l_scaled  = l_prev * scalar.exp(m_prev - m_next)
    l_next    = l_scaled + row_sum

    alpha = l_scaled / l_next
    beta  = 1.0 / l_next
```

This implements the online-softmax update from the Flash Attention paper:

- `vcgmax` (cross-lane max reduction) finds the row maximum.
- `max(m_prev, m_next)` combines with the running maximum.
- `vsubs` subtracts the scalar `m_next` from every lane (stabilized softmax).
- `vexp` computes `exp(s_shifted)` element-wise.
- `vcgadd` (cross-lane sum reduction) computes the row sum.
- `l_scaled` rescales the previous sum with the running-max correction factor.
- `alpha` and `beta` are the blending coefficients for the output update.

### Store results

```python
    pto.vsts(p_row, p_tile[row, 0:], col_mask)
    scalar.sts(m_next_tile[row, 0], m_next)
    scalar.sts(l_next_tile[row, 0], l_next)
    scalar.sts(alpha_tile[row, 0], alpha)
    scalar.sts(beta_tile[row, 0], beta)
```

- `vsts` stores the vector `p_row` back to UB under the column mask.
- `sts` stores each scalar to its respective UB tile.

**Boundary contract**: vreg values (`s_row`, `p_row`, `row_max`, `row_sum`) never escape the simd kernel. All persistent state is written to UB tiles.

## 11.7 L3c — `@pto.simt` sub-kernels

### `materialize_tile_bounds` — scalar metadata

```python
@pto.simt
def materialize_tile_bounds(meta_ptr, valid_rows, valid_cols):
    scalar.sts(meta_ptr + 0, 0)
    scalar.sts(meta_ptr + 4, valid_rows)
    scalar.sts(meta_ptr + 8, valid_cols)
```

Three scalar stores write the loop bounds into the metadata buffer. `meta_ptr` is a typed UB pointer; `+ 0`, `+ 4`, `+ 8` are byte offsets (three `i32` values). This is the simplest sub-kernel in the sketch — it handles scalar control metadata, not vector math.

### `blend_output_rows` — output accumulation

```python
@pto.simt
def blend_output_rows(o_prev_tile, pv_tile, alpha_tile, beta_tile,
                      o_next_tile, row_start, row_stop, valid_dim):
    with pto.for_(row_start, row_stop, step=1) as row:
        alpha = scalar.load(alpha_tile[row, 0])
        beta  = scalar.load(beta_tile[row, 0])

        with pto.for_(0, valid_dim, step=1) as col:
            o_prev = scalar.load(o_prev_tile[row, col])
            pv_val = scalar.load(pv_tile[row, col])
            o_next = alpha * o_prev + beta * pv_val
            scalar.sts(o_next_tile[row, col], o_next)
```

This is a scalar element-wise blend over the tile domain:

```
O_next[row, col] = alpha[row] * O_prev[row, col] + beta[row] * PV[row, col]
```

The SIMT kernel walks the tile element by element with nested `pto.for_` loops. Each iteration loads two scalars (`o_prev` and `pv_val`), computes the weighted sum, and stores the result. The `alpha`/`beta` coefficients are per-row (loaded once per row), while the blend is per-element.

**Why SIMT instead of SIMD?** The intent is to contrast with `online_softmax_rows`: softmax is dominated by row-wise vector reductions and exponentials — natural SIMD work. The final blend is a simple linear combination with per-row coefficients — expressing it as explicit scalar work-items makes the per-element access pattern explicit and leaves the compiler free to vectorize or fuse as it sees fit.

### Context manager alternative

For trivial sub-kernels like `materialize_tile_bounds`, a named function is overkill — the context manager form keeps the logic inline where it's used. Here is how the ukernel body would look with `materialize_tile_bounds` inlined:

```python
@pto.ukernel
def kv_block_process(...):
    pto.mte_load(k_part, k_tile)
    pto.mte_load(v_part, v_tile)
    pto.mem_bar(pto.BarrierType.SYNC)

    # Inline SIMT: materialize loop bounds (replaces the named @pto.simt function)
    with pto.simt():
        scalar.sts(meta_ptr + 0, 0)
        scalar.sts(meta_ptr + 4, valid_rows)
        scalar.sts(meta_ptr + 8, valid_cols)

    pto.mem_bar(pto.BarrierType.SYNC)

    qk_matmul(q_tile, k_tile, ...)
    ...
```

The `with pto.simt():` block is semantically identical to calling a `@pto.simt` function — the compiler treats it as an anonymous sub-kernel. For 3-line helpers that have no reuse, the context manager avoids the indirection of a separate function. For complex, reusable logic like `online_softmax_rows` or `qk_matmul`, the named decorator form remains the better fit.

## 11.8 Putting it all together: one KV block execution

For one KV block, the full execution sequence is:

| Step | Layer | Operation | Hardware |
|------|-------|-----------|----------|
| 1 | L1 | `tload(q_part, q_tile)` | MTE2 → UB |
| 2 | L2 | `mte_load(k_part, k_tile)` | MTE2 → UB |
| 3 | L2 | `mte_load(v_part, v_tile)` | MTE2 → UB |
| 4 | L2 | `mem_bar(SYNC)` | — |
| 5 | L3c | `materialize_tile_bounds` | SIMT |
| 6 | L3a | `qk_matmul` (mte_l1_l0a, mte_l1_l0b, mad, mte_l0c_ub) | Cube |
| 7 | L2 | `mem_bar(SYNC)` | — |
| 8 | L3b | `online_softmax_rows` (vlds, vcgmax, vexp, vcgadd, vsts, ...) | SIMD |
| 9 | L2 | `mem_bar(SYNC)` | — |
| 10 | L3a | `pv_matmul` | Cube |
| 11 | L2 | `mem_bar(SYNC)` | — |
| 12 | L3c | `blend_output_rows` | SIMT |
| 13 | L2 | `mem_bar(SYNC)` | — |

After all KV blocks: L1 issues `tstore(o_final_tile, o_part)` to write the result back to GM.

## 11.9 Design patterns in this sketch

**Ping-pong state for online accumulators**: `m_prev`/`m_next`, `l_prev`/`l_next`, `o_prev`/`o_next` make the state transition explicit. After each KV block, the caller swaps the ping-pong pair (via `kv_loop.update(...)`) rather than aliasing in place.

**Scratch reuse**: `rhs_l0b` serves both `K` (in `qk_matmul`) and `V` (in `pv_matmul`). `pv_acc_tile` reuses the accumulator from QK^T. The caller (L1) allocates once; the ukernel passes them to both cube sub-kernels.

**Tile-level boundary vs micro-instruction boundary**: `tload`/`tstore` appear only in `@pto.jit`. `mte_load`/`mte_store` appear only in `@pto.ukernel`. This is the key abstraction split: L1 operates on tiles, L2 operates on micro-instructions.

**No vreg across sub-kernel boundaries**: vector registers are local to each `@pto.simd` kernel. Data crosses sub-kernel boundaries through UB tiles — the boundary contract is enforced by the type system.

**L3 invocation flexibility**: This sketch uses the explicit `@pto.ukernel` → L3 path for full control over MTE and sync. For simpler kernels that don't need that control, L3 sub-kernels can be called directly from `@pto.jit` (the compiler handles MTE + sync) or written inline as context managers (`with pto.simd():`, etc.). See Chapter 3 for details.

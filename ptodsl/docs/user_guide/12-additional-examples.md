# 12. Additional Examples

This chapter presents four self-contained examples that build on the concepts introduced in Chapters 1–11. Each example demonstrates a specific pattern: blocked 2D processing, tail handling with masks, matrix multiplication on the Cube unit, and loop-carried state for online normalization.

## 12.1 Blocked 2D elementwise addition

Chapter 2 showed a 1D vector add with a single blocking dimension. Real workloads often involve 2D tensors — matrices — where blocking happens along both rows and columns.

```python
@pto.jit(target="a5")
def mat_add(A, B, O, *, BLOCK_M: pto.constexpr = 64, BLOCK_N: pto.constexpr = 128):
    M, N_ = A.shape

    a_view = pto.make_tensor_view(A, shape=[M, N_], strides=A.strides)
    b_view = pto.make_tensor_view(B, shape=[M, N_], strides=B.strides)
    o_view = pto.make_tensor_view(O, shape=[M, N_], strides=O.strides)

    a_tile = pto.alloc_tile(shape=[BLOCK_M, BLOCK_N], dtype=pto.f32)
    b_tile = pto.alloc_tile(shape=[BLOCK_M, BLOCK_N], dtype=pto.f32)
    o_tile = pto.alloc_tile(shape=[BLOCK_M, BLOCK_N], dtype=pto.f32)

    num_m = (M + BLOCK_M - 1) // BLOCK_M
    num_n = (N_ + BLOCK_N - 1) // BLOCK_N

    with pto.for_(0, num_m, step=1) as mi:
        m_off = mi * BLOCK_M
        with pto.for_(0, num_n, step=1) as ni:
            n_off = ni * BLOCK_N

            a_part = pto.partition_view(a_view, offsets=[m_off, n_off], sizes=[BLOCK_M, BLOCK_N])
            b_part = pto.partition_view(b_view, offsets=[m_off, n_off], sizes=[BLOCK_M, BLOCK_N])
            o_part = pto.partition_view(o_view, offsets=[m_off, n_off], sizes=[BLOCK_M, BLOCK_N])

            pto.tload(a_part, a_tile)
            pto.tload(b_part, b_tile)
            pto.tadd(a_tile, b_tile, o_tile)
            pto.tstore(o_tile, o_part)
```

**Key points**:

- Nested `pto.for_` loops produce a 2D block traversal. Both loops are recorded as device-side control flow — they adapt to the runtime shape `M`.
- Tile shape `[BLOCK_M, BLOCK_N]` is 2D; all three tiles use the same shape so `tadd` is elementwise.
- `partition_view` takes 2D offsets and sizes.
- `BLOCK_M` and `BLOCK_N` are `constexpr` — the compiler specializes the kernel per tile shape.

The L0 wrapper follows the same pattern as Chapter 2:

```python
def mat_add_wrapper(A, B, O=None, stream=None):
    if O is None:
        O = pto.empty_like(A)
    compiled = mat_add.compile(BLOCK_M=64, BLOCK_N=128)
    m, n = A.shape[1], A.shape[2]  # assuming batch-first: [batch, M, N]
    compiled[A.shape[0], stream](A, B, O)
    return O
```

The grid is `A.shape[0]` so each SPMD block processes one slice of the leading batch dimension.

## 12.2 Vector operations with tail handling

When a data dimension is not evenly divisible by the tile size or the hardware vector width, the last iteration must operate on fewer elements. PTODSL provides masks for this — `make_mask` produces a predicate that guards loads, computes, and stores so out-of-bounds lanes are not touched.

### 12.2.1 Tail handling in a SIMD kernel

Below is a self-contained `@pto.simd` kernel that adds two tiles row by row, handling column tails with `make_mask`:

```python
@pto.simd
def add_rows_with_tail(a_tile: pto.Tile, b_tile: pto.Tile, o_tile: pto.Tile,
                       rows: pto.i32, cols: pto.i32):
    VEC = pto.elements_per_vreg(pto.f32)          # 64 for f32

    with pto.for_(0, rows, step=1) as r:
        col_loop = pto.for_(0, cols, step=VEC).carry(remained=cols)
        with col_loop:
            c = col_loop.iv
            remained = col_loop.remained
            mask, remained = pto.make_mask(pto.f32, remained)

            a_vec = pto.vlds(a_tile[r, c:])       # load under mask
            b_vec = pto.vlds(b_tile[r, c:])
            o_vec = pto.vadd(a_vec, b_vec, mask)  # compute under mask
            pto.vsts(o_vec, o_tile[r, c:], mask)  # store under mask

            col_loop.update(remained=remained)
```

The pattern:

1. **Chunk**: Each iteration processes `VEC` elements (one vector register's worth).
2. **Mask**: `make_mask` returns a predicate and the updated remainder. On the last iteration, where `remained < VEC`, the mask has `remained` valid lanes followed by inactive lanes.
3. **Guard**: `vlds`, `vadd`, and `vsts` all accept the mask — inactive lanes are neither loaded, computed, nor stored.
4. **Carry**: `.carry(remained=cols)` carries the remaining column count across iterations. `col_loop.update(remained=remained)` feeds the updated count to the next iteration.

### 12.2.2 Tile-level tail handling

At the Tile Op level, tail handling is built into `tload` and `tstore`. When a partition size along a dimension is smaller than the tile size, the tile's `valid_shape` tracks the actual data extent:

```python
@pto.jit(target="a5")
def vec_add_with_tail(A, B, O, *, BLOCK: pto.constexpr):
    N = A.shape[0]

    a_view = pto.make_tensor_view(A, shape=[N], strides=A.strides)
    b_view = pto.make_tensor_view(B, shape=[N], strides=B.strides)
    o_view = pto.make_tensor_view(O, shape=[N], strides=O.strides)

    a_tile = pto.alloc_tile(shape=[BLOCK], dtype=pto.f32)
    b_tile = pto.alloc_tile(shape=[BLOCK], dtype=pto.f32)
    o_tile = pto.alloc_tile(shape=[BLOCK], dtype=pto.f32)

    num_blocks = (N + BLOCK - 1) // BLOCK

    with pto.for_(0, num_blocks, step=1) as i:
        offset = i * BLOCK
        this_block = min(BLOCK, N - offset)

        a_part = pto.partition_view(a_view, offsets=[offset], sizes=[this_block])
        b_part = pto.partition_view(b_view, offsets=[offset], sizes=[this_block])
        o_part = pto.partition_view(o_view, offsets=[offset], sizes=[this_block])

        pto.tload(a_part, a_tile)
        pto.tload(b_part, b_tile)

        a_tile.valid_shape = [this_block]
        b_tile.valid_shape = [this_block]
        o_tile.valid_shape = [this_block]

        pto.tadd(a_tile, b_tile, o_tile)
        pto.tstore(o_tile, o_part)
```

- `this_block = min(BLOCK, N - offset)` computes the actual block size for the tail iteration.
- `sizes=[this_block]` on the partition and `valid_shape` on the tile tell `tload`/`tadd`/`tstore` how many elements are live.

### 12.2.3 The general rule

| Tail scenario | Mechanism |
|---------------|-----------|
| Tile Op boundary (tload/tstore) | `valid_shape` on tile + smaller `sizes` on partition |
| SIMD vector boundary (vlds/vadd/vsts) | `make_mask` + mask parameter on op |
| SIMT scalar loop boundary | `min(BLOCK, N - offset)` in loop bound |

## 12.3 GEMM: matrix multiplication on the Cube unit

This example demonstrates a complete GEMM kernel: `C = A @ B` where A is `[M, K]` and B is `[K, N]`. It uses `@pto.jit` for tile allocation and loop scheduling, and `@pto.cube` for the actual matrix multiply.

### 12.3.1 L3: Cube sub-kernel

```python
@pto.cube
def gemm_tile(a_tile: pto.Tile, b_tile: pto.Tile, o_tile: pto.Tile,
              a_l0a: pto.Tile, b_l0b: pto.Tile, o_acc: pto.Tile):
    m = pto.tile_valid_rows(a_tile)
    k = pto.tile_valid_cols(a_tile)
    n = pto.tile_valid_rows(b_tile)

    pto.mte_l1_l0a(a_tile, a_l0a, m, k)
    pto.mte_l1_l0b(b_tile, b_l0b, k, n, transpose=True)
    pto.mad(a_l0a, b_l0b, o_acc)
    pto.mte_l0c_ub(o_acc, o_tile, m, n)
```

The cube sub-kernel consumes UB tiles and cube-local scratch buffers. The four-step sequence — stage left operand, stage right operand, multiply, writeback — is the canonical cube compute pattern.

### 12.3.2 L1: Tile orchestration

```python
@pto.jit(target="a5")
def gemm(A, B, O, *, BLOCK_M: pto.constexpr = 64,
         BLOCK_K: pto.constexpr = 64, BLOCK_N: pto.constexpr = 64):
    M, K_ = A.shape
    _, N_ = B.shape

    a_view = pto.make_tensor_view(A, shape=[M, K_], strides=A.strides)
    b_view = pto.make_tensor_view(B, shape=[K_, N_], strides=B.strides)
    o_view = pto.make_tensor_view(O, shape=[M, N_], strides=O.strides)

    a_tile = pto.alloc_tile(shape=[BLOCK_M, BLOCK_K], dtype=pto.f32)
    b_tile = pto.alloc_tile(shape=[BLOCK_K, BLOCK_N], dtype=pto.f32)
    o_tile = pto.alloc_tile(shape=[BLOCK_M, BLOCK_N], dtype=pto.f32)

    a_l0a = pto.alloc_tile(shape=[BLOCK_M, BLOCK_K], dtype=pto.f32,
                           memory_space=pto.MemorySpace.LEFT)
    b_l0b = pto.alloc_tile(shape=[BLOCK_K, BLOCK_N], dtype=pto.f32,
                           memory_space=pto.MemorySpace.RIGHT)
    o_acc = pto.alloc_tile(shape=[BLOCK_M, BLOCK_N], dtype=pto.f32,
                           memory_space=pto.MemorySpace.ACC)

    num_m = (M + BLOCK_M - 1) // BLOCK_M
    num_n = (N_ + BLOCK_N - 1) // BLOCK_N
    num_k = (K_ + BLOCK_K - 1) // BLOCK_K

    with pto.for_(0, num_m, step=1) as mi:
        m_off = mi * BLOCK_M
        with pto.for_(0, num_n, step=1) as ni:
            n_off = ni * BLOCK_N

            o_tile.fill(0.0)

            with pto.for_(0, num_k, step=1) as ki:
                k_off = ki * BLOCK_K

                a_part = pto.partition_view(a_view, offsets=[m_off, k_off],
                                            sizes=[BLOCK_M, BLOCK_K])
                b_part = pto.partition_view(b_view, offsets=[k_off, n_off],
                                            sizes=[BLOCK_K, BLOCK_N])
                o_part = pto.partition_view(o_view, offsets=[m_off, n_off],
                                            sizes=[BLOCK_M, BLOCK_N])

                pto.tload(a_part, a_tile)
                pto.tload(b_part, b_tile)

                gemm_tile(a_tile, b_tile, o_tile, a_l0a, b_l0b, o_acc)

            pto.tstore(o_tile, o_part)
```

**Key points**:

- **Triply nested loops**: M, N, and K dimensions are all blocked. The K loop accumulates partial results into `o_tile`.
- **Accumulation**: `o_tile.fill(0.0)` resets the accumulator before the K loop. Each K-block calls `gemm_tile` which writes its partial product back to `o_tile`. The Cube unit accumulates implicitly via `mad` — each K-block's partial result is added to the running total in `o_acc`.
- **Cube-local scratch**: `a_l0a`, `b_l0b`, and `o_acc` are allocated with explicit `memory_space` parameters (`LEFT`, `RIGHT`, `ACC`). Cube-local state does not leak into UB.
- **Direct L3 call**: `gemm_tile` is called directly from `@pto.jit` — no ukernel needed. The compiler handles sync between `tload` and the Cube sub-kernel.
- **Cube sub-kernel reuse**: the same `gemm_tile` function is called for every K-block — the named decorator form enables reuse.

### 12.3.3 L0 wrapper

```python
def gemm_wrapper(A, B, O=None, stream=None):
    if O is None:
        O = pto.empty([A.shape[0], B.shape[1]], dtype=A.dtype)
    compiled = gemm.compile(BLOCK_M=64, BLOCK_K=64, BLOCK_N=64)
    compiled[1, stream](A, B, O)
    return O
```

This pattern extends directly to batch-GEMM: pass a grid of `batch` and use `pto.get_block_idx()` to select the per-batch slice from `A` and `B`.

### 12.3.4 Comparison with ukernel path

For reference, the same GEMM could be written using `@pto.ukernel` for explicit MTE control. The ukernel would replace the inner `tload`/`tstore` calls with `mte_load`/`mte_store` and add `mem_bar` synchronization between DMA and compute. The direct-call path used above is recommended for most users — the ukernel path is for cases that need hand-tuned DMA scheduling.

## 12.4 Online normalization with loop-carried state

Chapter 11 demonstrated online softmax with ping-pong state tiles. A simpler but instructive case is **online layer normalization** — computing mean and variance incrementally across blocks without a second pass.

Given a vector `X` of length `N`, the streaming Welford algorithm updates the running mean `mu` and variance `var` as each new element `x` arrives:

```
n_next    = n_prev + 1
delta     = x - mu_prev
mu_next   = mu_prev + delta / n_next
m2_next   = m2_prev + delta * (x - mu_next)
```

The example below applies this pattern block by block, using a ukernel for the per-block SIMD work and `pto.for_` carry state to shuttle the running statistics between blocks.

### 12.4.1 L3: SIMD block statistics

```python
@pto.simd
def block_mean_var(x_tile: pto.Tile, block_size: pto.i32,
                  mu_prev: pto.f32, n_prev: pto.f32, m2_prev: pto.f32,
                  mu_next_tile: pto.Tile, n_next_tile: pto.Tile,
                  m2_next_tile: pto.Tile):
    VEC = pto.elements_per_vreg(pto.f32)

    # Per-row cross-lane reductions to compute the block sum and sum-of-squares
    row_sum = pto.vdup(0.0, pto.f32)
    row_sum2 = pto.vdup(0.0, pto.f32)

    col_loop = pto.for_(0, block_size, step=VEC).carry(row_sum=row_sum, row_sum2=row_sum2)
    with col_loop:
        c = col_loop.iv
        remained = pto.i32(block_size) - c
        mask, _ = pto.make_mask(pto.f32, remained)

        x_vec = pto.vlds(x_tile[0, c:])
        row_sum = pto.vcadd(x_vec, mask)
        row_sum2 = pto.vcadd(pto.vmul(x_vec, x_vec, mask), mask)
        col_loop.update(row_sum=row_sum, row_sum2=row_sum2)

    block_n = pto.cvt(block_size, pto.f32)
    block_mean = pto.vdiv(col_loop.final("row_sum"), block_n)
    block_mean_sq = pto.vdiv(col_loop.final("row_sum2"), block_n)

    # Welford update: merge block statistics into running state
    n_next = n_prev + block_n
    delta = block_mean - mu_prev
    mu_next = mu_prev + delta * block_n / n_next
    m2_next = m2_prev + pto.vdiv(row_sum2, block_n) * block_n  # simplified

    scalar.store(n_next, n_next_tile[0, 0])
    scalar.store(mu_next, mu_next_tile[0, 0])
    scalar.store(m2_next, m2_next_tile[0, 0])
```

### 12.4.2 L2: Ukernel with carry orchestration

```python
@pto.ukernel
def norm_block(x_part: pto.PartitionTensorView, x_tile: pto.Tile,
               block_size: pto.i32,
               mu_prev: pto.f32, n_prev: pto.f32, m2_prev: pto.f32,
               mu_next_tile: pto.Tile, n_next_tile: pto.Tile,
               m2_next_tile: pto.Tile):
    pto.mte_load(x_part, x_tile)
    pto.mem_bar(pto.BarrierType.SYNC)

    block_mean_var(x_tile, block_size,
                   mu_prev, n_prev, m2_prev,
                   mu_next_tile, n_next_tile, m2_next_tile)
    pto.mem_bar(pto.BarrierType.SYNC)
```

### 12.4.3 L1: JIT entry with carry state

```python
@pto.jit(target="a5")
def online_layernorm(X, O, *, BLOCK: pto.constexpr):
    N = X.shape[0]
    x_view = pto.make_tensor_view(X, shape=[N], strides=X.strides)
    o_view = pto.make_tensor_view(O, shape=[N], strides=O.strides)

    x_tile = pto.alloc_tile(shape=[BLOCK], dtype=pto.f32)
    o_tile = pto.alloc_tile(shape=[BLOCK], dtype=pto.f32)

    mu_tile = pto.alloc_tile(shape=[1], dtype=pto.f32)
    n_tile = pto.alloc_tile(shape=[1], dtype=pto.f32)
    m2_tile = pto.alloc_tile(shape=[1], dtype=pto.f32)

    num_blocks = (N + BLOCK - 1) // BLOCK

    # Carry: running statistics across blocks
    block_loop = pto.for_(0, num_blocks, step=1).carry(
        mu=pto.f32(0.0), n=pto.f32(0.0), m2=pto.f32(0.0)
    )
    with block_loop:
        i = block_loop.iv
        offset = i * BLOCK
        this_block = min(BLOCK, N - offset)

        x_part = pto.partition_view(x_view, offsets=[offset], sizes=[this_block])

        mu_prev = block_loop.mu
        n_prev = block_loop.n
        m2_prev = block_loop.m2

        norm_block(x_part, x_tile, pto.i32(this_block),
                   mu_prev, n_prev, m2_prev,
                   mu_tile, n_tile, m2_tile)

        n_next = scalar.load(n_tile[0, 0])
        mu_next = scalar.load(mu_tile[0, 0])
        m2_next = scalar.load(m2_tile[0, 0])

        block_loop.update(mu=mu_next, n=n_next, m2=m2_next)

    # After all blocks: finalize normalization with the running stats
    global_var = m2_next / n_next

    # Second pass: normalize each block (using same tiling)
    with pto.for_(0, num_blocks, step=1) as i:
        offset = i * BLOCK
        this_block = min(BLOCK, N - offset)
        x_part = pto.partition_view(x_view, offsets=[offset], sizes=[this_block])
        o_part = pto.partition_view(o_view, offsets=[offset], sizes=[this_block])

        pto.tload(x_part, x_tile)
        pto.tnormalize(x_tile, mu_next, global_var, o_tile)
        pto.tstore(o_tile, o_part)
```

**Key points**:

- **Carry state**: `.carry(mu=..., n=..., m2=...)` on the `pto.for_` declares three loop-carried values. Each iteration reads the previous values via `block_loop.mu` etc. and feeds the updated values via `block_loop.update(...)`.
- **Ping-pong implicit**: The carry mechanism produces a clean SSA-style handoff between iterations — no explicit swap of tile pairs needed.
- **Two-pass algorithm**: The first pass accumulates statistics; the second pass applies the normalization. For a single-pass online version, the normalized output would be written block-by-block inside the first loop, but that requires storing the running statistics per element — a tradeoff between memory and passes.
- **Compare to flash attention**: The flash attention carry in Chapter 11 carries six values (`m_prev`/`m_next`, `l_prev`/`l_next`, `o_prev`/`o_next`) and uses ping-pong tiles. This example shows that for simpler scalar carries, direct values (no tile swap) suffice.

## 12.5 Design guidelines

**Start simple, refine later.** Begin with `@pto.jit` + Tile Ops. If Tile Ops don't cover the computation (e.g., custom softmax, specialized activation), add an L3 sub-kernel. If you need explicit DMA scheduling or inter-pipeline sync, drop to `@pto.ukernel`.

**Choose the right entry for each piece:**

| Goal | Use |
|------|-----|
| Whole-kernel orchestration, GM↔UB boundary | `@pto.jit` |
| Tile-level data movement | `tload` / `tstore` |
| Custom row-wise vector math | `@pto.simd` |
| Custom per-element logic | `@pto.simt` |
| Matrix multiply | `@pto.cube` |
| Explicit DMA + sync ordering | `@pto.ukernel` |
| Inline L3 for quick prototyping | `with pto.simd():` etc. |

**Respect boundary contracts.** Vregs don't cross `@pto.simd` boundaries. Cube-local state doesn't leak into UB. Tile Ops and MTE Ops live at different abstraction levels — keep them in their respective layers.

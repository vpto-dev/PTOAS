# 2. Quick Start

This chapter walks through a minimal but complete PTODSL kernel — elementwise vector addition — covering the essential concepts you need to start writing your own kernels.

## 2.1 A first kernel: elementwise vector add

```python
from ptodsl import pto


@pto.jit(target="a5")
def vec_add(A, B, O, *, N: pto.constexpr):
    """O = A + B, elementwise, for vectors of length N."""

    # Describe the GM tensors.
    a_view = pto.make_tensor_view(A, shape=[N], strides=A.strides)
    b_view = pto.make_tensor_view(B, shape=[N], strides=B.strides)
    o_view = pto.make_tensor_view(O, shape=[N], strides=O.strides)

    # Allocate a UB tile to hold one block of each vector.
    a_tile = pto.alloc_tile(shape=[N], dtype=pto.f32)
    b_tile = pto.alloc_tile(shape=[N], dtype=pto.f32)
    o_tile = pto.alloc_tile(shape=[N], dtype=pto.f32)

    # Partition the GM views to cover the whole vector.
    a_part = pto.partition_view(a_view, offsets=[0], sizes=[N])
    b_part = pto.partition_view(b_view, offsets=[0], sizes=[N])
    o_part = pto.partition_view(o_view, offsets=[0], sizes=[N])

    # Load A and B from GM into UB tiles.
    pto.tload(a_part, a_tile)
    pto.tload(b_part, b_tile)

    # Elementwise add on the tiles.
    pto.tadd(a_tile, b_tile, o_tile)

    # Store the result back to GM.
    pto.tstore(o_tile, o_part)
```

Let us step through each piece.

### The entry point

```python
@pto.jit(target="a5")
def vec_add(A, B, O, *, N: pto.constexpr):
```

`@pto.jit` marks this function as a launchable PTO kernel. The positional parameters `A`, `B`, `O` are Python-native tensors — they arrive from NumPy, torch-npu, or any framework that provides a shape and strides. The keyword-only argument `N` is a compile-time constant declared with `pto.constexpr`; the compiler specializes the kernel for each value of `N`.

### Describing GM tensors

```python
a_view = pto.make_tensor_view(A, shape=[N], strides=A.strides)
```

`make_tensor_view` wraps a Python tensor into a `TensorView` — a descriptor that tells the kernel how to address the tensor in global memory. You provide the logical shape and the stride (in elements) of each dimension.

### Allocating on-chip buffers

```python
a_tile = pto.alloc_tile(shape=[N], dtype=pto.f32)
```

`alloc_tile` reserves space in the Unified Buffer (UB). A `Tile` is a 2D buffer that lives on-chip during kernel execution. Every tile has a `shape` and a `dtype`.

### Partitioning GM views

```python
a_part = pto.partition_view(a_view, offsets=[0], sizes=[N])
```

`partition_view` creates a sub-view of a `TensorView` at a given offset and size. It describes *which part* of the GM tensor a `tload` or `tstore` should operate on. For this simple whole-vector example the offset is zero and the size equals the full length; in a blocked kernel you would slide the offset through a loop.

### Moving data: tload and tstore

```python
pto.tload(a_part, a_tile)   # GM → UB
pto.tstore(o_tile, o_part)  # UB → GM
```

`tload` copies a block of data from GM (described by a partition) into a UB tile. `tstore` copies a UB tile back to GM. These are **Tile Ops** — they operate on entire tile buffers at once.

### Computing on tiles

```python
pto.tadd(a_tile, b_tile, o_tile)
```

`tadd` performs elementwise addition of two tiles. The result is written to a third tile. PTODSL provides a rich set of Tile-level compute instructions — `texp`, `trowsum`, `tcvt`, `tsel`, and many more — covered in Chapter 8.

## 2.2 A blocked version with a loop

The kernel above assumes the entire vector fits in one UB tile. For vectors longer than the maximum tile size, you need to process them in blocks. The length `N` is not known until the kernel is launched — it comes from the actual input tensor:

```python
@pto.jit(target="a5")
def vec_add_blocked(A, B, O, *, BLOCK: pto.constexpr):
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

        a_part = pto.partition_view(a_view, offsets=[offset], sizes=[BLOCK])
        b_part = pto.partition_view(b_view, offsets=[offset], sizes=[BLOCK])
        o_part = pto.partition_view(o_view, offsets=[offset], sizes=[BLOCK])

        pto.tload(a_part, a_tile)
        pto.tload(b_part, b_tile)
        pto.tadd(a_tile, b_tile, o_tile)
        pto.tstore(o_tile, o_part)
```

Here `N` is dynamic — it comes from `A.shape[0]` and can differ across launches. The loop bound `num_blocks` depends on `N`, so `pto.for_` records a structured loop in the IR rather than unrolling at trace time. The `BLOCK` parameter stays `constexpr` because it is a tuning knob, not data-dependent. Chapter 5 covers this distinction in detail.

## 2.3 Compile and launch

Once the kernel is defined, you compile it and then launch it:

```python
# Compile once, cache the result.
compiled = vec_add.compile(N=1024)

# Allocate or obtain input/output tensors (NumPy, torch-npu, ...).
import numpy as np
A = np.random.randn(1024).astype(np.float32)
B = np.random.randn(1024).astype(np.float32)
O = np.empty_like(A)

# Launch on the NPU.
compiled[1, None](A, B, O)
```

- `.compile(**constexprs)` traces the kernel body, lowers it through the PTOAS pipeline, and returns a compiled handle. Repeated calls with the same configuration hit the cache.
- `compiled[grid, stream](args...)` launches the compiled kernel. `grid` is the number of SPMD blocks; `stream` is the NPU stream (or `None` for the default).

## 2.4 SPMD launch

For workloads that can be parallelized across multiple blocks, specify a grid:

```python
# Process batch * heads slices in parallel.
compiled[batch * heads, stream](Q, K, V, O)
```

Inside the kernel, each block queries its index:

```python
block_idx = pto.get_block_idx()
block_num = pto.get_block_num()
```

This lets you map different data slices to different blocks — for example, one block per (batch, head) pair in flash attention.

## 2.5 Dropping down to micro-instructions

The examples above used Tile Ops (`tload`, `tadd`, `tstore`), which operate on entire tiles at once. When you need finer control — for instance, writing a custom softmax or an activation that maps directly to vector hardware — you can drop down to the micro-instruction level. This involves three layers working together:

```python
# L3: hardware-bound SIMD kernel — vector instructions on individual rows.
@pto.simd
def add_rows(a_tile: pto.Tile, b_tile: pto.Tile, o_tile: pto.Tile,
             rows: pto.i32, cols: pto.i32):
    VEC = pto.elements_per_vreg(pto.f32)
    with pto.for_(0, rows, step=1) as r:
        col_loop = pto.for_(0, cols, step=VEC).carry(remained=cols)
        with col_loop:
            c = col_loop.iv
            remained = col_loop.remained
            mask, remained = pto.make_mask(pto.f32, remained)
            a_vec = pto.vlds(a_tile[r, c:])
            b_vec = pto.vlds(b_tile[r, c:])
            o_vec = pto.vadd(a_vec, b_vec, mask)
            pto.vsts(o_vec, o_tile[r, c:], mask)
            col_loop.update(remained=remained)


# L2: ukernel — DMA staging, then dispatch the SIMD kernel.
@pto.ukernel
def add_block(a_part: pto.PartitionTensorView,
              b_part: pto.PartitionTensorView,
              o_part: pto.PartitionTensorView,
              a_tile: pto.Tile, b_tile: pto.Tile, o_tile: pto.Tile,
              rows: pto.i32, cols: pto.i32):
    pto.mte_load(a_part, a_tile)
    pto.mte_load(b_part, b_tile)
    pto.mem_bar(pto.BarrierType.SYNC)

    add_rows(a_tile, b_tile, o_tile, rows, cols)
    pto.mem_bar(pto.BarrierType.SYNC)

    pto.mte_store(o_tile, o_part)


# L1: JIT entry — tile allocation, partitioning, launch.
@pto.jit(target="a5")
def vec_add_micro(A, B, O, *, BLOCK: pto.constexpr):
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
        a_part = pto.partition_view(a_view, offsets=[offset], sizes=[BLOCK])
        b_part = pto.partition_view(b_view, offsets=[offset], sizes=[BLOCK])
        o_part = pto.partition_view(o_view, offsets=[offset], sizes=[BLOCK])
        add_block(a_part, b_part, o_part, a_tile, b_tile, o_tile, 1, BLOCK)
```

- **L1 `@pto.jit`**: allocates tiles, partitions the GM views, and loops over blocks — the same tile-level orchestration as Section 2.2, but now calling a ukernel instead of Tile Ops.

- **L2 `@pto.ukernel`**: stages data with `mte_load`, synchronizes with `mem_bar`, dispatches the SIMD kernel, synchronizes again, then writes back with `mte_store`. The ukernel owns the hardware-level sequencing.

- **L3 `@pto.simd`**: the outer `pto.for_` iterates over rows, the inner `pto.for_` iterates over column chunks of the hardware vector width (`elements_per_vreg`). Each iteration loads a vector-width slice into a `vreg`, does the addition under a mask (for tail elements), and stores the result back. Both loops are recorded as structured control flow IR — the compiler decides whether to keep them or unroll them.

Chapter 3 covers the full decorator family; Chapters 7–10 cover each operation family in detail.

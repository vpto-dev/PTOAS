# 3. Kernel Entry Points and Sub-Kernels

PTODSL provides five decorators that mark functions as PTO kernels, plus three context managers for inline use. This chapter is a reference for each entry point — its role, parameter contract, and boundary constraints.

## 3.1 Decorator family overview

```
@pto.jit          L1   Top-level JIT entry — compile, cache, launch
@pto.ukernel      L2   Micro-instruction orchestration (MTE + sync)
@pto.cube         L3   Matrix multiplication on the Cube unit
@pto.simd         L3   Vector math on the SIMD unit
@pto.simt         L3   Scalar compute on the SIMT unit
```

L3 sub-kernels can be invoked in two ways:

1. **As decorated functions** (`@pto.cube` / `@pto.simd` / `@pto.simt`) — reusable, named sub-kernels that can be called from `@pto.ukernel` or directly from `@pto.jit`.
2. **As context managers** (`with pto.cube():` / `with pto.simd():` / `with pto.simt():`) — inline L3 blocks for quick prototyping or one-off compute snippets inside any kernel.

Calling an L3 sub-kernel directly from `@pto.jit` skips the ukernel layer: you stage data with `tload`/`tstore` instead of `mte_load`/`mte_store`, and PTOAS handles the synchronization between Tile Ops and L3 compute automatically. This is the recommended path for most users — drop down to `@pto.ukernel` only when you need explicit control over micro-instruction ordering and synchronization.

## 3.2 `@pto.jit` — top-level JIT entry

### Role

`@pto.jit` marks a function as a launchable PTO kernel. It owns compilation (tracing + lowering), caching, and runtime launch binding. This is the only decorator that can be invoked directly from the host — all other decorators define sub-kernels that are called from within `@pto.jit` or `@pto.ukernel`.

### Signature

```python
@pto.jit(target="a5")
def kernel_name(
    tensor_arg_1,           # Python-native tensor (positional)
    tensor_arg_2,           # Python-native tensor (positional)
    ...,
    *,
    CONST_A: pto.constexpr = default,  # compile-time constant (keyword-only)
    CONST_B: pto.constexpr = default,  # compile-time constant (keyword-only)
):
```

**Positional parameters** are Python-native tensors — they arrive from NumPy, torch-npu, or any framework with `.shape` and `.strides`. Inside the body, wrap them with `make_tensor_view` to create GM descriptors.

**Keyword-only parameters** annotated with `pto.constexpr` are compile-time constants. They must be provided at `.compile()` time and cannot change between launches of the same compiled kernel. Use them for tile sizes, algorithmic knobs (e.g., `CAUSAL`), and other values that the compiler can specialize against.

### Compilation and launch

```python
# Compile (traces the body, lowers through PTOAS, caches the result)
compiled = kernel_name.compile(CONST_A=128, CONST_B=64)

# Launch on NPU
compiled[grid, stream](tensor_1, tensor_2, ...)
```

- `.compile(**constexprs)` — traces the kernel body with the given constexpr values, lowers the IR, and returns a compiled handle. Subsequent calls with the same (function identity, constexpr values) hit the cache.
- `compiled[grid, stream](args...)` — launches the compiled kernel. `grid` is the number of SPMD blocks (an integer); `stream` is the NPU stream (`None` for default).

### SPMD built-ins

Available inside a `@pto.jit` body:

| Built-in | Returns | Description |
|----------|---------|-------------|
| `pto.get_block_idx()` | `int` | Index of the current block (0-based) |
| `pto.get_block_num()` | `int` | Total number of blocks in the grid |
| `pto.get_subblock_idx()` | `int` | Index of the current sub-block |
| `pto.get_subblock_num()` | `int` | Total number of sub-blocks |

### Typical body

```python
@pto.jit(target="a5")
def my_kernel(A, B, O, *, BLOCK: pto.constexpr):
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

### Calling L3 sub-kernels directly

When you call an L3 sub-kernel directly from `@pto.jit`, data movement is handled by Tile Ops (`tload`/`tstore`) instead of MTE micro-instructions. PTOAS handles the synchronization between Tile Ops and L3 compute — the sub-kernel itself is unchanged:

```python
@pto.cube
def my_matmul(a_tile, b_tile, l0a, l0b, acc, o_tile):
    m = pto.tile_valid_rows(a_tile)
    k = pto.tile_valid_cols(a_tile)
    n = pto.tile_valid_rows(b_tile)
    pto.mte_l1_l0a(a_tile, l0a, m, k)
    pto.mte_l1_l0b(b_tile, l0b, k, n, transpose=True)
    pto.mad(l0a, l0b, acc)
    pto.mte_l0c_ub(acc, o_tile, m, n)

@pto.jit(target="a5")
def my_kernel(A, B, O, *, BLOCK: pto.constexpr):
    N = A.shape[0]
    a_view = pto.make_tensor_view(A, shape=[N], strides=A.strides)
    b_view = pto.make_tensor_view(B, shape=[N], strides=B.strides)
    o_view = pto.make_tensor_view(O, shape=[N], strides=O.strides)

    a_tile = pto.alloc_tile(shape=[BLOCK, BLOCK], dtype=pto.f32)
    b_tile = pto.alloc_tile(shape=[BLOCK, BLOCK], dtype=pto.f32)
    o_tile = pto.alloc_tile(shape=[BLOCK, BLOCK], dtype=pto.f32)
    l0a = pto.alloc_tile(shape=[BLOCK, BLOCK], dtype=pto.f32, memory_space=pto.MemorySpace.LEFT)
    l0b = pto.alloc_tile(shape=[BLOCK, BLOCK], dtype=pto.f32, memory_space=pto.MemorySpace.RIGHT)
    acc = pto.alloc_tile(shape=[BLOCK, BLOCK], dtype=pto.f32, memory_space=pto.MemorySpace.ACC)

    num_blocks = (N + BLOCK - 1) // BLOCK
    with pto.for_(0, num_blocks, step=1) as i:
        offset = i * BLOCK
        a_part = pto.partition_view(a_view, offsets=[offset, 0], sizes=[BLOCK, BLOCK])
        b_part = pto.partition_view(b_view, offsets=[offset, 0], sizes=[BLOCK, BLOCK])
        o_part = pto.partition_view(o_view, offsets=[offset, 0], sizes=[BLOCK, BLOCK])

        # Tile Ops stage data from GM to UB (replaces mte_load at L1)
        pto.tload(a_part, a_tile)
        pto.tload(b_part, b_tile)

        # Direct L3 call — PTOAS handles sync between tload and compute
        my_matmul(a_tile, b_tile, l0a, l0b, acc, o_tile)

        pto.tstore(o_tile, o_part)
```

This is the recommended path for users who want hardware-unit compute without writing explicit MTE Ops and manual sync. Mixing direct L3 calls with Tile Ops and ukernel calls in the same `@pto.jit` body is supported — the compiler unifies the lowering.

## 3.3 `@pto.ukernel` — micro-instruction orchestration

### Role

`@pto.ukernel` (short for *micro-instruction kernel*) is the entry point for writing PTO micro-instructions directly. Unlike `@pto.jit` where you work with tile-level ops (`tload`, `tadd`, etc.), a ukernel lets you write explicit MTE, SIMD, SIMT, and Cube instructions — staging data with `mte_load`, synchronizing with `mem_bar`, and dispatching L3 sub-kernels. This is an advanced programming mode for expert users who need precise control over instruction ordering and hardware-level data movement.

### Signature

```python
@pto.ukernel
def my_ukernel(
    part: pto.PartitionTensorView,   # GM partition descriptors
    tile: pto.Tile,                  # UB tile buffers
    scratch: pto.Tile,               # cube-local scratch (LEFT, RIGHT, ...)
    ptr: pto.ptr(dtype, space),      # typed UB pointers
    scalar: pto.i32,                 # PTO scalar values
):
```

Parameters are PTO-specific types — `Tile`, `PartitionTensorView`, `pto.ptr`, and PTO scalar types. Unlike `@pto.jit`, a ukernel does not accept Python-native tensors.

### Typical body

```python
@pto.ukernel
def process_block(k_part, v_part, k_tile, v_tile,
                  s_tile, o_tile, rows: pto.i32, cols: pto.i32):
    # Stage current block from GM to UB
    pto.mte_load(k_part, k_tile)
    pto.mte_load(v_part, v_tile)
    pto.mem_bar(pto.BarrierType.SYNC)

    # Dispatch sub-kernels
    qk_matmul(q_tile, k_tile, s_tile)
    pto.mem_bar(pto.BarrierType.SYNC)

    online_softmax(s_tile, o_tile, rows, cols)
    pto.mem_bar(pto.BarrierType.SYNC)

    # Write result back
    pto.mte_store(o_tile, o_part)
```

A ukernel stays below the tile-op boundary — GM↔UB movement is expressed with `mte_load`/`mte_store` (MTE Ops) rather than `tload`/`tstore`.

## 3.4 `@pto.cube` — Cube unit sub-kernel

### Role

`@pto.cube` marks a function that executes on the Cube unit (matrix multiplication engine). It consumes UB-resident tiles and explicit cube-local scratch buffers.

### Signature

```python
@pto.cube
def my_cube_kernel(
    input_tile: pto.Tile,            # UB tile (source data)
    output_tile: pto.Tile,           # UB tile (destination)
    left_scratch: pto.Tile,          # LEFT buffer (cube-local)
    right_scratch: pto.Tile,         # RIGHT buffer (cube-local)
    acc_scratch: pto.Tile,           # ACC buffer (cube-local)
):
```

All parameters are `Tile` references. Tiles marked as cube-local must be allocated with the appropriate `memory_space` (e.g., `pto.MemorySpace.LEFT`, `pto.MemorySpace.ACC`).

### Typical body

```python
@pto.cube
def qk_matmul(
    q_tile: pto.Tile,
    k_tile: pto.Tile,
    q_l0a: pto.Tile,
    k_l0b: pto.Tile,
    s_acc: pto.Tile,
    s_tile: pto.Tile,
):
    m = pto.tile_valid_rows(q_tile)
    k = pto.tile_valid_cols(q_tile)
    n = pto.tile_valid_rows(k_tile)

    pto.mte_l1_l0a(q_tile, q_l0a, m, k)
    pto.mte_l1_l0b(k_tile, k_l0b, k, n, transpose=True)
    pto.mad(q_l0a, k_l0b, s_acc)
    pto.mte_l0c_ub(s_acc, s_tile, m, n)
```

Cube-local state (LEFT, RIGHT, ACC, BIAS) never leaks into UB — it is the caller's responsibility to allocate scratch buffers and pass them in explicitly.

**Invocation modes**: `@pto.cube` functions can be:
- Called from `@pto.ukernel` (manual MTE + sync in the ukernel's hands).
- Called directly from `@pto.jit` (compiler infers MTE + sync).
- Used inline as a context manager: `with pto.cube():` (see Section 3.7).

## 3.5 `@pto.simd` — SIMD unit sub-kernel

### Role

`@pto.simd` marks a function that executes on the SIMD unit (vector engine). It operates on vector registers (`vreg`) loaded from UB tiles and stores results back to UB tiles. Vector registers are local to the function and never cross its boundary.

### Signature

```python
@pto.simd
def my_simd_kernel(
    input_tile: pto.Tile,            # UB tile
    output_tile: pto.Tile,           # UB tile
    rows: pto.i32,                   # PTO scalar
    cols: pto.i32,                   # PTO scalar
):
```

Parameters are UB `Tile` references and PTO scalar values (`pto.i32`, `pto.f32`, etc.). Scalar parameters may come from `lds` reads or compile-time constants.

### Typical body

```python
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
```

The boundary contract: `vreg` values (`a_vec`, `b_vec`, `o_vec`) are local to the function. The only way to persist data across a `@pto.simd` call is to write it back to a UB tile via `vsts` (or `psts`, etc.).

**Invocation modes**: `@pto.simd` functions can be:
- Called from `@pto.ukernel` (manual MTE + sync in the ukernel's hands).
- Called directly from `@pto.jit` (compiler infers MTE + sync).
- Used inline as a context manager: `with pto.simd():` (see Section 3.7).

## 3.6 `@pto.simt` — SIMT unit sub-kernel

### Role

`@pto.simt` marks a function that executes on the SIMT unit. SIMT (Single Instruction, Multiple Threads) is a programming model where you write instructions in scalar syntax, and the hardware executes them in parallel across many threads — analogous to how a GPU SM runs a CUDA kernel. Each instruction appears to operate on a single element (`lds`, `sts`, `a + b`), but the same instruction is issued across a large number of work-items simultaneously.

### Signature

```python
@pto.simt
def my_simt_kernel(
    tile: pto.Tile,                  # UB tile
    ptr: pto.ptr(dtype, space),      # typed UB pointer
    scalar: pto.i32,                 # PTO scalar
):
```

### Typical body

```python
@pto.simt
def blend_output_rows(
    o_prev_tile: pto.Tile, pv_tile: pto.Tile,
    alpha_tile: pto.Tile, beta_tile: pto.Tile,
    o_next_tile: pto.Tile,
    row_start: pto.i32, row_stop: pto.i32, valid_dim: pto.i32,
):
    with pto.for_(row_start, row_stop, step=1) as row:
        alpha = scalar.load(alpha_tile[row, 0])
        beta = scalar.load(beta_tile[row, 0])
        with pto.for_(0, valid_dim, step=1) as col:
            o_prev = scalar.load(o_prev_tile[row, col])
            pv_val = scalar.load(pv_tile[row, col])
            o_next = alpha * o_prev + beta * pv_val
            scalar.store(o_next, o_next_tile[row, col])
```

SIMT kernels read and write individual scalar elements from tiles. The unit executes the same scalar instruction across many work-items in parallel, making it efficient for per-element operations.

**Invocation modes**: `@pto.simt` functions can be:
- Called from `@pto.ukernel` (manual MTE + sync in the ukernel's hands).
- Called directly from `@pto.jit` (compiler infers MTE + sync).
- Used inline as a context manager: `with pto.simt():` (see Section 3.7).

## 3.7 Context manager syntax for L3 sub-kernels

In addition to the decorator form, each L3 sub-kernel unit provides a context manager: `with pto.cube():`, `with pto.simd():`, and `with pto.simt():`. These open an inline L3 block without requiring a separate named function — useful for quick prototyping, one-off compute snippets, or when the logic is too trivial to extract.

### Syntax

```python
with pto.simd():
    # Direct L3 instructions — vreg ops, scalar loads/stores
    a_vec = pto.vlds(a_tile[r, c:])
    b_vec = pto.vlds(b_tile[r, c:])
    o_vec = pto.vadd(a_vec, b_vec, mask)
    pto.vsts(o_vec, o_tile[r, c:], mask)
```

```python
with pto.simt():
    alpha = scalar.load(alpha_tile[row, 0])
    beta = scalar.load(beta_tile[row, 0])
    o_next = alpha * o_prev + beta * pv_val
    scalar.store(o_next, o_next_tile[row, col])
```

```python
with pto.cube():
    pto.mte_l1_l0a(q_tile, q_l0a, m, k)
    pto.mte_l1_l0b(k_tile, k_l0b, k, n, transpose=True)
    pto.mad(q_l0a, k_l0b, s_acc)
    pto.mte_l0c_ub(s_acc, s_tile, m, n)
```

### Semantics

- Inside the `with` block, instructions execute on the corresponding hardware unit.
- `vreg` values created inside `with pto.simd():` are scoped to the block — they do not escape.
- Cube-local scratch (`l0a`, `l0b`, `acc`) must be allocated by the caller before entering the block.
- The context manager form is equivalent to defining an inline anonymous sub-kernel. The compiler treats it identically to a named `@pto.simd` / `@pto.cube` / `@pto.simt` function.

### Comparison

| | Decorator form | Context manager form |
|---|---|---|
| Reuse | Named, callable from multiple call sites | Inline, single-use |
| Readability | Good for complex, multi-step logic | Good for short (3-10 line) snippets |
| Testing | Can be unit-tested independently | Tested only through the enclosing kernel |
| Cube-local args | Explicit parameters | Captured from enclosing scope |

The two forms can be freely mixed in the same `@pto.jit` or `@pto.ukernel` body.

## 3.8 Boundary contracts

Data crosses decorator boundaries only through UB-backed tiles or typed UB pointers:

| Boundary | Allowed |
|----------|---------|
| Host → `@pto.jit` | Python-native tensors |
| `@pto.jit` → `@pto.ukernel` | `Tile`, `PartitionTensorView`, `pto.ptr`, PTO scalars |
| `@pto.jit` → L3 sub-kernel (direct call) | `Tile`, PTO scalars (compiler handles MTE + sync) |
| `@pto.jit` → `with pto.{cube,sid,sitm}:` | `Tile` captured from enclosing scope |
| `@pto.ukernel` → L3 sub-kernel | `Tile`, PTO scalars |
| L3 sub-kernel → L3 sub-kernel | Not allowed (go through UB tiles via the caller) |
| `@pto.simd` → caller | Only via `vsts`/`psts` to UB tiles; `vreg` cannot escape |
| Cube-local → UB | Only via `mte_l0c_ub`; LEFT/RIGHT/ACC/BIAS are private |

## 3.9 `pto.constexpr`

`pto.constexpr` marks a `@pto.jit` keyword-only parameter as a compile-time constant. The compiler specializes the kernel for each combination of constexpr values, and the compiled artifact is cached by those values.

```python
@pto.jit(target="a5")
def kernel(A, *, BLOCK: pto.constexpr = 128, DTYPE: pto.constexpr = pto.f32):
    ...
```

- Must appear as a keyword-only argument (after `*`).
- Must have a default value.
- Must be provided at `.compile()` time if the caller needs to override the default.
- Cannot change between launches of the same compiled instance — compile a new variant for a different value.

`pto.constexpr` parameters can be used anywhere in the kernel body where a Python value is expected: tile shapes, loop bounds that are known at compile time, dtype arguments, etc. They are evaluated at trace time, so `for i in range(BLOCK)` would unroll `BLOCK` times.

In contrast, values derived from runtime tensor shapes (e.g., `A.shape[0]`) are dynamic — they vary per launch and should be used with `pto.for_` to produce device-side loops.

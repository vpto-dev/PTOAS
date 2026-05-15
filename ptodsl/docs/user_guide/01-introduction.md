# 1. Introduction

**PTO** is a virtual instruction set designed for the Ascend NPU — a hardware-abstracted programming model that exposes the full capability of the Cube, Vector, and Scalar compute units through a unified operation set. **PTODSL** is the Python frontend for PTO. It wraps the PTO instruction set in a Python-embedded DSL with tracing-based compilation, so you can write PTO programs using familiar Python syntax. Under the hood, PTODSL traces your kernel function into PTO IR, which the PTOAS compiler then lowers, optimizes, and emits as NPU executables. In short: PTO defines the *what* (the instruction set), PTODSL provides the *how* (the authoring experience), and together they give you direct access to all three NPU compute units without leaving Python.

## 1.1 Target hardware

The Ascend NPU is organized around three compute units and a shared on-chip buffer, connected through the Memory Transfer Engine (MTE):

```
                          ┌─────────────────────────┐
                          │     Global Memory (GM)    │
                          │       (off-chip HBM)      │
                          └────────────┬──────────────┘
                                       │
                            ┌──────────┴──────────┐
                            │    MTE (DMA engine)  │
                            └──────────┬──────────┘
                                       │
                          ┌────────────┴──────────────┐
                          │   Unified Buffer (UB)      │
                          │     (on-chip scratchpad)   │
                          └──┬───────────┬──────────┬──┘
                             │           │          │
                    ┌────────┴──┐  ┌─────┴──────┐  │
                    │ LEFT/RIGHT│  │            │  │
                    │  /ACC/BIAS│  │  Vector    │  │
                    │           │  │  (SIMD)    │  │
                    │   Cube    │  │            │  │
                    │           │  └────────────┘  │
                    └───────────┘                  │
                                        ┌──────────┴──┐
                                        │   SIMT      │
                                        │ (scalar PG)  │
                                        └─────────────┘
```

| Unit | Role | Typical workload |
|------|------|------------------|
| **Cube** | Matrix multiplication | GEMM, convolution |
| **SIMD** | Row-wise vector math | activation, normalization, reduction |
| **SIMT** | Scalar-programmable unit | pointwise tile walks, metadata |

- **Global Memory (GM)** is off-chip HBM. All input and output tensors reside here.
- **Unified Buffer (UB)** is the on-chip scratchpad shared by all three compute units. Tile buffers and intermediate results live here during kernel execution.
- **MTE** (Memory Transfer Engine) handles DMA transfers between GM and UB, and between UB regions.
- **Cube** has its own private on-chip buffers — LEFT, RIGHT, ACC, and BIAS — for staging matrix operands and accumulators.
- **SIMD** executes row-wise vector instructions directly on UB-resident data.
- **SIMT** is a scalar-programmable processor group that executes scalar instructions across many work-items in parallel. It is well-suited for per-element control logic, tile boundary metadata, and pointwise blends.

PTODSL gives you direct access to all three units and explicit control over data movement, without abstracting away the hardware boundaries.

## 1.2 Abstraction hierarchy

PTODSL organizes kernel code into three layers, each building on the one below it:

```
Python Wrapper              L0  user-facing wrapper (NumPy, torch-npu, pure Python)
  └─ @pto.jit                     L1  compile + cache + launch
       ├─ Tile Ops                     tile-level: tload, tstore, tadd, ...
       └─ @pto.ukernel                 L2  micro-instruction orchestration
            ├─ MTE Ops                 mte_load / mte_store / copy_gm_to_ubuf / ...
            ├─ @pto.cube               matrix products (mad, mte_l1_l0a, mte_l0c_ub, ...)
            ├─ @pto.simd               row-wise vector math (vlds, vadd, vexp, vsts, ...)
            └─ @pto.simt               scalar-like compute (lds, sts, pointwise blends, ...)
```

### L0 — Python wrapper

The outermost layer is plain Python. It handles ergonomic runtime concerns: allocating output tensors, extracting shapes and strides from framework tensors, compiling the JIT kernel, and launching it. Because L0 is just Python, you can freely mix in NumPy, torch-npu, or any other Python framework for pre- and post-processing, data preparation, or composing multiple kernel launches. This layer knows nothing about NPU internals — it is just a convenience function that most end users will call.

```python
def flash_attention(Q, K, V, *, O=None, causal=False):
    if O is None:
        O = pto.empty_like(Q)
    compiled = flash_attention_kernel.compile(
        BLOCK_Q=128, BLOCK_KV=128, CAUSAL=causal
    )
    compiled[batch * heads, stream](Q, K, V, O)
    return O
```

### L1 — `@pto.jit`

Decorating a function with `@pto.jit` marks it as a launchable PTO kernel. This decoration means:

- **Compilation**: the function body is traced once to record all PTO instructions, then lowered through the PTOAS compiler pipeline into an optimized NPU executable.
- **Caching**: compiled kernels are cached by key (function identity + constexpr parameter values), so repeated calls with the same configuration skip recompilation.
- **Launch binding**: the compiled kernel can be invoked with a grid and stream — `compiled[grid, stream](args...)` — which launches the executable on the NPU with the given SPMD grid.

The parameters of a `@pto.jit` function are Python-native tensors (not PTODSL-specific descriptors). The kernel body materializes `TensorView` descriptors from them via `make_tensor_view`, then partitions the problem with `partition_view`. Compile-time constants are declared as keyword-only arguments with `pto.constexpr`:

```python
@pto.jit(target="a5")
def flash_attention_kernel(
    Q, K, V, O,
    *,
    BLOCK_Q: pto.constexpr = 128,
    BLOCK_KV: pto.constexpr = 128,
    CAUSAL: pto.constexpr = False,
):
    ...
```

L1 is the primary layer for expressing **tile-level semantics**. Inside `@pto.jit`, you allocate tile buffers (`alloc_tile`), move data between GM and UB at block granularity (`tload`, `tstore`), and perform tile-level compute (`tadd`, `texp`, `trowsum`, etc.). When the built-in Tile Ops are not sufficient, you can drop down to `@pto.ukernel` to write custom tile-level semantics with micro-instructions.

The SPMD launch contract is also owned here: the runtime grid (e.g., `batch * heads` blocks) is declared at the call site, and block/subblock indices are queried via `pto.get_block_idx()` and friends.

### L2 — `@pto.ukernel`

`@pto.ukernel` (short for *micro-instruction kernel*) is the entry point for expressing **PTO micro-instruction semantics**. Where L1 works with tile buffers as opaque wholes, L2 gives you direct control over individual MTE, vector, and scalar instructions. This layer is intended for users who pursue peak performance and need precise control over low-level hardware details — instruction ordering, DMA scheduling, per-byte data placement, and synchronization.

Inside a ukernel, you write instructions targeting the three hardware units, and orchestrate data movement between them via **MTE Ops**:

- **MTE Ops** (`mte_load`, `mte_store`, `copy_gm_to_ubuf`, etc.) move data between GM and UB, or between UB regions, at the DMA engine level.
- **`@pto.cube`**, **`@pto.simd`**, and **`@pto.simt`** sub-kernels execute the actual compute on their respective hardware units.

The ukernel manages the execution sandwich for one block: staging data with MTE Ops, issuing synchronization barriers, dispatching sub-kernels, and managing loop-carried state between invocations.

### L3 — `@pto.cube` / `@pto.simd` / `@pto.simt`

These are hardware-bound compute sub-kernels, each mapped to a specific NPU compute unit:

- **`@pto.cube`** consumes UB tiles and explicit cube-local scratch (LEFT, RIGHT, ACC, BIAS). Typical operations: `mad`, `mte_l1_l0a`, `mte_l1_l0b`, `mte_l0c_ub`.

- **`@pto.simd`** operates on vector registers (`vreg`). Typical operations: `vlds`, `vadd`, `vexp`, `vcgmax`, `vsts`. Vector registers never cross the simd function boundary — persistent state is written back to UB tiles.

- **`@pto.simt`** is a scalar-programmable processor group that executes scalar instructions across many work-items in parallel. Typical operations: `lds`, `sts`, scalar arithmetic and comparison. Well-suited for per-element tile walks, boundary metadata, and pointwise blends.

L3 sub-kernels can be invoked in two ways: as named decorated functions (`@pto.cube` / `@pto.simd` / `@pto.simt`) — reusable and callable from `@pto.ukernel` or directly from `@pto.jit` — or inline as context managers (`with pto.cube():` / `with pto.simd():` / `with pto.simt():`) for quick prototyping. When called directly from `@pto.jit`, you stage data with `tload`/`tstore` instead of `mte_load`/`mte_store`; PTOAS handles the synchronization between Tile Ops and L3 compute automatically.

The boundary contract is strict: vreg values do not escape a simd kernel, cube-local state does not leak into UB, and data crosses layer boundaries only through UB-backed tiles or typed UB pointers.

## 1.3 Tracing execution model

PTODSL uses a **tracing** compilation model. When you call `kernel.compile(...)`, PTODSL executes your Python function body once to record every PTO instruction into an intermediate representation — this pass is called *tracing*. The traced IR is then lowered and optimized into device code. Once compiled, invoking `compiled[grid, stream](args...)` launches the already-built device code directly on the NPU.

This has one critical implication for how you write control flow and scalar logic:

- **Python native control flow** (`for`, `if`, Python arithmetic) runs at trace time. A `for i in range(4)` loop gets unrolled — the device code contains four copies of the body, not a loop instruction. An `if` branch condition is evaluated at trace time, and only the taken branch is recorded.

- **`pto.for_` / `pto.if_`** are recorded as structured control-flow IR. They preserve loop and branch semantics into the compiler pipeline, where the PTOAS compiler may further optimize them — unrolling, folding, or keeping them as runtime control flow depending on what is known at compile time.

- **Python scalar expressions** (`alpha * x`, `1.0 / sqrt(d)`) are evaluated at trace time and their results are baked into the IR as constants — the compiler never sees the original expression.

- **PTO scalar instructions** (`scalar.load(...)`, `scalar.max(...)`, `scalar.exp(...)`) are recorded as scalar IR and enter the compiler pipeline, where they may be constant-folded or lowered to runtime scalar operations depending on whether their inputs are compile-time known.

A simple rule of thumb: **Python constructs are resolved before the compiler sees them. PTO constructs are recorded into IR and the compiler decides.**

Chapter 5 (Control Flow) and Chapter 6 (Scalar & Pointer Operations) cover this in detail.

## 1.4 A worked example

The flash attention kernel from Section 1.2 is not just an architectural diagram — it is a complete, runnable design sketch distributed with PTODSL (`demos/flash_attention_sketch.py`). Here is how the layers map to actual code:

**L1 (`@pto.jit`)** allocates tiles for the Q block, KV block, online-softmax state (m/l/o ping-pong tiles), and cube-local scratch. It loops over Q blocks (outer `pto.for_`) and KV blocks (inner `pto.for_` with carry state), calling `kv_block_process` for each KV block and using `tload`/`tstore` at the GM boundary.

**L2 (`@pto.ukernel`)** stages the current K and V blocks with `mte_load`, issues `mem_bar` for synchronization, then sequences four sub-kernel calls: `qk_matmul` (cube), `online_softmax_rows` (simd), `pv_matmul` (cube), `blend_output_rows` (simt).

**L3a (`@pto.cube`)** performs `mte_l1_l0a` / `mte_l1_l0b` / `mad` / `mte_l0c_ub` for both QK^T and P@V products.

**L3b (`@pto.simd`)** implements the online softmax update: per-row max, exp, sum, and alpha/beta computation using vector ops (`vlds`, `vcgmax`, `vexp`, `vcgadd`, `vsts`).

**L3c (`@pto.simt`)** blends the old and new output accumulators with per-element `lds`/`sts` and scalar arithmetic.

Chapter 11 walks through this example in full detail.

## 1.5 Reading guide

| If you are... | Start with... |
|---------------|---------------|
| New to PTODSL | Chapter 2 (Quick Start), then Chapter 3 (Kernel Entries) |

| Writing your first kernel | Chapter 2 → Chapter 4 (Type System) → Chapter 5 (Control Flow) |
| Looking up a specific operation | Chapters 6–10 (organized by topic) |
| Understanding the flash attention reference | Chapter 11 |

**Chapter overview:**

| Chapter | Topic |
|---------|-------|
| 1 | Introduction (this chapter) |
| 2 | Quick Start — a minimal working kernel |
| 3 | Kernel entry points: `@pto.jit`, `@pto.ukernel`, `@pto.cube`, `@pto.simd`, `@pto.simt` |
| 4 | Type system and buffer management: scalars, tiles, views, allocation |
| 5 | Control flow: trace-time Python vs device-side `pto.for_` / `pto.if_` |
| 6 | Scalar and pointer operations |
| 7 | Data movement: tile loads/stores, DMA, vector loads/stores, cube data movement |
| 8 | Compute operations: tile-level, vector, and cube arithmetic |
| 9 | Predicate and mask operations |
| 10 | Synchronization: barriers, flags, memory fences |
| 11 | Flash attention walkthrough |
| 12 | Additional examples |
| 13 | Migration from the old `@pto.vkernel`/`@pto.ckernel` API |
| 14 | Common errors and compatibility notes |

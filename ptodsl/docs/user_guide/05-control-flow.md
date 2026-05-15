# 5. Control Flow

PTODSL uses a **tracing** compilation model. When you call `kernel.compile(...)`, PTODSL executes your Python function body once to record every PTO instruction — this pass is called *tracing*. The recorded program is then lowered and optimized into device code. Once compiled, launching the kernel runs the already-built device code directly on the NPU.

This has one critical implication for how you write loops and branches:

- **Python native `for`/`if`** runs at trace time. A `for i in range(4)` loop gets unrolled — the device code contains four copies of the body, not a loop instruction. An `if` condition is evaluated at trace time, and only the taken branch is recorded.
- **`pto.for_` / `pto.if_`** produce device-side control flow. The loop bound or branch condition can be a runtime value, and the hardware will execute the loop or take the branch dynamically.

**Simple rule: Python control flow = trace time (compile-time). `pto.*` control flow = device-side (runtime).**

## 5.1 Python native `for` — trace-time unrolling

When you write a plain Python `for` loop inside a kernel body, Python executes it immediately during tracing. Each iteration records its instructions separately, so the device code gets a linear sequence with the body repeated:

```python
@pto.jit(target="a5")
def unrolled_kernel(A, O, *, N: pto.constexpr):
    a_view = pto.make_tensor_view(A, shape=[N], strides=A.strides)
    o_view = pto.make_tensor_view(O, shape=[N], strides=O.strides)

    # N is constexpr, so range(N) is known at trace time.
    # The loop unrolls: the device gets N copies of the body.
    for i in range(N):
        a_part = pto.partition_view(a_view, offsets=[i], sizes=[1])
        o_part = pto.partition_view(o_view, offsets=[i], sizes=[1])
        a_tile = pto.alloc_tile(shape=[1], dtype=pto.f32)
        o_tile = pto.alloc_tile(shape=[1], dtype=pto.f32)
        pto.tload(a_part, a_tile)
        pto.tadd(a_tile, a_tile, o_tile)
        pto.tstore(o_tile, o_part)
```

This works when the loop bound is a compile-time constant (like a `constexpr` parameter). But if `N` comes from a tensor shape and varies per launch, `range(N)` would trace a different number of iterations each time — you would get a cache miss and recompilation on every new value. For dynamic bounds, use `pto.for_`.

## 5.2 `pto.for_` — device-side loops

`pto.for_` records a structured loop that executes on the device. Its bound can be any expression involving runtime values (tensor shapes, scalar computations, block indices), and the compiler may optimize it further — unrolling when the bound is known at compile time, or keeping it as a runtime loop otherwise.

### Basic form

```python
with pto.for_(start, stop, step) as iv:
    # iv is the loop index (0-based relative to start)
    ...
```

- `start`, `stop`, `step` are PTO scalar expressions. They are evaluated on the device.
- The loop body executes `(stop - start + step - 1) // step` times.
- Use with `step=1` unless you need a strided iteration.

Compare the two approaches:

```python
# Trace-time unrolling — BLOCK must be constexpr
for i in range(BLOCK):
    ...

# Device-side loop — num_blocks can be dynamic
with pto.for_(0, num_blocks, step=1) as i:
    offset = i * BLOCK
    ...
```

### Nested loops

```python
with pto.for_(0, rows, step=1) as r:
    with pto.for_(0, cols, step=1) as c:
        val = scalar.load(tile[r, c])
        ...
```

Both loops execute on the device. The outer loop bound `rows` and inner loop bound `cols` can be runtime values.

### Loop with carry state

When a loop needs to propagate state from one iteration to the next, use the `.carry(...)` method. This is the PTODSL equivalent of a loop that accumulates or updates variables across iterations:

```python
kv_loop = pto.for_(0, num_blocks, step=1).carry(
    m=m_prev_tile,
    l=l_prev_tile,
    o=o_prev_tile,
)
with kv_loop:
    i = kv_loop.iv         # current iteration index
    m_cur = kv_loop.m      # value carried in from previous iteration
    l_cur = kv_loop.l
    o_cur = kv_loop.o

    # ... compute m_next, l_next, o_next from m_cur, l_cur, o_cur ...

    kv_loop.update(
        m=m_next_tile,
        l=l_next_tile,
        o=o_next_tile,
    )

# After the loop, retrieve the final carried values
final_o = kv_loop.final("o")
```

`.carry(name=initial_value)` declares named state variables that are passed from one iteration to the next. Inside the loop body, access the current value with `loop.name`. At the end of the body, call `loop.update(name=new_value)` to set what the next iteration receives. After the loop exits, `loop.final("name")` retrieves the value from the last iteration.

This pattern is central to algorithms like online softmax, where each KV block updates running statistics (row max, sum, output accumulator). The ping-pong tile pattern — allocating two tiles and swapping them each iteration — is the idiomatic way to manage this state:

```python
# Allocate ping-pong state tiles
m_prev = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32)
m_next = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32)
l_prev = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32)
l_next = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32)

# Initialize prev tiles
m_prev.fill(float("-inf"))
l_prev.fill(0.0)

loop = pto.for_(0, num_blocks, step=1).carry(m=m_prev, l=l_prev)
with loop:
    m_cur = loop.m
    l_cur = loop.l

    # ... compute new m and l into m_next, l_next ...

    loop.update(m=m_next, l=l_next)
```

### Chunked inner loop with carry (tail handling)

For SIMD kernels that process data in vector-width chunks, use a carry loop to track the remaining element count across column iterations:

```python
VEC = pto.elements_per_vreg(pto.f32)
col_loop = pto.for_(0, cols, step=VEC).carry(remained=cols)
with col_loop:
    c = col_loop.iv
    remained = col_loop.remained
    mask, remained = pto.make_mask(pto.f32, remained)
    vec = pto.vlds(tile[r, c:])
    # ... operate under mask ...
    pto.vsts(vec, out_tile[r, c:], mask)
    col_loop.update(remained=remained)
```

`make_mask(dtype, n)` returns two values: the predicate mask for the current chunk and the updated remaining count. Passing the updated count back via `col_loop.update(remained=...)` feeds it into the next iteration, so each chunk correctly computes how many elements are left.

## 5.3 `pto.if_` — device-side conditionals

`pto.if_` records a device-side conditional branch. Unlike a Python `if`, the condition can be a runtime PTO scalar, and both branches are recorded into the program so the hardware can choose at runtime.

The condition must be a PTO scalar value (e.g., the result of a comparison like `scalar.gt(a, b)` or a value loaded from a tile). Python booleans evaluated at trace time should use a plain `if` instead.

### Value merge across branches

When a variable is assigned inside both branches of `pto.if_`/`pto.else_`, the assignments are recorded and the variable holds the merged value after the conditional block. This is the standard SSA-style merge — the downstream code sees whichever value was produced by the taken branch:

```python
@pto.simt
def conditional_scale(
    tile: pto.Tile,
    threshold: pto.f32,
    scale: pto.f32,
    rows: pto.i32,
    cols: pto.i32,
):
    with pto.for_(0, rows, step=1) as r:
        with pto.for_(0, cols, step=1) as c:
            val = scalar.load(tile[r, c])
            big = scalar.gt(val, threshold)

            with pto.if_(big):
                # Branch A: scale the value up
                val = val * scale
            with pto.else_():
                # Branch B: leave it as-is
                pass

            # val is usable here — it is the merged result from both branches.
            # If big was true,  val = original * scale.
            # If big was false, val = original (passed through unchanged).
            scalar.store(val, tile[r, c])
```

In this example, `val` is reassigned in the `if_` branch but left untouched in the `else_` branch. After the conditional block, `val` correctly represents the merged result and is stored back to the tile. You can reassign the same variable in both branches as well — the downstream code always sees the correct value.

### Expression form

For simple either-or selection, `pto.if_` also works as an expression that directly returns the merged value:

```python
result = pto.if_(cond, then_value, else_value)
```

This is equivalent to the block form above and is convenient when each branch simply produces a different scalar or tile reference.

## 5.4 `pto.constexpr` and tracing

`pto.constexpr` parameters (Section 3.8) are compile-time constants. They are fixed at `.compile()` time and cannot change between launches of the same compiled kernel. Because their values are known during tracing, they interact naturally with Python control flow:

```python
@pto.jit(target="a5")
def kernel(A, *, BLOCK: pto.constexpr = 128, UNROLL: pto.constexpr = False):
    N = A.shape[0]
    num_blocks = (N + BLOCK - 1) // BLOCK

    if UNROLL:
        # Trace-time: UNROLL is known, so this branch resolves at compile time.
        # Each iteration records separately — the loop is fully unrolled.
        for i in range(num_blocks):
            ...
    else:
        # Device-side: a single loop instruction is recorded.
        with pto.for_(0, num_blocks, step=1) as i:
            ...
```

This lets you write a single kernel that specializes into different strategies based on constexpr knobs.

## 5.5 Summary

| Construct | When evaluated | Use for |
|-----------|---------------|---------|
| Python `for` | Trace time | Bounds known at compile time (constexpr), deliberate unrolling |
| Python `if` | Trace time | Conditions known at compile time, variant selection |
| `pto.for_` | Device-side | Dynamic bounds, runtime loop counts |
| `pto.for_(...).carry(...)` | Device-side | Loops with accumulated state across iterations |
| `pto.if_` | Device-side | Runtime conditions, data-dependent branching |

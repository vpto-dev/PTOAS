# 6. Scalar and Pointer Operations

Chapter 5 established the rule: Python constructs are resolved at trace time, PTO constructs produce device-side behavior. This chapter applies that distinction to scalars and pointers — when to use a plain Python number, when to use a `scalar.*` operation, and how to work with typed pointers.

## 6.1 Python scalars vs PTO scalars

A **Python scalar** is any value computed by Python during tracing: a literal (`3.14159`), a shape dimension (`A.shape[0]`), a constexpr parameter (`BLOCK`), or an arithmetic expression built from these (`1.0 / sqrt(dim)`). These are evaluated at trace time and their results are baked into the device code as constants.

A **PTO scalar** is a value that lives on the device at runtime. It comes from a `scalar.load` read, a device-side computation (`scalar.max`, `scalar.exp`), or a runtime query (`pto.get_block_idx()`). PTO scalars flow through the recorded program and are not resolved until the kernel executes.

### The mixed expression

In practice, a single expression can mix both kinds:

```python
alpha * o_prev + beta * pv_val
# ^ Python float (trace-time constant, e.g. 1.0 / sqrt(dim))
#        ^ PTO scalar (loaded from tile at runtime)
#                  ^ PTO scalar (loaded from tile at runtime)
```

`alpha` is a Python float computed from compile-time information — it becomes an immediate constant in the device code. `o_prev` and `pv_val` are PTO scalars read from tiles at runtime. The `*` and `+` operators are recorded as device-side multiply-add instructions. The tracer sees the whole expression and produces the appropriate device instructions, embedding the constant operand where possible.

### Rule of thumb

| If the value... | Use... | Example |
|-----------------|--------|---------|
| Is known at compile time | Python scalar | `BLOCK`, `1.0 / sqrt(dim)`, `A.shape[0]` |
| Comes from device memory | PTO scalar | `scalar.load(tile[r, c])` |
| Depends on a runtime value | PTO scalar | `scalar.max(m_prev, row_max)` |
| Is a block/subblock index | PTO scalar | `pto.get_block_idx()` |

When in doubt, ask: *can this value change between launches of the same compiled kernel?* If yes, it must be a PTO scalar.

## 6.2 Scalar access: load and store

`scalar.load` reads a single scalar element from a typed pointer or tile location. `scalar.store` writes a scalar back. These are the canonical scalar memory ops for SIMT authoring. The offset is counted in elements, not bytes.

#### `scalar.load(ptr: PtrType, offset: Index) -> ScalarType`

**Description**: Loads one scalar element from a typed pointer at the given element offset.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `ptr` | `PtrType` | Typed pointer (`pto.ptr<T, space>`) or the result of `tile.as_ptr()` |
| `offset` | `Index` | Element displacement from `ptr` |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `value` | `ScalarType` | The loaded scalar, matching the pointer's element type |

**Tile-index form** — the preferred syntax when loading from a tile:

```python
val = scalar.load(tile[row, col])
```

`tile[row, col]` selects one element. Row and column indices are PTO scalars (or Python integers that the tracer promotes). This form is equivalent to computing the pointer and offset from the tile's base address and layout.

**Pointer forms**:

```python
val = scalar.load(ptr, offset)       # explicit offset
val = scalar.load(ptr + offset)      # pointer arithmetic shorthand
```

---

#### `scalar.store(value: ScalarType, ptr: PtrType, offset: Index) -> None`

**Description**: Stores one scalar element to a typed pointer at the given element offset.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `value` | `ScalarType` | Scalar value to write |
| `ptr` | `PtrType` | Typed destination pointer |
| `offset` | `Index` | Element displacement from `ptr` |

**Returns**: None (side-effect operation).

**Tile-index form**:

```python
scalar.store(value, tile[row, col])
```

**Pointer forms**:

```python
scalar.store(value, ptr, offset)
```

---

### Typical SIMT usage

`scalar.load` and `scalar.store` are the primary data access pattern inside `@pto.simt` kernels. Each `load`/`store` operates on one element per work-item, but the SIMT unit executes the same instruction across many work-items in parallel:

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

When writing to a raw pointer (e.g., a small metadata buffer obtained via `as_ptr()`), use the pointer-plus-offset form:

```python
meta_ptr = meta_tile.as_ptr()
scalar.store(0, meta_ptr, 0)                    # store at element offset 0
scalar.store(valid_rows, meta_ptr, 4)           # store at element offset 4
row_start = scalar.load(meta_ptr, 0)
row_stop  = scalar.load(meta_ptr, 4)
```

## 6.3 Scalar arithmetic and comparisons

### Python operators for basic arithmetic

Addition, subtraction, multiplication, and division of PTO scalars use standard Python syntax. The tracer records the corresponding device-side instructions automatically:

```python
o_next = alpha * o_prev + beta * pv_val      # multiply-add
l_scaled = l_prev * scalar.exp(m_prev - m_next)  # subtraction inside exp
step = (N + BLOCK - 1) // BLOCK               # Python int arithmetic (trace-time)
```

When both operands are PTO scalars (loaded from device memory or produced by another device-side op), `+`, `-`, `*`, `/` produce device-side arithmetic instructions. When one operand is a Python scalar (trace-time constant), the tracer embeds it as an immediate.

### Math functions: `scalar.*`

Non-trivial scalar math functions live under the `scalar` namespace (imported as `from pto import scalar` or accessed as `pto.scalar`):

#### `scalar.max(a: ScalarType, b: ScalarType) -> ScalarType`

**Description**: Returns the maximum of two scalars.

#### `scalar.min(a: ScalarType, b: ScalarType) -> ScalarType`

**Description**: Returns the minimum of two scalars.

#### `scalar.exp(x: ScalarType) -> ScalarType`

**Description**: Exponential, e^x.

#### `scalar.log(x: ScalarType) -> ScalarType`

**Description**: Natural logarithm.

#### `scalar.sqrt(x: ScalarType) -> ScalarType`

**Description**: Square root.

#### `scalar.abs(x: ScalarType) -> ScalarType`

**Description**: Absolute value.

#### `scalar.gt(a: ScalarType, b: ScalarType) -> pto.i1`

**Description**: Greater-than comparison. Returns `pto.i1`.

#### `scalar.lt(a: ScalarType, b: ScalarType) -> pto.i1`

**Description**: Less-than comparison. Returns `pto.i1`.

#### `scalar.eq(a: ScalarType, b: ScalarType) -> pto.i1`

**Description**: Equality comparison. Returns `pto.i1`.

**Example**:

```python
m_next = scalar.max(m_prev, row_max)
l_scaled = l_prev * scalar.exp(m_prev - m_next)
need_scale = scalar.gt(val, threshold)
```

For readability in files with many scalar operations, assign `pto.scalar` to a short local name:

```python
scalar = pto.scalar

m_next = scalar.max(m_prev, row_max)
l_scaled = l_prev * scalar.exp(m_prev - m_next)
```

These are the scalar-path counterparts of the vector math operations covered in Chapter 8. Use them inside `@pto.simt` kernels and in `@pto.ukernel` orchestration code where you need to compute a loop bound or a scalar coefficient from runtime data.

## 6.4 Pointer operations

Typed pointers (Section 4.4) carry both an element type and a memory space. This section covers the operations that create and manipulate them.

### Obtaining pointers: as_ptr()

Tiles and tensor views expose their base address via `as_ptr()`:

```python
gm_ptr = partition.as_ptr()    # GM pointer from a PartitionTensorView
ub_ptr = tile.as_ptr()         # UB pointer from a Tile
```

`as_ptr()` is the preferred way to get a typed pointer from a high-level descriptor. The result carries the correct element type and memory space from the source.

---

#### `pto.addptr(ptr: PtrType, offset: Index) -> PtrType`

**Description**: Advances a pointer by a number of elements (not bytes).

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `ptr` | `PtrType` | Source pointer |
| `offset` | `Index` | Number of elements to advance |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `new_ptr` | `PtrType` | Pointer advanced by `offset` elements |

**Example**:

```python
ptr = pto.addptr(base_ptr, 1024)  # advances by 1024 * sizeof(T) bytes
```

The `+` shorthand on pointers also counts in elements, not bytes.

---

#### `pto.castptr(address: Index, ptr_type: Type) -> PtrType`

**Description**: Creates a typed pointer from an integer address or reinterprets a pointer as a different type.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `address` | `Index` | Integer address or existing pointer value |
| `ptr_type` | `Type` | Target pointer type, e.g. `pto.ptr(pto.f32, pto.MemorySpace.UB)` |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `ptr` | `PtrType` | Typed pointer value |

This is an advanced operation. Prefer `as_ptr()` when the source already carries type information.

## 6.5 Compile-time queries

These functions return values that are known at trace time from type information or hardware constants.

#### `pto.bytewidth(dtype: Type) -> int`

**Description**: Returns the size in bytes of a single element of the given data type. The result is a Python `int` evaluated at trace time.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `dtype` | `Type` | Data type, e.g. `pto.f32`, `pto.f16`, `pto.i8` |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `size` | `int` | Element size in bytes |

**Example**:

```python
bw = pto.bytewidth(pto.f32)   # 4
bw = pto.bytewidth(pto.f16)   # 2
bw = pto.bytewidth(pto.i8)    # 1
```

---

#### `pto.elements_per_vreg(dtype: Type) -> int`

**Description**: Returns how many elements of `dtype` fit in one 256-byte vector register. The result is a Python `int` evaluated at trace time.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `dtype` | `Type` | Data type, e.g. `pto.f32`, `pto.f16`, `pto.i8` |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `elems` | `int` | Number of elements per vector register |

**Example**:

```python
vec = pto.elements_per_vreg(pto.f32)   # 64
vec = pto.elements_per_vreg(pto.f16)   # 128
vec = pto.elements_per_vreg(pto.i8)    # 256
```

This is the standard stride for chunking column loops in SIMD kernels:

```python
VEC = pto.elements_per_vreg(pto.f32)
with pto.for_(0, cols, step=VEC) as c:
    ...
```

## 6.6 Per-element tile traversal in @pto.simt

`@pto.simt` kernels are the natural home for per-element scalar work. A typical pattern uses nested `pto.for_` loops to walk over a tile row by row, column by column:

```python
@pto.simt
def elementwise_scale(
    src_tile: pto.Tile,
    dst_tile: pto.Tile,
    scale: pto.f32,
    rows: pto.i32,
    cols: pto.i32,
):
    with pto.for_(0, rows, step=1) as r:
        with pto.for_(0, cols, step=1) as c:
            val = scalar.load(src_tile[r, c])
            scaled = val * scale
            scalar.store(scaled, dst_tile[r, c])
```

This reads each element from `src_tile`, multiplies by `scale`, and writes to `dst_tile`. The SIMT unit executes the body in parallel across work-items, so this scalar-looking code achieves high throughput — each work-item handles a different `(r, c)` pair.

For operations that need per-row metadata alongside per-element computation, lift the row-level scalar out of the inner loop:

```python
@pto.simt
def blend_with_per_row_coeffs(
    o_prev_tile: pto.Tile,
    pv_tile: pto.Tile,
    alpha_tile: pto.Tile,    # [rows, 1] — one coefficient per row
    beta_tile: pto.Tile,     # [rows, 1]
    o_next_tile: pto.Tile,
    rows: pto.i32,
    cols: pto.i32,
):
    with pto.for_(0, rows, step=1) as r:
        alpha = scalar.load(alpha_tile[r, 0])   # read once per row
        beta = scalar.load(beta_tile[r, 0])     # read once per row
        with pto.for_(0, cols, step=1) as c:
            o_prev = scalar.load(o_prev_tile[r, c])
            pv_val = scalar.load(pv_tile[r, c])
            o_next = alpha * o_prev + beta * pv_val
            scalar.store(o_next, o_next_tile[r, c])
```

This hoists `alpha` and `beta` out of the inner loop — the row coefficients are loaded once and broadcast across all columns in that row.

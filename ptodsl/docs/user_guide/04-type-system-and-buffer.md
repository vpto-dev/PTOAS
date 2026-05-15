# 4. Type System and Buffer Management

This chapter covers every type you can use in a PTODSL kernel, plus the operations for managing buffers in global memory (GM) and on-chip Unified Buffer (UB).

## 4.1 Scalar types

### Numeric scalar types

| DSL Type | Description | Bit Width |
|----------|-------------|-----------|
| `pto.i1` | Boolean | 1 |
| `pto.i8` | 8-bit signless integer | 8 |
| `pto.si8` | 8-bit signed integer | 8 |
| `pto.ui8` | 8-bit unsigned integer | 8 |
| `pto.i16` | 16-bit signless integer | 16 |
| `pto.si16` | 16-bit signed integer | 16 |
| `pto.ui16` | 16-bit unsigned integer | 16 |
| `pto.i32` | 32-bit signless integer | 32 |
| `pto.si32` | 32-bit signed integer | 32 |
| `pto.ui32` | 32-bit unsigned integer | 32 |
| `pto.i64` | 64-bit signless integer | 64 |
| `pto.si64` | 64-bit signed integer | 64 |
| `pto.ui64` | 64-bit unsigned integer | 64 |
| `pto.f16` | Half-precision float | 16 |
| `pto.bf16` | Brain float 16 | 16 |
| `pto.f32` | Single-precision float | 32 |

Python literals are automatically typed by the tracer: `bool` → `pto.i1`, `int` → context-dependent (typically `pto.i32` or `pto.i64`), `float` → `pto.f32`.

For explicit typing, use type constructors:

```python
x = pto.i32(1024)
y = pto.ui16(7)
z: pto.i32 = 1024
```

### Low-precision types (storage only)

The following types are available for storage and data movement, but **not** for computation. Use them to reduce memory bandwidth; convert to a compute-capable type before arithmetic.

| DSL Type | Description |
|----------|-------------|
| `pto.hif8` | HiFloat8 format |
| `pto.f4e1m2x2` | 4-bit float (E1M2, 2-wide packed) |
| `pto.f4e2m1x2` | 4-bit float (E2M1, 2-wide packed) |
| `pto.f8e4m3` | 8-bit float (E4M3) |
| `pto.f8e5m2` | 8-bit float (E5M2) |

### Integer literal guidance

Prefer plain integer literals. Hex string literals are reserved for explicit bit-pattern authoring:

```python
count = pto.i32(1024)
delta = pto.i16(-12)
hi_bit = pto.i32("0x80000000")   # bit-pattern: -2147483648
```

### Floating-point literal forms

```python
a = pto.f16(-1.5)
b = pto.f32("inf")
c = pto.f32("-inf")
d = pto.f32("nan")
# Bit-pattern hex
f16_neg_inf = pto.f16("0xFC00")
```

## 4.2 Vector register type

Vector registers hold a fixed 256-byte payload. `pto.vreg(dtype)` infers the element count automatically:

| `dtype` | Result | Elements |
|---------|--------|----------|
| `pto.f32` / `pto.i32` / ... | `vreg<64xT>` | 64 |
| `pto.f16` / `pto.bf16` / `pto.i16` / ... | `vreg<128xT>` | 128 |
| `pto.i8` / `pto.si8` / `pto.ui8` | `vreg<256xT>` | 256 |

Constraint: `element_count × bitwidth(dtype) = 2048`.

Use `pto.elements_per_vreg(dtype)` to query the element count:

```python
lanes = pto.elements_per_vreg(pto.f32)  # 64
```

### vbitcast

Reinterpret the bits of a vector register as a different element type:

```python
fvec = pto.vlds(ptr, offset)            # !pto.vreg<64xf32>
ivec = pto.vbitcast(fvec, pto.i32)      # !pto.vreg<64xi32>
f16_vec = pto.vbitcast(fvec, pto.f16)   # !pto.vreg<128xf16>
```

`vbitcast` preserves the exact bit pattern (type punning). Use `vcvt` for numeric value conversion.

## 4.3 Mask (predicate) types

Masks are typed by bit granularity and must match the vector element width:

| DSL Type | Granularity | Used with |
|----------|-------------|-----------|
| `pto.mask_b8` | 8-bit | `i8`, `si8`, `ui8` |
| `pto.mask_b16` | 16-bit | `f16`, `bf16`, `i16`, `si16`, `ui16` |
| `pto.mask_b32` | 32-bit | `f32`, `i32`, `si32`, `ui32` |

Bitcast between mask types with `pto.pbitcast`:

```python
mask_b16 = pto.pbitcast(mask_b8, pto.mask_b16)
```

## 4.4 Pointer types

Pointers combine an element type and a memory space:

```python
ptr_gm  = pto.ptr(pto.f32, pto.MemorySpace.GM)
ptr_ub  = pto.ptr(pto.f16, pto.MemorySpace.UB)
```

### MemorySpace enum

| Enum Value | Description |
|------------|-------------|
| `MemorySpace.GM` | Global Memory (off-chip HBM) |
| `MemorySpace.UB` | Unified Buffer (on-chip scratchpad) |
| `MemorySpace.MAT` | Cube L1 / cbuf staging buffer |
| `MemorySpace.LEFT` | Cube L0A left-operand buffer |
| `MemorySpace.RIGHT` | Cube L0B right-operand buffer |
| `MemorySpace.ACC` | Cube L0C accumulator buffer |
| `MemorySpace.BIAS` | Cube bias table buffer |

## 4.5 TensorView

`TensorView` is a descriptor for a tensor in Global Memory. Create one inside a `@pto.jit` body with `make_tensor_view`:

```python
@pto.jit(target="a5")
def kernel(A, *, BLOCK: pto.constexpr):
    tv = pto.make_tensor_view(A, shape=[N], strides=A.strides)
```

`make_tensor_view` wraps a Python-native tensor. You provide the logical shape and the stride of each dimension in **elements** (not bytes). The resulting `TensorView` can be partitioned for `tload`/`tstore`.

### TensorView attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `shape` | `tuple[int, ...]` | Logical dimensions (up to 5D) |
| `element_type` | `Type` | Element dtype (e.g., `pto.f32`) |
| `strides` | `tuple[int, ...]` | Stride of each dimension, in elements |

Strides support non-contiguous tensors. Pass `strides=A.strides` from the source tensor for the default row-major layout, or supply explicit strides for sub-views. Use `tv.as_ptr()` to obtain a typed GM pointer for use with MTE Ops in a ukernel.

## 4.6 PartitionTensorView

`partition_view` creates a sub-view of a TensorView at a given offset and size. It describes *which part* of the GM tensor a `tload` or `tstore` should operate on:

```python
part = pto.partition_view(tv, offsets=[row_offset, 0], sizes=[BLOCK, dim])
```

The result is a `PartitionTensorView` — a lightweight descriptor, not a data buffer. It carries the partition's shape, strides, and element type (inherited from the source TensorView). Use `part.as_ptr()` to obtain a typed GM pointer for MTE Ops in a ukernel.

## 4.7 Tile

A `Tile` is an on-chip buffer allocated in UB or cube-local memory. Allocate tiles with `alloc_tile`:

```python
# UB tile
a_tile = pto.alloc_tile(shape=[BLOCK, dim], dtype=pto.f32)

# Cube-local scratch with explicit memory space
q_l0a = pto.alloc_tile(shape=[Br, dim], dtype=pto.f16, memory_space=pto.MemorySpace.LEFT)
s_acc = pto.alloc_tile(shape=[Br, Bc], dtype=pto.f32, memory_space=pto.MemorySpace.ACC)
```

`alloc_tile` returns a `Tile` object. The `shape` must be a compile-time constant. The default memory space is UB.

### Tile attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `shape` | `tuple[int, ...]` | Physical tile dimensions (compile-time constant) |
| `element_type` | `Type` | Element dtype |
| `memory_space` | `MemorySpace` | Where the tile lives (UB, LEFT, RIGHT, ACC, BIAS) |
| `valid_shape` | `tuple[int, ...]` | Logical data region, ≤ `shape` in each dimension |

### Tile methods

| Method | Description |
|--------|-------------|
| `tile.fill(value)` | Fill the entire tile with a scalar value |
| `tile.as_ptr()` | Obtain a typed pointer to the tile's base address |

```python
m_prev_tile.fill(float("-inf"))
l_prev_tile.fill(0.0)

rows = q_tile.valid_shape[0]
cols = k_tile.valid_shape[1]

meta_ptr = meta_tile.as_ptr()
```

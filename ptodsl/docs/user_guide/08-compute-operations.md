# 8. Compute Operations

Chapters 6 and 7 covered scalars, pointers, and data movement. This chapter covers everything that actually *computes* — arithmetic, math functions, reductions, comparisons, and matrix multiplication — organized by abstraction level: tile ops (L1), vector ops (L3 SIMD), and cube ops (L3 cube).

## 8.1 Tile-level compute (L1)

Tile compute ops are the primary arithmetic surface inside `@pto.jit`. They operate on `Tile` buffers in UB and follow a consistent pattern: each op reads one or more source tiles, optionally a scalar, and writes a destination tile. Shapes and valid regions must be compatible across all operands.

### 8.1.1 Binary tile-tile arithmetic

Element-wise operations between two tiles of the same shape.

#### `pto.tadd(src0: Tile, src1: Tile, dst: Tile) -> None`
#### `pto.tsub(src0: Tile, src1: Tile, dst: Tile) -> None`
#### `pto.tmul(src0: Tile, src1: Tile, dst: Tile) -> None`
#### `pto.tmax(src0: Tile, src1: Tile, dst: Tile) -> None`
#### `pto.tmin(src0: Tile, src1: Tile, dst: Tile) -> None`

**Description**: Element-wise `dst[i,j] = src0[i,j] <op> src1[i,j]`.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `src0` | `Tile` | First source tile |
| `src1` | `Tile` | Second source tile |
| `dst` | `Tile` | Destination tile (must be pre-allocated, shape-compatible) |

**Returns**: None (writes to `dst`).

**Example**:

```python
pto.tadd(a_tile, b_tile, o_tile)
pto.tmul(scale_tile, data_tile, scaled_tile)
```

---

#### `pto.tdiv(src0: Tile, src1: Tile, dst: Tile, *, div_precision: DivPrecision = DivPrecision.Default) -> None`

**Description**: Element-wise division. `div_precision` can be `Default` or `HighPrecision` (f16/f32 only).

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `src0` | `Tile` | Numerator tile |
| `src1` | `Tile` | Denominator tile |
| `dst` | `Tile` | Destination tile |
| `div_precision` | `DivPrecision` | `Default` (default) or `HighPrecision` |

**Returns**: None.

---

### 8.1.2 Tile-scalar arithmetic

Element-wise operations between a tile and a scalar.

#### `pto.tadds(src: Tile, scalar: ScalarType, dst: Tile) -> None`
#### `pto.tsubs(src: Tile, scalar: ScalarType, dst: Tile) -> None`
#### `pto.tmuls(src: Tile, scalar: ScalarType, dst: Tile) -> None`
#### `pto.tmaxs(src: Tile, scalar: ScalarType, dst: Tile) -> None`
#### `pto.tmins(src: Tile, scalar: ScalarType, dst: Tile) -> None`

**Description**: Element-wise `dst[i,j] = src[i,j] <op> scalar`.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `Tile` | Source tile |
| `scalar` | `ScalarType` | Scalar operand (Python number or PTO scalar) |
| `dst` | `Tile` | Destination tile |

**Returns**: None.

---

#### `pto.tdivs(numer: Tile | ScalarType, denom: Tile | ScalarType, dst: Tile, *, div_precision: DivPrecision = DivPrecision.Default) -> None`

**Description**: Element-wise tile-scalar division. Accepts both `(tile, scalar)` and `(scalar, tile)` operand orders.

---

### 8.1.3 Unary math

Single-source element-wise math functions.

#### `pto.texp(src: Tile, dst: Tile, *, exp_precision: ExpPrecision = ExpPrecision.Default) -> None`
#### `pto.tlog(src: Tile, dst: Tile, *, log_precision: LogPrecision = LogPrecision.Default) -> None`
#### `pto.tsqrt(src: Tile, dst: Tile, *, sqrt_precision: SqrtPrecision = SqrtPrecision.Default) -> None`
#### `pto.trsqrt(src: Tile, dst: Tile, *, rsqrt_precision: RsqrtPrecision = RsqrtPrecision.Default) -> None`
#### `pto.trecip(src: Tile, dst: Tile, *, recip_precision: RecipPrecision = RecipPrecision.Default) -> None`

**Description**: Element-wise `exp`, `ln`, `sqrt`, `1/sqrt`, `1/x`.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `Tile` | Source tile |
| `dst` | `Tile` | Destination tile |
| `*_precision` | op-specific precision enum | `Default` or `HighPrecision` |

**Returns**: None.

---

#### `pto.tabs(src: Tile, dst: Tile) -> None`
#### `pto.tneg(src: Tile, dst: Tile) -> None`

**Description**: Element-wise absolute value and negation. No precision mode attribute.

---

### 8.1.4 Activation

#### `pto.trelu(src: Tile, dst: Tile) -> None`

**Description**: `dst[i,j] = max(0, src[i,j])`. Supported on f16, f32, i32.

#### `pto.tlrelu(src: Tile, slope: float, dst: Tile) -> None`

**Description**: Leaky ReLU — `dst[i,j] = src[i,j] >= 0 ? src[i,j] : slope * src[i,j]`.

---

### 8.1.5 Row and column reductions

Reductions collapse one dimension of a 2D tile, producing a tile with one row or one column.

#### Row reductions

#### `pto.trowsum(src: Tile, tmp: Tile, dst: Tile) -> None`
#### `pto.trowmax(src: Tile, tmp: Tile, dst: Tile) -> None`
#### `pto.trowmin(src: Tile, tmp: Tile, dst: Tile) -> None`
#### `pto.trowprod(src: Tile, tmp: Tile, dst: Tile) -> None`
#### `pto.trowargmax(src: Tile, tmp: Tile, dst: Tile) -> None`
#### `pto.trowargmin(src: Tile, tmp: Tile, dst: Tile) -> None`

**Description**: For each row `i`, reduce across columns: `dst[i, 0] = <reduce>_j src[i, j]`. `trowargmax`/`trowargmin` return the column index of the extremum.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `Tile` | Source tile (`[rows, cols]`) |
| `tmp` | `Tile` | Scratch tile for intermediate reduction state |
| `dst` | `Tile` | Destination tile (`[rows, 1]`) |

**Returns**: None.

---

#### Column reductions

#### `pto.tcolsum(src: Tile, dst: Tile) -> None`
#### `pto.tcolmax(src: Tile, dst: Tile) -> None`
#### `pto.tcolmin(src: Tile, dst: Tile) -> None`
#### `pto.tcolprod(src: Tile, dst: Tile) -> None`

**Description**: For each column `j`, reduce across rows: `dst[0, j] = <reduce>_i src[i, j]`.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `Tile` | Source tile (`[rows, cols]`) |
| `dst` | `Tile` | Destination tile (`[1, cols]`) |

**Returns**: None.

---

### 8.1.6 Broadcast and expansion

Expansion ops take a narrow source (scalar, row vector, or column vector) and broadcast it to a full tile shape. They are useful for applying per-row or per-column coefficients to a tile.

#### Scalar broadcast

#### `pto.texpands(scalar: ScalarType, dst: Tile) -> None`

**Description**: `dst[i,j] = scalar` — fills every element of `dst` with the same scalar value.

---

#### Row expansion

#### `pto.trowexpand(src: Tile, dst: Tile) -> None`

**Description**: `dst[row, col] = src[row, 0]` — broadcasts each row's single value across all columns of `dst`.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `Tile` | Source tile (`[rows, 1]`) |
| `dst` | `Tile` | Destination tile (`[rows, cols]`) |

**Returns**: None.

---

#### Column expansion

#### `pto.tcolexpand(src: Tile, dst: Tile) -> None`

**Description**: `dst[row, col] = src[0, col]` — broadcasts each column's single value across all rows of `dst`.

---

#### Row-expand arithmetic

These combine broadcasting with an arithmetic operation: `src1` is a per-row coefficient tile (`[rows, 1]`) that gets expanded row-wise before the element-wise op with `src0`.

| Op | Semantics |
|----|-----------|
| `pto.trowexpandadd(src0, src1, dst)` | `dst = src0 + expand_rows(src1)` |
| `pto.trowexpandsub(src0, src1, dst)` | `dst = src0 - expand_rows(src1)` |
| `pto.trowexpandmul(src0, src1, dst)` | `dst = src0 * expand_rows(src1)` |
| `pto.trowexpanddiv(src0, src1, dst)` | `dst = src0 / expand_rows(src1)` (f-only) |
| `pto.trowexpandmax(src0, src1, dst)` | `dst = max(src0, expand_rows(src1))` |
| `pto.trowexpandmin(src0, src1, dst)` | `dst = min(src0, expand_rows(src1))` |
| `pto.trowexpandexpdif(src0, src1, dst)` | `dst = exp(src0 - expand_rows(src1))` (f-only) |

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `src0` | `Tile` | Full-shape source tile (`[rows, cols]`) |
| `src1` | `Tile` | Per-row coefficient tile (`[rows, 1]`) |
| `dst` | `Tile` | Destination tile (`[rows, cols]`) |

**Returns**: None.

**Example** — apply per-row scale and bias:

```python
# alpha_tile: [rows, 1], beta_tile: [rows, 1], data_tile: [rows, cols]
pto.trowexpandmul(data_tile, alpha_tile, scaled_tile)
pto.trowexpandadd(scaled_tile, beta_tile, result_tile)
```

---

#### Column-expand arithmetic

Same pattern as row-expand arithmetic, but `src1` is a per-column coefficient tile (`[1, cols]`):

| Op | Semantics |
|----|-----------|
| `pto.tcolexpandadd(src0, src1, dst)` | `dst = src0 + expand_cols(src1)` |
| `pto.tcolexpandsub(src0, src1, dst)` | `dst = src0 - expand_cols(src1)` |
| `pto.tcolexpandmul(src0, src1, dst)` | `dst = src0 * expand_cols(src1)` |
| `pto.tcolexpanddiv(src0, src1, dst)` | `dst = src0 / expand_cols(src1)` (f-only) |
| `pto.tcolexpandmax(src0, src1, dst)` | `dst = max(src0, expand_cols(src1))` |
| `pto.tcolexpandmin(src0, src1, dst)` | `dst = min(src0, expand_cols(src1))` |
| `pto.tcolexpandexpdif(src0, src1, dst)` | `dst = exp(src0 - expand_cols(src1))` (f-only) |

---

### 8.1.7 Selection

#### `pto.tsel(mask: Tile, src0: Tile, src1: Tile, tmp: Tile, dst: Tile) -> None`

**Description**: Element-wise ternary: `dst[i,j] = mask[i,j] ? src0[i,j] : src1[i,j]`. The `mask` is an integer tile where zero means false and non-zero means true.

#### `pto.tsels(mask: Tile, src: Tile, scalar: ScalarType, tmp: Tile, dst: Tile) -> None`

**Description**: Element-wise select with scalar fallback: `dst[i,j] = mask[i,j] ? src[i,j] : scalar`.

---

### 8.1.8 Type conversion

#### `pto.tcvt(src: Tile, dst: Tile, *, rmode: RoundMode = RoundMode.NONE) -> None`

**Description**: Element-wise type conversion. The destination tile's `dtype` determines the target type.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `Tile` | Source tile |
| `dst` | `Tile` | Destination tile (with target dtype) |
| `rmode` | `RoundMode` | Rounding mode: `NONE`, `RINT`, `ROUND`, `FLOOR`, `CEIL`, `TRUNC`, `ODD`, `CAST_RINT` |

**Returns**: None.

---

### 8.1.9 Tile compute quick reference

| Category | Operations |
|----------|------------|
| Binary tile-tile | `tadd`, `tsub`, `tmul`, `tdiv`, `tmax`, `tmin` |
| Tile-scalar | `tadds`, `tsubs`, `tmuls`, `tdivs`, `tmaxs`, `tmins` |
| Unary math | `texp`, `tlog`, `tsqrt`, `trsqrt`, `trecip`, `tabs`, `tneg` |
| Activation | `trelu`, `tlrelu` |
| Row reductions | `trowsum`, `trowmax`, `trowmin`, `trowprod`, `trowargmax`, `trowargmin` |
| Column reductions | `tcolsum`, `tcolmax`, `tcolmin`, `tcolprod` |
| Broadcast | `texpands`, `trowexpand`, `tcolexpand` |
| Row-expand arith | `trowexpandadd`, `trowexpandsub`, `trowexpandmul`, `trowexpanddiv`, `trowexpandmax`, `trowexpandmin`, `trowexpandexpdif` |
| Col-expand arith | `tcolexpandadd`, `tcolexpandsub`, `tcolexpandmul`, `tcolexpanddiv`, `tcolexpandmax`, `tcolexpandmin`, `tcolexpandexpdif` |
| Selection | `tsel`, `tsels` |
| Type conversion | `tcvt` |
| Bitwise | `tnot`, `tand`, `tor`, `txor`, `tshl`, `tshr`, `tands`, `tors`, `txors`, `tshls`, `tshrs` |
| Partial elementwise | `tpartadd`, `tpartmul`, `tpartmax`, `tpartmin` |
| Fill/padding | `tfillpad`, `tfillpad_expand`, `tfillpad_inplace` |

---

## 8.2 Vector compute (L3 — `@pto.simd`)

Vector compute ops operate on `VRegType` values inside `@pto.simd` sub-kernels. Every vector op takes a `MaskType` predicate that gates which lanes participate; masked-off lanes produce an unspecified result (use the result only where the mask is true, or feed it to a masked store).

All vector ops in this section follow the pattern established in Section 7.3 for tile-index and pointer-form addressing. The signatures below use the vector-register form — tile-index forms load into `vreg` first, then compute.

### 8.2.1 Unary vector ops

#### `pto.vexp(vec: VRegType, mask: MaskType) -> VRegType`
#### `pto.vln(vec: VRegType, mask: MaskType) -> VRegType`
#### `pto.vsqrt(vec: VRegType, mask: MaskType) -> VRegType`
#### `pto.vabs(vec: VRegType, mask: MaskType) -> VRegType`
#### `pto.vneg(vec: VRegType, mask: MaskType) -> VRegType`
#### `pto.vrec(vec: VRegType, mask: MaskType) -> VRegType`
#### `pto.vrsqrt(vec: VRegType, mask: MaskType) -> VRegType`
#### `pto.vrelu(vec: VRegType, mask: MaskType) -> VRegType`
#### `pto.vnot(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Element-wise unary operation under mask. `vrec` = reciprocal, `vrsqrt` = inverse square root, `vrelu` = `max(0, x)`.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `mask` | `MaskType` | Predicate mask (granularity must match element type) |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Result vector |

**Example**:

```python
exp_vec = pto.vexp(s_row, col_mask)
```

---

### 8.2.2 Binary vector ops

#### `pto.vadd(v0: VRegType, v1: VRegType, mask: MaskType) -> VRegType`
#### `pto.vsub(v0: VRegType, v1: VRegType, mask: MaskType) -> VRegType`
#### `pto.vmul(v0: VRegType, v1: VRegType, mask: MaskType) -> VRegType`
#### `pto.vdiv(v0: VRegType, v1: VRegType, mask: MaskType) -> VRegType`
#### `pto.vmax(v0: VRegType, v1: VRegType, mask: MaskType) -> VRegType`
#### `pto.vmin(v0: VRegType, v1: VRegType, mask: MaskType) -> VRegType`

**Description**: Element-wise binary operation: `result[i] = v0[i] <op> v1[i]` for lanes where `mask[i]` is true.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `v0` | `VRegType` | First operand vector |
| `v1` | `VRegType` | Second operand vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Result vector |

---

**Bitwise binary ops** (integer types only):

| Op | Semantics |
|----|-----------|
| `pto.vand(v0, v1, mask) -> VRegType` | `v0 & v1` |
| `pto.vor(v0, v1, mask) -> VRegType` | `v0 \| v1` |
| `pto.vxor(v0, v1, mask) -> VRegType` | `v0 ^ v1` |
| `pto.vshl(vec, shift, mask) -> VRegType` | `vec << shift` (per-element) |
| `pto.vshr(vec, shift, mask) -> VRegType` | `vec >> shift` (per-element) |

---

### 8.2.3 Vector-scalar ops

#### `pto.vadds(vec: VRegType, scalar: ScalarType, mask: MaskType) -> VRegType`
#### `pto.vsubs(vec: VRegType, scalar: ScalarType, mask: MaskType) -> VRegType`
#### `pto.vmuls(vec: VRegType, scalar: ScalarType, mask: MaskType) -> VRegType`
#### `pto.vmaxs(vec: VRegType, scalar: ScalarType, mask: MaskType) -> VRegType`
#### `pto.vmins(vec: VRegType, scalar: ScalarType, mask: MaskType) -> VRegType`

**Description**: Element-wise `result[i] = vec[i] <op> scalar`. The scalar is broadcast to all active lanes.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `scalar` | `ScalarType` | Scalar operand (uniform across all lanes) |
| `mask` | `MaskType` | Predicate mask |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Result vector |

**Example** — subtract row max from score row (online softmax):

```python
s_shifted = pto.vsubs(s_row, m_next, col_mask)
```

---

#### `pto.vlrelu(vec: VRegType, alpha: ScalarType, mask: MaskType) -> VRegType`

**Description**: Leaky ReLU — `vec[i] >= 0 ? vec[i] : alpha * vec[i]`.

---

### 8.2.4 Full-vector and group reductions

#### Full-vector reductions

#### `pto.vcadd(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Full-vector sum reduction. Result placed in lane 0.

#### `pto.vcmax(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Full-vector max with argmax. Result lane 0 = max value, lane 1 = max index.

#### `pto.vcmin(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Full-vector min with argmin. Result lane 0 = min value, lane 1 = min index.

---

#### Group reductions (per-VLane)

These reduce within each hardware vector lane group (typically 8 groups per vector). Useful when a vector register holds multiple independent sub-vectors that need separate reductions.

#### `pto.vcgadd(vec: VRegType, mask: MaskType) -> VRegType`
#### `pto.vcgmax(vec: VRegType, mask: MaskType) -> VRegType`
#### `pto.vcgmin(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Per-group sum, max, or min. Each group's result is placed in the first lane of that group.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Vector with per-group reduction results |

**Example** — row max and row sum from online softmax:

```python
row_max = pto.vcgmax(s_row, col_mask)   # per-group max → first lane of each group
row_sum = pto.vcgadd(p_row, col_mask)   # per-group sum → first lane of each group
```

---

#### `pto.vcpadd(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Inclusive prefix sum (scan). `result[i] = sum_{k=0}^{i} vec[k]` for active lanes. f16 and f32 only.

---

### 8.2.5 Fused and compound ops

These combine an arithmetic operation with a math function or activation in a single instruction.

#### `pto.vexpdif(vec: VRegType, max_vec: VRegType, mask: MaskType, *, part: PartMode = PartMode.EVEN) -> VRegType`

**Description**: `exp(vec[i] - max_vec[i])` — the stable softmax numerator. `part` controls which half of the vector is computed: `EVEN` or `ODD`. Result type is always f32.

---

#### `pto.vaxpy(alpha: ScalarType, x: VRegType, y: VRegType, mask: MaskType) -> VRegType`

**Description**: Fused multiply-add: `alpha * x[i] + y[i]`.

---

#### `pto.vaddrelu(v0: VRegType, v1: VRegType, mask: MaskType) -> VRegType`

**Description**: `max(0, v0[i] + v1[i])` — fused add + ReLU.

#### `pto.vsubrelu(v0: VRegType, v1: VRegType, mask: MaskType) -> VRegType`

**Description**: `max(0, v0[i] - v1[i])` — fused sub + ReLU.

---

### 8.2.6 Comparison and selection

#### `pto.vcmp(v0: VRegType, v1: VRegType, seed_mask: MaskType, cmp_mode: CmpMode) -> MaskType`

**Description**: Element-wise comparison producing a predicate mask. `seed_mask` selects which lanes participate; the result inherits its granularity (e.g., `mask_b32` for f32).

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `v0` | `VRegType` | First operand |
| `v1` | `VRegType` | Second operand |
| `seed_mask` | `MaskType` | Seed mask gating participation |
| `cmp_mode` | `CmpMode` | `EQ`, `NE`, `LT`, `LE`, `GT`, `GE` |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `pred` | `MaskType` | Result predicate mask |

---

#### `pto.vcmps(vec: VRegType, scalar: ScalarType, seed_mask: MaskType, cmp_mode: CmpMode) -> MaskType`

**Description**: Vector-scalar comparison. Same semantics as `vcmp` with a uniform scalar second operand.

---

#### `pto.vsel(true_v: VRegType, false_v: VRegType, mask: MaskType) -> VRegType`

**Description**: Per-lane select: `mask[i] ? true_v[i] : false_v[i]`.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `true_v` | `VRegType` | Values when mask is true |
| `false_v` | `VRegType` | Values when mask is false |
| `mask` | `MaskType` | Selection predicate |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Selected vector |

---

### 8.2.7 Vector compute quick reference

| Category | Operations |
|----------|------------|
| Unary | `vexp`, `vln`, `vsqrt`, `vabs`, `vneg`, `vrec`, `vrsqrt`, `vrelu`, `vnot`, `vmov`, `vcls`, `vbcnt` |
| Binary | `vadd`, `vsub`, `vmul`, `vdiv`, `vmax`, `vmin`, `vand`, `vor`, `vxor`, `vshl`, `vshr`, `vmod` |
| Vector-scalar | `vadds`, `vsubs`, `vmuls`, `vmaxs`, `vmins`, `vshls`, `vshrs`, `vlrelu`, `vands`, `vors`, `vxors` |
| Broadcast | `vbr`, `vdup` |
| Full reduction | `vcadd`, `vcmax`, `vcmin` |
| Group reduction | `vcgadd`, `vcgmax`, `vcgmin` |
| Scan | `vcpadd` |
| Fused | `vexpdif`, `vaxpy`, `vprelu`, `vaddrelu`, `vsubrelu`, `vmulconv`, `vaddreluconv` |
| Compare/select | `vcmp`, `vcmps`, `vsel`, `vselr`, `vselrv2` |
| Carry | `vaddc`, `vsubc`, `vaddcs`, `vsubcs` |
| Extended arith | `vmull`, `vmula` |
| Conversion | `vcvt`, `vtrc`, `vbitcast`, `pbitcast` |
| Index gen | `vci` |
| Rearrangement | `vintlv`, `vdintlv`, `vintlvv2`, `vdintlvv2`, `vsqz`, `vusqz`, `vpack`, `vsunpack`, `vzunpack`, `vperm`, `vshift`, `vslide`, `vsort32`, `vmrgsort`, `vtranspose` |

---

## 8.3 Cube compute (L3 — `@pto.cube`)

The Cube unit performs matrix multiplication. Its operands are typed pointers into cube-local buffers — L0A (left operand), L0B (right operand), L0C (accumulator), and BIAS. Cube data movement (`mte_l1_l0a`, `mte_l1_l0b`, `mte_l0c_ub`, etc.) was covered in Section 7.5; this section covers the compute instruction itself.

### 8.3.1 Matrix multiply: `pto.mad`

#### `pto.mad(lhs: PtrType, rhs: PtrType, dst: PtrType, m: int, k: int, n: int) -> None`

**Description**: Zero-initialized matrix multiply: `dst[M×N] = lhs[M×K] * rhs[K×N]`. `lhs` is an L0A pointer, `rhs` is an L0B pointer, `dst` is an L0C pointer.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `lhs` | `PtrType` (L0A) | Left operand matrix (M × K) |
| `rhs` | `PtrType` (L0B) | Right operand matrix (K × N) |
| `dst` | `PtrType` (L0C) | Destination accumulator (M × N) |
| `m` | `int` | M dimension size |
| `k` | `int` | K dimension (inner/reduction dimension) |
| `n` | `int` | N dimension size |

**Returns**: None (writes to `dst` in L0C).

---

#### `pto.mad_acc(lhs: PtrType, rhs: PtrType, dst: PtrType, m: int, k: int, n: int) -> None`

**Description**: Accumulating matrix multiply: `dst[M×N] += lhs[M×K] * rhs[K×N]`. `dst` must already hold a prior accumulation result.

---

#### `pto.mad_bias(lhs: PtrType, rhs: PtrType, dst: PtrType, bias: PtrType, m: int, k: int, n: int) -> None`

**Description**: Bias-initialized matrix multiply: `dst[M×N] = lhs[M×K] * rhs[K×N] + bias[M×N]`. `bias` is a BIAS pointer.

---

### 8.3.2 Typical cube matmul pattern

A full cube matmul follows a three-stage pattern: stage operands into L0A/L0B, compute, write back to UB.

```python
@pto.cube
def qk_matmul(q_tile, k_tile, q_l0a, k_l0b, s_acc, s_tile):
    m = pto.tile_valid_rows(q_tile)
    k = pto.tile_valid_cols(q_tile)
    n = pto.tile_valid_rows(k_tile)

    # Stage: UB → L0A / L0B
    pto.mte_l1_l0a(q_tile, q_l0a, m, k)
    pto.mte_l1_l0b(k_tile, k_l0b, k, n, transpose=True)

    # Compute: L0A × L0B → L0C
    pto.mad(q_l0a, k_l0b, s_acc, m, k, n)

    # Writeback: L0C → UB
    pto.mte_l0c_ub(s_acc, s_tile, m, n)
```

The `mte_l1_l0a`/`mte_l1_l0b` stage operands from UB into cube-local buffers. `mad` performs the matrix multiply into L0C. `mte_l0c_ub` writes the result back to a UB tile for downstream processing.

---

### 8.3.3 Cube compute quick reference

| Operation | Semantics |
|-----------|-----------|
| `pto.mad(lhs, rhs, dst, m, k, n)` | `dst = lhs * rhs` (zero-init) |
| `pto.mad_acc(lhs, rhs, dst, m, k, n)` | `dst += lhs * rhs` (accumulating) |
| `pto.mad_bias(lhs, rhs, dst, bias, m, k, n)` | `dst = lhs * rhs + bias` |
| `pto.mad_mx(lhs, rhs, dst, m, k, n)` | MX-format zero-init matmul |
| `pto.mad_mx_acc(lhs, rhs, dst, m, k, n)` | MX-format accumulating matmul |
| `pto.mad_mx_bias(lhs, rhs, dst, bias, m, k, n)` | MX-format bias-init matmul |

MX variants require MX-enabled dtypes (f8) and pre-loaded scale payloads. For most users, the standard `mad`, `mad_acc`, and `mad_bias` are the primary interface.

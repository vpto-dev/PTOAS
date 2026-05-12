# 16. Cube Matrix Multiply (MAT)

> **Category:** Cube unit ops — staged load/store and matrix multiply

---

## Wrapper-Layer Compute Ops

The `pto.mad*` family describes a logical matrix multiply over already prepared
cube tiles:

- `%lhs` is the logical `M x K` left matrix tile in `left`.
- `%rhs` is the logical `K x N` right matrix tile in `right`.
- `%dst` is the logical `M x N` accumulator tile in `acc`.
- `%m`, `%n`, and `%k` are element counts in the logical M/N/K dimensions, not
  byte sizes.
- The matrix data type is inferred from the pointer element types. Do not pass
  a separate type selector. Invalid target-profile type combinations are not
  valid programs.

Common optional clauses:

| Clause | Values | Semantic effect |
|--------|--------|-----------------|
| `unit_flag(...)` | `check_only`, `check_and_set` | Enables producer-side unit-flag participation for this result tile. `check_only` only performs the producer-side availability check; `check_and_set` also publishes the produced tile for downstream consumers. Omit it when the schedule does not use unit-flag synchronization. |
| `disable_gemv` | flag | Selects the normal matmul `%lhs` consumption contract for `M = 1` GEMV operations. When `M = 1`, omitting this flag selects the GEMV `%lhs` consumption contract: `%lhs` must point to an L0A tile organized in the target GEMV A-vector format. On the current A5 contract, the logical `A[1, K]` values must be provided as a sequential K-element vector. Supplying a normal matmul L0A organization while GEMV mode is selected is invalid. Adding `disable_gemv` makes `%lhs` use the normal matmul consumption contract instead. For `M != 1`, the op uses the normal matmul `%lhs` consumption contract. The mathematical result is unchanged: all modes compute `A @ B`; only the required L0A organization for `%lhs` changes in the `M = 1` GEMV case. |
| `sat` / `nosat` | flags | Optional CUBE floating exceptional-value mode. With `sat`, exceptional input values participating in the multiply are normalized before arithmetic (`+/-INF -> +/-finite type max`, `NaN -> 0`), and finite arithmetic overflow saturates to the finite type range instead of producing INF. With `nosat`, INF/NaN inputs are preserved and overflow can produce INF/NaN through the multiply-accumulate. Omit both to use the surrounding execution mode. These flags are valid only for floating/MX MAD forms, not integer MAD. |
| `tf32_mode(...)` | `round_even`, `round_away` | Valid only for non-MX `f32 x f32 -> f32`. FP32 inputs are rounded to TF32 precision before multiplication; accumulation/output remain FP32. `round_even` uses nearest-even tie handling; `round_away` uses nearest-away tie handling. |
| `n_dir` | flag | Requests N-direction result production. This does not change the mathematical matrix value; it changes the producer ordering expected by schedules that combine unit flags with later layout movement. Omit it for the default M-direction production order. |

`pto.mad_mx*` additionally applies microscaling. The scale operands are not
explicit operands of the matmul op: they are associated with the `%lhs` and
`%rhs` tiles by the MX load/layout contract. Logically, each group of 32
K-direction values shares one scale value. In the current target profile, the
supported MX data tile type is `f8E4M3FN`:

```text
mx_matmul(lhs, rhs)[m, n] =
  sum_k (lhs[m, k] * scale_lhs[m, floor(k / 32)]) *
        (rhs[k, n] * scale_rhs[floor(k / 32), n])
```

The corresponding scale data must have been loaded into the matching MX side
buffers before the `pto.mad_mx*` op executes, and it must be aligned with the
data tile selected by `%lhs` / `%rhs`.

### `pto.mad`

- **syntax:**
```mlir
pto.mad %lhs, %rhs, %dst, %m, %n, %k
  unit_flag(check_only | check_and_set)?
  disable_gemv?
  (sat | nosat)?
  tf32_mode(round_even | round_away)?
  n_dir?
  : !pto.ptr<A, left>, !pto.ptr<B, right>, !pto.ptr<C, acc>, i64, i64, i64
```
- **semantics:** Zero-init cube matmul, `dst = lhs * rhs`.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%lhs` | ptr | L0A input (`left`) |
| `%rhs` | ptr | L0B input (`right`) |
| `%dst` | ptr | L0C accumulator (`acc`) |
| `%m` | i64 | M size |
| `%n` | i64 | N size |
| `%k` | i64 | K size |
| `unit_flag` | clause | Optional unit-flag producer control; see common clauses |
| `disable_gemv` | flag | Optional GEMV disable; see common clauses |
| `sat` / `nosat` | flag | Optional floating exceptional-value mode override; see common clauses |
| `tf32_mode` | clause | Optional TF32 input rounding; see common clauses |
| `n_dir` | flag | Optional N-direction producer order; see common clauses |

**Constraints:**

- Address spaces must be `left`, `right`, `acc`.
- `unit_flag(check_only)` and `unit_flag(check_and_set)` are the supported forms.
- `tf32_mode(...)` requires `f32` lhs, rhs, and dst element types.
- `sat` / `nosat` requires floating or MX lhs/rhs/dst element types; integer `s8 -> s32` MAD does not accept this clause.

**Example:**

```mlir
pto.mad %l0a, %l0b, %l0c, %c16_i64, %c16_i64, %c16_i64
  : !pto.ptr<f16, left>, !pto.ptr<f16, right>, !pto.ptr<f32, acc>, i64, i64, i64
```

---

### `pto.mad_acc`

- **syntax:**
```mlir
pto.mad_acc %lhs, %rhs, %dst, %m, %n, %k
  unit_flag(check_only | check_and_set)?
  disable_gemv?
  (sat | nosat)?
  tf32_mode(round_even | round_away)?
  n_dir?
  : !pto.ptr<A, left>, !pto.ptr<B, right>, !pto.ptr<C, acc>, i64, i64, i64
```
- **semantics:** Accumulating cube matmul, `dst += lhs * rhs`.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%lhs` | ptr | L0A input (`left`) |
| `%rhs` | ptr | L0B input (`right`) |
| `%dst` | ptr | L0C accumulator (`acc`) |
| `%m` | i64 | M size |
| `%n` | i64 | N size |
| `%k` | i64 | K size |
| `unit_flag` | clause | Optional unit-flag producer control; see common clauses |
| `disable_gemv` | flag | Optional GEMV disable; see common clauses |
| `sat` / `nosat` | flag | Optional floating exceptional-value mode override; see common clauses |
| `tf32_mode` | clause | Optional TF32 input rounding; see common clauses |
| `n_dir` | flag | Optional N-direction producer order; see common clauses |

**Constraints:**

- Same address space/type family requirements as `pto.mad`.
- `tf32_mode(...)` requires `f32` lhs, rhs, and dst element types.
- `sat` / `nosat` requires floating or MX lhs/rhs/dst element types.

**Example:**

```mlir
pto.mad_acc %l0a, %l0b, %l0c, %c16_i64, %c16_i64, %c16_i64 unit_flag(check_only)
  : !pto.ptr<f16, left>, !pto.ptr<f16, right>, !pto.ptr<f32, acc>, i64, i64, i64
```

---

### `pto.mad_bias`

- **syntax:**
```mlir
pto.mad_bias %lhs, %rhs, %dst, %bias, %m, %n, %k
  unit_flag(check_only | check_and_set)?
  disable_gemv?
  (sat | nosat)?
  tf32_mode(round_even | round_away)?
  n_dir?
  : !pto.ptr<A, left>, !pto.ptr<B, right>, !pto.ptr<C, acc>, !pto.ptr<C, bias>, i64, i64, i64
```
- **semantics:** Bias-init cube matmul, `dst[m, n] = lhs[m, k] * rhs[k, n] + bias[n]`.
  The bias operand is not an `M x N` matrix. It points to an `N`-element
  per-output-channel bias vector in the bias buffer, and that vector is
  broadcast across the M dimension.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%lhs` / `%rhs` / `%dst` / `%m` / `%n` / `%k` | - | Same meaning as `pto.mad` |
| `%bias` | ptr | Bias-buffer pointer (`!pto.ptr<C, bias>`) for the `N`-element bias vector. |
| `unit_flag` | clause | Optional unit-flag producer control; see common clauses |
| `disable_gemv` | flag | Optional GEMV disable; see common clauses |
| `sat` / `nosat` | flag | Optional floating exceptional-value mode override; see common clauses |
| `tf32_mode` | clause | Optional TF32 input rounding; see common clauses |
| `n_dir` | flag | Optional N-direction producer order; see common clauses |

**Constraints:**

- `%bias` must be in `bias` address space.
- `%bias` element type must match `%dst` element type.
- `%bias` must satisfy the target alignment requirement for the bias buffer.
- Only `N` bias values are consumed. To model a logical `M x N` result, use
  `bias[None, :]` in the reference calculation, not an independent `M x N`
  bias matrix.
- `tf32_mode(...)` requires `f32` lhs, rhs, and dst element types.
- `sat` / `nosat` requires floating or MX lhs/rhs/dst element types.

**Example:**

```mlir
pto.mad_bias %l0a, %l0b, %l0c, %bt, %c16_i64, %c16_i64, %c16_i64
  : !pto.ptr<f16, left>, !pto.ptr<f16, right>, !pto.ptr<f32, acc>, !pto.ptr<f32, bias>, i64, i64, i64
```

---

### `pto.mad_mx`

- **syntax:**
```mlir
pto.mad_mx %lhs, %rhs, %dst, %m, %n, %k
  unit_flag(check_only | check_and_set)?
  disable_gemv?
  (sat | nosat)?
  n_dir?
  : !pto.ptr<A, left>, !pto.ptr<B, right>, !pto.ptr<C, acc>, i64, i64, i64
```
- **semantics:** Zero-init MX cube matmul, `dst = mx_matmul(lhs, rhs)`.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%lhs` | ptr | MX L0A input (`left`) |
| `%rhs` | ptr | MX L0B input (`right`) |
| `%dst` | ptr | L0C accumulator (`acc`) |
| `%m` | i64 | M size |
| `%n` | i64 | N size |
| `%k` | i64 | K size; MX scaling groups values in K-direction chunks of 32 |
| `unit_flag` | clause | Optional unit-flag producer control; see common clauses |
| `disable_gemv` | flag | Optional GEMV disable; see common clauses |
| `sat` / `nosat` | flag | Optional floating exceptional-value mode override; see common clauses |
| `n_dir` | flag | Optional N-direction producer order; see common clauses |

**Constraints:**

- Operands must use a target-supported MX dtype combination.
- The corresponding MX scale data for `%lhs` and `%rhs` must already be
  prepared and aligned with the data tiles selected by those pointers.

**Example:**

```mlir
pto.mad_mx %l0a, %l0b, %l0c, %c16_i64, %c16_i64, %c64_i64
  : !pto.ptr<f8E4M3FN, left>, !pto.ptr<f8E4M3FN, right>, !pto.ptr<f32, acc>, i64, i64, i64
```

---

### `pto.mad_mx_acc`

- **syntax:**
```mlir
pto.mad_mx_acc %lhs, %rhs, %dst, %m, %n, %k
  unit_flag(check_only | check_and_set)?
  disable_gemv?
  (sat | nosat)?
  n_dir?
  : !pto.ptr<A, left>, !pto.ptr<B, right>, !pto.ptr<C, acc>, i64, i64, i64
```
- **semantics:** Accumulating MX cube matmul, `dst += mx_matmul(lhs, rhs)`.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%lhs` | ptr | MX L0A input (`left`) |
| `%rhs` | ptr | MX L0B input (`right`) |
| `%dst` | ptr | L0C accumulator (`acc`) |
| `%m` | i64 | M size |
| `%n` | i64 | N size |
| `%k` | i64 | K size; MX scaling groups values in K-direction chunks of 32 |
| `unit_flag` | clause | Optional unit-flag producer control; see common clauses |
| `disable_gemv` | flag | Optional GEMV disable; see common clauses |
| `sat` / `nosat` | flag | Optional floating exceptional-value mode override; see common clauses |
| `n_dir` | flag | Optional N-direction producer order; see common clauses |

**Constraints:** same as `pto.mad_mx`; `sat` / `nosat` is valid because MX is a floating exceptional-value mode.

**Example:**

```mlir
pto.mad_mx_acc %l0a, %l0b, %l0c, %c16_i64, %c16_i64, %c64_i64
  : !pto.ptr<f8E4M3FN, left>, !pto.ptr<f8E4M3FN, right>, !pto.ptr<f32, acc>, i64, i64, i64
```

---

### `pto.mad_mx_bias`

- **syntax:**
```mlir
pto.mad_mx_bias %lhs, %rhs, %dst, %bias, %m, %n, %k
  unit_flag(check_only | check_and_set)?
  disable_gemv?
  (sat | nosat)?
  n_dir?
  : !pto.ptr<A, left>, !pto.ptr<B, right>, !pto.ptr<C, acc>, !pto.ptr<C, bias>, i64, i64, i64
```
- **semantics:** Bias-init MX cube matmul. Like `pto.mad_bias`, the bias is an
  `N`-element per-output-channel vector in the bias buffer and is broadcast
  across M:
  `dst[m, n] = mx_matmul(lhs, rhs)[m, n] + bias[n]`.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%lhs` | ptr | MX L0A input (`left`) |
| `%rhs` | ptr | MX L0B input (`right`) |
| `%dst` | ptr | L0C accumulator (`acc`) |
| `%bias` | ptr | Bias-buffer pointer (`bias`) for the `N`-element bias vector |
| `%m` | i64 | M size |
| `%n` | i64 | N size |
| `%k` | i64 | K size; MX scaling groups values in K-direction chunks of 32 |
| `unit_flag` | clause | Optional unit-flag producer control; see common clauses |
| `disable_gemv` | flag | Optional GEMV disable; see common clauses |
| `sat` / `nosat` | flag | Optional floating exceptional-value mode override; see common clauses |
| `n_dir` | flag | Optional N-direction producer order; see common clauses |

**Constraints:** same as `pto.mad_mx` plus the `pto.mad_bias` bias-buffer/broadcast
constraints; `sat` / `nosat` is valid because MX is a floating exceptional-value mode.

**Example:**

```mlir
pto.mad_mx_bias %l0a, %l0b, %l0c, %bt, %c16_i64, %c16_i64, %c64_i64
  : !pto.ptr<f8E4M3FN, left>, !pto.ptr<f8E4M3FN, right>, !pto.ptr<f32, acc>, !pto.ptr<f32, bias>, i64, i64, i64
```

---

## Cube Bridge Wrapper Ops

### `pto.cube_load`

- **syntax:**
```mlir
pto.cube_load %src, %dst, %len_burst
  nburst(%count, %src_stride, %dst_stride)
  [loop(%count_i, %src_stride_i, %dst_stride_i)]*
  : !pto.ptr<T, gm>, !pto.ptr<T, mat>, i64, i64, i64, i64
```
- **semantics:** Structured GM-to-L1 (`cbuf`) wrapper.
  The op copies one or more contiguous byte ranges from GM to L1. Each inner
  burst copies `%len_burst` bytes. `nburst` repeats that burst, and each
  optional `loop(...)` wraps the inner transfer pattern in an outer repeated
  pattern.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | GM source pointer |
| `%dst` | ptr | L1 destination pointer (`mat`) |
| `%len_burst` | i64 | Contiguous transfer length in bytes |
| `nburst(%count, %src_stride, %dst_stride)` | i64 triple | Inner burst count and source/destination gaps between bursts. The gap is applied after each `%len_burst` byte burst. |
| `loop(%count_i, %src_stride_i, %dst_stride_i)` | i64 triple | Optional outer loop triplet. Strides are byte offsets applied between repetitions of the enclosed transfer pattern. |

**Constraints:**

- For a contiguous 16-element f16 vector, use `%len_burst = 32`, not `16`.
- `%count = 1` with zero gaps describes one contiguous copy of `%len_burst`
  bytes. Increase `%count` for multiple same-sized bursts; use `loop(...)` for
  higher-dimensional tiling around the inner burst pattern.

**Example:**

```mlir
pto.cube_load %bias_gm, %l1_bias, %c32_i64
  nburst(%c1_i64, %c0_i64, %c0_i64)
  : !pto.ptr<f16, gm>, !pto.ptr<f16, mat>, i64, i64, i64, i64
```

---

### `pto.cube_store`

- **syntax:**
```mlir
pto.cube_store %src, %dst, %len_burst
  nburst(%count, %src_stride, %dst_stride)
  [loop(%count_i, %src_stride_i, %dst_stride_i)]*
  : !pto.ptr<T, mat>, !pto.ptr<T, ub>, i64, i64, i64, i64
```
- **semantics:** Structured L1 (`cbuf`) to UB wrapper. The transfer shape uses
  the same burst/repeat model as `pto.cube_load`, but the source is L1 and the
  destination is UB.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | L1 source pointer (`mat`) |
| `%dst` | ptr | UB destination pointer |
| `%len_burst` | i64 | Contiguous transfer length in bytes |
| `nburst(%count, %src_stride, %dst_stride)` | i64 triple | Inner burst count and source/destination gaps between bursts |
| `loop(%count_i, %src_stride_i, %dst_stride_i)` | i64 triple | Optional outer loop triplet. Strides are byte offsets between repetitions of the enclosed transfer pattern. |

**Constraints:**

- The source and destination address spaces must match the L1-to-UB dataflow.

**Example:**

```mlir
pto.cube_store %l1_src, %ub_dst, %c16_i64
  nburst(%c1_i64, %c0_i64, %c0_i64)
  : !pto.ptr<f16, mat>, !pto.ptr<f16, ub>, i64, i64, i64, i64
```

---

### `pto.cube_load_frac`

- **syntax:**
```mlir
pto.cube_load_frac %src, %dst, nd2nz|dn2nz, shape(%n_value, %d_value), src_layout(%src_inner_stride[, %src_outer_stride]), dst_group(%group_count, %dst_loop2_stride, %dst_loop3_stride, %dst_loop4_stride), ctrl(%l2_cache_ctrl, %smallc0_en)
  : !pto.ptr<T, gm>, !pto.ptr<T, mat>, ...
```
- **semantics:** Structured fractal-load wrapper for `nd2nz` / `dn2nz`.
  It copies a logical 2-D source region into L1 using the target fractal
  layout expected by subsequent cube loads. `nd2nz` treats the source as
  logical N-major rows with D as the inner dimension; `dn2nz` treats the same
  logical region with D/N interpretation swapped before writing NZ layout.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | GM source pointer |
| `%dst` | ptr | L1 destination pointer (`mat`) |
| `nd2nz` / `dn2nz` | enum token | Fractal load mode |
| `shape(%n_value, %d_value)` | i64 pair | Logical N and D shape of the source region being transformed |
| `src_layout(%src_inner_stride[, %src_outer_stride])` | i64 / i64 pair | Source address strides in bytes. For row-major f16 `[N, D]`, `%src_inner_stride = D * 2`. |
| `dst_group(%group_count, %dst_loop2_stride, %dst_loop3_stride, %dst_loop4_stride)` | i64 tuple | Destination fractal grouping and nested destination strides in C0-size units. These values organize where generated NZ fractals land in L1; they do not select a separate destination memory block. |
| `ctrl(%l2_cache_ctrl, %smallc0_en)` | i64, i1 | L2 cache policy hint and small-C0 mode enable |

**Constraints:**

- `src_inner_stride` and `src_outer_stride` are byte strides. Do not pass
  element counts. For example, a row-major `16 x 16` f16 matrix uses
  `src_layout(32)`.
- The destination pointer still selects the base L1 address. Destination
  grouping/strides are offsets relative to that base.
- `smallc0_en` is only valid for the target-supported small-C0 cases; the
  combination is invalid when `d_value > 4`.

**Example:**

```mlir
pto.cube_load_frac %src, %dst, nd2nz,
  shape(%n, %d),
  src_layout(%sis),
  dst_group(%g, %l2s, %l3s, %l4s),
  ctrl(%l2, %small)
  : !pto.ptr<f16, gm>, !pto.ptr<f16, mat>, nd2nz, shape i64, i64, src_layout(i64), dst_group i64, i64, i64, i64, ctrl i64, i1
```

---

### `pto.bias_load`

- **syntax:**
```mlir
pto.bias_load %src, %dst, %len_burst
  nburst(%count, %src_gap, %dst_gap)
  : !pto.ptr<T, mat>, !pto.ptr<U, bias>, i64, i64, i64, i64
```
- **semantics:** Structured helper for L1 (`cbuf`) to bias-buffer load. It
  prepares the per-output-channel bias vector consumed by `pto.mad_bias` and
  `pto.mad_mx_bias`. For f16/bf16 sources, the source data in L1 is stored
  compactly and can be converted to f32 values in the bias buffer.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | L1 source pointer (`mat`) |
| `%dst` | ptr | Bias destination pointer (`bias`) |
| `%len_burst` | i64 | Source burst length in the bias-load unit. For `f16->f32`, one unit covers 32B of compact f16 source data and produces 16 f32 bias values. |
| `%count` | i64 | Burst count |
| `%src_gap` | i64 | Source gap between bursts in the bias-load unit |
| `%dst_gap` | i64 | Destination gap between bursts in the bias-load unit; must preserve target N0 alignment |

**Constraints:**

- Supported type pairs: `f32->f32`, `i32->i32`, `f16->f32`, `bf16->f32`.
- For `f16->f32`, each compact f16 source value is converted to f32 before
  being written to the bias buffer.
- This op only prepares the bias buffer. The `mad_bias` consumer reads `N`
  consecutive bias values and broadcasts them across M.
- The bias buffer contains channel bias values, not an `M x N` result-shaped
  tile. Load exactly the N-channel bias data needed by the consumer tile.

**Example:**

```mlir
pto.bias_load %l1_bias, %bt, %c16_i64 nburst(%c1_i64, %c0_i64, %c0_i64)
  : !pto.ptr<f16, mat>, !pto.ptr<f32, bias>, i64, i64, i64, i64
```

This example loads one 32B source burst, i.e. 16 compact f16 bias values, and
materializes 16 f32 bias-buffer values for a `N = 16` `mad_bias`.

---

### `pto.fp_load`

- **syntax:**
```mlir
pto.fp_load %src, %dst, %len_burst
  nburst(%count, %src_gap, %dst_gap)
  : !pto.ptr<T, mat>, !pto.ptr<U, scaling>, i64, i64, i64, i64
```
- **semantics:** Structured helper for L1 (`cbuf`) to Fixpipe Buffer (`scaling`) load.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | L1 source pointer (`mat`) |
| `%dst` | ptr | Fixpipe-buffer destination pointer (`scaling`) |
| `%len_burst` | i64 | Burst length |
| `%count` | i64 | Burst count |
| `%src_gap` | i64 | Source gap |
| `%dst_gap` | i64 | Destination gap |

**Constraints:**

- `%src` must be in `mat`, `%dst` must be in `scaling`.

**Example:**

```mlir
pto.fp_load %l1_fp, %fb_fp, %c2_i64 nburst(%c1_i64, %c0_i64, %c0_i64)
  : !pto.ptr<f32, mat>, !pto.ptr<ui64, scaling>, i64, i64, i64, i64
```

---

### `pto.left_load`

- **syntax:**
```mlir
pto.left_load %src, %dst, %m, %k
  : !pto.ptr<T, mat>, !pto.ptr<T, left>, i64, i64
```
- **semantics:** Structured L1-to-L0A wrapper.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | L1 source pointer (`mat`) |
| `%dst` | ptr | L0A destination pointer (`left`) |
| `%m` | i64 | M tile size |
| `%k` | i64 | K tile size |

**Constraints:**

- `%src` must be in `mat`, `%dst` must be in `left`.

**Example:**

```mlir
pto.left_load %l1_a, %l0a, %c16_i64, %c16_i64
  : !pto.ptr<f16, mat>, !pto.ptr<f16, left>, i64, i64
```

---

### `pto.right_load`

- **syntax:**
```mlir
pto.right_load %src, %dst, %k, %n
  : !pto.ptr<T, mat>, !pto.ptr<T, right>, i64, i64
```
- **semantics:** Structured L1-to-L0B wrapper.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | L1 source pointer (`mat`) |
| `%dst` | ptr | L0B destination pointer (`right`) |
| `%k` | i64 | K tile size |
| `%n` | i64 | N tile size |

**Constraints:**

- `%src` must be in `mat`, `%dst` must be in `right`.

**Example:**

```mlir
pto.right_load %l1_b, %l0b, %c16_i64, %c16_i64
  : !pto.ptr<f16, mat>, !pto.ptr<f16, right>, i64, i64
```

---

### `pto.left_load_mx`

- **syntax:**
```mlir
pto.left_load_mx %src, %dst, %m, %k
  : !pto.ptr<T, mat>, !pto.ptr<T, left>, i64, i64
```
- **semantics:** MX-mode L1-to-L0A wrapper.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | MX-formatted L1 source pointer (`mat`) |
| `%dst` | ptr | MX L0A destination pointer (`left`) |
| `%m` | i64 | M tile size |
| `%k` | i64 | K tile size |

**Constraints:**

- `%src` must be in `mat`, `%dst` must be in `left`.

**Example:**

```mlir
pto.left_load_mx %l1_a, %l0a, %c16_i64, %c64_i64
  : !pto.ptr<f8E4M3FN, mat>, !pto.ptr<f8E4M3FN, left>, i64, i64
```

---

### `pto.right_load_mx`

- **syntax:**
```mlir
pto.right_load_mx %src, %dst, %k, %n
  : !pto.ptr<T, mat>, !pto.ptr<T, right>, i64, i64
```
- **semantics:** MX-mode L1-to-L0B wrapper.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | MX-formatted L1 source pointer (`mat`) |
| `%dst` | ptr | MX L0B destination pointer (`right`) |
| `%k` | i64 | K tile size |
| `%n` | i64 | N tile size |

**Constraints:**

- `%src` must be in `mat`, `%dst` must be in `right`.

**Example:**

```mlir
pto.right_load_mx %l1_b, %l0b, %c64_i64, %c16_i64
  : !pto.ptr<f8E4M3FN, mat>, !pto.ptr<f8E4M3FN, right>, i64, i64
```

---

### `pto.acc_store`

`pto.acc_store*` 是结构化的 fixpipe 写回族，用来把 `pto.mad*` 产出的 L0C(`acc`) 结果写到不同目标空间。
从语义上看，这条流水按下面的顺序组织：

1. 读取 `%src` 指向的 L0C 累加结果，并按 `%m/%n` 解释逻辑输出区域。
2. 如指定 `unit_flag(...)`，在消费这次 L0C 结果前执行完成态检查；`check_and_clear` 还会在消费后清除该完成态，便于后续下一轮生产/消费配对。
3. 如指定 `pre_quant(%payload, mode = ...)`，先对 L0C 元素做预量化。这里的 `%payload` 是该量化模式所需的标量参数或 scaling 指针；标量模式允许直接传 `f16`、`bf16`、`f32`，向量模式要求 `scaling` 指针。
4. 如指定 `pre_relu(...)`，在写回前对结果做 ReLU 预处理。`scalar_relu`/`vector_relu` 需要额外 payload；`normal_relu`/`no_relu` 不接 payload。`scalar_relu` 允许直接传 `f16`、`bf16`、`f32` alpha，`vector_relu` 要求 `scaling` 指针。`clip = %clip` 是 `pre_relu(...)` 子句的一部分，用于在支持的目标元素类型上启用 clip 阶段。
5. 按 `nz2nd` / `nz2dn` / `nz2nz` 将 L0C 中的 NZ 累加布局转换成目标布局，并结合 `%src_stride`、`%dst_stride` 以及可选 `loop3(...)`/`%loop0_src_stride`/`%split` 控制跨 tile 的遍历方式。
6. 如指定 `sat`，则在最终写回目标元素类型时启用饱和语义。
7. 将结果写入目标空间。`acc_store` 写 L1(`mat`)，`acc_store_ub` 写 UB，`acc_store_gm` 写 GM；GM 路径还可额外指定原子更新语义。

- **syntax:**
```mlir
pto.acc_store %src, %dst, %m, %n, %src_stride, %dst_stride
    [, unit_flag(check_only | check_and_clear)]?
    [, pre_quant(%payload, mode = <quant_pre_mode>)]?
    [, pre_relu(%payload, mode = <relu_pre_mode> [, clip = %clip])]?
    [, nz2nd | nz2dn(%loop0_src_stride) | nz2nz(%split)?]
    [, loop3(%count, %src_stride3, %dst_stride3)]?
    [, sat]?
  : ...
```
- **semantics:** Structured L0C (`acc`) to L1 (`cbuf`) wrapper.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | buffer-like | L0C source buffer (`acc`)，可为 typed `!pto.ptr` 或等价 memref |
| `%dst` | buffer-like | L1 destination buffer (`mat`)，可为 typed `!pto.ptr` 或等价 memref |
| `%m` | i64 | M size |
| `%n` | i64 | N size |
| `%src_stride` | i64 | 源 NZ 布局在 fixpipe 写回过程中的主 stride 参数 |
| `%dst_stride` | i64 | 目标布局在 fixpipe 写回过程中的主 stride 参数 |
| `unit_flag(...)` | optional clause | 是否在消费这次 L0C 结果前检查完成态；`check_and_clear` 还会在消费后清除完成态 |
| `pre_quant(%payload, mode = ...)` | optional clause | 写回前的预量化；payload 为该量化模式需要的浮点标量或 `scaling` 指针 |
| `pre_relu(..., mode = ...[, clip = %clip])` | optional clause | 写回前的 ReLU 预处理；`clip` 只能作为 `pre_relu` 的一部分出现 |
| `nz2nd` / `nz2dn(%loop0_src_stride)` / `nz2nz(%split)?` | mode clause | L0C NZ 布局到目标布局的写回模式 |
| `loop3(%count, %src_stride3, %dst_stride3)` | optional i64 triple | 额外的外层重复写回控制，用于跨 tile 迭代 |
| `sat` | optional flag | 最终写回到目标元素类型时启用饱和语义 |

**Constraints:**

- 子句顺序固定为 `unit_flag` -> `pre_quant` -> `pre_relu` -> `mode` -> `loop3` -> `sat`。
- `pre_quant` 必须同时提供 payload 和 `mode`。
- `pre_quant` 仅支持 L0C 源元素类型为 `f32` 或 `i32`。
- 标量 `pre_quant` 模式要求 payload 为 `f16`/`bf16`/`f32` 标量；其中 `f16`/`bf16` 会在 lowering 中先扩成 `f32`，再以 32-bit 浮点 bit pattern 形式配置到 fixpipe 标量参数寄存器。向量 `pre_quant` 模式要求 `scaling !pto.ptr<ui64>` payload。
- `pre_relu` 的 payload 规则取决于 `mode`：
  - `no_relu` / `normal_relu` 不接受 payload。
  - `scalar_relu` 要求 payload 为 `f16`/`bf16`/`f32` 标量；其中 `f16`/`bf16` 会在 lowering 中先扩成 `f32`，再以 32-bit 浮点 bit pattern 形式配置到 fixpipe 标量参数寄存器。
  - `vector_relu` 要求 `scaling !pto.ptr<ui64>` payload。
- `clip` 只能出现在 `pre_relu(...)` 中。
- `clip` 仅支持目标元素类型为 `f16`、`ui8`、或有符号 `i4/i8/i16`。
- `clip` payload 必须与目标元素类型匹配：
  - `f16` 目标要求 `f16` payload。
  - `ui8` 目标要求 `ui16` 风格的 16-bit payload；在 PTO IR 中通常写成 `signless i16`。
  - 有符号 `i4/i8/i16` 目标要求有符号或 signless 的 `i4/i8/i16` payload。
- `loop3(...)` 必须三个操作数同时提供。
- `nz2dn` 必须提供 `%loop0_src_stride`；`nz2nd`/`nz2nz` 不接受它。
- 当 `nz2dn(%loop0_src_stride)` 中 `%loop0_src_stride != 1` 时，`unit_flag` 必须关闭。
- `nz2nz` 不接受 `loop3(...)`，且目标元素类型必须是 `f32`。

**Example:**

```mlir
pto.acc_store %l0c, %l1_out, %c16_i64, %c16_i64, %c16_i64, %c16_i64, nz2dn(%c64_i64), loop3(%c3_i64, %c4_i64, %c5_i64)
  : !pto.ptr<f32, acc>, !pto.ptr<f32, mat>, i64, i64, i64, i64, i64, i64, i64, i64
```

---

### `pto.acc_store_gm`

- **syntax:**
```mlir
pto.acc_store_gm %src, %dst, %m, %n, %src_stride, %dst_stride, %sid, %l2_cache_ctrl
    [, unit_flag(check_only | check_and_clear)]?
    [, pre_quant(%payload, mode = <quant_pre_mode>)]?
    [, pre_relu(%payload, mode = <relu_pre_mode> [, clip = %clip])]?
    [, nz2nd | nz2dn(%loop0_src_stride) | nz2nz(%split)?]
    [, loop3(%count, %src_stride3, %dst_stride3)]?
    [, sat]?
    [, atomic(type = <atomic_type>, op = <atomic_op>)]?
  : ...
```
- **semantics:** Structured L0C (`acc`) to GM wrapper.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | buffer-like | L0C source buffer (`acc`)，可为 typed `!pto.ptr` 或等价 memref |
| `%dst` | buffer-like | GM destination buffer，可为 typed `!pto.ptr` 或等价 memref |
| `%m` | i64 | M size |
| `%n` | i64 | N size |
| `%src_stride` | i64 | 源 NZ 布局在 fixpipe 写回过程中的主 stride 参数 |
| `%dst_stride` | i64 | 目标布局在 fixpipe 写回过程中的主 stride 参数 |
| `%sid` | i64 | GM 写回使用的 stream/session 标识参数 |
| `%l2_cache_ctrl` | i64 | GM 路径的 L2 cache 策略参数 |
| (optional clauses) | — | 与 `pto.acc_store` 相同的语义子句，外加 GM 独有的 `atomic(...)` |

**Constraints:**

- GM output path controls (`sid`, `l2_cache_ctrl`) must be provided.
- `atomic(type = ..., op = ...)` 只允许出现在 `pto.acc_store_gm`。
- `atomic` 必须同时提供 `type` 和 `op`。
- 当前 `op` 取值为 `add` / `max` / `min`；`type` 取值为 `f32` / `f16` / `bf16` / `s32` / `s16` / `s8`。

**Example:**

```mlir
pto.acc_store_gm %l0c, %c_gm, %c16_i64, %c16_i64, %c16_i64, %c16_i64, %c0_i64, %c0_i64, nz2nd
  : !pto.ptr<f32, acc>, !pto.ptr<f32, gm>, i64, i64, i64, i64, i64, i64
```

---

### `pto.acc_store_ub`

- **syntax:**
```mlir
pto.acc_store_ub %src, %dst, %m, %n, %src_stride, %dst_stride, %dual_dst_mode, %sub_blockid
    [, unit_flag(check_only | check_and_clear)]?
    [, pre_quant(%payload, mode = <quant_pre_mode>)]?
    [, pre_relu(%payload, mode = <relu_pre_mode> [, clip = %clip])]?
    [, nz2nd | nz2dn(%loop0_src_stride) | nz2nz(%split)?]
    [, loop3(%count, %src_stride3, %dst_stride3)]?
    [, sat]?
  : ...
```
- **semantics:** Structured L0C (`acc`) to UB wrapper.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | buffer-like | L0C source buffer (`acc`)，可为 typed `!pto.ptr` 或等价 memref |
| `%dst` | buffer-like | UB destination buffer，可为 typed `!pto.ptr` 或等价 memref |
| `%m` | i64 | M size |
| `%n` | i64 | N size |
| `%src_stride` | i64 | 源 NZ 布局在 fixpipe 写回过程中的主 stride 参数 |
| `%dst_stride` | i64 | 目标布局在 fixpipe 写回过程中的主 stride 参数 |
| `%dual_dst_mode` | i64 | UB 路径的双目标写回模式参数 |
| `%sub_blockid` | i64 | UB 路径的子块选择参数 |
| (optional clauses) | — | 与 `pto.acc_store` 相同，但不支持 `atomic(...)` |

**Constraints:**

- 不支持 `atomic(...)`。

**Example:**

```mlir
pto.acc_store_ub %l0c, %ub_out, %c16_i64, %c16_i64, %c16_i64, %c16_i64, %c0_i64, %c0_i64, nz2nd
  : !pto.ptr<f32, acc>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64, i64
```

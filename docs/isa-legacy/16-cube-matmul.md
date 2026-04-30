# 16A. Cube Matmul Raw Ops (MAT)

> **Category:** Cube unit raw ops — low-level data movement and matrix-side configuration
> **Audience:** Backend / lowering / bridge-op implementers

---

## Scope

This document lists the raw cube matmul ops used by wrapper interfaces in
`16-cube-matmul.md`.

If you are writing user-facing VPTO kernels, prefer wrapper ops such as
`pto.cube_load`, `pto.left_load`, `pto.right_load`, `pto.acc_store*`, and
`pto.cube_load_frac`.

---

## Raw Staging / Load Ops

### `pto.copy_gm_to_cbuf`

- **syntax:**
```mlir
pto.copy_gm_to_cbuf %src, %dst, %n_burst, %len_burst, %src_stride, %dst_stride
  : !pto.ptr<T, gm>, !pto.ptr<T, mat>, i64, i64, i64, i64
```
- **semantics:** Copy matrix tile data from GM to L1 (`cbuf`).

### `pto.load_cbuf_to_ca`

- **syntax:**
```mlir
pto.load_cbuf_to_ca %src, %dst, %m_start, %k_start, %m_step, %k_step, %src_stride, %dst_stride
  : !pto.ptr<T, mat>, !pto.ptr<T, left>, i64, i64, i64, i64, i64, i64
```
- **semantics:** Load L1 (`cbuf`) tile to L0A.

### `pto.load_cbuf_to_cb`

- **syntax:**
```mlir
pto.load_cbuf_to_cb %src, %dst, %m_start, %k_start, %m_step, %k_step, %src_stride, %dst_stride
  : !pto.ptr<T, mat>, !pto.ptr<T, right>, i64, i64, i64, i64, i64, i64
```
- **semantics:** Load L1 (`cbuf`) tile to L0B.

### `pto.load_cbuf_to_ca_mx`

- **syntax:**
```mlir
pto.load_cbuf_to_ca_mx %src, %dst, %m, %k
  : !pto.ptr<T, mat>, !pto.ptr<T, left>, i64, i64
```
- **semantics:** Load L1 (`cbuf`) tile to L0A using MX path.

### `pto.load_cbuf_to_cb_mx`

- **syntax:**
```mlir
pto.load_cbuf_to_cb_mx %src, %dst, %x_start_position, %y_start_position, %x_step, %y_step, %src_stride, %dst_stride
  : !pto.ptr<T, mat>, !pto.ptr<T, right>, i64, i64, i64, i64, i64, i64
```
- **semantics:** Load L1 (`cbuf`) tile to L0B using MX path with explicit hardware control fields.

---

## Raw L0C Writeback / Move Ops

### `pto.copy_matrix_cc_to_gm`

- **syntax:**
```mlir
pto.copy_matrix_cc_to_gm %src, %dst, %xm, %xt
  : !pto.ptr<T, acc>, !pto.ptr<T, gm>, i64, i64
```
- **semantics:** Write L0C (`acc`) tile back to GM.

### `pto.copy_matrix_cc_to_cbuf`

- **syntax:**
```mlir
pto.copy_matrix_cc_to_cbuf %src, %dst, %config0, %config1
  : !pto.ptr<T, acc>, !pto.ptr<T, mat>, i64, i64
```
- **semantics:** Move L0C (`acc`) tile to L1 (`cbuf`).

### `pto.copy_matrix_cc_to_ub`

- **syntax:**
```mlir
pto.copy_matrix_cc_to_ub %src, %dst, %config0, %config1
  : !pto.ptr<T, acc>, !pto.ptr<T, ub>, i64, i64
```
- **semantics:** Move L0C (`acc`) tile to UB.

---

## Raw CBUF Outbound Ops

### `pto.copy_cbuf_to_bt`

- **syntax:**
```mlir
pto.copy_cbuf_to_bt %src, %dst, %len_burst, %n_burst, %src_gap, %dst_gap
  : !pto.ptr<T, mat>, !pto.ptr<U, bias>, i64, i64, i64, i64
```
- **semantics:** Move L1 (`cbuf`) data to BT buffer.

### `pto.copy_cbuf_to_fbuf`

- **syntax:**
```mlir
pto.copy_cbuf_to_fbuf %src, %dst, %n_burst, %len_burst, %src_gap, %dst_gap
  : !pto.ptr<T, mat>, !pto.ptr<T, ub>, i64, i64, i64, i64
```
- **semantics:** Move L1 (`cbuf`) data to FB-related destination path.

### `pto.copy_gm_to_cbuf_multi_nd2nz`

- **syntax:**
```mlir
pto.copy_gm_to_cbuf_multi_nd2nz %src, %dst, %sid, %loop1_src_stride, %l2_cache_ctrl, %n_value, %d_value, %loop4_src_stride, %smallc0_en
  : !pto.ptr<T, gm>, !pto.ptr<T, mat>, i64, i64, i64, i64, i64, i64, i1
```
- **semantics:** Multi-fractal `ND2NZ` staging from GM to L1 (`cbuf`).

### `pto.copy_gm_to_cbuf_multi_dn2nz`

- **syntax:**
```mlir
pto.copy_gm_to_cbuf_multi_dn2nz %src, %dst, %sid, %loop1_src_stride, %l2_cache_ctrl, %n_value, %d_value, %loop4_src_stride, %smallc0_en
  : !pto.ptr<T, gm>, !pto.ptr<T, mat>, i64, i64, i64, i64, i64, i64, i1
```
- **semantics:** Multi-fractal `DN2NZ` staging from GM to L1 (`cbuf`).

---

## Raw vs Wrapper Mapping

| Wrapper op | Typical raw op(s) |
|---|---|
| `pto.cube_load` | `pto.copy_gm_to_cbuf` + loop setup |
| `pto.left_load` | `pto.load_cbuf_to_ca` |
| `pto.right_load` | `pto.load_cbuf_to_cb` |
| `pto.left_load_mx` | `pto.load_cbuf_to_ca_mx` |
| `pto.right_load_mx` | `pto.load_cbuf_to_cb_mx` |
| `pto.cube_load_frac` | `pto.copy_gm_to_cbuf_multi_nd2nz` / `pto.copy_gm_to_cbuf_multi_dn2nz` + config setup |
| `pto.bias_load` | `pto.copy_cbuf_to_bt` |
| `pto.acc_store` | `pto.copy_matrix_cc_to_cbuf` (+ related config) |
| `pto.acc_store_gm` | `pto.copy_matrix_cc_to_gm` (+ related config) |
| `pto.acc_store_ub` | `pto.copy_matrix_cc_to_ub` (+ related config) |


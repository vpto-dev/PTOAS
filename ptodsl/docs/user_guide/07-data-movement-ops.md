# 7. Data Movement Operations

This chapter covers every operation that moves data between memory spaces in PTODSL — tile-level transfers, DMA micro-instructions, vector loads and stores, and cube data movement. Operations are organized by abstraction level: tile ops (L1), DMA ops (L2), vector memory ops (L3 SIMD), and cube memory ops (L3 cube).

## 7.1 Tile-level movement: tload and tstore

Tile ops move entire blocks between Global Memory and the Unified Buffer in a single call. They are the primary data movement interface inside `@pto.jit`.

#### `pto.tload(partition: PartitionTensorView, tile: Tile) -> None`

**Description**: Copies data from a GM partition into a UB tile. The transfer size is determined by the partition's `sizes` and the tile's shape — they must be compatible.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `partition` | `PartitionTensorView` | Source region in GM |
| `tile` | `Tile` | Destination buffer in UB |

**Returns**: None (side-effect operation).

**Example**:

```python
a_part = pto.partition_view(a_view, offsets=[offset], sizes=[BLOCK])
a_tile = pto.alloc_tile(shape=[BLOCK], dtype=pto.f32)
pto.tload(a_part, a_tile)
```

---

#### `pto.tstore(tile: Tile, partition: PartitionTensorView) -> None`

**Description**: Copies data from a UB tile back to a GM partition. The tile's `valid_shape` determines how many elements are written; elements outside `valid_shape` are not stored.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `tile` | `Tile` | Source buffer in UB |
| `partition` | `PartitionTensorView` | Destination region in GM |

**Returns**: None (side-effect operation).

**Example**:

```python
pto.tstore(o_tile, o_part)
```

---

Both `tload` and `tstore` operate at **tile granularity** — they are the idiomatic choice inside `@pto.jit` loops. When you need finer control over DMA scheduling, drop down to the micro-instruction level.

## 7.2 DMA micro-instructions (ukernel)

Inside `@pto.ukernel`, data movement between memory spaces is expressed with grouped DMA instructions on typed pointers. There are four operations covering the four data-movement directions:

| Operation | Direction | Stride unit | Padding |
|-----------|-----------|-------------|---------|
| `pto.mte_gm_ub` | GM → UB | bytes | Supported |
| `pto.mte_ub_gm` | UB → GM | bytes | — (de-padded on read) |
| `pto.mte_ub_ub` | UB → UB | 32B units | — |
| `pto.mte_ub_l1` | UB → L1 | 32B units | — |

All four share a common structure: a required innermost `nburst(...)` group that defines the repeated burst transfer, plus optional outer `loop(...)` groups for multi-level repetition. `pto.mte_gm_ub` additionally supports `pad(...)` for UB row padding.

> **Convenience wrappers**: `pto.mte_load(src, dst)` and `pto.mte_store(src, dst)` are Python-level shorthands that expand to `mte_gm_ub` / `mte_ub_gm` with inferred strides. The reference operations below are the full grouped MTE interfaces.

### 7.2.1 GM → UB: `pto.mte_gm_ub`

#### `pto.mte_gm_ub(gm_src: PtrType, ub_dst: PtrType, l2_cache_ctl: int, len_burst: int, *, nburst: tuple[int, int, int], loops: list[tuple[int, int, int]] | None = None, pad: tuple[ScalarType, int, int] | tuple[ScalarType] | None = None) -> None`

**Description**: Grouped DMA transfer from Global Memory to Unified Buffer. `nburst(...)` defines the innermost repeated burst (count, source stride in bytes, destination stride in bytes). Optional `loop(...)` groups add outer repetition levels. Optional `pad(...)` fills the gap between `len_burst` and `dst_stride` up to the 32B-aligned boundary.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `gm_src` | `PtrType` (gm) | GM source pointer |
| `ub_dst` | `PtrType` (ub) | UB destination pointer (must be 32B-aligned) |
| `l2_cache_ctl` | `int` | L2 cache allocate control (2 bits) |
| `len_burst` | `int` | Contiguous bytes transferred per burst row |
| `nburst` | `tuple[int, int, int]` | `(n_burst, src_stride, dst_stride)` — innermost burst group (required) |
| `loops` | `list[tuple[int, int, int]]` or `None` | Optional outer loop groups, each `(count, src_stride, dst_stride)`. Ordered inner to outer |
| `pad` | `tuple[ScalarType, int, int]` or `tuple[ScalarType]` or `None` | Optional padding: `(pad_value, left_count, right_count)` or `(pad_value,)`. Omitted counts default to 0 |

**Returns**: None (side-effect operation).

**Constraints**:
- `nburst` is always required.
- `loop` groups are ordered from inner (wrapping `nburst`) to outer.
- If `pad` specifies either left or right count, both must be provided.

**Example** — load a 32×32 f32 tile from contiguous GM into contiguous UB:

```python
pto.mte_gm_ub(gm_ptr, ub_ptr, 0, 128,
              nburst=(32, 128, 128))
# 32 rows, 128 bytes per row, contiguous in both GM and UB
```

**Example** — load a 64×128 f16 tile from a larger GM matrix (1024×512) into UB:

```python
pto.mte_gm_ub(gm_ptr, ub_ptr, 0, 256,
              nburst=(64, 1024, 256))
# 64 rows of 256 bytes each.
# GM: each row is 1024 bytes apart (full matrix row stride).
# UB: rows are packed contiguously (256-byte stride).
```

**Example** — load with padding (100 valid f16 columns into a 128-wide UB tile):

```python
pto.mte_gm_ub(gm_ptr, ub_ptr, 0, 200,
              nburst=(64, 200, 256),
              pad=(0.0, 0, 0))
# 64 rows, 200 valid bytes per row, 256-byte UB stride.
# Gap (56 bytes) between len_burst and dst_stride is zero-padded.
```

**Example** — multi-level loop: load 4 batches of 8×128 f16 tiles:

```python
pto.mte_gm_ub(gm_ptr, ub_ptr, 0, 256,
              nburst=(8, 256, 256),
              loops=[(4, 2048, 2048)])
# Innermost: 8 rows × 256B (one tile).
# Outer loop: 4 iterations, each advancing 2048 bytes in both GM and UB.
```

---

### 7.2.2 UB → GM: `pto.mte_ub_gm`

#### `pto.mte_ub_gm(ub_src: PtrType, gm_dst: PtrType, len_burst: int, *, nburst: tuple[int, int, int], loops: list[tuple[int, int, int]] | None = None) -> None`

**Description**: Grouped DMA transfer from Unified Buffer to Global Memory. The MTE reads `len_burst` bytes from each UB row (skipping any padding), writing only valid data to GM.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `ub_src` | `PtrType` (ub) | UB source pointer (must be 32B-aligned) |
| `gm_dst` | `PtrType` (gm) | GM destination pointer |
| `len_burst` | `int` | Contiguous bytes transferred per burst row |
| `nburst` | `tuple[int, int, int]` | `(n_burst, src_stride, dst_stride)` — innermost burst group (required) |
| `loops` | `list[tuple[int, int, int]]` or `None` | Optional outer loop groups, ordered inner to outer |

**Returns**: None (side-effect operation).

**Example** — store a 32×32 f32 tile from UB to GM:

```python
pto.mte_ub_gm(ub_ptr, gm_ptr, 128,
              nburst=(32, 128, 128))
```

**Example** — store a 64×128 f16 tile back to a larger GM matrix:

```python
pto.mte_ub_gm(ub_ptr, gm_ptr, 256,
              nburst=(64, 256, 1024))
# UB: contiguous rows (256-byte stride).
# GM: rows spaced at 1024-byte intervals (full matrix width).
```

---

### 7.2.3 UB → UB: `pto.mte_ub_ub`

#### `pto.mte_ub_ub(ub_src: PtrType, ub_dst: PtrType, len_burst: int, *, nburst: tuple[int, int, int]) -> None`

**Description**: Grouped UB-to-UB copy. Stride and gap values are in units of 32 bytes.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `ub_src` | `PtrType` (ub) | UB source pointer (must be 32B-aligned) |
| `ub_dst` | `PtrType` (ub) | UB destination pointer (must be 32B-aligned) |
| `len_burst` | `int` | Burst length in units of 32 bytes |
| `nburst` | `tuple[int, int, int]` | `(n_burst, src_gap, dst_gap)` — count, source gap, destination gap (all in 32B units) |

**Returns**: None (side-effect operation).

Each burst copies `len_burst * 32` bytes. The next burst starts at `src + (len_burst + src_gap) * 32` and `dst + (len_burst + dst_gap) * 32`.

**Example**:

```python
pto.mte_ub_ub(ub_src, ub_dst, 8,
              nburst=(16, 0, 4))
# 16 bursts, each copying 8×32=256 bytes.
# Source: contiguous (src_gap=0).
# Destination: 4×32=128-byte gap between bursts.
```

---

### 7.2.4 UB → L1: `pto.mte_ub_l1`

#### `pto.mte_ub_l1(ub_src: PtrType, l1_dst: PtrType, len_burst: int, *, nburst: tuple[int, int, int]) -> None`

**Description**: Grouped UB-to-L1 (CBUF) copy. Identical structure to `mte_ub_ub` but the destination is L1 cube buffer space.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `ub_src` | `PtrType` (ub) | UB source pointer (must be 32B-aligned) |
| `l1_dst` | `PtrType` (l1) | L1 destination pointer (must be 32B-aligned) |
| `len_burst` | `int` | Burst length in units of 32 bytes |
| `nburst` | `tuple[int, int, int]` | `(n_burst, src_gap, dst_gap)` — all in 32B units |

**Returns**: None (side-effect operation).

---

### 7.2.5 The nburst / loop / pad model

All grouped DMA operations follow a nested-loop execution model. `nburst` is the innermost group; each `loop` wraps the previous group as an outer iteration level.

For `mte_gm_ub` and `mte_ub_gm`, strides are **byte distances** from the start of one burst row to the start of the next:

```
GM → UB (nburst only):

  for r in range(n_burst):
      memcpy(ub_dst + r * dst_stride,
             gm_src + r * src_stride,
             len_burst)
      if pad enabled:
          memset(ub_dst + r * dst_stride + len_burst,
                 pad_value,
                 dst_stride_aligned - len_burst)
```

Each additional `loop(count, src_stride, dst_stride)` adds one outer `for` level that advances both base pointers by the corresponding strides.

For `mte_ub_ub` and `mte_ub_l1`, the parameters are in **32-byte units**. Each burst copies `len_burst * 32` bytes, and the next burst starts at `src + (len_burst + src_gap) * 32` / `dst + (len_burst + dst_gap) * 32`.

**UB address alignment**: For all four operations, every UB address (source and destination) must be 32-byte aligned. The `pad(...)` on `mte_gm_ub` ensures each UB row is padded to the 32B-aligned boundary of `dst_stride`, so subsequent rows stay aligned.

### 7.2.6 Typical ukernel DMA pattern

```python
@pto.ukernel
def process_block(k_part, v_part, k_tile, v_tile, o_tile, o_part,
                  rows: pto.i32, cols: pto.i32):
    # Stage K and V blocks from GM to UB
    pto.mte_gm_ub(k_part.as_ptr(), k_tile.as_ptr(), 0,
                  cols * pto.bytewidth(pto.f16),
                  nburst=(rows, cols * pto.bytewidth(pto.f16),
                          cols * pto.bytewidth(pto.f16)))
    pto.mte_gm_ub(v_part.as_ptr(), v_tile.as_ptr(), 0,
                  cols * pto.bytewidth(pto.f16),
                  nburst=(rows, cols * pto.bytewidth(pto.f16),
                          cols * pto.bytewidth(pto.f16)))
    pto.mem_bar(pto.BarrierType.SYNC)

    # ... compute on tiles ...

    pto.mem_bar(pto.BarrierType.SYNC)
    pto.mte_ub_gm(o_tile.as_ptr(), o_part.as_ptr(),
                  cols * pto.bytewidth(pto.f32),
                  nburst=(rows, cols * pto.bytewidth(pto.f32),
                          cols * pto.bytewidth(pto.f32)))
```

## 7.3 Vector loads (simd)

Inside `@pto.simd`, data moves between UB tiles and vector registers (`vreg`). Vector loads read a contiguous chunk of a tile row into a `vreg`; the chunk size equals the hardware vector width for the element type (e.g., 64 elements for `f32`, 128 for `f16`).

### Tile-index syntax

All vector load and store operations support the element-indexing syntax, which eliminates manual byte-offset calculation:

```python
vec = pto.vlds(tile[row, col:])       # load from row, starting at column col
vec = pto.vlds(tile[start:])          # 1D tile, starting at element start
```

The compiler automatically computes the byte offset from the tile's shape, element type, and layout. The `:` indicates a full vector-width range — the number of elements loaded is `elements_per_vreg(dtype)`.

---

#### `pto.vlds(tile[row, col:], dist: VLoadDist | None = None) -> VRegType`
#### `pto.vlds(tile[start:], dist: VLoadDist | None = None) -> VRegType`
#### `pto.vlds(buf: PtrType, offset: Index, dist: VLoadDist | None = None) -> VRegType`

**Description**: Stateless vector load from UB. Reads one vector-width slice.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `tile[row, col:]` | Tile index | 2D tile row with starting column (vector-width range) |
| `tile[start:]` | Tile index | 1D tile with starting element (vector-width range) |
| `buf` | `PtrType` (UB) | Pointer to buffer in UB (pointer form) |
| `offset` | `Index` | Byte offset (pointer form) |
| `dist` | `VLoadDist` or `None` | Optional load distribution: `NORM` (default), `UNPK_B8`/`UNPK_B16`/`UNPK_B32`, `BRC_B8`/`BRC_B16`/`BRC_B32` |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `vec` | `VRegType` | Loaded vector register |

---

#### `pto.vldsx2(tile[row, col:], dist: DeinterleaveDist) -> (VRegType, VRegType)`
#### `pto.vldsx2(tile[start:], dist: DeinterleaveDist) -> (VRegType, VRegType)`
#### `pto.vldsx2(buf: PtrType, offset: Index, dist: DeinterleaveDist) -> (VRegType, VRegType)`

**Description**: Dual vector load with deinterleave (AoS → SoA). Loads interleaved data and deinterleaves into two vectors.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `tile[row, col:]` | Tile index | 2D tile row with starting column (vector-width range) |
| `tile[start:]` | Tile index | 1D tile with starting element (vector-width range) |
| `buf` | `PtrType` (UB) | Pointer to buffer in UB (pointer form) |
| `offset` | `Index` | Byte offset (pointer form) |
| `dist` | `DeinterleaveDist` | `DINTLV` (alternating elements) or `BDINTLV` (block deinterleave) |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `low` | `VRegType` | Even-indexed elements |
| `high` | `VRegType` | Odd-indexed elements |

---

#### `pto.vldas(tile[row, col:]) -> AlignType`
#### `pto.vldas(tile[start:]) -> AlignType`
#### `pto.vldas(buf: PtrType) -> AlignType`

**Description**: Primes the alignment buffer for a subsequent unaligned load stream. Returns alignment state consumed by `vldus`.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `tile[row, col:]` | Tile index | 2D tile row with starting column |
| `tile[start:]` | Tile index | 1D tile with starting element |
| `buf` | `PtrType` | Pointer to buffer in UB |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `align` | `AlignType` | Alignment state for use with `vldus` |

---

#### `pto.vldus(tile[row, col:], align: AlignType) -> (VRegType, AlignType, PtrType)`
#### `pto.vldus(tile[start:], align: AlignType) -> (VRegType, AlignType, PtrType)`
#### `pto.vldus(buf: PtrType, align: AlignType) -> (VRegType, AlignType, PtrType)`

**Description**: Unaligned load with alignment state threading. Requires alignment state from `vldas` or a previous `vldus`.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `tile[row, col:]` | Tile index | 2D tile row with starting column (vector-width range) |
| `tile[start:]` | Tile index | 1D tile with starting element (vector-width range) |
| `buf` | `PtrType` (UB) | Pointer to buffer in UB (pointer form) |
| `offset` | `Index` | Byte offset (pointer form) |
| `align` | `AlignType` | Alignment state from `vldas` or previous `vldus` |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `vec` | `VRegType` | Assembled vector |
| `align_out` | `AlignType` | Updated alignment state for next load |
| `base_out` | `PtrType` | Post-update base pointer |

**Example**:

```python
align = pto.vldas(tile[row, col:])
vec, align, base = pto.vldus(tile[row, col:], align)
```

---

#### `pto.vsld(tile[row, col], stride: StrideMode) -> VRegType`
#### `pto.vsld(tile[pos], stride: StrideMode) -> VRegType`
#### `pto.vsld(buf: PtrType, offset: Index, stride: StrideMode) -> VRegType`

**Description**: Strided scalar load with broadcast. Loads a single element using a strided access pattern and broadcasts to all vector lanes.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `tile[row, col]` | Tile index | 2D single-element index |
| `tile[pos]` | Tile index | 1D single-element index |
| `stride` | `StrideMode` | `S3_B16`, `S4_B64`, `S8_B32`, or `S2_B64` |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `vec` | `VRegType` | Broadcast vector |

---

#### `pto.vgather2(buf: PtrType, offsets: Index, active_lanes: Index) -> VRegType`

**Description**: Indexed gather from UB using per-lane offsets. Only the first `active_lanes` lanes participate.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `buf` | `PtrType` (UB) | Source buffer |
| `offsets` | `Index` | Per-lane element offsets (vector register) |
| `active_lanes` | `Index` | Number of participating lanes |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `vec` | `VRegType` | Gathered vector |

---

#### `pto.vgather2_bc(buf: PtrType, offsets: Index, mask: MaskType) -> VRegType`

**Description**: Indexed gather with mask. Masked-off lanes are zero-filled.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `buf` | `PtrType` (UB) | Source buffer |
| `offsets` | `Index` | Per-lane element offsets (vector register) |
| `mask` | `MaskType` | Mask gating lane participation |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `vec` | `VRegType` | Gathered vector |

---

#### `pto.vgatherb(buf: PtrType, offsets: Index) -> VRegType`

**Description**: Byte-granularity gather load.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `buf` | `PtrType` | Source buffer |
| `offsets` | `Index` | Byte offsets |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `vec` | `VRegType` | Gathered vector |

---

#### `pto.vsldb(tile[row, col], offset: Index, mask: MaskType) -> VRegType`
#### `pto.vsldb(tile[pos], offset: Index, mask: MaskType) -> VRegType`
#### `pto.vsldb(buf: PtrType, offset: Index, mask: MaskType) -> VRegType`

**Description**: Block-strided load. The `offset` encodes a packed stride/control word, not a plain byte displacement. Masked-off blocks are zeroed.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `offset` | `Index` | Packed stride/control word |
| `mask` | `MaskType` | Mask controlling which blocks participate |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `vec` | `VRegType` | Block-strided vector |

## 7.4 Vector stores (simd)

Vector stores write `vreg` contents back to UB tiles. Like loads, they support tile-index syntax.

#### `pto.vsts(vec: VRegType, tile[row, col:], mask: MaskType, dist: VStoreDist | None = None) -> None`
#### `pto.vsts(vec: VRegType, tile[start:], mask: MaskType, dist: VStoreDist | None = None) -> None`
#### `pto.vsts(vec: VRegType, buf: PtrType, offset: Index, mask: MaskType, dist: VStoreDist | None = None) -> None`

**Description**: Stateless vector store to UB. The mask gates which lanes are written.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Vector to store |
| `tile[row, col:]` | Tile index | 2D destination (vector-width range) |
| `tile[start:]` | Tile index | 1D destination (vector-width range) |
| `buf` | `PtrType` (UB) | Destination buffer (pointer form) |
| `offset` | `Index` | Byte offset (pointer form) |
| `mask` | `MaskType` | Predicate mask gating writes |
| `dist` | `VStoreDist` or `None` | Optional store distribution: `NORM_B32` (default), `PK_B16`/`PK_B32`/`PK_B64`, `ONE_POINT_B8`/`ONE_POINT_B16`/`ONE_POINT_B32` |

**Returns**: None (side-effect operation).

---

#### `pto.psts(mask: MaskType, buf: PtrType, offset: Index) -> None`

**Description**: Predicate store. Writes the packed predicate payload of `mask` to UB memory.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `mask` | `MaskType` | Predicate payload to store |
| `buf` | `PtrType` (UB) | Destination buffer |
| `offset` | `Index` | Byte offset |

**Returns**: None (side-effect operation).

---

#### `pto.vstsx2(low: VRegType, high: VRegType, tile[row, col:], dist: InterleaveDist, mask: MaskType) -> None`
#### `pto.vstsx2(low: VRegType, high: VRegType, tile[start:], dist: InterleaveDist, mask: MaskType) -> None`
#### `pto.vstsx2(low: VRegType, high: VRegType, buf: PtrType, offset: Index, dist: InterleaveDist, mask: MaskType) -> None`

**Description**: Dual interleaving store (SoA → AoS). Interleaves two vectors into one destination.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `low` | `VRegType` | First vector (even elements) |
| `high` | `VRegType` | Second vector (odd elements) |
| `tile[row, col:]` | Tile index | 2D destination (vector-width range) |
| `tile[start:]` | Tile index | 1D destination (vector-width range) |
| `buf` | `PtrType` (UB) | Destination buffer (pointer form) |
| `offset` | `Index` | Byte offset (pointer form) |
| `dist` | `InterleaveDist` | `INTLV` |
| `mask` | `MaskType` | Predicate mask |

**Returns**: None (side-effect operation).

---

#### `pto.vsst(scalar: ScalarType, tile[row, col:], mask: MaskType) -> None`
#### `pto.vsst(scalar: ScalarType, tile[start:], mask: MaskType) -> None`
#### `pto.vsst(scalar: ScalarType, buf: PtrType, offset: Index, mask: MaskType) -> None`

**Description**: Scalar broadcast store. Stores a scalar value replicated to all lanes under `mask`.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `scalar` | `ScalarType` | Scalar value to broadcast |
| `tile[row, col:]` | Tile index | 2D destination (vector-width range) |
| `tile[start:]` | Tile index | 1D destination (vector-width range) |
| `buf` | `PtrType` (UB) | Destination buffer (pointer form) |
| `offset` | `Index` | Byte offset (pointer form) |
| `mask` | `MaskType` | Predicate mask |

**Returns**: None (side-effect operation).

---

#### `pto.vsstb(scalar: ScalarType, tile[row, col:], mask: MaskType) -> None`
#### `pto.vsstb(scalar: ScalarType, tile[start:], mask: MaskType) -> None`
#### `pto.vsstb(scalar: ScalarType, buf: PtrType, offset: Index, mask: MaskType) -> None`

**Description**: Enhanced scalar broadcast store. Same semantics as `vsst`.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `scalar` | `ScalarType` | Scalar value to broadcast |
| `tile[row, col:]` | Tile index | 2D destination (vector-width range) |
| `tile[start:]` | Tile index | 1D destination (vector-width range) |
| `buf` | `PtrType` (UB) | Destination buffer (pointer form) |
| `offset` | `Index` | Byte offset (pointer form) |
| `mask` | `MaskType` | Predicate mask |

**Returns**: None (side-effect operation).

---

#### `pto.vsta(align: AlignType, tile[row, col:]) -> None`
#### `pto.vsta(align: AlignType, tile[start:]) -> None`
#### `pto.vsta(align: AlignType, buf: PtrType, offset: Index) -> None`

**Description**: Flush alignment state to memory. Commits buffered tail bytes from an unaligned store stream. Consumes the alignment state.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `align` | `AlignType` | Pending store-alignment state |
| `tile[row, col:]` | Tile index | 2D destination (vector-width range) |
| `tile[start:]` | Tile index | 1D destination (vector-width range) |
| `buf` | `PtrType` (UB) | Destination buffer (pointer form) |
| `offset` | `Index` | Byte offset (pointer form) |

**Returns**: None (side-effect operation).

---

#### `pto.vstas(align: AlignType, tile[row, col:], offset: Index) -> None`
#### `pto.vstas(align: AlignType, tile[start:], offset: Index) -> None`
#### `pto.vstas(align: AlignType, buf: PtrType, offset: Index) -> None`

**Description**: Scalar-register-offset form of alignment-state flush. Same buffered-tail semantics as `vsta` with an explicit scalar offset.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `align` | `AlignType` | Pending store-alignment state |
| `tile[row, col:]` | Tile index | 2D destination (vector-width range) |
| `tile[start:]` | Tile index | 1D destination (vector-width range) |
| `buf` | `PtrType` (UB) | Destination buffer (pointer form) |
| `offset` | `Index` | Byte offset (all forms) |

**Returns**: None (side-effect operation).

---

#### `pto.vstar(align: AlignType, tile[row, col:]) -> None`
#### `pto.vstar(align: AlignType, tile[start:]) -> None`
#### `pto.vstar(align: AlignType, buf: PtrType) -> None`

**Description**: Register-update form of alignment-state flush. Consumes the implicit update state from the matching store stream.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `align` | `AlignType` | Pending store-alignment state |

| `tile[row, col:]` | Tile index | 2D destination (vector-width range) |
| `tile[start:]` | Tile index | 1D destination (vector-width range) |
| `buf` | `PtrType` (UB) | Destination buffer (pointer form) |

**Returns**: None (side-effect operation).

---

#### `pto.vscatter(vec: VRegType, buf: PtrType, offsets: Index, active_lanes: Index) -> None`

**Description**: Indexed scatter to UB. Stores vector lanes to irregular locations using per-lane offsets.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Source vector to scatter |
| `buf` | `PtrType` (UB) | Destination buffer |
| `offsets` | `Index` | Per-lane element offsets (vector register) |
| `active_lanes` | `Index` | Number of participating lanes |

**Returns**: None (side-effect operation).

---

### Stateful store family

For streaming unaligned stores with explicit alignment threading:

#### `pto.vstu(align_in: AlignType, base_in: PtrType, vec: VRegType, buf: PtrType, mode: Index) -> (AlignType, PtrType)`

**Description**: Unaligned store with explicit threaded alignment/base state. Returns updated state for the next store in the stream.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `align_in` | `AlignType` | Incoming store-alignment state |
| `base_in` | `PtrType` | Current stream base pointer |
| `vec` | `VRegType` | Vector to store |
| `buf` | `PtrType` (UB) | Destination buffer |
| `mode` | `Index` | Post-update mode |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `align_out` | `AlignType` | Updated buffered-tail state |
| `base_out` | `PtrType` | Post-update base pointer |

---

#### `pto.vstus(align_in: AlignType, base_in: PtrType, vec: VRegType, buf: PtrType, offset: Index) -> (AlignType, PtrType)`

**Description**: Scalar-offset unaligned store. Same roles as `vstu` with explicit scalar displacement.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `align_in` | `AlignType` | Incoming store-alignment state |
| `base_in` | `PtrType` | Current stream base pointer |
| `vec` | `VRegType` | Vector to store |
| `buf` | `PtrType` (UB) | Destination buffer |
| `offset` | `Index` | Scalar displacement |
| `mode` | `Index` | Post-update mode |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `align_out` | `AlignType` | Updated buffered-tail state |
| `base_out` | `PtrType` | Post-update base pointer |

---

#### `pto.vstur(align_in: AlignType, vec: VRegType, buf: PtrType, mode: PostUpdateMode = PostUpdateMode.NO_POST_UPDATE) -> AlignType`

**Description**: Register-update unaligned store. Updates only residual alignment state without base pointer update.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `align_in` | `AlignType` | Incoming store-alignment state |
| `vec` | `VRegType` | Vector to store |
| `buf` | `PtrType` (UB) | Destination buffer |
| `mode` | `PostUpdateMode` | `NO_POST_UPDATE` (default) or `POST_UPDATE` |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `align_out` | `AlignType` | Updated buffered-tail state |

---

#### `pto.pstu(align_in: AlignType, mask: MaskType, buf: PtrType) -> (AlignType, PtrType)`

**Description**: Predicate unaligned store with alignment state threading.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `align_in` | `AlignType` | Incoming store-alignment state |
| `mask` | `MaskType` | Predicate mask to store |
| `buf` | `PtrType` (UB) | Destination buffer |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `align_out` | `AlignType` | Updated alignment state |
| `base_out` | `PtrType` | Post-update base pointer |

---

**Unaligned store stream pattern** — prime, thread, flush:

```python
align, base = pto.vstu(align0, base0, vec0, ub_ptr, mode)
align, base = pto.vstu(align, base, vec1, ub_ptr, mode)
pto.vsta(align, ub_ptr, flush_offset)
```

### Distribution enums reference

| Enum | Values | Used with |
|------|--------|-----------|
| `VLoadDist` | `NORM`, `UNPK_B8`, `UNPK_B16`, `UNPK_B32`, `BRC_B8`, `BRC_B16`, `BRC_B32`, `US_B8`, `US_B16`, `DS_B8`, `DS_B16` | `vlds` |
| `VStoreDist` | `NORM_B8`, `NORM_B16`, `NORM_B32`, `ONE_POINT_B8`, `ONE_POINT_B16`, `ONE_POINT_B32`, `PK_B16`, `PK_B32`, `PK_B64`, `PK4_B32`, `MRG4CHN_B8`, `MRG2CHN_B8`, `MRG2CHN_B16` | `vsts` |
| `DeinterleaveDist` | `DINTLV`, `BDINTLV` | `vldsx2` |
| `InterleaveDist` | `INTLV` | `vstsx2` |
| `StrideMode` | `S3_B16`, `S4_B64`, `S8_B32`, `S2_B64` | `vsld` |
| `PostUpdateMode` | `NO_POST_UPDATE`, `POST_UPDATE` | `vstur` |

## 7.5 Cube data movement (cube)

Inside `@pto.cube`, data flows through a hierarchy of private buffers: GM → L1 (cbuf) → L0A/L0B (operand buffers) → L0C (accumulator) → UB or back to GM.

### Staging: GM → L1 and L1 → UB

#### `pto.mte_gm_l1(src: PtrType, dst: PtrType, len_burst: int, *, nburst: tuple[int, int, int] = (1, 0, 0), loops: list[tuple[int, int, int]] | None = None) -> None`

**Description**: Structured GM-to-L1 (cbuf) data movement.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `PtrType` (GM) | Global Memory source pointer |
| `dst` | `PtrType` (L1) | L1 (cbuf) destination pointer |
| `len_burst` | `int` | Burst length in bytes |
| `nburst` | `tuple[int, int, int]` | `(count, src_stride, dst_stride)` |
| `loops` | `list[tuple[int, int, int]]` or `None` | Optional nested loop parameters |

**Returns**: None (side-effect operation).

---

#### `pto.mte_gm_l1_frac(src: PtrType, dst: PtrType, mode: FractalMode, *, shape: tuple[int, int], src_layout: tuple[int, int], dst_group: tuple[int, int, int, int], ctrl: tuple[int, bool]) -> None`

**Description**: Fractal GM-to-L1 load for specialized layouts (`ND2NZ`, `DN2NZ`).

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `PtrType` (GM) | Global Memory source pointer |
| `dst` | `PtrType` (L1) | L1 destination pointer |
| `mode` | `FractalMode` | `ND2NZ` or `DN2NZ` |
| `shape` | `tuple[int, int]` | `(n_value, d_value)` |
| `src_layout` | `tuple[int, int]` | `(inner_stride, outer_stride)` |
| `dst_group` | `tuple[int, int, int, int]` | `(group_count, loop2_stride, loop3_stride, loop4_stride)` |
| `ctrl` | `tuple[int, bool]` | `(l2_cache_ctrl, smallc0_en)` |

**Returns**: None (side-effect operation).

---

#### `pto.mte_l1_ub(src: PtrType, dst: PtrType, len_burst: int, *, nburst: tuple[int, int, int] = (1, 0, 0), loops: list[tuple[int, int, int]] | None = None) -> None`

**Description**: Structured L1 (cbuf) to UB data movement.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `PtrType` (L1) | L1 source pointer |
| `dst` | `PtrType` (UB) | UB destination pointer |
| `len_burst` | `int` | Burst length in bytes |
| `nburst` | `tuple[int, int, int]` | `(count, src_stride, dst_stride)` |
| `loops` | `list[tuple[int, int, int]]` or `None` | Optional nested loop parameters |

**Returns**: None (side-effect operation).

---

### Operand loading: L1 → L0A / L0B

#### `pto.mte_l1_l0a(src: PtrType, dst: PtrType, m: int, k: int) -> None`

**Description**: Structured L1-to-L0A (left-operand buffer) load.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `PtrType` (L1) | L1 source pointer |
| `dst` | `PtrType` (L0A) | L0A destination pointer |
| `m` | `int` | M dimension size |
| `k` | `int` | K dimension size |

**Returns**: None (side-effect operation).

---

#### `pto.mte_l1_l0b(src: PtrType, dst: PtrType, k: int, n: int, *, transpose: bool = False) -> None`

**Description**: Structured L1-to-L0B (right-operand buffer) load.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `PtrType` (L1) | L1 source pointer |
| `dst` | `PtrType` (L0B) | L0B destination pointer |
| `k` | `int` | K dimension size |
| `n` | `int` | N dimension size |
| `transpose` | `bool` | Whether to load in transposed order |

**Returns**: None (side-effect operation).

---

#### `pto.mte_l1_l0a_mx(src: PtrType, dst: PtrType, m: int, k: int) -> None`
#### `pto.mte_l1_l0b_mx(src: PtrType, dst: PtrType, k: int, n: int) -> None`

**Description**: MX-mode variants of `mte_l1_l0a` and `mte_l1_l0b` for MX-capable dtypes. Parameters same as their non-MX counterparts.

---

### Bias loading

#### `pto.mte_l1_bias(src: PtrType, dst: PtrType, len_burst: int, *, nburst: tuple[int, int, int] = (1, 0, 0)) -> None`

**Description**: Structured L1 (cbuf) to bias table load.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `PtrType` (L1) | L1 source pointer |
| `dst` | `PtrType` (BIAS) | Bias table destination pointer |
| `len_burst` | `int` | Burst length in bytes |
| `nburst` | `tuple[int, int, int]` | `(count, src_gap, dst_gap)` |

**Returns**: None (side-effect operation).

---

### Accumulator writeback: L0C → L1 / GM / UB

#### `pto.mte_l0c_l1(src: PtrType, dst: PtrType, m: int, n: int, src_stride: int, dst_stride: int, *, mode: FractalMode = FractalMode.NZ2ND, loop0_src_stride: int | None = None, split: int | None = None, loop3: tuple[int, int, int] | None = None) -> None`

**Description**: Structured L0C (acc) to L1 (cbuf) writeback.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `PtrType` (L0C) | L0C accumulator source pointer |
| `dst` | `PtrType` (L1) | L1 destination pointer |
| `m` | `int` | M dimension size |
| `n` | `int` | N dimension size |
| `src_stride` | `int` | Source stride |
| `dst_stride` | `int` | Destination stride |
| `mode` | `FractalMode` | `NZ2ND` (default), `NZ2DN`, or `NZ2NZ` |

**Returns**: None (side-effect operation).

---

#### `pto.mte_l0c_gm(src: PtrType, dst: PtrType, m: int, n: int, src_stride: int, dst_stride: int, *, sid: int = 0, l2_cache_ctrl: int = 0, mode: FractalMode = FractalMode.NZ2ND, loop0_src_stride: int | None = None, split: int | None = None, loop3: tuple[int, int, int] | None = None) -> None`

**Description**: Structured L0C (acc) to GM writeback.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `PtrType` (L0C) | L0C accumulator source pointer |
| `dst` | `PtrType` (gm) | GM destination pointer |
| `m` | `int` | M dimension size |
| `n` | `int` | N dimension size |
| `src_stride` | `int` | Source stride |
| `dst_stride` | `int` | Destination stride |
| `sid` | `int` | Stream ID (default 0) |
| `l2_cache_ctrl` | `int` | L2 cache control (default 0) |
| `mode` | `FractalMode` | `NZ2ND` (default), `NZ2DN`, or `NZ2NZ` |
| `loop0_src_stride` | `int` or `None` | Loop level 0 source stride |
| `split` | `int` or `None` | Split parameter |
| `loop3` | `tuple[int, int, int]` or `None` | Loop level 3 parameters |

**Returns**: None (side-effect operation).

---

#### `pto.mte_l0c_ub(src: PtrType, dst: PtrType, m: int, n: int, src_stride: int, dst_stride: int, *, dual_dst_mode: int = 0, sub_blockid: int = 0, mode: FractalMode = FractalMode.NZ2ND, loop0_src_stride: int | None = None, channel_split_en: int | None = None, loop3: tuple[int, int, int] | None = None) -> None`

**Description**: Structured L0C (acc) directly to UB. This is the most common writeback path for cube kernels that feed results into subsequent processing.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `PtrType` (L0C) | L0C accumulator source pointer |
| `dst` | `PtrType` (ub) | UB destination pointer |
| `m` | `int` | M dimension size |
| `n` | `int` | N dimension size |
| `src_stride` | `int` | Source stride |
| `dst_stride` | `int` | Destination stride |
| `dual_dst_mode` | `int` | Dual destination mode (default 0) |
| `sub_blockid` | `int` | Sub-block ID (default 0) |
| `mode` | `FractalMode` | `NZ2ND` (default), `NZ2DN`, or `NZ2NZ` |
| `loop0_src_stride` | `int` or `None` | Loop level 0 source stride |
| `channel_split_en` | `int` or `None` | Channel split enable (required for `NZ2NZ` mode) |
| `loop3` | `tuple[int, int, int]` or `None` | Loop level 3 parameters |

**Returns**: None (side-effect operation).

---

### Cube data movement quick reference

| Data Flow | Operation | Src Space | Dst Space |
|-----------|-----------|-----------|-----------|
| GM → L1 | `mte_gm_l1` | gm | l1 |
| GM → L1 (fractal) | `mte_gm_l1_frac` | gm | l1 |
| L1 → UB | `mte_l1_ub` | l1 | ub |
| L1 → L0A | `mte_l1_l0a` | l1 | l0a |
| L1 → L0B | `mte_l1_l0b` | l1 | l0b |
| L1 → L0A (MX) | `mte_l1_l0a_mx` | l1 | l0a |
| L1 → L0B (MX) | `mte_l1_l0b_mx` | l1 | l0b |
| L1 → Bias | `mte_l1_bias` | l1 | bt |
| L0C → L1 | `mte_l0c_l1` | l0c | l1 |
| L0C → GM | `mte_l0c_gm` | l0c | gm |
| L0C → UB | `mte_l0c_ub` | l0c | ub |

### Typical cube dataflow in a matmul

A full cube matmul (`@pto.cube`) follows this dataflow pattern:

```python
@pto.cube
def qk_matmul(q_tile, k_tile, q_l0a, k_l0b, s_acc, s_tile):
    m = q_tile.valid_shape[0]
    k = q_tile.valid_shape[1]
    n = k_tile.valid_shape[0]

    pto.mte_l1_l0a(q_tile, q_l0a, m, k)          # UB tile → L0A
    pto.mte_l1_l0b(k_tile, k_l0b, k, n, transpose=True)  # UB tile → L0B
    pto.mad(q_l0a, k_l0b, s_acc)                # L0A × L0B → L0C
    pto.mte_l0c_ub(s_acc, s_tile, m, n)       # L0C → UB tile
```

The `mte_l1_l0a`/`mte_l1_l0b` operations take UB `Tile` references directly (not raw pointers) — the tile-to-cube-local transfer is implicit. `mad` performs the matrix multiply. `mte_l0c_ub` writes the result back to a UB tile.

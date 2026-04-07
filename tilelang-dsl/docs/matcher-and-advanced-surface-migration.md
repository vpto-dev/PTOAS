# TileLang DSL Matcher And Advanced-Surface Migration

## Scope

This document explains how to move from the original v1 core contract
(`add-tilelang-dsl-core-foundation` +
`add-tilelang-dsl-authoring-vpto-lowering`) to the matcher and
advanced-surface capability implemented by
`extend-tilelang-dsl-matcher-and-advanced-surface`.

It focuses on:
- matcher-driven kernel selection
- implicit vecscope inference
- raw pointer / low-level DMA authoring
- advanced vector-family coverage that is implemented today
- the remaining deferred boundary

## What Changed

The original v1 core profile assumed:
- one monomorphic `dtypes` signature
- no matcher registry or selection API
- explicit `pto.strict_vecscope` for vector code
- no raw-pointer or low-level DMA authoring surface
- no advanced vector-family lowering beyond the fixed elementwise set

The current package now adds:
- `KernelRegistry`
- `pto.select_kernel(...)`
- multi-signature `dtypes`
- `AnyFloat`, `AnyInt`, `AnyType`, `AnyMask`
- `TypeVar(...)`
- `constraints=[...]`
- `priority=<int>`
- implicit vecscope inference in `advanced=True` kernels
- `ptr(...)` / `PointerType`
- `castptr`, `addptr`
- low-level DMA config/copy surface
- compare/select, predicate movement, carry, and rearrangement families

## Matcher Migration

### Before

The original v1 contract only supported one concrete signature:

```python
@pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f32)])
def kernel(inp: pto.TensorView, out: pto.Tile):
    return None
```

### After

You can now register multiple polymorphic descriptors and let the matcher pick
the concrete specialization:

```python
@pto.vkernel(
    op="eltwise",
    dtypes=[
        (pto.AnyFloat, pto.AnyFloat),
        (pto.AnyInt, pto.AnyInt),
    ],
    constraints=[lambda attrs: attrs.get("enabled", True)],
    priority=10,
)
def kernel(inp: pto.TensorView, out: pto.Tile):
    return None

selected = pto.select_kernel(
    "a5",
    "eltwise",
    (pto.f32, pto.f32),
    context_attrs={"enabled": True},
)
```

Matcher rules in the implemented package:
- matching is deterministic
- selection order is `target -> op -> dtypes -> constraints -> priority`
- highest-priority ties raise an explicit error
- `TypeVar` only binds within one signature

## Vecscope Migration

### Before

Vector code needed an explicit `pto.strict_vecscope` boundary:

```python
with pto.strict_vecscope(tile, tile, 0, 256, 64) as (src, dst, lb, ub, step):
    for lane in range(lb, ub, step):
        mask = pto.make_mask(pto.f32, pto.PAT.ALL)
        vec = pto.vlds(src, lane)
        pto.vsts(vec, dst, lane, mask)
```

### After

In `advanced=True` kernels, the frontend now infers `pto.vecscope` for
contiguous vector-active regions:

```python
@pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f32)], advanced=True)
def kernel(src: pto.Tile, dst: pto.Tile):
    mask = pto.make_mask(pto.f32, pto.PAT.ALL)
    vec = pto.vlds(src[0, 0:])
    pto.vsts(vec, dst[0, 0:], mask)
```

Inference boundaries in the implemented package:
- scalar statements cut inference
- `if` / `for` structure is respected
- sync and DMA statements cut inference
- explicit `pto.strict_vecscope` remains a hard boundary

Use `pto.strict_vecscope` when you need a deterministic region ABI or do not
want inference to merge adjacent vector chains.

## Pointer And DMA Migration

### New Pointer Surface

The package now exposes:
- `pto.ptr(dtype, memory_space)`
- pointer-typed parameters such as `pto.ptr(pto.f32, pto.MemorySpace.UB)`
- `pto.castptr(...)`
- `pto.addptr(...)`

Example:

```python
@pto.vkernel(op="copy", dtypes=[(pto.f32, pto.i64)], advanced=True)
def kernel(dst: pto.ptr(pto.f32, pto.MemorySpace.UB), addr: pto.i64):
    src = pto.castptr(addr, pto.ptr(pto.f32, pto.MemorySpace.UB))
    next_src = pto.addptr(src, 64)
    mask = pto.make_mask(pto.f32, pto.PAT.ALL)
    vec = pto.vlds(src, 0)
    pto.vsts(vec, next_src, 0, mask)
```

### New Low-Level DMA Surface

The package now lowers:
- `set_loop2_stride_outtoub`
- `set_loop1_stride_outtoub`
- `set_loop_size_outtoub`
- `set_loop2_stride_ubtoout`
- `set_loop1_stride_ubtoout`
- `set_loop_size_ubtoout`
- `copy_gm_to_ubuf`
- `copy_ubuf_to_gm`
- `copy_ubuf_to_ubuf`

High-level `dma_load` / `dma_store` remain the preferred default. Use the
low-level surface only when you need manual DMA programming.

## Advanced Vector Families

The currently implemented advanced-family groups are:
- compare/select:
  `vcmp`, `vcmps`, `vsel`, `vselr`, `vselrv2`
- predicate movement:
  `pnot`, `psel`, `ppack`, `punpack`
- carry family:
  `vaddc`, `vsubc`, `vaddcs`, `vsubcs`
- rearrangement:
  `vintlv`, `vdintlv`, `vintlvv2`, `vdintlvv2`

These lower directly to authoring-form VPTO and are covered by
`tilelang-dsl/tests/test_tilelang_dsl_v1.py`.

## Still Deferred

The following boundary remains intentionally deferred:
- reduction family authoring

Reason:
- the current repo does not expose a public authoring-form VPTO reduction op
  that TileLang DSL can target directly
- existing reduction logic lives in other lowering paths such as OpLib / EmitC
  and cannot be treated as the public TileLang DSL authoring contract

Current package behavior:
- reduction-family surface remains an explicit frontend reject
- no extra helper IR is introduced to fake reduction support

## Recommended Reading Order

For the current package contract, read in this order:
1. `tilelang-dsl/docs/v1-surface.md`
2. `tilelang-dsl/docs/v1-lowering.md`
3. `tilelang-dsl/docs/matcher-and-advanced-surface-migration.md`
4. `docs/tilelang-dsl-guide.md`

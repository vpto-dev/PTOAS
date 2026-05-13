# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Lazy MLIR type descriptors and eager type constructors.

Type descriptors (``_DType`` subclasses) can be created *before* any MLIR
Context exists – they only resolve to concrete ``mlir.ir.Type`` objects when
``_resolve()`` is called inside an active context.  This lets users write::

    def softmax(arg0: pto.ptr(pto.float32, "GM"), ...):
        ...

where the annotation is evaluated at *import* time (no active context), and
the actual type is materialised later by the ``@pto.to_ir`` decorator.
"""

from ._bootstrap import make_context  # ensure MLIR is on sys.path

from mlir.dialects import pto as _pto
from mlir.ir import (
    F16Type,
    F32Type,
    IndexType,
    IntegerType,
    ShapedType,
    Type,
)

# ── Address-space name → AddressSpace enum ───────────────────────────────────
_ADDR_SPACE = {
    "ub":  _pto.AddressSpace.VEC,  # UB == unified buffer == VEC in PTO
    "gm":  _pto.AddressSpace.GM,
    "vec": _pto.AddressSpace.VEC,
    "mat": _pto.AddressSpace.MAT,
    "GM":  _pto.AddressSpace.GM,
    "UB":  _pto.AddressSpace.VEC,
    "VEC": _pto.AddressSpace.VEC,
    "MAT": _pto.AddressSpace.MAT,
}


# ── Lazy type descriptor base ─────────────────────────────────────────────────

class _DType:
    """Deferred MLIR type: only resolves inside an active MLIR context."""

    def __init__(self, factory):
        self._factory = factory

    def resolve(self) -> Type:
        return self._factory()

    def __repr__(self):
        return f"<pto.dtype {self._factory}>"


class _PtrDescriptor(_DType):
    def __init__(self, elem, space: str):
        self._elem = elem
        self._space = space

    def resolve(self) -> Type:
        elem = _resolve(self._elem)
        space_enum = _ADDR_SPACE.get(self._space)
        if space_enum is None:
            raise ValueError(
                f"Unknown address space '{self._space}'; "
                f"known: {list(_ADDR_SPACE)}"
            )
        space_attr = _pto.AddressSpaceAttr.get(space_enum)
        return _pto.PtrType.get(elem, memory_space=space_attr)

    def __repr__(self):
        return f"<pto.ptr {self._elem} {self._space}>"


class _VRegDescriptor(_DType):
    def __init__(self, lanes: int, elem):
        self._lanes = lanes
        self._elem = elem

    def resolve(self) -> Type:
        elem = _resolve(self._elem)
        return Type.parse(f"!pto.vreg<{self._lanes}x{elem}>")

    def __repr__(self):
        return f"<pto.vreg {self._lanes}x{self._elem}>"


def _resolve(dtype) -> Type:
    """Coerce a ``_DType`` descriptor or a concrete ``mlir.ir.Type`` to a Type."""
    if isinstance(dtype, _DType):
        return dtype.resolve()
    return dtype  # already an mlir.ir.Type


# ── Scalar dtype singletons ───────────────────────────────────────────────────

float32 = _DType(F32Type.get)
float16 = _DType(F16Type.get)
int8    = _DType(lambda: IntegerType.get_signless(8))
int16   = _DType(lambda: IntegerType.get_signless(16))
int32   = _DType(lambda: IntegerType.get_signless(32))
int64   = _DType(lambda: IntegerType.get_signless(64))
index   = _DType(IndexType.get)


# ── Type constructor functions ────────────────────────────────────────────────

def ptr(elem, space: str = "ub") -> _PtrDescriptor:
    """Return a lazy descriptor for ``!pto.ptr<elem, space>``."""
    return _PtrDescriptor(elem, space)


def vreg_type(lanes: int, elem) -> _VRegDescriptor:
    """Return a lazy descriptor for ``!pto.vreg<lanesxelem>``."""
    return _VRegDescriptor(lanes, elem)


def mask_type(bits: str = "b32") -> Type:
    """Return ``!pto.mask<bits>`` (b8 | b16 | b32).  Requires active context."""
    return Type.parse(f"!pto.mask<{bits}>")


def tile_buf_type(shape, dtype, valid_shape, *,
                  blayout: str = "RowMajor",
                  address_space: str = "ub",
                  slayout: str = "NoneBox",
                  fractal_size: int = 512,
                  pad: str = "Null") -> Type:
    """
    Construct a ``!pto.tile_buf<…>`` type via the Python bindings.

    ``valid_shape`` entries may be ``-1`` for dynamic (``?``) dimensions.
    ``blayout="ColMajor"`` prints as ``blayout=col_major``.

    Requires an active MLIR context.
    """
    elem = _resolve(dtype)
    space_enum = _ADDR_SPACE.get(address_space)
    if space_enum is None:
        raise ValueError(
            f"Unknown address_space '{address_space}'; known: {list(_ADDR_SPACE)}"
        )
    space_attr = _pto.AddressSpaceAttr.get(space_enum)
    cfg = _pto.TileBufConfigAttr.get(
        _pto.BLayoutAttr.get(getattr(_pto.BLayout, blayout)),
        _pto.SLayoutAttr.get(getattr(_pto.SLayout, slayout)),
        fractal_size,
        _pto.PadValueAttr.get(getattr(_pto.PadValue, pad)),
    )
    return _pto.TileBufType.get(shape, elem, space_attr, valid_shape, cfg)


def tensor_view_type(rank: int, elem) -> Type:
    """``!pto.tensor_view<?x…xelem>`` with *rank* all-dynamic dims."""
    return _pto.TensorViewType.get(rank, _resolve(elem))


def part_tensor_view_type(rank: int, elem) -> Type:
    """``!pto.partition_tensor_view<?x…xelem>`` with *rank* all-dynamic dims."""
    kDynamic = ShapedType.get_dynamic_size()
    return _pto.PartitionTensorViewType.get([kDynamic] * rank, _resolve(elem))


__all__ = [
    "_DType", "_resolve",
    "float32", "float16", "int8", "int16", "int32", "int64", "index",
    "ptr", "vreg_type", "mask_type",
    "tile_buf_type", "tensor_view_type", "part_tensor_view_type",
]

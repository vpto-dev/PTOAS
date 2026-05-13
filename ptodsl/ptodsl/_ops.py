# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
PTO operation wrappers.

Every function in this module emits one or more MLIR operations at the
active insertion point and returns the primary SSA result(s).

Design rules:
- Vector math ops infer the result type from the first operand's type.
- ``vlds`` / ``vbrc_load`` still require an explicit ``vreg_type`` argument
  because the result type cannot be inferred from the pointer alone.
- ``make_tensor_view`` infers the TensorViewType from ``len(shape)`` and the
  pointer's element type.
- ``partition_view`` infers the PartitionTensorViewType from the source type.
"""

from ._bootstrap import make_context  # noqa: F401 – ensure MLIR on sys.path
from ._types import _resolve, mask_type, part_tensor_view_type, tensor_view_type

from mlir.dialects import arith, pto as _pto
from mlir.ir import (
    Attribute,
    IndexType,
    IntegerType,
    ShapedType,
    StringAttr,
)

# Pipe name shorthands → canonical PIPE_* names
_PIPE_ALIASES = {
    "MTE1": "PIPE_MTE1",
    "MTE2": "PIPE_MTE2",
    "MTE3": "PIPE_MTE3",
    "MTE4": "PIPE_MTE4",
    "V":    "PIPE_V",
    "M":    "PIPE_M",
    "S":    "PIPE_S",
    "ALL":  "PIPE_ALL",
}


def _pipe_attr(name: str):
    canonical = _PIPE_ALIASES.get(name, name)
    if not canonical.startswith("PIPE_"):
        canonical = "PIPE_" + canonical
    return _pto.PipeAttr.get(getattr(_pto.PIPE, canonical))


def _event_attr(event_id: int):
    return getattr(_pto, f"EVENT_ID{event_id}")


# ── Constants ────────────────────────────────────────────────────────────────

def const(value: int, *, dtype=None):
    """
    Emit an ``arith.constant``.

    ``dtype`` is a ``_DType`` descriptor or a concrete ``mlir.ir.Type``.
    Defaults to ``index`` when omitted.
    """
    from ._types import index as _idx_dtype
    mlir_type = _resolve(dtype) if dtype is not None else _resolve(_idx_dtype)
    return arith.ConstantOp(mlir_type, value).result


# ── Pointer ops ───────────────────────────────────────────────────────────────

def castptr(int_addr, result_ptr_type):
    """``pto.castptr`` – cast an integer address to a typed PTO pointer."""
    return _pto.CastPtrOp(_resolve(result_ptr_type), int_addr).result


def addptr(base_ptr, index_offset):
    """``pto.addptr`` – advance a pointer by an index offset."""
    return _pto.AddPtrOp(base_ptr, index_offset).result


# ── Vector load / store ───────────────────────────────────────────────────────

def vlds(src_ptr, offset, result_vreg_type):
    """``pto.vlds`` – vector load from *src_ptr* at *offset*."""
    return _pto.VldsOp(_resolve(result_vreg_type), src_ptr, offset).result


def vbrc_load(src_ptr, offset, result_vreg_type):
    """``pto.vlds {dist="BRC_B32"}`` – broadcast a scalar into all lanes."""
    return _pto.VldsOp(_resolve(result_vreg_type), src_ptr, offset,
                       dist="BRC_B32").result


def vsts(val, dst_ptr, offset, mask):
    """``pto.vsts`` – vector store."""
    _pto.VstsOp(val, dst_ptr, offset, mask)


def vsts_1pt(val, dst_ptr, offset, mask):
    """``pto.vsts {dist="1PT_B32"}`` – store only the lowest lane."""
    _pto.VstsOp(val, dst_ptr, offset, mask, dist="1PT_B32")


# ── Mask / predicate ops ──────────────────────────────────────────────────────

def plt_b32(scalar):
    """
    ``pto.plt_b32`` – predicate-load from a 32-bit scalar.

    Returns ``(mask_value, scalar_out)``.  ``scalar_out`` is often unused
    and can be discarded with ``_``.
    """
    plt_op = _pto.PltB32Op(mask_type("b32"), IntegerType.get_signless(32), scalar)
    return plt_op.mask, plt_op.scalar_out


def pset_b32(pattern: str):
    """``pto.pset_b32 "PATTERN"`` → ``!pto.mask<b32>``."""
    return _pto.PsetB32Op(mask_type("b32"), pattern).result


# ── Vector math (result type inferred from first operand) ─────────────────────

def vadd(lhs, rhs, mask, result_type=None):
    """``pto.vadd`` – element-wise add."""
    rt = result_type if result_type is not None else lhs.type
    return _pto.VaddOp(_resolve(rt), lhs, rhs, mask).result


def vmul(lhs, rhs, mask):
    """``pto.vmul`` – element-wise multiply."""
    return _pto.VmulOp(lhs.type, lhs, rhs, mask).result


def vmax(lhs, rhs, mask):
    """``pto.vmax`` – element-wise maximum."""
    return _pto.VmaxOp(lhs.type, lhs, rhs, mask).result


def vdiv(lhs, rhs, mask):
    """``pto.vdiv`` – element-wise divide."""
    return _pto.VdivOp(lhs.type, lhs, rhs, mask).result


def vcmax(v, mask):
    """``pto.vcmax`` – cross-lane maximum reduction."""
    return _pto.VcmaxOp(v.type, v, mask).result


def vcadd(v, mask):
    """``pto.vcadd`` – cross-lane add (sum reduction)."""
    return _pto.VcaddOp(v.type, v, mask).result


def vdup(v, mask, *, position=None):
    """``pto.vdup`` – duplicate a lane value into all lanes.

    Pass ``position="LOWEST"`` to broadcast the lowest (lane-0) element.
    """
    return _pto.VdupOp(v.type, v, mask, position=position).result


def vexpdif(inp, ref, mask, part: str = "ODD"):
    """``pto.vexpdif`` – ``exp(inp - ref)`` selecting ODD or EVEN lanes."""
    return _pto.VexpdifOp(inp.type, inp, ref, mask, part).result


# ── Tile-domain operations ────────────────────────────────────────────────────

def make_tensor_view(ptr, *, shape, strides):
    """
    ``pto.make_tensor_view`` – wrap a pointer as a tensor view.

    Type is inferred: rank from ``len(shape)``, element type from ``ptr``.
    """
    rank = len(shape)
    elem = _pto.PtrType(ptr.type).element_type
    tv_type = tensor_view_type(rank, elem)
    return _pto.MakeTensorViewOp(tv_type, ptr, list(shape), list(strides)).result


def partition_view(tv, *, offsets, sizes):
    """
    ``pto.partition_view`` – slice a tensor view.

    Type is inferred from the source tensor-view type.
    """
    src_type = _pto.TensorViewType(tv.type)
    rank = src_type.rank
    elem = src_type.element_type
    ptv_type = part_tensor_view_type(rank, elem)
    return _pto.PartitionViewOp(ptv_type, tv, list(offsets), list(sizes)).result


def alloc_tile(tile_type, *, addr, valid_row, valid_col=None):
    """``pto.alloc_tile``."""
    return _pto.AllocTileOp(_resolve(tile_type), addr=addr, valid_row=valid_row,
                            valid_col=valid_col).result


def tload(part, tile):
    """``pto.tload ins(part) outs(tile)``."""
    _pto.TLoadOp(None, part, tile)


def tstore(tile, part):
    """``pto.tstore ins(tile) outs(part)``."""
    _pto.TStoreOp(None, tile, part)


def tile_ptr(tile, result_ptr_type):
    """``pto.tile_buf_addr`` – materialise a UB pointer from a tile handle."""
    return _pto.TileBufAddrOp(_resolve(result_ptr_type), tile).result


# ── Hardware / sync ───────────────────────────────────────────────────────────

def get_block_idx():
    """``pto.get_block_idx`` → i64 block index."""
    return _pto.GetBlockIdxOp().result


def barrier_all():
    """``pto.barrier #pto.pipe<PIPE_ALL>``."""
    _pto.BarrierOp(_pipe_attr("ALL"))


def set_flag(src: str, dst: str, *, event_id: int = 0):
    """``pto.set_flag[src, dst, event_id]``.

    Accepts short pipe names (``"MTE2"``, ``"V"``, …) or full ``"PIPE_MTE2"``
    names.  ``event_id`` is an integer in ``[0, 7]``.
    """
    _pto.set_flag(_pipe_attr(src), _pipe_attr(dst), _event_attr(event_id))


def wait_flag(src: str, dst: str, *, event_id: int = 0):
    """``pto.wait_flag[src, dst, event_id]``."""
    _pto.wait_flag(_pipe_attr(src), _pipe_attr(dst), _event_attr(event_id))


__all__ = [
    "const",
    "castptr", "addptr",
    "vlds", "vbrc_load", "vsts", "vsts_1pt",
    "plt_b32", "pset_b32",
    "vadd", "vmul", "vmax", "vdiv",
    "vcmax", "vcadd", "vdup", "vexpdif",
    "make_tensor_view", "partition_view",
    "alloc_tile", "tload", "tstore", "tile_ptr",
    "get_block_idx", "barrier_all", "set_flag", "wait_flag",
]

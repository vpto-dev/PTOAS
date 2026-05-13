# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Scalar arithmetic helpers – exposed as ``pto.scalar.*`` (or ``s = pto.scalar``).

All functions operate on raw ``mlir.ir.Value`` objects and emit the
corresponding arith dialect operations at the active insertion point.
"""

from ._bootstrap import make_context  # ensure MLIR is on sys.path  # noqa: F401
from ._types import _resolve

from mlir.dialects import arith
from mlir.ir import IndexType, IntegerType

_CMPI_PREDICATES = {
    "eq":  arith.CmpIPredicate.eq,
    "ne":  arith.CmpIPredicate.ne,
    "slt": arith.CmpIPredicate.slt,
    "sle": arith.CmpIPredicate.sle,
    "sgt": arith.CmpIPredicate.sgt,
    "sge": arith.CmpIPredicate.sge,
    "ult": arith.CmpIPredicate.ult,
    "ule": arith.CmpIPredicate.ule,
    "ugt": arith.CmpIPredicate.ugt,
    "uge": arith.CmpIPredicate.uge,
}


def muli(lhs, rhs):
    """arith.muli"""
    return arith.MulIOp(lhs, rhs).result


def addi(lhs, rhs):
    """arith.addi"""
    return arith.AddIOp(lhs, rhs).result


def subi(lhs, rhs):
    """arith.subi"""
    return arith.SubIOp(lhs, rhs).result


def index_cast(type_or_val, val=None):
    """
    arith.index_cast.

    Two calling conventions::

        index_cast(result_type, value)   # explicit result type
        index_cast(value)                # result type = index (1-arg shorthand)
    """
    if val is None:
        # 1-arg form: cast to index
        return arith.IndexCastOp(IndexType.get(), type_or_val).result
    return arith.IndexCastOp(_resolve(type_or_val), val).result


def cmpi(pred: str, lhs, rhs):
    """
    arith.cmpi with a named predicate string.

    ``pred`` is one of: ``"eq"``, ``"ne"``, ``"slt"``, ``"sle"``,
    ``"sgt"``, ``"sge"``, ``"ult"``, ``"ule"``, ``"ugt"``, ``"uge"``.
    """
    predicate = _CMPI_PREDICATES.get(pred)
    if predicate is None:
        raise ValueError(
            f"Unknown cmpi predicate '{pred}'; known: {list(_CMPI_PREDICATES)}"
        )
    return arith.CmpIOp(predicate, lhs, rhs).result


def cmpi_sgt(lhs, rhs):
    """arith.cmpi sgt (signed greater-than)."""
    return arith.CmpIOp(arith.CmpIPredicate.sgt, lhs, rhs).result


def select(cond, true_val, false_val):
    """arith.select"""
    return arith.SelectOp(cond, true_val, false_val).result


__all__ = ["muli", "addi", "subi", "index_cast", "cmpi", "cmpi_sgt", "select"]

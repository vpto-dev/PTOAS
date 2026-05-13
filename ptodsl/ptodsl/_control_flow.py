# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Control-flow context managers for PTO kernels.

All CMs work with the current MLIR insertion point; no context threading needed.

Public API
──────────
``vecscope()``            – ``pto.vecscope { … }``
``for_(lo, hi, step, *, iter_args)``
                          – ``scf.for`` with optional iter_args
``if_(cond, *, results)`` – ``scf.if`` with optional results + else
``yield_(*vals)``         – ``scf.yield``
"""

from ._bootstrap import make_context  # noqa: F401
from ._types import _resolve

from mlir.dialects import pto as _pto, scf
from mlir.ir import InsertionPoint


# ── vecscope ──────────────────────────────────────────────────────────────────

class _VecScopeCM:
    """Context manager for ``pto.vecscope { … }``."""

    def __enter__(self):
        self._op = _pto.VecScopeOp()
        self._block = self._op.body.blocks.append()
        self._ip = InsertionPoint(self._block)
        self._ip.__enter__()
        return None

    def __exit__(self, *exc):
        self._ip.__exit__(*exc)


def vecscope() -> _VecScopeCM:
    """Return a context manager that emits ``pto.vecscope { … }``."""
    return _VecScopeCM()


# ── for_ ──────────────────────────────────────────────────────────────────────

class LoopHandle:
    """
    Handle for a ``scf.for`` loop with iter_args.

    Attributes available *after* the ``with pto.for_(…) as loop:`` block::

        loop.iv         – induction variable
        loop.iter_args  – tuple of inner (mutable) SSA values
        loop.results    – tuple of ForOp results (after loop exit)
    """

    def __init__(self, for_op):
        self._op = for_op

    @property
    def iv(self):
        return self._op.induction_variable

    @property
    def iter_args(self):
        return tuple(self._op.inner_iter_args)

    @property
    def results(self):
        return tuple(self._op.results)


class _ForCM:
    def __init__(self, start, stop, step, iter_args):
        self._start = start
        self._stop = stop
        self._step = step
        self._iter_args = list(iter_args) if iter_args is not None else []
        self._for_op = None
        self._ip = None

    def __enter__(self):
        self._for_op = scf.ForOp(
            self._start, self._stop, self._step,
            self._iter_args if self._iter_args else None,
        )
        self._ip = InsertionPoint(self._for_op.body)
        self._ip.__enter__()
        if not self._iter_args:
            return self._for_op.induction_variable
        return LoopHandle(self._for_op)

    def __exit__(self, *exc):
        if not self._iter_args:
            scf.YieldOp([])
        self._ip.__exit__(*exc)


def for_(start, stop, *, step, iter_args=None) -> _ForCM:
    """
    ``scf.for`` context manager.

    Without ``iter_args`` – yields the induction variable; ``scf.yield`` is
    inserted automatically::

        with pto.for_(c0, c16, step=c1) as i:
            ...

    With ``iter_args`` – yields a :class:`LoopHandle`; the caller must emit
    ``pto.yield_(…)`` before the block closes::

        with pto.for_(c0, c128, step=c64, iter_args=(a, b)) as loop:
            x, y = loop.iter_args
            ...
            pto.yield_(nx, ny)
        fa, fb = loop.results
    """
    return _ForCM(start, stop, step, iter_args)


# ── if_ ───────────────────────────────────────────────────────────────────────

class _BlockCM:
    """Enters the InsertionPoint of a single block for ``with br.then_:`` style."""

    def __init__(self, block):
        self._block = block
        self._ip = None

    def __enter__(self):
        self._ip = InsertionPoint(self._block)
        self._ip.__enter__()

    def __exit__(self, *exc):
        self._ip.__exit__(*exc)


class BranchHandle:
    """
    Handle for ``scf.if`` with results and an else branch.

    Usage::

        with pto.if_(cond, results=(vf32, vf32)) as br:
            with br.then_:
                ...
                pto.yield_(a, b)
            with br.else_:
                pto.yield_(c, d)
        x, y = br.results
    """

    def __init__(self, if_op):
        self._op = if_op
        self.then_ = _BlockCM(if_op.then_block)
        self.else_ = _BlockCM(if_op.else_block)

    @property
    def results(self):
        return tuple(self._op.results)


class _IfCM:
    def __init__(self, cond, result_types):
        self._cond = cond
        self._result_types = [_resolve(t) for t in result_types] if result_types else []
        self._if_op = None
        self._ip = None

    def __enter__(self):
        if self._result_types:
            # if/else with results: create IfOp but don't enter any block;
            # the caller manages blocks via br.then_ / br.else_
            self._if_op = scf.IfOp(self._cond, self._result_types, hasElse=True)
            return BranchHandle(self._if_op)
        else:
            # simple if without results: enter then_block automatically
            self._if_op = scf.IfOp(self._cond)
            self._ip = InsertionPoint(self._if_op.then_block)
            self._ip.__enter__()
            return None

    def __exit__(self, *exc):
        if not self._result_types:
            scf.YieldOp([])
            self._ip.__exit__(*exc)
        # for if/else with results: blocks are managed by BranchHandle; nothing to do


def if_(cond, *, results=None) -> _IfCM:
    """
    ``scf.if`` context manager.

    Without ``results`` – simple if with no else; ``scf.yield`` is inserted
    automatically::

        with pto.if_(has_rows):
            ...

    With ``results`` – if/else pair that produces SSA values; the caller must
    manage ``br.then_`` and ``br.else_`` and emit ``pto.yield_(…)`` in each::

        with pto.if_(has_chunk, results=(vf32, vf32)) as br:
            with br.then_:
                ...
                pto.yield_(merged_max, merged_sum)
            with br.else_:
                pto.yield_(running_max, running_sum)
        x, y = br.results
    """
    return _IfCM(cond, results)


# ── yield_ ────────────────────────────────────────────────────────────────────

def yield_(*vals):
    """Emit ``scf.yield`` with the given values."""
    scf.YieldOp(list(vals))


__all__ = [
    "vecscope", "LoopHandle", "BranchHandle",
    "for_", "if_", "yield_",
]

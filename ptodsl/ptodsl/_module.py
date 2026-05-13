# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
``@pto.to_ir`` decorator and module-level IR builders.

The decorator:
1. Inspects the function signature – annotations are ``_DType`` lazy
   descriptors or concrete ``mlir.ir.Type`` objects.
2. Creates the MLIR context and module.
3. Calls the Python function body with actual MLIR SSA values.
4. Verifies the module and caches it as ``fn._ir_module``.
5. Adds ``__str__`` so ``print(my_kernel)`` prints the MLIR text.

Module structure is selected by ``func_attr``:
- ``func_attr="pto.aicore"``  → flat module + ``pto.aicore`` function attribute
  (used by softmax-style kernels)
- otherwise                  → nested double-module (used by vPTO TADD-style)
"""

import inspect

from ._bootstrap import make_context
from ._types import _resolve

from mlir.dialects import func, pto as _pto
from mlir.ir import (
    Attribute,
    InsertionPoint,
    Location,
    Module,
    Operation,
    StringAttr,
    UnitAttr,
)


def _call_body(ir_fn, fn, arg_types):
    """Add entry block to *ir_fn* and call *fn* with the SSA arguments."""
    entry = ir_fn.add_entry_block()
    with InsertionPoint(entry):
        fn(*entry.arguments)
        func.ReturnOp([])


def _build_flat_module(fn_name, arg_types, fn, arch, kernel_kind):
    """
    Flat ``module attributes {pto.target_arch, pto.kernel_kind}`` with a
    single function that carries ``pto.aicore``.
    """
    m = Module.create()
    m.operation.attributes["pto.target_arch"] = StringAttr.get(arch)
    m.operation.attributes["pto.kernel_kind"] = Attribute.parse(
        f"#pto.kernel_kind<{kernel_kind}>"
    )
    fn_ty = func.FunctionType.get(arg_types, [])
    with InsertionPoint(m.body):
        ir_fn = func.FuncOp(fn_name, fn_ty)
        ir_fn.attributes["pto.aicore"] = UnitAttr.get()
    _call_body(ir_fn, fn, arg_types)
    return m


def _build_nested_module(fn_name, arg_types, fn, arch, kernel_kind):
    """
    Nested ``module { module { func … } }`` structure used by vPTO kernels
    without function arguments (e.g. TADD).
    """
    outer = Module.create()
    outer.operation.attributes["pto.target_arch"] = StringAttr.get(arch)

    with InsertionPoint(outer.body):
        # Module.create() ignores the active InsertionPoint, so create
        # the inner module via Operation.create("builtin.module") instead.
        inner_op = Operation.create("builtin.module", regions=1)
        inner_op.attributes["pto.target_arch"] = StringAttr.get(arch)
        inner_op.attributes["pto.kernel_kind"] = Attribute.parse(
            f"#pto.kernel_kind<{kernel_kind}>"
        )
        inner_body = inner_op.regions[0].blocks.append()

        with InsertionPoint(inner_body):
            fn_ty = func.FunctionType.get(arg_types, [])
            ir_fn = func.FuncOp(fn_name, fn_ty)

    _call_body(ir_fn, fn, arg_types)
    return outer


def to_ir(name=None, *, kernel_kind: str = "vector", arch: str = "a5",
          func_attr: str = None):
    """
    Decorator that eagerly lowers a Python function to an MLIR module.

    Parameters
    ----------
    name:        IR function name (defaults to the Python function name).
    kernel_kind: ``"vector"`` or ``"cube"`` – sets ``pto.kernel_kind``.
    arch:        Target architecture string, e.g. ``"a5"``.
    func_attr:   Optional function attribute.  Pass ``"pto.aicore"`` to
                 select the flat-module structure with the aicore attribute.

    The decorated function is replaced by a :class:`KernelHandle` that:

    - prints as the MLIR module text (``print(my_kernel)``),
    - exposes ``my_kernel.build()`` returning the ``mlir.ir.Module``,
    - exposes ``my_kernel._ir_module`` for direct access.
    """

    def decorator(fn):
        fn_name = name or fn.__name__
        sig = inspect.signature(fn)
        ctx = make_context()
        with ctx, Location.unknown():
            arg_types = [
                _resolve(p.annotation)
                for p in sig.parameters.values()
                if p.annotation is not inspect.Parameter.empty
            ]
            if func_attr == "pto.aicore":
                mod = _build_flat_module(fn_name, arg_types, fn, arch, kernel_kind)
            else:
                mod = _build_nested_module(fn_name, arg_types, fn, arch, kernel_kind)
            mod.operation.verify()

        return KernelHandle(fn.__name__, mod)

    return decorator


class KernelHandle:
    """
    Represents a compiled PTO kernel.

    ``print(handle)`` emits the MLIR module text.
    ``handle.build()`` returns the ``mlir.ir.Module`` (for ``check_ir.py``).
    ``handle._ir_module`` is the raw module for direct access.
    """

    def __init__(self, py_name: str, module):
        self._py_name = py_name
        self._ir_module = module

    def build(self):
        """Return the compiled ``mlir.ir.Module``."""
        return self._ir_module

    def __str__(self):
        return str(self._ir_module)

    def __repr__(self):
        return str(self._ir_module)


__all__ = ["to_ir", "KernelHandle"]

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
MLIR path bootstrap and context factory.

Adds the ptoas install directory to sys.path so that the mlir package is
importable regardless of how the ptodsl package itself was installed.
"""

import os
import sys

_INSTALL = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "install", "mlir")
)
if os.path.isdir(_INSTALL) and _INSTALL not in sys.path:
    sys.path.insert(0, _INSTALL)

from mlir.dialects import pto as _pto_dialect  # noqa: E402
from mlir.ir import Context, Location           # noqa: E402


def make_context() -> Context:
    """Create a fresh MLIR Context with the PTO dialect loaded."""
    ctx = Context()
    _pto_dialect.register_dialect(ctx, load=True)
    return ctx


__all__ = ["make_context"]

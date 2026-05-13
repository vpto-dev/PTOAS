# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
IR correctness check for all ptodsl example scripts.

Run from the repository root or from this directory:
    python3 ptodsl/check_ir.py              # from ptoas_a5/
    python3 check_ir.py                     # from ptoas_a5/ptodsl/

Each example's ``build()`` function is called; its output is compared against
the corresponding hand-written reference ``.pto`` file.

Comparison methodology
──────────────────────
Both the generated module and the reference file are parsed by the MLIR Python
API (``Module.parse``), then printed back to a string.  This round-trip:

  • Strips ``//`` comments present in hand-written ``.pto`` files
  • Normalises SSA value names  (``%block_idx`` → ``%0``, …)
  • Normalises attribute ordering

The resulting canonical strings are compared with ``==``.  A unified diff of
the first 60 diverging lines is printed on failure.
"""

import difflib
import importlib
import os
import sys

# ── Path setup ────────────────────────────────────────────────────────────────

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLES = os.path.join(_HERE, "examples")
_MLIR_INSTALL = os.path.join(_HERE, "..", "install", "mlir")

for _p in (_MLIR_INSTALL, _HERE, _EXAMPLES):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mlir.ir import Context, Module          # noqa: E402
from mlir.dialects import pto as _pto_mod    # noqa: E402


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize(mlir_text: str) -> str:
    """Parse *mlir_text* with MLIR and return the canonical printed form."""
    with Context() as ctx:
        _pto_mod.register_dialect(ctx, load=True)
        return str(Module.parse(mlir_text))


def _strip_comments(text: str) -> str:
    """Remove ``//`` comment lines found in hand-written ``.pto`` files."""
    return "\n".join(
        line for line in text.splitlines() if not line.strip().startswith("//")
    )


# ── Test cases ────────────────────────────────────────────────────────────────
# Each entry: (label, module_name, reference_pto_path)

_REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_TADD_REF   = os.path.join(_REPO_ROOT, "test/lit/vpto/expand_tileop_to_vpto_result.pto")
_SOFTMAX_REF = os.path.join(_REPO_ROOT,
                             "test/tilelang_st/npu/a5/src/st/testcase/softmax/softmax.pto")

CASES = [
    ("TADD  low-level ", "tadd_lowlevel",   _TADD_REF),
    ("TADD  dsl-style ", "tadd_dsl",        _TADD_REF),
    ("softmax low-level", "softmax_lowlevel", _SOFTMAX_REF),
    ("softmax dsl-style", "softmax_dsl",     _SOFTMAX_REF),
]


# ── Runner ────────────────────────────────────────────────────────────────────

def run_checks(cases=CASES) -> bool:
    """Execute every check case; return ``True`` if all passed."""
    all_passed = True

    for label, module_name, ref_path in cases:
        # -- import the example and call build() --
        try:
            # Re-import on every run so state doesn't leak between cases
            spec = importlib.util.spec_from_file_location(
                module_name, os.path.join(_EXAMPLES, f"{module_name}.py")
            )
            builder = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(builder)
            generated_text = str(builder.build())
        except Exception as exc:
            print(f"  FAIL  {label}  [builder error: {exc}]")
            all_passed = False
            continue

        # -- load and prepare the reference --
        try:
            ref_raw = open(ref_path).read()
        except FileNotFoundError:
            print(f"  FAIL  {label}  [reference not found: {ref_path}]")
            all_passed = False
            continue

        ref_clean = _strip_comments(ref_raw)

        # -- normalise both through the MLIR parser --
        try:
            gen_norm = _normalize(generated_text)
            ref_norm = _normalize(ref_clean)
        except Exception as exc:
            print(f"  FAIL  {label}  [MLIR parse error: {exc}]")
            all_passed = False
            continue

        # -- compare --
        if gen_norm == ref_norm:
            print(f"  PASS  {label}")
        else:
            all_passed = False
            diff_lines = list(
                difflib.unified_diff(
                    ref_norm.splitlines(),
                    gen_norm.splitlines(),
                    fromfile="reference",
                    tofile="generated",
                    lineterm="",
                )
            )
            snippet = "\n".join(diff_lines[:60])
            print(f"  FAIL  {label}\n{snippet}")
            if len(diff_lines) > 60:
                print(f"        … ({len(diff_lines) - 60} more diff lines)")

    return all_passed


if __name__ == "__main__":
    print("ptodsl IR check")
    print("=" * 50)
    passed = run_checks()
    print("=" * 50)
    print("Result:", "ALL PASS" if passed else "SOME TESTS FAILED")
    sys.exit(0 if passed else 1)

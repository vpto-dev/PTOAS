"""Support-matrix definitions and diagnostics for TileLang DSL v1."""

from __future__ import annotations

FOLLOW_UP_CHANGE = "extend-tilelang-dsl-matcher-and-advanced-surface"

SUPPORTED_TOPLEVEL_PTO_CALLS = frozenset(
    {
        "strict_vecscope",
        "dma_load",
        "dma_store",
        "set_flag",
        "wait_flag",
        "pipe_barrier",
        "barrier",
    }
)

SUPPORTED_VECSCOPE_PTO_CALLS = frozenset(
    {
        "make_mask",
        "vlds",
        "vsts",
        "vabs",
        "vrelu",
        "vexp",
        "vnot",
        "vadd",
        "vsub",
        "vmul",
        "vdiv",
        "vmax",
        "vmin",
        "vand",
        "vor",
        "vxor",
        "vadds",
        "vsubs",
        "vmuls",
        "vdivs",
        "vmaxs",
        "vmins",
    }
)

ADVANCED_VECSCOPE_PTO_CALLS = frozenset(
    {
        "vcmp",
        "vcmps",
        "vsel",
        "vselr",
        "vselrv2",
        "pnot",
        "psel",
        "ppack",
        "punpack",
        "vaddc",
        "vsubc",
        "vaddcs",
        "vsubcs",
        "vintlv",
        "vdintlv",
        "vintlvv2",
        "vdintlvv2",
    }
)

ADVANCED_EXPR_PTO_CALLS = frozenset(
    {
        "ptr",
        "castptr",
        "addptr",
    }
)

ADVANCED_TOPLEVEL_PTO_CALLS = frozenset(
    {
        "copy_gm_to_ubuf",
        "copy_ubuf_to_gm",
        "copy_ubuf_to_ubuf",
        "set_loop2_stride_outtoub",
        "set_loop1_stride_outtoub",
        "set_loop_size_outtoub",
        "set_loop2_stride_ubtoout",
        "set_loop1_stride_ubtoout",
        "set_loop_size_ubtoout",
    }
)

DEFERRED_PTO_SURFACES = frozenset(
    {
        "vreduce",
    }
)


def unsupported_feature_message(feature: str) -> str:
    return (
        f"{feature} is not supported in TileLang DSL v1; "
        f"see follow-up change `{FOLLOW_UP_CHANGE}`"
    )


def deferred_surface_message(name: str) -> str:
    return unsupported_feature_message(f"advanced family surface `pto.{name}`")


def advanced_mode_message(name: str) -> str:
    return f"surface `pto.{name}` requires advanced=True in TileLang DSL"


__all__ = [
    "DEFERRED_PTO_SURFACES",
    "FOLLOW_UP_CHANGE",
    "ADVANCED_EXPR_PTO_CALLS",
    "ADVANCED_TOPLEVEL_PTO_CALLS",
    "ADVANCED_VECSCOPE_PTO_CALLS",
    "SUPPORTED_TOPLEVEL_PTO_CALLS",
    "SUPPORTED_VECSCOPE_PTO_CALLS",
    "advanced_mode_message",
    "deferred_surface_message",
    "unsupported_feature_message",
]

"""Authoring-form VPTO lowering skeleton for TileLang DSL v1."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .semantic import (
    SemanticAssignStmt,
    SemanticAttributeAccess,
    SemanticBinaryExpr,
    SemanticBindingRef,
    SemanticCallExpr,
    SemanticDmaConfigStmt,
    SemanticDmaLoadStmt,
    SemanticDmaStoreStmt,
    SemanticExpr,
    SemanticExprStmt,
    SemanticForStmt,
    SemanticIfStmt,
    SemanticIndexType,
    SemanticIfResult,
    SemanticKernel,
    SemanticLiteralExpr,
    SemanticLowLevelCopyStmt,
    SemanticMaskType,
    SemanticMetaType,
    SemanticPipeBarrierStmt,
    SemanticPtrType,
    SemanticReturnStmt,
    SemanticScalarType,
    SemanticSetFlagStmt,
    SemanticStmt,
    SemanticVecscopeStmt,
    SemanticStrictVecscopeStmt,
    SemanticSubscriptAccess,
    SemanticSymbolExpr,
    SemanticTensorSliceExpr,
    SemanticTensorViewType,
    SemanticTileType,
    SemanticType,
    SemanticTupleExpr,
    SemanticTupleType,
    SemanticVRegType,
    SemanticVectorStoreStmt,
    SemanticWaitFlagStmt,
)
from .types import MaskPattern, ScalarType


_I1_TYPE = SemanticScalarType(dtype=ScalarType("i1"))
_I32_TYPE = SemanticScalarType(dtype=ScalarType("i32"))
_I64_TYPE = SemanticScalarType(dtype=ScalarType("i64"))


def _format_symbol_name(symbol_name: str) -> str:
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$.]*", symbol_name):
        return f"@{symbol_name}"
    escaped = symbol_name.replace("\\", "\\\\").replace('"', '\\"')
    return f'@"{escaped}"'


@dataclass(frozen=True)
class AuthoringModule:
    """Lowering result that owns authoring-form VPTO text emission."""

    kernel: SemanticKernel

    def render(self) -> str:
        return _AuthoringRenderer(self.kernel).render()


@dataclass(frozen=True)
class _RenderedValue:
    name: str
    type: SemanticType


class _AuthoringRenderer:
    def __init__(self, kernel: SemanticKernel):
        self.kernel = kernel
        self._constant_lines: list[str] = []
        self._constant_cache: dict[tuple[str, object], str] = {}
        self._temp_counter = 0
        self._loop_counter = 0

    def render(self) -> str:
        parameter_list = ", ".join(
            f"{param.ssa_name}: {self._render_type(param.type)}"
            for param in self.kernel.parameters
        )
        env = {
            param.name: _RenderedValue(name=param.ssa_name, type=param.type)
            for param in self.kernel.parameters
        }
        body_lines = self._render_block(self.kernel.body, env, indent=4)

        lines = [
            f"// tilelang.target = {self.kernel.target}",
            f"// tilelang.op = {self.kernel.op}",
            f"// tilelang.dtypes = {self.kernel.dtype_signature}",
            f"// tilelang.verify = {self.kernel.verify_enabled}",
            f"// tilelang.advanced = {self.kernel.advanced_enabled}",
        ]
        for binding in self.kernel.tile_bindings:
            lines.append(
                "// tilelang.specialize "
                f"{binding.name} shape={binding.shape} memory_space={binding.memory_space} "
                f"config={binding.config}"
            )
        lines.append(f'module attributes {{pto.target_arch = "{self.kernel.target}"}} {{')
        lines.append(
            f"  func.func {_format_symbol_name(self.kernel.symbol_name)}({parameter_list}) {{"
        )
        lines.extend(self._constant_lines)
        lines.extend(body_lines)
        lines.append("  }")
        lines.append("}")
        lines.append("")
        return "\n".join(lines)

    def _render_block(
        self,
        statements: tuple[SemanticStmt, ...],
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        lines: list[str] = []
        for stmt in statements:
            lines.extend(self._render_stmt(stmt, env, indent=indent))
        return lines

    def _render_stmt(
        self,
        stmt: SemanticStmt,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        if isinstance(stmt, SemanticAssignStmt):
            return self._render_assign(stmt, env, indent=indent)
        if isinstance(stmt, SemanticExprStmt):
            self._lower_expr(stmt.expr, env, indent=indent)
            return []
        if isinstance(stmt, SemanticDmaLoadStmt):
            return self._render_dma_load(stmt, env, indent=indent)
        if isinstance(stmt, SemanticDmaStoreStmt):
            return self._render_dma_store(stmt, env, indent=indent)
        if isinstance(stmt, SemanticVectorStoreStmt):
            return self._render_vector_store(stmt, env, indent=indent)
        if isinstance(stmt, SemanticSetFlagStmt):
            return [
                self._indent(indent)
                + f'pto.set_flag["{stmt.src_pipe}", "{stmt.dst_pipe}", "{stmt.event}"]'
            ]
        if isinstance(stmt, SemanticWaitFlagStmt):
            return [
                self._indent(indent)
                + f'pto.wait_flag["{stmt.src_pipe}", "{stmt.dst_pipe}", "{stmt.event}"]'
            ]
        if isinstance(stmt, SemanticPipeBarrierStmt):
            return [self._indent(indent) + f"pto.barrier #pto.pipe<{stmt.pipe}>"]
        if isinstance(stmt, SemanticDmaConfigStmt):
            return self._render_dma_config(stmt, env, indent=indent)
        if isinstance(stmt, SemanticLowLevelCopyStmt):
            return self._render_low_level_copy(stmt, env, indent=indent)
        if isinstance(stmt, SemanticReturnStmt):
            if stmt.value is None:
                return [self._indent(indent) + "return"]
            value = self._lower_expr(stmt.value, env, indent=indent)
            return [self._indent(indent) + f"return {value.name} : {self._render_type(value.type)}"]
        if isinstance(stmt, SemanticVecscopeStmt):
            return self._render_vecscope(stmt, env, indent=indent)
        if isinstance(stmt, SemanticStrictVecscopeStmt):
            return self._render_strict_vecscope(stmt, env, indent=indent)
        if isinstance(stmt, SemanticForStmt):
            return self._render_for(stmt, env, indent=indent)
        if isinstance(stmt, SemanticIfStmt):
            return self._render_if(stmt, env, indent=indent)
        raise ValueError(f"unsupported semantic statement {type(stmt).__name__}")

    def _render_dma_config(
        self,
        stmt: SemanticDmaConfigStmt,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        lines: list[str] = []
        first = self._lower_to_i64(stmt.first, env, indent=indent, into=lines)
        second = self._lower_to_i64(stmt.second, env, indent=indent, into=lines)
        lines.append(
            self._indent(indent)
            + f"pto.{stmt.name} {first.name}, {second.name} : i64, i64"
        )
        return lines

    def _render_low_level_copy(
        self,
        stmt: SemanticLowLevelCopyStmt,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        lines: list[str] = []
        source = self._lower_expr(stmt.source, env, indent=indent, into=lines)
        destination = self._lower_expr(stmt.destination, env, indent=indent, into=lines)

        rendered_operands = []
        rendered_types = []
        for index, operand in enumerate(stmt.operands):
            if stmt.name == "copy_gm_to_ubuf" and index == 5:
                lowered = self._lower_to_i1(operand, env, indent=indent, into=lines)
            else:
                lowered = self._lower_to_i64(operand, env, indent=indent, into=lines)
            rendered_operands.append(lowered.name)
            rendered_types.append(self._render_type(lowered.type))

        operand_text = ", ".join([source.name, destination.name, *rendered_operands])
        type_text = ", ".join(
            [self._render_type(source.type), self._render_type(destination.type), *rendered_types]
        )
        lines.append(
            self._indent(indent)
            + f"pto.{stmt.name} {operand_text} : {type_text}"
        )
        return lines

    def _render_assign(
        self,
        stmt: SemanticAssignStmt,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        if len(stmt.targets) != 1:
            if isinstance(stmt.value, SemanticTupleExpr):
                return self._render_tuple_expr_assign(stmt, env, indent=indent)
            return self._render_multi_result_assign(stmt, env, indent=indent)
        target = stmt.targets[0]
        if isinstance(target.type, SemanticMetaType):
            env[target.name] = _RenderedValue(name=target.ssa_name, type=target.type)
            return []
        lines: list[str] = []
        lowered = self._lower_expr(
            stmt.value,
            env,
            indent=indent,
            desired_name=target.ssa_name,
            into=lines,
        )
        env[target.name] = lowered
        return lines

    def _render_tuple_expr_assign(
        self,
        stmt: SemanticAssignStmt,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        if not isinstance(stmt.value, SemanticTupleExpr):
            raise NotImplementedError("tuple expression assignment expects a SemanticTupleExpr")
        if len(stmt.targets) != len(stmt.value.elements):
            raise NotImplementedError("tuple expression assignment arity mismatch")

        lines: list[str] = []
        for target, element in zip(stmt.targets, stmt.value.elements):
            lowered = self._lower_expr(
                element,
                env,
                indent=indent,
                desired_name=target.ssa_name,
                into=lines,
            )
            env[target.name] = lowered
        return lines

    def _render_multi_result_assign(
        self,
        stmt: SemanticAssignStmt,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        if not isinstance(stmt.value, SemanticCallExpr):
            raise NotImplementedError("multi-result assignment expects a call expression in TileLang DSL v1")
        if stmt.value.namespace != "pto":
            raise NotImplementedError(
                f"multi-result assignment for `pto.{stmt.value.name}` is not supported in TileLang DSL v1"
            )
        if len(stmt.targets) != 2:
            raise NotImplementedError("multi-result lowering expects exactly two assignment targets")
        if not isinstance(stmt.value.type, SemanticTupleType) or len(stmt.value.type.elements) != 2:
            raise NotImplementedError("multi-result lowering expects a two-result tuple type")

        if stmt.value.name == "make_mask":
            dtype_expr, remaining_expr = stmt.value.args
            if not self._is_dtype_meta_expr(dtype_expr):
                raise NotImplementedError("make_mask dtype lowering expects a dtype symbol")

            lines: list[str] = []
            remaining = self._lower_remaining_to_i32(remaining_expr, env, indent=indent, into=lines)
            mask_target, remaining_target = stmt.targets
            mask_type, remaining_type = stmt.value.type.elements
            suffix = self._mask_suffix(mask_type)
            lines.append(
                self._indent(indent)
                + f"{mask_target.ssa_name}, {remaining_target.ssa_name} = pto.plt_{suffix} {remaining.name} : "
                + f"i32 -> {self._render_type(mask_type)}, {self._render_type(remaining_type)}"
            )
            env[mask_target.name] = _RenderedValue(name=mask_target.ssa_name, type=mask_type)
            env[remaining_target.name] = _RenderedValue(name=remaining_target.ssa_name, type=remaining_type)
            return lines

        if stmt.value.name in {"vaddc", "vsubc"}:
            lines = []
            lhs = self._lower_expr(stmt.value.args[0], env, indent=indent, into=lines)
            rhs = self._lower_expr(stmt.value.args[1], env, indent=indent, into=lines)
            mask = self._lower_expr(stmt.value.args[2], env, indent=indent, into=lines)
            result_target, carry_target = stmt.targets
            result_type, carry_type = stmt.value.type.elements
            lines.append(
                self._indent(indent)
                + f"{result_target.ssa_name}, {carry_target.ssa_name} = pto.{stmt.value.name} "
                + f"{lhs.name}, {rhs.name}, {mask.name} : "
                + f"{self._render_type(lhs.type)}, {self._render_type(rhs.type)}, {self._render_type(mask.type)} "
                + f"-> {self._render_type(result_type)}, {self._render_type(carry_type)}"
            )
            env[result_target.name] = _RenderedValue(name=result_target.ssa_name, type=result_type)
            env[carry_target.name] = _RenderedValue(name=carry_target.ssa_name, type=carry_type)
            return lines

        if stmt.value.name in {"vaddcs", "vsubcs"}:
            lines = []
            lhs = self._lower_expr(stmt.value.args[0], env, indent=indent, into=lines)
            rhs = self._lower_expr(stmt.value.args[1], env, indent=indent, into=lines)
            carry_in = self._lower_expr(stmt.value.args[2], env, indent=indent, into=lines)
            mask = self._lower_expr(stmt.value.args[3], env, indent=indent, into=lines)
            result_target, carry_target = stmt.targets
            result_type, carry_type = stmt.value.type.elements
            lines.append(
                self._indent(indent)
                + f"{result_target.ssa_name}, {carry_target.ssa_name} = pto.{stmt.value.name} "
                + f"{lhs.name}, {rhs.name}, {carry_in.name}, {mask.name} : "
                + f"{self._render_type(lhs.type)}, {self._render_type(rhs.type)}, "
                + f"{self._render_type(carry_in.type)}, {self._render_type(mask.type)} "
                + f"-> {self._render_type(result_type)}, {self._render_type(carry_type)}"
            )
            env[result_target.name] = _RenderedValue(name=result_target.ssa_name, type=result_type)
            env[carry_target.name] = _RenderedValue(name=carry_target.ssa_name, type=carry_type)
            return lines

        if stmt.value.name in {"vintlv", "vdintlv"}:
            lines = []
            lhs = self._lower_expr(stmt.value.args[0], env, indent=indent, into=lines)
            rhs = self._lower_expr(stmt.value.args[1], env, indent=indent, into=lines)
            low_target, high_target = stmt.targets
            low_type, high_type = stmt.value.type.elements
            lines.append(
                self._indent(indent)
                + f"{low_target.ssa_name}, {high_target.ssa_name} = pto.{stmt.value.name} "
                + f"{lhs.name}, {rhs.name} : {self._render_type(lhs.type)}, {self._render_type(rhs.type)} "
                + f"-> {self._render_type(low_type)}, {self._render_type(high_type)}"
            )
            env[low_target.name] = _RenderedValue(name=low_target.ssa_name, type=low_type)
            env[high_target.name] = _RenderedValue(name=high_target.ssa_name, type=high_type)
            return lines

        raise NotImplementedError(
            f"multi-result assignment for `pto.{stmt.value.name}` is not supported in TileLang DSL v1"
        )

    def _render_dma_load(
        self,
        stmt: SemanticDmaLoadStmt,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        lines: list[str] = []
        src = self._lower_expr(stmt.src.base, env, indent=indent, into=lines)
        dst = self._lower_expr(stmt.dst, env, indent=indent, into=lines)
        row_count, col_count = self._dma_transfer_extents(stmt.src, stmt.dst.type)
        element_bytes = self._dtype_byte_width(stmt.src.type.element_dtype)
        burst_bytes = col_count * element_bytes

        c0_i64 = self._materialize_constant(0, _I64_TYPE)
        c1_i64 = self._materialize_constant(1, _I64_TYPE)
        n_burst = self._materialize_constant(row_count, _I64_TYPE)
        len_burst = self._materialize_constant(burst_bytes, _I64_TYPE)
        false_bit = self._materialize_constant(False, _I1_TYPE)

        lines.extend(
            [
                self._indent(indent)
                + f"pto.set_loop_size_outtoub {c1_i64}, {c1_i64} : i64, i64",
                self._indent(indent)
                + "pto.copy_gm_to_ubuf "
                + f"{src.name}, {dst.name}, {c0_i64}, {n_burst}, {len_burst}, {c0_i64}, {c0_i64}, "
                + f"{false_bit}, {c0_i64}, {len_burst}, {len_burst} : "
                + f"{self._render_type(src.type)}, {self._render_type(dst.type)}, "
                + "i64, i64, i64, i64, i64, i1, i64, i64, i64",
            ]
        )
        return lines

    def _render_dma_store(
        self,
        stmt: SemanticDmaStoreStmt,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        lines: list[str] = []
        src = self._lower_expr(stmt.src, env, indent=indent, into=lines)
        dst = self._lower_expr(stmt.dst.base, env, indent=indent, into=lines)
        row_count, col_count = self._dma_transfer_extents(stmt.dst, stmt.src.type)
        element_bytes = self._dtype_byte_width(stmt.dst.type.element_dtype)
        burst_bytes = col_count * element_bytes

        c0_i64 = self._materialize_constant(0, _I64_TYPE)
        c1_i64 = self._materialize_constant(1, _I64_TYPE)
        n_burst = self._materialize_constant(row_count, _I64_TYPE)
        len_burst = self._materialize_constant(burst_bytes, _I64_TYPE)

        lines.extend(
            [
                self._indent(indent)
                + f"pto.set_loop_size_ubtoout {c1_i64}, {c1_i64} : i64, i64",
                self._indent(indent)
                + "pto.copy_ubuf_to_gm "
                + f"{src.name}, {dst.name}, {c0_i64}, {n_burst}, {len_burst}, {c0_i64}, "
                + f"{len_burst}, {len_burst} : {self._render_type(src.type)}, {self._render_type(dst.type)}, "
                + "i64, i64, i64, i64, i64, i64",
            ]
        )
        return lines

    def _render_vector_store(
        self,
        stmt: SemanticVectorStoreStmt,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        lines: list[str] = []
        value = self._lower_expr(stmt.value, env, indent=indent, into=lines)
        destination = self._lower_expr(stmt.destination, env, indent=indent, into=lines)
        offset = self._lower_expr(stmt.offset, env, indent=indent, into=lines)
        mask = self._lower_expr(stmt.mask, env, indent=indent, into=lines)
        lines.append(
            self._indent(indent)
            + "pto.vsts "
            + f"{value.name}, {destination.name}[{offset.name}], {mask.name} : "
            + f"{self._render_type(value.type)}, {self._render_type(destination.type)}, {self._render_type(mask.type)}"
        )
        return lines

    def _tensor_slice_extents(self, expr: SemanticTensorSliceExpr) -> tuple[int, int]:
        if expr.type.rank != 2 or len(expr.type.extents) != 2:
            raise NotImplementedError("TileLang DSL v1 DMA lowering currently only supports rank-2 TensorView slices")
        return expr.type.extents

    def _dma_transfer_extents(
        self,
        slice_expr: SemanticTensorSliceExpr,
        tile_type: SemanticTileType,
    ) -> tuple[int, int]:
        row_count, col_count = self._tensor_slice_extents(slice_expr)
        if row_count is not None and col_count is not None:
            return row_count, col_count
        if tile_type.shape is None or len(tile_type.shape) != 2:
            raise NotImplementedError("DMA lowering requires a statically specialized rank-2 Tile shape")
        return tile_type.shape

    def _render_strict_vecscope(
        self,
        stmt: SemanticStrictVecscopeStmt,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        lines: list[str] = []
        capture_values = [
            self._lower_expr(expr, env, indent=indent, into=lines)
            for expr in stmt.captures
        ]
        capture_names = ", ".join(value.name for value in capture_values)
        block_args = ", ".join(
            f"{binding.ssa_name}: {self._render_type(binding.type)}"
            for binding in stmt.block_arguments
        )
        function_type = ", ".join(
            self._render_type(binding.type) for binding in stmt.block_arguments
        )

        scope_env = {
            binding.name: _RenderedValue(name=binding.ssa_name, type=binding.type)
            for binding in stmt.block_arguments
        }

        lines.append(self._indent(indent) + f"pto.strict_vecscope({capture_names}) {{")
        lines.append(self._indent(indent) + f"^bb0({block_args}):")
        lines.extend(self._render_block(stmt.body, scope_env, indent=indent + 2))
        lines.append(self._indent(indent) + f"}} : ({function_type}) -> ()")
        return lines

    def _render_vecscope(
        self,
        stmt: SemanticVecscopeStmt,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        scope_env = dict(env)
        lines = [self._indent(indent) + "pto.vecscope {"]
        lines.extend(self._render_block(stmt.body, scope_env, indent=indent + 2))
        lines.append(self._indent(indent) + "}")
        return lines

    def _render_for(
        self,
        stmt: SemanticForStmt,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        lines: list[str] = []
        lower_bound = self._lower_expr(stmt.lower_bound, env, indent=indent, into=lines)
        upper_bound = self._lower_expr(stmt.upper_bound, env, indent=indent, into=lines)
        step = self._lower_expr(stmt.step, env, indent=indent, into=lines)

        body_env = dict(env)
        body_env[stmt.induction_variable.name] = _RenderedValue(
            name=stmt.induction_variable.ssa_name,
            type=stmt.induction_variable.type,
        )

        if not stmt.loop_carried:
            lines.append(
                self._indent(indent)
                + f"scf.for {stmt.induction_variable.ssa_name} = {lower_bound.name} "
                f"to {upper_bound.name} step {step.name} {{"
            )
            lines.extend(self._render_block(stmt.body, body_env, indent=indent + 2))
            lines.append(self._indent(indent) + "}")
            return lines

        if len(stmt.loop_carried) != 1:
            raise NotImplementedError(
                "TileLang DSL v1 lowering currently supports at most one loop-carried binding"
            )

        carried_binding = stmt.loop_carried[0]
        initial_value = env[carried_binding.name]
        iter_arg_name = f"%{carried_binding.name}_iter_{self._loop_counter}"
        self._loop_counter += 1
        body_env[carried_binding.name] = _RenderedValue(
            name=iter_arg_name,
            type=carried_binding.type,
        )

        lines.append(
            self._indent(indent)
            + f"{carried_binding.ssa_name}:1 = scf.for {stmt.induction_variable.ssa_name} = "
            f"{lower_bound.name} to {upper_bound.name} step {step.name} "
            f"iter_args({iter_arg_name} = {initial_value.name}) -> "
            f"({self._render_type(carried_binding.type)}) {{"
        )
        lines.extend(self._render_block(stmt.body, body_env, indent=indent + 2))
        yielded_value = body_env[carried_binding.name]
        lines.append(
            self._indent(indent + 2)
            + f"scf.yield {yielded_value.name} : {self._render_type(yielded_value.type)}"
        )
        lines.append(self._indent(indent) + "}")
        env[carried_binding.name] = _RenderedValue(
            name=carried_binding.ssa_name,
            type=carried_binding.type,
        )
        return lines

    def _render_if(
        self,
        stmt: SemanticIfStmt,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
    ) -> list[str]:
        cond_lines: list[str] = []
        condition = self._lower_condition(stmt.condition, env, indent=indent, into=cond_lines)
        then_env = dict(env)
        else_env = dict(env)

        if not stmt.results:
            lines = list(cond_lines)
            lines.append(self._indent(indent) + f"scf.if {condition.name} {{")
            lines.extend(self._render_block(stmt.then_body, then_env, indent=indent + 2))
            if stmt.else_body:
                lines.append(self._indent(indent) + "} else {")
                lines.extend(self._render_block(stmt.else_body, else_env, indent=indent + 2))
            lines.append(self._indent(indent) + "}")
            return lines

        if len(stmt.results) != 1:
            raise NotImplementedError(
                "TileLang DSL v1 lowering currently supports at most one merged if/else binding"
            )

        result = stmt.results[0]
        lines = list(cond_lines)
        lines.append(
            self._indent(indent)
            + f"{result.result_binding.ssa_name} = scf.if {condition.name} -> "
            + f"({self._render_type(result.result_binding.type)}) {{"
        )
        lines.extend(self._render_block(stmt.then_body, then_env, indent=indent + 2))
        then_value = then_env.get(result.result_binding.name, then_env.get(result.then_binding.name))
        if then_value is None:
            then_value = _RenderedValue(result.then_binding.ssa_name, result.then_binding.type)
        lines.append(
            self._indent(indent + 2)
            + f"scf.yield {then_value.name} : {self._render_type(then_value.type)}"
        )
        lines.append(self._indent(indent) + "} else {")
        lines.extend(self._render_block(stmt.else_body, else_env, indent=indent + 2))
        else_value = else_env.get(result.result_binding.name, else_env.get(result.else_binding.name))
        if else_value is None:
            else_value = _RenderedValue(result.else_binding.ssa_name, result.else_binding.type)
        lines.append(
            self._indent(indent + 2)
            + f"scf.yield {else_value.name} : {self._render_type(else_value.type)}"
        )
        lines.append(self._indent(indent) + "}")
        env[result.result_binding.name] = _RenderedValue(
            name=result.result_binding.ssa_name,
            type=result.result_binding.type,
        )
        return lines

    def _lower_condition(
        self,
        expr: SemanticExpr,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
        into: list[str],
    ) -> _RenderedValue:
        value = self._lower_expr(expr, env, indent=indent, into=into)
        if isinstance(value.type, SemanticScalarType) and value.type.dtype.name == "i1":
            return value

        zero_type: SemanticType
        predicate: str
        if isinstance(value.type, SemanticIndexType):
            zero_type = SemanticIndexType()
            predicate = "arith.cmpi ne"
        elif isinstance(value.type, SemanticScalarType):
            zero_type = value.type
            if value.type.dtype.name in {"f16", "bf16", "f32"}:
                predicate = "arith.cmpf une"
            else:
                predicate = "arith.cmpi ne"
        else:
            raise NotImplementedError(f"unsupported if condition type {value.type!r}")

        zero = self._materialize_constant(0, zero_type)
        result_name = self._new_temp()
        into.append(
            self._indent(indent)
            + f"{result_name} = {predicate}, {value.name}, {zero} : {self._render_type(value.type)}"
        )
        return _RenderedValue(name=result_name, type=_I1_TYPE)

    def _lower_expr(
        self,
        expr: SemanticExpr,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
        desired_name: str | None = None,
        into: list[str] | None = None,
    ) -> _RenderedValue:
        if isinstance(expr, SemanticBindingRef):
            return env.get(expr.binding.name, _RenderedValue(expr.binding.ssa_name, expr.type))
        if isinstance(expr, SemanticLiteralExpr):
            if desired_name is not None and into is not None:
                into.append(
                    self._indent(indent)
                    + f"{desired_name} = arith.constant {self._format_constant(expr.value, expr.type)} : "
                    f"{self._render_type(expr.type)}"
                )
                return _RenderedValue(name=desired_name, type=expr.type)
            return _RenderedValue(
                name=self._materialize_constant(expr.value, expr.type),
                type=expr.type,
            )
        if isinstance(expr, SemanticSubscriptAccess):
            return self._lower_subscript_access(
                expr,
                env,
                indent=indent,
                desired_name=desired_name,
                into=into,
            )
        if isinstance(expr, SemanticBinaryExpr):
            if into is None:
                into = []
            lhs = self._lower_expr(expr.lhs, env, indent=indent, into=into)
            rhs = self._lower_expr(expr.rhs, env, indent=indent, into=into)
            result_name = desired_name or self._new_temp()
            into.append(
                self._indent(indent)
                + f"{result_name} = {self._render_binary_op(expr.op, expr.type)} "
                f"{lhs.name}, {rhs.name} : {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)
        if isinstance(expr, SemanticCallExpr):
            return self._lower_call_expr(expr, env, indent=indent, desired_name=desired_name, into=into)
        if isinstance(expr, SemanticAttributeAccess):
            raise NotImplementedError("bare shape attribute values are not materialized directly")
        if isinstance(expr, SemanticTensorSliceExpr):
            raise NotImplementedError("TensorView slices are only lowered through DMA statements in TileLang DSL v1")
        if isinstance(expr, SemanticSymbolExpr):
            raise NotImplementedError("symbol expressions are only lowered through specialized TileLang DSL ops")
        raise NotImplementedError(f"unsupported semantic expression {type(expr).__name__}")

    def _lower_call_expr(
        self,
        expr: SemanticCallExpr,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
        desired_name: str | None,
        into: list[str] | None,
    ) -> _RenderedValue:
        if expr.namespace != "pto":
            raise NotImplementedError(f"unsupported call namespace {expr.namespace!r}")
        if isinstance(expr.type, SemanticTupleType):
            raise NotImplementedError("multi-result call values must be assigned directly in TileLang DSL v1")
        if into is None:
            into = []
        result_name = desired_name or self._new_temp()

        if expr.name == "make_mask":
            dtype_expr, pattern_expr = expr.args
            if not self._is_dtype_meta_expr(dtype_expr):
                raise NotImplementedError("make_mask dtype lowering expects a dtype symbol")
            if not isinstance(pattern_expr, SemanticSymbolExpr) or not isinstance(pattern_expr.value, MaskPattern):
                raise NotImplementedError("make_mask pattern lowering expects a MaskPattern symbol")
            suffix = expr.type.granularity
            into.append(
                self._indent(indent)
                + f'{result_name} = pto.pset_{suffix} "{pattern_expr.value.value}" : {self._render_type(expr.type)}'
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name == "vlds":
            source = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            offset = self._lower_expr(expr.args[1], env, indent=indent, into=into)
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.vlds {source.name}[{offset.name}] : "
                + f"{self._render_type(source.type)} -> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name == "castptr":
            value = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            if isinstance(expr.type, SemanticPtrType) and isinstance(value.type, SemanticIndexType):
                value = self._coerce_rendered_to_i64(value, indent=indent, into=into)
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.castptr {value.name} : "
                + f"{self._render_type(value.type)} -> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name == "addptr":
            pointer = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            offset = self._lower_expr(expr.args[1], env, indent=indent, into=into)
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.addptr {pointer.name}, {offset.name} : "
                + f"{self._render_type(pointer.type)} -> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name in {"ppack", "punpack"}:
            value = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            part = self._render_string_literal(expr.args[1])
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.{expr.name} {value.name}, {part} : "
                + f"{self._render_type(value.type)} -> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name == "pnot":
            value = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            mask = self._lower_expr(expr.args[1], env, indent=indent, into=into)
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.pnot {value.name}, {mask.name} : "
                + f"{self._render_type(value.type)}, {self._render_type(mask.type)} -> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name == "psel":
            src0 = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            src1 = self._lower_expr(expr.args[1], env, indent=indent, into=into)
            mask = self._lower_expr(expr.args[2], env, indent=indent, into=into)
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.psel {src0.name}, {src1.name}, {mask.name} : "
                + f"{self._render_type(src0.type)}, {self._render_type(src1.type)}, {self._render_type(mask.type)} "
                + f"-> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name == "vcmp":
            lhs = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            rhs = self._lower_expr(expr.args[1], env, indent=indent, into=into)
            seed = self._lower_expr(expr.args[2], env, indent=indent, into=into)
            cmp_mode = self._render_string_literal(expr.args[3])
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.vcmp {lhs.name}, {rhs.name}, {seed.name}, {cmp_mode} : "
                + f"{self._render_type(lhs.type)}, {self._render_type(rhs.type)}, {self._render_type(seed.type)} "
                + f"-> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name == "vcmps":
            vector = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            scalar = self._lower_expr(expr.args[1], env, indent=indent, into=into)
            seed = self._lower_expr(expr.args[2], env, indent=indent, into=into)
            cmp_mode = self._render_string_literal(expr.args[3])
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.vcmps {vector.name}, {scalar.name}, {seed.name}, {cmp_mode} : "
                + f"{self._render_type(vector.type)}, {self._render_type(scalar.type)}, {self._render_type(seed.type)} "
                + f"-> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name == "vsel":
            src0 = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            src1 = self._lower_expr(expr.args[1], env, indent=indent, into=into)
            mask = self._lower_expr(expr.args[2], env, indent=indent, into=into)
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.vsel {src0.name}, {src1.name}, {mask.name} : "
                + f"{self._render_type(src0.type)}, {self._render_type(src1.type)}, {self._render_type(mask.type)} "
                + f"-> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name in {"vselr", "vselrv2"}:
            src0 = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            src1 = self._lower_expr(expr.args[1], env, indent=indent, into=into)
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.{expr.name} {src0.name}, {src1.name} : "
                + f"{self._render_type(src0.type)}, {self._render_type(src1.type)} -> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name in {"vintlvv2", "vdintlvv2"}:
            lhs = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            rhs = self._lower_expr(expr.args[1], env, indent=indent, into=into)
            part = self._render_string_literal(expr.args[2])
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.{expr.name} {lhs.name}, {rhs.name}, {part} : "
                + f"{self._render_type(lhs.type)}, {self._render_type(rhs.type)} -> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name in {"vabs", "vrelu", "vexp", "vnot"}:
            value = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            mask = self._lower_expr(expr.args[1], env, indent=indent, into=into)
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.{expr.name} {value.name}, {mask.name} : "
                + f"{self._render_type(value.type)}, {self._render_type(mask.type)} -> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name in {"vadd", "vsub", "vmul", "vdiv", "vmax", "vmin", "vand", "vor", "vxor"}:
            lhs = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            rhs = self._lower_expr(expr.args[1], env, indent=indent, into=into)
            mask = self._lower_expr(expr.args[2], env, indent=indent, into=into)
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.{expr.name} {lhs.name}, {rhs.name}, {mask.name} : "
                + f"{self._render_type(lhs.type)}, {self._render_type(rhs.type)}, {self._render_type(mask.type)} "
                + f"-> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        if expr.name in {"vadds", "vsubs", "vmuls", "vdivs", "vmaxs", "vmins"}:
            value = self._lower_expr(expr.args[0], env, indent=indent, into=into)
            scalar = self._lower_expr(expr.args[1], env, indent=indent, into=into)
            mask = self._lower_expr(expr.args[2], env, indent=indent, into=into)
            into.append(
                self._indent(indent)
                + f"{result_name} = pto.{expr.name} {value.name}, {scalar.name}, {mask.name} : "
                + f"{self._render_type(value.type)}, {self._render_type(scalar.type)}, {self._render_type(mask.type)} "
                + f"-> {self._render_type(expr.type)}"
            )
            return _RenderedValue(name=result_name, type=expr.type)

        raise NotImplementedError(f"unsupported pto call `{expr.name}` in lowering")

    def _render_string_literal(self, expr: SemanticExpr) -> str:
        if isinstance(expr, SemanticLiteralExpr) and isinstance(expr.value, str):
            escaped = expr.value.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        if isinstance(expr, SemanticBindingRef) and isinstance(expr.binding.value, str):
            escaped = expr.binding.value.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        raise NotImplementedError("expected a string literal for TileLang DSL advanced-family lowering")

    def _lower_to_i1(
        self,
        expr: SemanticExpr,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
        into: list[str],
    ) -> _RenderedValue:
        value = self._lower_expr(expr, env, indent=indent, into=into)
        if isinstance(value.type, SemanticScalarType) and value.type.dtype.name == "i1":
            return value
        raise NotImplementedError("expected an i1 operand during TileLang DSL v1 lowering")

    def _lower_to_i64(
        self,
        expr: SemanticExpr,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
        into: list[str],
    ) -> _RenderedValue:
        value = self._lower_expr(expr, env, indent=indent, into=into)
        return self._coerce_rendered_to_i64(value, indent=indent, into=into)

    def _coerce_rendered_to_i64(
        self,
        value: _RenderedValue,
        *,
        indent: int,
        into: list[str],
    ) -> _RenderedValue:
        if isinstance(value.type, SemanticScalarType) and value.type.dtype.name == "i64":
            return value
        if isinstance(value.type, SemanticIndexType):
            cast_name = self._new_temp()
            into.append(
                self._indent(indent)
                + f"{cast_name} = arith.index_castui {value.name} : index to i64"
            )
            return _RenderedValue(name=cast_name, type=_I64_TYPE)
        raise NotImplementedError("expected an i64 or index operand during TileLang DSL v1 lowering")

    def _lower_remaining_to_i32(
        self,
        expr: SemanticExpr,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
        into: list[str],
    ) -> _RenderedValue:
        value = self._lower_expr(expr, env, indent=indent, into=into)
        if isinstance(value.type, SemanticScalarType) and value.type.dtype.name == "i32":
            return value
        if isinstance(value.type, SemanticIndexType):
            cast_name = self._new_temp()
            into.append(
                self._indent(indent)
                + f"{cast_name} = arith.index_cast {value.name} : index to i32"
            )
            return _RenderedValue(name=cast_name, type=_I32_TYPE)
        raise NotImplementedError("tail make_mask lowering expects an i32 or index remaining operand")

    def _mask_suffix(self, ty: SemanticType) -> str:
        if not isinstance(ty, SemanticMaskType):
            raise NotImplementedError("tail make_mask lowering expects a mask result type")
        return ty.granularity

    def _is_dtype_meta_expr(self, expr: SemanticExpr) -> bool:
        if isinstance(expr, SemanticSymbolExpr):
            return isinstance(expr.value, ScalarType) and expr.type.kind == "dtype"
        if isinstance(expr, SemanticBindingRef):
            return (
                isinstance(expr.type, SemanticMetaType)
                and expr.type.kind == "dtype"
                and isinstance(expr.binding.value, ScalarType)
            )
        return False

    def _lower_subscript_access(
        self,
        expr: SemanticSubscriptAccess,
        env: dict[str, _RenderedValue],
        *,
        indent: int,
        desired_name: str | None,
        into: list[str] | None,
    ) -> _RenderedValue:
        value = self._extract_shape_subscript_value(expr, env)
        if isinstance(value, _RenderedValue):
            return value
        if desired_name is not None and into is not None:
            into.append(
                self._indent(indent)
                + f"{desired_name} = arith.constant {self._format_constant(value, expr.type)} : "
                f"{self._render_type(expr.type)}"
            )
            return _RenderedValue(name=desired_name, type=expr.type)
        return _RenderedValue(
            name=self._materialize_constant(value, expr.type),
            type=expr.type,
        )

    def _tensor_shape_binding_name(self, tensor_name: str, axis: int) -> str:
        return f"__shape_{tensor_name}_{axis}"

    def _extract_shape_subscript_value(
        self,
        expr: SemanticSubscriptAccess,
        env: dict[str, _RenderedValue],
    ) -> int | _RenderedValue:
        if not isinstance(expr.base, SemanticAttributeAccess):
            raise NotImplementedError("only shape indexing is supported in TileLang DSL v1 lowering")
        if expr.base.attr != "shape":
            raise NotImplementedError("only `.shape[...]` indexing is supported in TileLang DSL v1 lowering")
        if not isinstance(expr.index, SemanticLiteralExpr) or not isinstance(expr.index.value, int):
            raise NotImplementedError("shape indices must be integer literals in TileLang DSL v1 lowering")
        if not isinstance(expr.base.base, SemanticBindingRef):
            raise NotImplementedError("shape indexing expects a bound TensorView or Tile value")

        base_binding = expr.base.base.binding
        base_value = env.get(base_binding.name, _RenderedValue(base_binding.ssa_name, base_binding.type))
        base_type = base_value.type
        index = expr.index.value

        if isinstance(base_type, SemanticTileType):
            if base_type.shape is None:
                raise NotImplementedError("dynamic Tile shapes are not supported in TileLang DSL v1 lowering")
            return base_type.shape[index]

        if isinstance(base_type, SemanticTensorViewType):
            hidden_name = self._tensor_shape_binding_name(base_binding.name, index)
            hidden_value = env.get(hidden_name)
            if hidden_value is None:
                raise NotImplementedError(
                    f"missing TensorView shape binding for '{base_binding.name}.shape[{index}]'"
                )
            return hidden_value

        raise NotImplementedError("shape indexing expects a Tile or TensorView operand")

    def _materialize_constant(self, value: object, ty: SemanticType) -> str:
        cache_key = (self._render_type(ty), value)
        if cache_key in self._constant_cache:
            return self._constant_cache[cache_key]

        name = self._constant_name(value, ty)
        self._constant_cache[cache_key] = name
        self._constant_lines.append(
            self._indent(4)
            + f"{name} = arith.constant {self._format_constant(value, ty)} : {self._render_type(ty)}"
        )
        return name

    def _constant_name(self, value: object, ty: SemanticType) -> str:
        if isinstance(ty, SemanticIndexType):
            stem = f"c{value}"
        elif isinstance(ty, SemanticScalarType):
            if ty.dtype.name == "i1" and isinstance(value, bool):
                stem = "true" if value else "false"
            else:
                stem = f"c{value}_{ty.dtype.name}"
        else:
            stem = "cst"
        name = f"%{stem}"
        existing = {line.split(" = ", 1)[0].strip() for line in self._constant_lines}
        if name not in existing:
            return name
        suffix = 0
        while f"{name}_{suffix}" in existing:
            suffix += 1
        return f"{name}_{suffix}"

    def _format_constant(self, value: object, ty: SemanticType) -> str:
        if isinstance(ty, SemanticIndexType):
            return str(value)
        if isinstance(ty, SemanticScalarType):
            if ty.dtype.name == "i1" and isinstance(value, bool):
                return "true" if value else "false"
            return str(value)
        raise NotImplementedError(f"unsupported constant type {ty!r}")

    def _render_binary_op(self, op: str, ty: SemanticType) -> str:
        if isinstance(ty, (SemanticIndexType, SemanticScalarType)):
            if op == "add":
                return "arith.addi"
            if op == "sub":
                return "arith.subi"
            if op == "mul":
                return "arith.muli"
            if op == "floordiv":
                return "arith.floordivsi"
        raise NotImplementedError(f"unsupported binary op '{op}' for type {ty!r}")

    def _render_type(self, ty: SemanticType) -> str:
        if isinstance(ty, SemanticIndexType):
            return "index"
        if isinstance(ty, SemanticScalarType):
            return ty.dtype.name
        if isinstance(ty, SemanticPtrType):
            return f"!pto.ptr<{ty.element_dtype.name}, {ty.memory_space}>"
        if isinstance(ty, SemanticTensorViewType):
            return f"!pto.ptr<{ty.element_dtype.name}, gm>"
        if isinstance(ty, SemanticTileType):
            memory_space = ty.memory_space or "ub"
            return f"!pto.ptr<{ty.element_dtype.name}, {memory_space}>"
        if isinstance(ty, SemanticMaskType):
            return f"!pto.mask<{ty.granularity}>"
        if isinstance(ty, SemanticVRegType):
            return f"!pto.vreg<{ty.lanes}x{ty.element_dtype.name}>"
        raise NotImplementedError(f"unsupported semantic type {ty!r}")

    def _dtype_byte_width(self, dtype: ScalarType) -> int:
        widths = {
            "i8": 1,
            "i16": 2,
            "i32": 4,
            "i64": 8,
            "f16": 2,
            "bf16": 2,
            "f32": 4,
        }
        width = widths.get(dtype.name)
        if width is None:
            raise NotImplementedError(f"unsupported DMA dtype '{dtype.name}' in TileLang DSL v1 lowering")
        return width

    def _indent(self, indent: int) -> str:
        return " " * indent

    def _new_temp(self) -> str:
        name = f"%tmp_{self._temp_counter}"
        self._temp_counter += 1
        return name


def lower_semantic_kernel(kernel: SemanticKernel) -> AuthoringModule:
    """Lower the semantic model to the current authoring-form VPTO builder."""

    return AuthoringModule(kernel=kernel)


__all__ = ["AuthoringModule", "lower_semantic_kernel"]

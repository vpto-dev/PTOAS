"""Kernel descriptor surface for TileLang DSL v1."""

from __future__ import annotations

import os
import inspect
import ast
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from .types import (
    MemorySpace,
    PointerType,
    ScalarType,
    TensorView,
    Tile,
    TileConfig,
    TileSpecialization,
    TypeVariable,
    WildcardType,
)
from .frontend_ast import build_frontend_kernel_node
from .lowering import lower_semantic_kernel
from .semantic import analyze_frontend_kernel
from .support_matrix import (
    ADVANCED_EXPR_PTO_CALLS,
    ADVANCED_TOPLEVEL_PTO_CALLS,
    ADVANCED_VECSCOPE_PTO_CALLS,
    DEFERRED_PTO_SURFACES,
    SUPPORTED_TOPLEVEL_PTO_CALLS,
    SUPPORTED_VECSCOPE_PTO_CALLS,
    advanced_mode_message,
    deferred_surface_message,
)


_UNSET = object()
_PTOAS_BIN_ENV = "PTOAS_BIN"


def _validate_dtype_pattern(dtype: Any) -> ScalarType | WildcardType | TypeVariable:
    if isinstance(dtype, (ScalarType, WildcardType, TypeVariable)):
        return dtype
    raise TypeError(f"unsupported dtype pattern {dtype!r}")


class TileLangFrontendError(ValueError):
    """Source-located frontend diagnostic for TileLang DSL."""

    def __init__(self, path: str, line: int, column: int, message: str):
        self.path = path
        self.line = line
        self.column = column
        self.message = message
        super().__init__(f"{path}:{line}:{column}: {message}")


@dataclass(frozen=True)
class _FunctionSourceInfo:
    path: str
    start_line: int
    function_def: ast.FunctionDef

    def location(self, node: ast.AST) -> tuple[int, int]:
        line = self.start_line + getattr(node, "lineno", 1) - 1
        column = getattr(node, "col_offset", 0) + 1
        return line, column

    def error(self, node: ast.AST, message: str) -> TileLangFrontendError:
        line, column = self.location(node)
        return TileLangFrontendError(self.path, line, column, message)

    def parameter_node(self, param_name: str) -> ast.AST | None:
        for arg in self.function_def.args.args:
            if arg.arg == param_name:
                return arg.annotation or arg
        return None


class _KernelBodyValidator(ast.NodeVisitor):
    def __init__(self, source_info: _FunctionSourceInfo, *, advanced_enabled: bool):
        self.source_info = source_info
        self.advanced_enabled = advanced_enabled
        self._vecscope_depth = 0

    def validate(self) -> None:
        for stmt in self.source_info.function_def.body:
            self.visit(stmt)

    def visit_While(self, node: ast.While) -> None:
        raise self.source_info.error(node, "unsupported Python syntax `while` in TileLang DSL v1")

    def visit_ListComp(self, node: ast.ListComp) -> None:
        raise self.source_info.error(
            node, "unsupported Python syntax `list comprehension` in TileLang DSL v1"
        )

    def visit_DictComp(self, node: ast.DictComp) -> None:
        raise self.source_info.error(
            node, "unsupported Python syntax `dict comprehension` in TileLang DSL v1"
        )

    def visit_SetComp(self, node: ast.SetComp) -> None:
        raise self.source_info.error(
            node, "unsupported Python syntax `set comprehension` in TileLang DSL v1"
        )

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        raise self.source_info.error(
            node, "unsupported Python syntax `generator expression` in TileLang DSL v1"
        )

    def visit_For(self, node: ast.For) -> None:
        if not isinstance(node.target, ast.Name):
            raise self.source_info.error(node.target, "for target must be a single name")
        if not isinstance(node.iter, ast.Call) or not isinstance(node.iter.func, ast.Name):
            raise self.source_info.error(node.iter, "only Python range(lb, ub, step) loops are supported")
        if node.iter.func.id != "range":
            raise self.source_info.error(node.iter, "only Python range(lb, ub, step) loops are supported")
        if len(node.iter.args) != 3:
            raise self.source_info.error(node.iter, "range() expects exactly 3 arguments in TileLang DSL v1")
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_If(self, node: ast.If) -> None:
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_With(self, node: ast.With) -> None:
        if len(node.items) != 1:
            raise self.source_info.error(node, "only single with item is supported in TileLang DSL v1")
        item = node.items[0]
        if not isinstance(item.context_expr, ast.Call):
            raise self.source_info.error(item.context_expr, "with context must be a call in TileLang DSL v1")
        if not (
            isinstance(item.context_expr.func, ast.Attribute)
            and isinstance(item.context_expr.func.value, ast.Name)
            and item.context_expr.func.value.id == "pto"
            and item.context_expr.func.attr == "strict_vecscope"
        ):
            raise self.source_info.error(
                item.context_expr,
                "only pto.strict_vecscope is supported as a with-context in TileLang DSL v1",
            )
        if not isinstance(item.optional_vars, ast.Tuple):
            raise self.source_info.error(item, "pto.strict_vecscope requires tuple binding in 'as'")
        for elt in item.optional_vars.elts:
            if not isinstance(elt, ast.Name):
                raise self.source_info.error(elt, "pto.strict_vecscope bindings must be names")
        self._vecscope_depth += 1
        try:
            for stmt in node.body:
                self.visit(stmt)
        finally:
            self._vecscope_depth -= 1

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id == "pto" and node.func.attr in SUPPORTED_TOPLEVEL_PTO_CALLS:
                return
            if node.func.value.id == "pto" and node.func.attr in SUPPORTED_VECSCOPE_PTO_CALLS:
                if self.advanced_enabled:
                    return
                if self._vecscope_depth <= 0:
                    raise self.source_info.error(
                        node,
                        f"vector op surface `pto.{node.func.attr}` requires explicit pto.strict_vecscope in TileLang DSL v1",
                    )
                return
            if node.func.value.id == "pto" and node.func.attr in ADVANCED_VECSCOPE_PTO_CALLS:
                if self.advanced_enabled:
                    return
                raise self.source_info.error(
                    node,
                    advanced_mode_message(node.func.attr),
                )
            if node.func.value.id == "pto" and (
                node.func.attr in ADVANCED_EXPR_PTO_CALLS
                or node.func.attr in ADVANCED_TOPLEVEL_PTO_CALLS
            ):
                if self.advanced_enabled:
                    return
                raise self.source_info.error(
                    node,
                    advanced_mode_message(node.func.attr),
                )
            if node.func.value.id == "pto" and node.func.attr in DEFERRED_PTO_SURFACES:
                raise self.source_info.error(
                    node,
                    deferred_surface_message(node.func.attr),
                )
            if node.func.value.id == "pto":
                raise self.source_info.error(
                    node,
                    f"unsupported op surface `pto.{node.func.attr}` in TileLang DSL v1",
                )
            raise self.source_info.error(
                node,
                f"arbitrary external call `{node.func.value.id}.{node.func.attr}` is not supported "
                "in TileLang DSL v1",
            )

        if isinstance(node.func, ast.Name):
            if node.func.id == "range":
                return
            raise self.source_info.error(
                node,
                f"arbitrary external call `{node.func.id}` is not supported in TileLang DSL v1",
            )

        raise self.source_info.error(
            node,
            "unsupported call surface in TileLang DSL v1",
        )


def _load_function_source_info(py_fn: Callable[..., Any]) -> _FunctionSourceInfo | None:
    try:
        source_lines, start_line = inspect.getsourcelines(py_fn)
        path = inspect.getsourcefile(py_fn) or inspect.getfile(py_fn)
    except (OSError, IOError, TypeError):
        return None

    source = textwrap.dedent("".join(source_lines))
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == py_fn.__name__:
            return _FunctionSourceInfo(path=path, start_line=start_line, function_def=node)
    return None


def _validate_function_body(
    source_info: _FunctionSourceInfo | None,
    *,
    advanced_enabled: bool,
) -> None:
    if source_info is None:
        return
    _KernelBodyValidator(
        source_info,
        advanced_enabled=advanced_enabled,
    ).validate()


def _raise_tile_param_error(
    source_info: _FunctionSourceInfo | None,
    param_name: str,
    message: str,
    fallback_exception: type[Exception] = ValueError,
) -> None:
    if source_info is not None:
        node = source_info.parameter_node(param_name)
        if node is not None:
            raise source_info.error(node, message)
    raise fallback_exception(message)


def _freeze_dtypes(dtypes: Any) -> tuple[tuple[Any, ...], ...]:
    if not isinstance(dtypes, (list, tuple)):
        raise TypeError("dtypes must be a sequence of signature tuples")

    frozen_signatures = []
    for signature in dtypes:
        if not isinstance(signature, (list, tuple)):
            raise TypeError("each dtypes entry must be a signature tuple")
        frozen_signature = tuple(signature)
        for dtype in frozen_signature:
            _validate_dtype_pattern(dtype)
        frozen_signatures.append(frozen_signature)

    if not frozen_signatures:
        raise ValueError("dtypes must contain at least one signature tuple")

    return tuple(frozen_signatures)


@dataclass(frozen=True)
class BoundKernelParameter:
    """One parameter after v1 monomorphic dtype binding."""

    name: str
    kind: str
    annotation: Any
    dtype: ScalarType

    @property
    def element_dtype(self) -> ScalarType | None:
        if self.kind in ("tensorview", "tile", "ptr"):
            return self.dtype
        return None


@dataclass(frozen=True)
class KernelParameterSpec:
    """One validated Python function parameter before dtype selection."""

    name: str
    kind: str
    annotation: Any


@dataclass(frozen=True)
class VKernelDescriptor:
    """Descriptor returned by `@tilelang_dsl.vkernel`."""

    target: str
    op: str
    dtypes: tuple[tuple[Any, ...], ...]
    name: str
    verify_enabled: bool
    advanced_enabled: bool
    _parameter_specs: tuple[KernelParameterSpec, ...]
    _py_fn: Callable[..., Any] = field(repr=False)
    _source_info: _FunctionSourceInfo | None = field(repr=False, compare=False, default=None)
    specializations: tuple[tuple[str, TileSpecialization], ...] = ()
    constraints: tuple[Callable[[Mapping[str, Any]], Any], ...] = field(default=(), repr=False)
    priority: int = 0
    _selected_dtype_signature: tuple[ScalarType, ...] | None = None
    _parameters: tuple[BoundKernelParameter, ...] | None = field(default=None, repr=False)

    @property
    def py_fn(self) -> Callable[..., Any]:
        return self._py_fn

    @property
    def dtype_signature(self) -> tuple[ScalarType, ...]:
        if self._selected_dtype_signature is None:
            raise ValueError(
                "descriptor requires pto.select_kernel(...) to choose a concrete dtype signature "
                "before materialization"
            )
        return self._selected_dtype_signature

    @property
    def parameters(self) -> tuple[BoundKernelParameter, ...]:
        if self._parameters is None:
            raise ValueError(
                "descriptor requires pto.select_kernel(...) to bind concrete parameter dtypes "
                "before materialization"
            )
        return self._parameters

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "op": self.op,
            "dtypes": self.dtypes,
            "name": self.name,
            "verify": self.verify_enabled,
            "advanced": self.advanced_enabled,
            "constraints": self.constraints,
            "priority": self.priority,
        }

    @property
    def tile_parameters(self) -> tuple[BoundKernelParameter, ...]:
        return tuple(param for param in self.parameters if param.kind == "tile")

    @property
    def specializations_by_name(self) -> dict[str, TileSpecialization]:
        return dict(self.specializations)

    def _tile_parameter_names(self) -> tuple[str, ...]:
        return tuple(param.name for param in self._parameter_specs if param.kind == "tile")

    def _bind_selected_dtype_signature(
        self,
        dtype_signature: tuple[ScalarType, ...],
    ) -> "VKernelDescriptor":
        bound_parameters = _bind_parameters(self._parameter_specs, dtype_signature)
        return VKernelDescriptor(
            target=self.target,
            op=self.op,
            dtypes=self.dtypes,
            name=self.name,
            verify_enabled=self.verify_enabled,
            advanced_enabled=self.advanced_enabled,
            _parameter_specs=self._parameter_specs,
            _py_fn=self._py_fn,
            _source_info=self._source_info,
            specializations=self.specializations,
            constraints=self.constraints,
            priority=self.priority,
            _selected_dtype_signature=dtype_signature,
            _parameters=bound_parameters,
        )

    def specialize(self, **bindings: Any) -> "VKernelDescriptor":
        tile_param_names = set(self._tile_parameter_names())
        if not tile_param_names:
            if bindings:
                unknown = ", ".join(sorted(bindings))
                raise TypeError(
                    f"specialize() received bindings for non-Tile parameters: {unknown}"
                )
            return self

        unknown = sorted(set(bindings) - tile_param_names)
        if unknown:
            unknown_names = ", ".join(unknown)
            raise TypeError(
                f"specialize() only accepts bare Tile parameters; got: {unknown_names}"
            )

        updated = self.specializations_by_name
        for name, binding in bindings.items():
            updated[name] = _coerce_tile_specialization(name, binding, self._source_info)

        return VKernelDescriptor(
            target=self.target,
            op=self.op,
            dtypes=self.dtypes,
            name=self.name,
            verify_enabled=self.verify_enabled,
            advanced_enabled=self.advanced_enabled,
            _parameter_specs=self._parameter_specs,
            _source_info=self._source_info,
            specializations=tuple(sorted(updated.items())),
            constraints=self.constraints,
            priority=self.priority,
            _selected_dtype_signature=self._selected_dtype_signature,
            _parameters=self._parameters,
            _py_fn=self._py_fn,
        )

    def _require_specialized_tiles(self, api_name: str) -> None:
        tile_names = list(self._tile_parameter_names())
        if not tile_names:
            return

        specialized = self.specializations_by_name
        missing = [name for name in tile_names if name not in specialized]
        if missing:
            missing_names = ", ".join(missing)
            _raise_tile_param_error(
                self._source_info,
                missing[0],
                f"{api_name}() requires specialize() bindings for bare Tile parameters: "
                f"{missing_names}",
            )

    def _build_authoring_module(self):
        self.parameters
        frontend_kernel = build_frontend_kernel_node(self)
        semantic_kernel = analyze_frontend_kernel(frontend_kernel)
        return lower_semantic_kernel(semantic_kernel)

    def mlir_text(self) -> str:
        self._require_specialized_tiles("mlir_text")
        return self._build_authoring_module().render()

    def mlir_module(self) -> "MaterializedMLIRModule":
        self._require_specialized_tiles("mlir_module")
        return MaterializedMLIRModule(text=self.mlir_text(), target=self.target)

    def verify(self, *, ptoas_bin: str | Path | None = None) -> "VerificationResult":
        self._require_specialized_tiles("verify")
        return self.mlir_module().verify(ptoas_bin=ptoas_bin)

    def emit(self, path: str | Path) -> None:
        self._require_specialized_tiles("emit")
        output_path = Path(path)
        output_path.write_text(self.mlir_text(), encoding="utf-8")


class KernelRegistry:
    """Explicit registry for TileLang kernel descriptors."""

    def __init__(self, descriptors: tuple[VKernelDescriptor, ...] = ()):
        self._descriptors: list[VKernelDescriptor] = []
        for descriptor in descriptors:
            self.register(descriptor)

    def register(self, descriptor: VKernelDescriptor) -> VKernelDescriptor:
        if not isinstance(descriptor, VKernelDescriptor):
            raise TypeError("KernelRegistry.register() expects a VKernelDescriptor")
        self._descriptors.append(descriptor)
        return descriptor

    @property
    def descriptors(self) -> tuple[VKernelDescriptor, ...]:
        return tuple(self._descriptors)

    def __iter__(self):
        return iter(self._descriptors)

    def __len__(self) -> int:
        return len(self._descriptors)


_DEFAULT_KERNEL_REGISTRY = KernelRegistry()


@dataclass(frozen=True)
class MaterializedMLIRModule:
    text: str
    target: str = "a5"

    def __str__(self) -> str:
        return self.text

    def verify(self, *, ptoas_bin: str | Path | None = None) -> "VerificationResult":
        return _run_ptoas_verifier(self.text, target=self.target, ptoas_bin=ptoas_bin)


@dataclass(frozen=True)
class VerificationResult:
    status: str
    available: bool
    passed: bool
    message: str
    command: tuple[str, ...] | None = None
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.available and self.passed

    def __bool__(self) -> bool:
        return self.ok


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_ptoas_bin(ptoas_bin: str | Path | None) -> Path:
    if ptoas_bin is not None:
        return Path(ptoas_bin)
    env_path = os.environ.get(_PTOAS_BIN_ENV)
    if env_path:
        return Path(env_path)
    return _repo_root() / "build/tools/ptoas/ptoas"


def _unavailable_result(
    message: str,
    *,
    command: tuple[str, ...] | None = None,
    stderr: str = "",
) -> VerificationResult:
    return VerificationResult(
        status="unavailable",
        available=False,
        passed=False,
        message=message,
        command=command,
        stderr=stderr,
    )


def _failed_result(
    message: str,
    *,
    command: tuple[str, ...],
    returncode: int,
    stdout: str,
    stderr: str,
) -> VerificationResult:
    return VerificationResult(
        status="failed",
        available=True,
        passed=False,
        message=message,
        command=command,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _passed_result(
    *,
    command: tuple[str, ...],
    stdout: str,
    stderr: str,
) -> VerificationResult:
    return VerificationResult(
        status="passed",
        available=True,
        passed=True,
        message="generated IR passed the repo VPTO authoring-stage legality verifier",
        command=command,
        returncode=0,
        stdout=stdout,
        stderr=stderr,
    )


def _is_verifier_unavailable_process_failure(stderr: str) -> bool:
    lowered = stderr.lower()
    return (
        "error while loading shared libraries" in lowered
        or "cannot open shared object file" in lowered
        or "image not found" in lowered
        or "dll load failed" in lowered
    )


def _run_ptoas_verifier(
    mlir_text: str,
    *,
    target: str,
    ptoas_bin: str | Path | None,
) -> VerificationResult:
    binary = _resolve_ptoas_bin(ptoas_bin)
    command = (
        str(binary),
        "--pto-arch",
        target,
        "--pto-backend=vpto",
        "--emit-vpto",
    )
    if not binary.exists():
        return _unavailable_result(
            f"verifier unavailable: missing ptoas binary at {binary}",
            command=command,
        )
    if not os.access(binary, os.X_OK):
        return _unavailable_result(
            f"verifier unavailable: ptoas binary is not executable: {binary}",
            command=command,
        )

    try:
        with tempfile.TemporaryDirectory(prefix="tilelang_dsl_verify_") as tmpdir:
            tmpdir_path = Path(tmpdir)
            input_path = tmpdir_path / "kernel.mlir"
            output_path = tmpdir_path / "verified.mlir"
            input_path.write_text(mlir_text, encoding="utf-8")
            full_command = command + (str(input_path), "-o", str(output_path))
            completed = subprocess.run(
                full_command,
                cwd=_repo_root(),
                text=True,
                capture_output=True,
                check=False,
            )
    except OSError as exc:
        return _unavailable_result(
            f"verifier unavailable: failed to execute ptoas: {exc}",
            command=command,
            stderr=str(exc),
        )

    stderr = completed.stderr.strip()
    stdout = completed.stdout.strip()
    if completed.returncode == 0:
        return _passed_result(command=full_command, stdout=stdout, stderr=stderr)
    if _is_verifier_unavailable_process_failure(stderr):
        return _unavailable_result(
            "verifier unavailable: failed to launch repo ptoas legality path",
            command=full_command,
            stderr=stderr,
        )
    message = stderr or stdout or "generated IR failed the repo VPTO authoring-stage legality verifier"
    return _failed_result(
        message,
        command=full_command,
        returncode=completed.returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _validate_target(target: str) -> str:
    if not isinstance(target, str):
        raise TypeError("target must be a string")
    if target != "a5":
        raise ValueError("TileLang DSL v1 currently only supports target='a5'")
    return target


def _validate_op(op: Any) -> str:
    if not isinstance(op, str) or not op:
        raise TypeError("op must be a non-empty string")
    return op


def _validate_name(py_fn: Callable[..., Any], name: Any) -> str:
    if name is None:
        return py_fn.__name__
    if not isinstance(name, str) or not name:
        raise TypeError("name must be a non-empty string")
    return name


def _validate_verify(verify: Any) -> bool:
    if not isinstance(verify, bool):
        raise TypeError("verify must be a bool")
    return verify


def _validate_advanced(advanced: Any) -> bool:
    if not isinstance(advanced, bool):
        raise TypeError("advanced must be a bool")
    return advanced


def _validate_constraints(constraints: Any) -> tuple[Callable[[Mapping[str, Any]], Any], ...]:
    if constraints is _UNSET:
        return ()
    if not isinstance(constraints, (list, tuple)):
        raise TypeError("constraints must be a sequence of predicate callables")

    frozen_constraints = []
    for index, constraint in enumerate(constraints):
        if not callable(constraint):
            raise TypeError(f"constraints[{index}] must be callable")
        frozen_constraints.append(constraint)
    return tuple(frozen_constraints)


def _validate_priority(priority: Any) -> int:
    if priority is _UNSET:
        return 0
    if isinstance(priority, bool) or not isinstance(priority, int):
        raise TypeError("priority must be an int")
    return priority


def _coerce_memory_space(value: Any, param_name: str) -> MemorySpace:
    if isinstance(value, MemorySpace):
        return value
    if isinstance(value, str):
        normalized = value.strip().upper()
        try:
            return MemorySpace[normalized]
        except KeyError as exc:
            raise ValueError(
                f"specialization for '{param_name}' uses unsupported memory_space {value!r}"
            ) from exc
    raise TypeError(
        f"specialization for '{param_name}' must provide MemorySpace or string memory_space"
    )


def _coerce_tile_config(value: Any, param_name: str) -> TileConfig | None:
    if value is None:
        return None
    if isinstance(value, TileConfig):
        return value
    if isinstance(value, dict):
        return TileConfig.from_mapping(value)
    raise TypeError(
        f"specialization for '{param_name}' must provide TileConfig, dict, or None for config"
    )


def _coerce_tile_specialization(
    param_name: str,
    binding: Any,
    source_info: _FunctionSourceInfo | None,
) -> TileSpecialization:
    if isinstance(binding, TileSpecialization):
        spec = binding
    elif isinstance(binding, dict):
        if "shape" not in binding:
            _raise_tile_param_error(
                source_info,
                param_name,
                f"specialization for '{param_name}' must provide a static physical Tile shape",
                TypeError,
            )
        if "memory_space" not in binding:
            _raise_tile_param_error(
                source_info,
                param_name,
                f"specialization for '{param_name}' must provide memory_space",
                TypeError,
            )
        spec = TileSpecialization(
            shape=tuple(binding["shape"]),
            memory_space=_coerce_memory_space(binding["memory_space"], param_name),
            config=_coerce_tile_config(binding.get("config"), param_name),
        )
    else:
        _raise_tile_param_error(
            source_info,
            param_name,
            f"specialization for '{param_name}' must be a TileSpecialization or dict",
            TypeError,
        )

    if not spec.shape:
        _raise_tile_param_error(
            source_info,
            param_name,
            f"illegal Tile profile for '{param_name}': shape must be non-empty",
        )
    for dim in spec.shape:
        if not isinstance(dim, int) or isinstance(dim, bool):
            _raise_tile_param_error(
                source_info,
                param_name,
                f"dynamic physical Tile shape is not supported for '{param_name}'",
                TypeError,
            )
        if dim <= 0:
            _raise_tile_param_error(
                source_info,
                param_name,
                f"illegal Tile profile for '{param_name}': dimensions must be positive",
            )
    if len(spec.shape) not in (1, 2):
        _raise_tile_param_error(
            source_info,
            param_name,
            f"illegal Tile profile for '{param_name}': v1 only supports rank-1 or rank-2 Tile shapes",
        )
    if spec.memory_space != MemorySpace.UB:
        _raise_tile_param_error(
            source_info,
            param_name,
            f"illegal Tile profile for '{param_name}': v1 only supports MemorySpace.UB",
        )
    return spec


def _validate_scalar_dtype(dtype: Any, param_name: str) -> ScalarType:
    if not isinstance(dtype, ScalarType):
        raise TypeError(
            f"dtypes entry for parameter '{param_name}' must be a TileLang scalar dtype"
        )
    return dtype


def _freeze_operand_types(operand_types: Any) -> tuple[ScalarType, ...]:
    if not isinstance(operand_types, (list, tuple)):
        raise TypeError("operand_types must be a sequence of TileLang scalar dtypes")
    return tuple(_validate_scalar_dtype(dtype, f"operand_types[{index}]") for index, dtype in enumerate(operand_types))


def _matches_wildcard(pattern: WildcardType, actual: ScalarType) -> bool:
    if pattern.name == "AnyType":
        return True
    if pattern.name == "AnyFloat":
        return actual.name in {"f16", "bf16", "f32"}
    if pattern.name == "AnyInt":
        return actual.name.startswith("i")
    if pattern.name == "AnyMask":
        return actual.name == "i1"
    raise TypeError(f"unsupported wildcard matcher {pattern.name!r}")


def _match_dtype_signature(
    dtype_signature: tuple[Any, ...],
    operand_types: tuple[ScalarType, ...],
) -> tuple[ScalarType, ...] | None:
    if len(dtype_signature) != len(operand_types):
        return None

    typevar_bindings: dict[str, ScalarType] = {}
    for pattern, actual in zip(dtype_signature, operand_types):
        if isinstance(pattern, ScalarType):
            if pattern != actual:
                return None
            continue
        if isinstance(pattern, WildcardType):
            if not _matches_wildcard(pattern, actual):
                return None
            continue
        if isinstance(pattern, TypeVariable):
            bound = typevar_bindings.get(pattern.name)
            if bound is None:
                typevar_bindings[pattern.name] = actual
                continue
            if bound != actual:
                return None
            continue
        raise TypeError(f"unsupported dtype pattern {pattern!r}")
    return operand_types


def _match_descriptor_dtype_signature(
    descriptor: VKernelDescriptor,
    operand_types: tuple[ScalarType, ...],
) -> tuple[ScalarType, ...] | None:
    for dtype_signature in descriptor.dtypes:
        matched = _match_dtype_signature(dtype_signature, operand_types)
        if matched is not None:
            return matched
    return None


def _validate_parameter_spec(param: inspect.Parameter) -> KernelParameterSpec:
    if param.kind not in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        raise TypeError(
            f"parameter '{param.name}' uses unsupported parameter kind for TileLang DSL v1"
        )
    if param.default is not inspect._empty:
        raise TypeError(
            f"parameter '{param.name}' must not declare a default value in TileLang DSL v1"
        )
    if param.annotation is inspect._empty:
        raise TypeError(
            f"parameter '{param.name}' must declare a TileLang DSL type annotation"
        )

    annotation = param.annotation
    if annotation is TensorView:
        return KernelParameterSpec(
            name=param.name,
            kind="tensorview",
            annotation=annotation,
        )
    if annotation is Tile:
        return KernelParameterSpec(
            name=param.name,
            kind="tile",
            annotation=annotation,
        )
    if isinstance(annotation, PointerType):
        return KernelParameterSpec(
            name=param.name,
            kind="ptr",
            annotation=annotation,
        )
    if isinstance(annotation, ScalarType):
        return KernelParameterSpec(
            name=param.name,
            kind="scalar",
            annotation=annotation,
        )

    raise TypeError(
        f"parameter '{param.name}' uses unsupported annotation {annotation!r}"
    )


def _collect_parameter_specs(py_fn: Callable[..., Any]) -> tuple[KernelParameterSpec, ...]:
    signature = inspect.signature(py_fn)
    return tuple(_validate_parameter_spec(param) for param in signature.parameters.values())


def _validate_dtype_arity(
    parameter_specs: tuple[KernelParameterSpec, ...],
    dtypes: tuple[tuple[Any, ...], ...],
) -> None:
    for dtype_signature in dtypes:
        if len(dtype_signature) != len(parameter_specs):
            raise ValueError(
                "each dtypes signature must match the decorated function parameter count"
            )


def _bind_parameter(
    param_spec: KernelParameterSpec,
    dtype: Any,
) -> BoundKernelParameter:
    scalar_dtype = _validate_scalar_dtype(dtype, param_spec.name)
    if param_spec.kind == "tensorview":
        return BoundKernelParameter(
            name=param_spec.name,
            kind=param_spec.kind,
            annotation=param_spec.annotation,
            dtype=scalar_dtype,
        )
    if param_spec.kind == "tile":
        return BoundKernelParameter(
            name=param_spec.name,
            kind=param_spec.kind,
            annotation=param_spec.annotation,
            dtype=scalar_dtype,
        )
    if param_spec.kind == "ptr":
        if param_spec.annotation.element_dtype != scalar_dtype:
            raise TypeError(
                f"pointer parameter '{param_spec.name}' annotation {param_spec.annotation!r} "
                f"does not match selected dtype {scalar_dtype!r}"
            )
        return BoundKernelParameter(
            name=param_spec.name,
            kind=param_spec.kind,
            annotation=param_spec.annotation,
            dtype=scalar_dtype,
        )
    if param_spec.annotation != scalar_dtype:
        raise TypeError(
            f"scalar parameter '{param_spec.name}' annotation {param_spec.annotation!r} "
            f"does not match selected dtype {scalar_dtype!r}"
        )
    return BoundKernelParameter(
        name=param_spec.name,
        kind=param_spec.kind,
        annotation=param_spec.annotation,
        dtype=scalar_dtype,
    )


def _bind_parameters(
    parameter_specs: tuple[KernelParameterSpec, ...],
    dtype_signature: tuple[ScalarType, ...],
) -> tuple[BoundKernelParameter, ...]:
    if len(dtype_signature) != len(parameter_specs):
        raise ValueError(
            "selected dtype signature must match the decorated function parameter count"
        )
    return tuple(
        _bind_parameter(param_spec, dtype)
        for param_spec, dtype in zip(parameter_specs, dtype_signature)
    )


def _build_descriptor(
    py_fn: Callable[..., Any],
    *,
    target: str,
    op: Any,
    dtypes: Any,
    name: Any,
    verify: Any,
    advanced: Any,
    constraints: Any,
    priority: Any,
) -> VKernelDescriptor:
    if not callable(py_fn):
        raise TypeError("@vkernel can only decorate callables")

    source_info = _load_function_source_info(py_fn)
    advanced_enabled = _validate_advanced(advanced)
    _validate_function_body(source_info, advanced_enabled=advanced_enabled)
    frozen_dtypes = _freeze_dtypes(dtypes)
    parameter_specs = _collect_parameter_specs(py_fn)
    _validate_dtype_arity(parameter_specs, frozen_dtypes)

    selected_dtype_signature: tuple[ScalarType, ...] | None = None
    bound_parameters: tuple[BoundKernelParameter, ...] | None = None
    if len(frozen_dtypes) == 1 and all(isinstance(dtype, ScalarType) for dtype in frozen_dtypes[0]):
        selected_dtype_signature = tuple(frozen_dtypes[0])
        bound_parameters = _bind_parameters(parameter_specs, selected_dtype_signature)

    return VKernelDescriptor(
        target=_validate_target(target),
        op=_validate_op(op),
        dtypes=frozen_dtypes,
        name=_validate_name(py_fn, name),
        verify_enabled=_validate_verify(verify),
        advanced_enabled=advanced_enabled,
        _parameter_specs=parameter_specs,
        _py_fn=py_fn,
        _source_info=source_info,
        constraints=_validate_constraints(constraints),
        priority=_validate_priority(priority),
        _selected_dtype_signature=selected_dtype_signature,
        _parameters=bound_parameters,
    )


def _evaluate_constraints(
    descriptor: VKernelDescriptor,
    context_attrs: Mapping[str, Any],
) -> bool:
    for index, constraint in enumerate(descriptor.constraints):
        try:
            result = constraint(context_attrs)
        except Exception as exc:
            raise TypeError(
                f"constraint {index} for kernel {descriptor.name!r} raised {type(exc).__name__}: {exc}"
            ) from exc
        if not result:
            return False
    return True


def _format_descriptor_identity(descriptor: VKernelDescriptor) -> str:
    dtype_signature = descriptor._selected_dtype_signature
    if dtype_signature is None:
        dtype_signature = tuple("?" for _ in descriptor.dtypes[0]) if descriptor.dtypes else ()
    return f"{descriptor.name}(priority={descriptor.priority}, dtypes={dtype_signature!r})"


def select_kernel(
    target: str,
    op: str,
    operand_types: Any,
    context_attrs: Mapping[str, Any] | None = None,
    registry: KernelRegistry | None = None,
) -> VKernelDescriptor:
    """Select one registered kernel descriptor for the given query."""

    normalized_target = _validate_target(target)
    normalized_op = _validate_op(op)
    normalized_operand_types = _freeze_operand_types(operand_types)

    if context_attrs is None:
        normalized_context_attrs: dict[str, Any] = {}
    elif isinstance(context_attrs, Mapping):
        normalized_context_attrs = dict(context_attrs)
    else:
        raise TypeError("context_attrs must be a mapping or None")

    active_registry = _DEFAULT_KERNEL_REGISTRY if registry is None else registry
    if not isinstance(active_registry, KernelRegistry):
        raise TypeError("registry must be a KernelRegistry or None")

    type_matched_candidates = [
        descriptor._bind_selected_dtype_signature(matched_signature)
        if descriptor._selected_dtype_signature != matched_signature
        else descriptor
        for descriptor in active_registry
        if descriptor.target == normalized_target
        and descriptor.op == normalized_op
        for matched_signature in (_match_descriptor_dtype_signature(descriptor, normalized_operand_types),)
        if matched_signature is not None
    ]

    if not type_matched_candidates:
        raise LookupError(
            "select_kernel() found no registered kernel for "
            f"target={normalized_target!r}, op={normalized_op!r}, operand_types={normalized_operand_types!r}"
        )

    constrained_candidates = [
        descriptor
        for descriptor in type_matched_candidates
        if _evaluate_constraints(descriptor, normalized_context_attrs)
    ]
    if not constrained_candidates:
        raise LookupError(
            "select_kernel() found no registered kernel after constraint evaluation for "
            f"target={normalized_target!r}, op={normalized_op!r}, operand_types={normalized_operand_types!r}"
        )

    highest_priority = max(descriptor.priority for descriptor in constrained_candidates)
    winners = [
        descriptor
        for descriptor in constrained_candidates
        if descriptor.priority == highest_priority
    ]
    if len(winners) > 1:
        winner_set = ", ".join(sorted(_format_descriptor_identity(descriptor) for descriptor in winners))
        raise LookupError(
            "select_kernel() found multiple highest-priority kernels for "
            f"target={normalized_target!r}, op={normalized_op!r}, operand_types={normalized_operand_types!r}: "
            f"{winner_set}"
        )
    return winners[0]


def vkernel(
    py_fn: Callable[..., Any] | None = None,
    *,
    target: str = "a5",
    op: str | None = None,
    dtypes: Any = None,
    name: str | None = None,
    verify: bool = True,
    advanced: bool = False,
    constraints: Any = _UNSET,
    priority: Any = _UNSET,
) -> VKernelDescriptor | Callable[[Callable[..., Any]], VKernelDescriptor]:
    """Create a TileLang DSL v1 kernel descriptor.

    v1 keeps only the minimal descriptor metadata surface:
    `target`, `op`, `dtypes`, `constraints`, `priority`, `name`, `verify`,
    and opt-in `advanced`.
    """

    def wrap(fn: Callable[..., Any]) -> VKernelDescriptor:
        descriptor = _build_descriptor(
            fn,
            target=target,
            op=op,
            dtypes=dtypes,
            name=name,
            verify=verify,
            advanced=advanced,
            constraints=constraints,
            priority=priority,
        )
        return _DEFAULT_KERNEL_REGISTRY.register(descriptor)

    if py_fn is None:
        return wrap
    return wrap(py_fn)


__all__ = [
    "BoundKernelParameter",
    "KernelRegistry",
    "MaterializedMLIRModule",
    "TileLangFrontendError",
    "VKernelDescriptor",
    "select_kernel",
    "vkernel",
]

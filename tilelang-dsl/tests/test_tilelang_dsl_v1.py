import tempfile
import unittest
from unittest import mock
from importlib import util
from pathlib import Path

import tilelang_dsl as pto
import tilelang_dsl.kernel as kernel_impl
from tilelang_dsl.frontend_ast import build_frontend_kernel_node
from tilelang_dsl.lowering import AuthoringModule, lower_semantic_kernel
from tilelang_dsl.semantic import (
    SemanticAssignStmt,
    SemanticCallExpr,
    SemanticDmaConfigStmt,
    SemanticDmaLoadStmt,
    SemanticDmaStoreStmt,
    SemanticForStmt,
    SemanticIfStmt,
    SemanticIndexType,
    SemanticLowLevelCopyStmt,
    SemanticMaskType,
    SemanticPipeBarrierStmt,
    SemanticPtrType,
    SemanticScalarType,
    SemanticSetFlagStmt,
    SemanticStrictVecscopeStmt,
    SemanticTensorViewType,
    SemanticTileType,
    SemanticVecscopeStmt,
    SemanticVectorStoreStmt,
    SemanticWaitFlagStmt,
    analyze_frontend_kernel,
)


class TileLangDSLPackageTests(unittest.TestCase):
    def test_package_exports_surface(self) -> None:
        self.assertIsNotNone(pto.__file__)
        self.assertTrue(hasattr(pto, "vkernel"))
        self.assertTrue(hasattr(pto, "KernelRegistry"))
        self.assertTrue(hasattr(pto, "select_kernel"))
        self.assertTrue(hasattr(pto, "TensorView"))
        self.assertTrue(hasattr(pto, "Tile"))
        self.assertTrue(hasattr(pto, "TileSpecialization"))
        self.assertTrue(hasattr(pto, "PointerType"))
        self.assertTrue(hasattr(pto, "ptr"))
        self.assertTrue(hasattr(pto, "get_lanes"))
        self.assertTrue(hasattr(pto, "PAT"))
        self.assertTrue(hasattr(pto, "PIPE"))
        self.assertTrue(hasattr(pto, "EVENT"))


class TileLangDSLMatcherEntryTests(unittest.TestCase):
    def test_select_kernel_returns_descriptor_from_default_registry(self) -> None:
        @pto.vkernel(op="matcher_entry_default_registry_unique", dtypes=[(pto.f32, pto.i32)])
        def kernel(inp: pto.TensorView, scale: pto.i32):
            return None

        selected = pto.select_kernel(
            "a5",
            "matcher_entry_default_registry_unique",
            (pto.f32, pto.i32),
        )

        self.assertIs(selected, kernel)

    def test_select_kernel_uses_explicit_registry_without_falling_back(self) -> None:
        @pto.vkernel(op="matcher_entry_registry_isolation_unique", dtypes=[(pto.f32,)])
        def default_kernel(inp: pto.TensorView):
            return None

        empty_registry = pto.KernelRegistry()
        with self.assertRaises(LookupError) as ctx:
            pto.select_kernel(
                "a5",
                "matcher_entry_registry_isolation_unique",
                (pto.f32,),
                registry=empty_registry,
            )
        self.assertIn("found no registered kernel", str(ctx.exception))

        isolated_registry = pto.KernelRegistry()
        isolated_registry.register(default_kernel)
        selected = pto.select_kernel(
            "a5",
            "matcher_entry_registry_isolation_unique",
            (pto.f32,),
            registry=isolated_registry,
        )

        self.assertIs(selected, default_kernel)
        self.assertEqual(len(isolated_registry.descriptors), 1)

    def test_select_kernel_binds_concrete_signature_from_multi_signature_descriptor(self) -> None:
        @pto.vkernel(
            op="matcher_multi_signature_unique",
            dtypes=[
                (pto.f16, pto.f16),
                (pto.f32, pto.f32),
            ],
        )
        def kernel(inp: pto.TensorView, tile: pto.Tile):
            return None

        selected = pto.select_kernel(
            "a5",
            "matcher_multi_signature_unique",
            (pto.f32, pto.f32),
        )

        self.assertEqual(selected.dtype_signature, (pto.f32, pto.f32))
        self.assertEqual(
            [(param.name, param.kind, param.dtype) for param in selected.parameters],
            [("inp", "tensorview", pto.f32), ("tile", "tile", pto.f32)],
        )
        specialized = selected.specialize(
            tile=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB)
        )
        self.assertIn("memref<8x16xf32", specialized.mlir_text())

    def test_select_kernel_matches_wildcards_deterministically(self) -> None:
        @pto.vkernel(
            op="matcher_wildcard_unique",
            dtypes=[
                (pto.AnyInt, pto.AnyType),
                (pto.AnyFloat, pto.AnyType),
            ],
        )
        def kernel(lhs: pto.TensorView, rhs: pto.Tile):
            return None

        selected = pto.select_kernel(
            "a5",
            "matcher_wildcard_unique",
            (pto.f32, pto.i32),
        )

        self.assertEqual(selected.dtype_signature, (pto.f32, pto.i32))
        self.assertEqual(selected.parameters[0].dtype, pto.f32)
        self.assertEqual(selected.parameters[1].dtype, pto.i32)

    def test_select_kernel_enforces_typevar_consistency_per_signature(self) -> None:
        @pto.vkernel(
            op="matcher_typevar_unique",
            dtypes=[(pto.TypeVar("T"), pto.TypeVar("T"))],
        )
        def kernel(lhs: pto.TensorView, rhs: pto.Tile):
            return None

        selected = pto.select_kernel(
            "a5",
            "matcher_typevar_unique",
            (pto.f32, pto.f32),
        )
        self.assertEqual(selected.dtype_signature, (pto.f32, pto.f32))

        with self.assertRaises(LookupError) as ctx:
            pto.select_kernel(
                "a5",
                "matcher_typevar_unique",
                (pto.f32, pto.i32),
            )
        self.assertIn("found no registered kernel", str(ctx.exception))

    def test_polymorphic_descriptor_requires_select_kernel_before_materialization(self) -> None:
        @pto.vkernel(
            op="matcher_materialization_gate_unique",
            dtypes=[(pto.AnyFloat, pto.AnyFloat)],
        )
        def kernel(inp: pto.TensorView, out: pto.TensorView):
            return None

        with self.assertRaises(ValueError) as ctx:
            kernel.mlir_text()
        self.assertIn("requires pto.select_kernel(...)", str(ctx.exception))

    def test_select_kernel_evaluates_constraints_before_priority(self) -> None:
        def requires_large_batch(context_attrs):
            return context_attrs.get("batch", 0) >= 1024

        @pto.vkernel(
            op="matcher_constraint_priority_unique",
            dtypes=[(pto.AnyFloat, pto.AnyFloat)],
            constraints=[requires_large_batch],
            priority=100,
        )
        def high_priority_kernel(inp: pto.TensorView, out: pto.TensorView):
            return None

        @pto.vkernel(
            op="matcher_constraint_priority_unique",
            dtypes=[(pto.AnyFloat, pto.AnyFloat)],
            constraints=[],
            priority=10,
        )
        def fallback_kernel(inp: pto.TensorView, out: pto.TensorView):
            return None

        selected = pto.select_kernel(
            "a5",
            "matcher_constraint_priority_unique",
            (pto.f32, pto.f32),
            context_attrs={"batch": 128},
        )
        self.assertIs(selected.py_fn, fallback_kernel.py_fn)
        self.assertEqual(selected.priority, 10)

        selected = pto.select_kernel(
            "a5",
            "matcher_constraint_priority_unique",
            (pto.f32, pto.f32),
            context_attrs={"batch": 4096},
        )
        self.assertIs(selected.py_fn, high_priority_kernel.py_fn)
        self.assertEqual(selected.priority, 100)

    def test_select_kernel_raises_tie_error_for_equal_highest_priority(self) -> None:
        @pto.vkernel(
            op="matcher_priority_tie_unique",
            dtypes=[(pto.AnyFloat, pto.AnyFloat)],
            priority=50,
        )
        def lhs(inp: pto.TensorView, out: pto.TensorView):
            return None

        @pto.vkernel(
            op="matcher_priority_tie_unique",
            dtypes=[(pto.AnyFloat, pto.AnyFloat)],
            priority=50,
        )
        def rhs(inp: pto.TensorView, out: pto.TensorView):
            return None

        with self.assertRaises(LookupError) as ctx:
            pto.select_kernel(
                "a5",
                "matcher_priority_tie_unique",
                (pto.f32, pto.f32),
            )
        self.assertIn("multiple highest-priority kernels", str(ctx.exception))
        self.assertIn("lhs(priority=50", str(ctx.exception))
        self.assertIn("rhs(priority=50", str(ctx.exception))

    def test_select_kernel_reports_no_candidate_after_constraint_evaluation(self) -> None:
        @pto.vkernel(
            op="matcher_constraint_empty_unique",
            dtypes=[(pto.AnyFloat, pto.AnyFloat)],
            constraints=[lambda context_attrs: context_attrs.get("enabled", False)],
            priority=1,
        )
        def kernel(inp: pto.TensorView, out: pto.TensorView):
            return None

        with self.assertRaises(LookupError) as ctx:
            pto.select_kernel(
                "a5",
                "matcher_constraint_empty_unique",
                (pto.f32, pto.f32),
                context_attrs={"enabled": False},
            )
        self.assertIn("after constraint evaluation", str(ctx.exception))


class TileLangDSLDescriptorTests(unittest.TestCase):
    def test_descriptor_metadata_and_parameter_binding(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f16, pto.i32)], verify=False)
        def kernel(inp: pto.TensorView, tile: pto.Tile, scale: pto.i32):
            return None

        self.assertEqual(kernel.target, "a5")
        self.assertEqual(kernel.op, "eltwise")
        self.assertEqual(kernel.name, "kernel")
        self.assertFalse(kernel.verify_enabled)
        self.assertFalse(kernel.advanced_enabled)
        self.assertEqual(kernel.metadata["verify"], False)
        self.assertEqual(kernel.metadata["advanced"], False)
        self.assertEqual(kernel.dtype_signature, (pto.f32, pto.f16, pto.i32))
        self.assertEqual(
            [(param.name, param.kind, param.dtype) for param in kernel.parameters],
            [("inp", "tensorview", pto.f32), ("tile", "tile", pto.f16), ("scale", "scalar", pto.i32)],
        )
        self.assertEqual(kernel.parameters[0].element_dtype, pto.f32)
        self.assertEqual(kernel.parameters[1].element_dtype, pto.f16)
        self.assertIsNone(kernel.parameters[2].element_dtype)

    def test_pointer_parameter_annotation_binds_as_ptr_kind(self) -> None:
        @pto.vkernel(op="ptr_surface", dtypes=[(pto.f32, pto.i64)], advanced=True)
        def kernel(src: pto.ptr(pto.f32, pto.MemorySpace.UB), addr: pto.i64):
            return None

        self.assertEqual(kernel.parameters[0].kind, "ptr")
        self.assertEqual(kernel.parameters[0].dtype, pto.f32)
        self.assertEqual(kernel.parameters[0].annotation, pto.ptr(pto.f32, pto.MemorySpace.UB))
        self.assertEqual(kernel.parameters[0].element_dtype, pto.f32)

    def test_specialization_enables_materialization_apis(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f16)])
        def kernel(inp: pto.TensorView, tile: pto.Tile):
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(16, 32),
                memory_space=pto.MemorySpace.UB,
                config=pto.TileConfig.from_mapping({"layout": "row_major"}),
            )
        )

        self.assertIn("tile", specialized.specializations_by_name)
        text = specialized.mlir_text()
        self.assertIn("// tilelang.target = a5", text)
        self.assertIn("// tilelang.specialize tile shape=(16, 32) memory_space=ub", text)
        self.assertIn('module attributes {pto.target_arch = "a5"} {', text)
        self.assertIn("func.func @kernel(%arg0: !pto.ptr<f32, gm>, %arg1: !pto.ptr<f16, ub>) {", text)
        module = specialized.mlir_module()
        self.assertEqual(type(module).__name__, "MaterializedMLIRModule")
        mocked_result = kernel_impl.VerificationResult(
            status="passed",
            available=True,
            passed=True,
            message="ok",
            command=("ptoas",),
            returncode=0,
        )
        with mock.patch("tilelang_dsl.kernel._run_ptoas_verifier", return_value=mocked_result):
            self.assertTrue(module.verify())
            self.assertTrue(specialized.verify())
            self.assertEqual(module.verify().status, "passed")
            self.assertEqual(specialized.verify().status, "passed")

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "kernel.mlir"
            specialized.emit(out)
            self.assertEqual(out.read_text(encoding="utf-8"), text)

    def test_verify_reports_structured_unavailable_when_ptoas_is_missing(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f16)])
        def kernel(inp: pto.TensorView, tile: pto.Tile):
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(16, 32),
                memory_space=pto.MemorySpace.UB,
            )
        )

        result = specialized.verify(ptoas_bin="/definitely-missing/ptoas")
        self.assertFalse(result)
        self.assertEqual(result.status, "unavailable")
        self.assertFalse(result.available)
        self.assertFalse(result.passed)
        self.assertIn("verifier unavailable", result.message)

    def test_descriptor_materialization_flows_through_pipeline(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f16, pto.i32)])
        def kernel(inp: pto.TensorView, tile: pto.Tile, scale: pto.i32):
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(8, 16),
                memory_space=pto.MemorySpace.UB,
            )
        )

        frontend_kernel = build_frontend_kernel_node(specialized)
        self.assertEqual(frontend_kernel.name, "kernel")
        self.assertEqual(
            [(param.name, param.kind) for param in frontend_kernel.parameters],
            [("inp", "tensorview"), ("tile", "tile"), ("scale", "scalar")],
        )
        self.assertEqual(frontend_kernel.tile_specializations[0].shape, (8, 16))

        semantic_kernel = analyze_frontend_kernel(frontend_kernel)
        self.assertEqual(semantic_kernel.symbol_name, "kernel")
        self.assertEqual(semantic_kernel.tile_bindings[0].memory_space, "ub")

        authoring_module = lower_semantic_kernel(semantic_kernel)
        self.assertIsInstance(authoring_module, AuthoringModule)
        self.assertEqual(authoring_module.render(), specialized.mlir_text())
        self.assertIn("return", authoring_module.render())

    def test_semantic_pipeline_binds_parameter_loop_and_strict_vecscope_types(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f16, pto.i32)])
        def kernel(inp: pto.TensorView, tile: pto.Tile, scale: pto.i32):
            rows = tile.shape[0]
            step = rows
            with pto.strict_vecscope(inp, tile, scale, 0, rows, step) as (
                vin,
                vtmp,
                factor,
                lb,
                ub,
                vec_step,
            ):
                for lane in range(lb, ub, vec_step):
                    current = factor
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(8, 16),
                memory_space=pto.MemorySpace.UB,
            )
        )

        frontend_kernel = build_frontend_kernel_node(specialized)
        self.assertEqual(len(frontend_kernel.body), 4)

        semantic_kernel = analyze_frontend_kernel(frontend_kernel)
        self.assertIsInstance(semantic_kernel.parameters[0].type, SemanticTensorViewType)
        self.assertIsInstance(semantic_kernel.parameters[1].type, SemanticTileType)
        self.assertEqual(semantic_kernel.parameters[1].type.shape, (8, 16))
        self.assertIsInstance(semantic_kernel.parameters[2].type, SemanticScalarType)

        rows_assign = semantic_kernel.body[0]
        self.assertIsInstance(rows_assign, SemanticAssignStmt)
        self.assertIsInstance(rows_assign.targets[0].type, SemanticIndexType)
        self.assertTrue(rows_assign.targets[0].ssa_name.startswith("%rows_"))

        vecscope_stmt = semantic_kernel.body[2]
        self.assertIsInstance(vecscope_stmt, SemanticStrictVecscopeStmt)
        self.assertEqual(
            [binding.name for binding in vecscope_stmt.block_arguments],
            ["vin", "vtmp", "factor", "lb", "ub", "vec_step"],
        )
        self.assertIsInstance(vecscope_stmt.block_arguments[0].type, SemanticTensorViewType)
        self.assertIsInstance(vecscope_stmt.block_arguments[1].type, SemanticTileType)
        self.assertIsInstance(vecscope_stmt.block_arguments[2].type, SemanticScalarType)
        self.assertIsInstance(vecscope_stmt.block_arguments[3].type, SemanticIndexType)
        self.assertIsInstance(vecscope_stmt.block_arguments[4].type, SemanticIndexType)
        self.assertIsInstance(vecscope_stmt.block_arguments[5].type, SemanticIndexType)
        self.assertTrue(vecscope_stmt.block_arguments[0].ssa_name.startswith("%vin_"))

        loop_stmt = vecscope_stmt.body[0]
        self.assertIsInstance(loop_stmt, SemanticForStmt)
        self.assertEqual(loop_stmt.induction_variable.name, "lane")
        self.assertIsInstance(loop_stmt.induction_variable.type, SemanticIndexType)
        self.assertTrue(loop_stmt.induction_variable.ssa_name.startswith("%lane_"))
        self.assertEqual(loop_stmt.loop_carried, ())

        text = specialized.mlir_text()
        self.assertIn("%rows_", text)
        self.assertIn("= arith.constant 8 : index", text)
        self.assertIn("pto.strict_vecscope(%arg0, %arg1, %arg2, %c0, %rows_", text)
        self.assertIn("^bb0(", text)
        self.assertIn("scf.for %lane_", text)
        self.assertIn("to %ub_6 step %vec_step_7 {", text)

    def test_dma_load_and_store_lower_to_dma_programming_and_copy_ops(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f32, pto.f32)])
        def kernel(inp: pto.TensorView, out: pto.TensorView, tile: pto.Tile):
            pto.dma_load(inp[0:16, 0:16], tile)
            pto.dma_store(tile, out[0:16, 0:16])
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(16, 16),
                memory_space=pto.MemorySpace.UB,
            )
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        self.assertIsInstance(semantic_kernel.body[0], SemanticDmaLoadStmt)
        self.assertIsInstance(semantic_kernel.body[1], SemanticDmaStoreStmt)

        text = specialized.mlir_text()
        self.assertIn(
            "func.func @kernel(%arg0: !pto.ptr<f32, gm>, %arg1: !pto.ptr<f32, gm>, %arg2: !pto.ptr<f32, ub>) {",
            text,
        )
        self.assertIn("pto.set_loop_size_outtoub %c1_i64, %c1_i64 : i64, i64", text)
        self.assertIn(
            "pto.copy_gm_to_ubuf %arg0, %arg2, %c0_i64, %c16_i64, %c64_i64, %c0_i64, %c0_i64, %false, %c0_i64, %c64_i64, %c64_i64",
            text,
        )
        self.assertIn("pto.set_loop_size_ubtoout %c1_i64, %c1_i64 : i64, i64", text)
        self.assertIn(
            "pto.copy_ubuf_to_gm %arg2, %arg1, %c0_i64, %c16_i64, %c64_i64, %c0_i64, %c64_i64, %c64_i64",
            text,
        )

    def test_dynamic_tensorview_shape_profile_supports_runtime_bound_and_slice(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f32)])
        def kernel(inp: pto.TensorView, tile: pto.Tile):
            rows = inp.shape[0]
            pto.dma_load(inp[0:rows, 0:16], tile)
            for lane in range(0, rows, 1):
                current = lane
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(16, 16),
                memory_space=pto.MemorySpace.UB,
            )
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        self.assertEqual(
            [(param.name, param.kind) for param in semantic_kernel.parameters],
            [("inp", "tensorview"), ("tile", "tile"), ("__shape_inp_0", "tensorview_shape")],
        )

        rows_assign = semantic_kernel.body[0]
        self.assertIsInstance(rows_assign, SemanticAssignStmt)
        self.assertIsInstance(rows_assign.targets[0].type, SemanticIndexType)

        dma_stmt = semantic_kernel.body[1]
        self.assertIsInstance(dma_stmt, SemanticDmaLoadStmt)
        self.assertEqual(dma_stmt.src.type.extents, (None, 16))

        loop_stmt = semantic_kernel.body[2]
        self.assertIsInstance(loop_stmt, SemanticForStmt)

        text = specialized.mlir_text()
        self.assertIn(
            "func.func @kernel(%arg0: !pto.ptr<f32, gm>, %arg1: !pto.ptr<f32, ub>, %arg2: index) {",
            text,
        )
        self.assertIn(
            "pto.copy_gm_to_ubuf %arg0, %arg1, %c0_i64, %c16_i64, %c64_i64, %c0_i64, %c0_i64, %false, %c0_i64, %c64_i64, %c64_i64",
            text,
        )
        self.assertIn("scf.for %lane_", text)
        self.assertIn("to %arg2 step %c1 {", text)

    def test_make_mask_vlds_vsts_and_vector_families_lower_inside_strict_vecscope(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f32, pto.f32)])
        def kernel(inp: pto.TensorView, tile: pto.Tile, scale: pto.f32):
            pto.dma_load(inp[0:16, 0:16], tile)
            with pto.strict_vecscope(tile, tile, scale, 0, 256, 64) as (
                src,
                dst,
                factor,
                lb,
                ub,
                step,
            ):
                for lane in range(lb, ub, step):
                    mask = pto.make_mask(pto.f32, pto.PAT.ALL)
                    vec = pto.vlds(src, lane)
                    biased = pto.vadds(vec, factor, mask)
                    summed = pto.vadd(biased, vec, mask)
                    activated = pto.vrelu(summed, mask)
                    pto.vsts(activated, dst, lane, mask)
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(16, 16),
                memory_space=pto.MemorySpace.UB,
            )
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        vecscope = semantic_kernel.body[1]
        self.assertIsInstance(vecscope, SemanticStrictVecscopeStmt)
        loop_stmt = vecscope.body[0]
        self.assertIsInstance(loop_stmt, SemanticForStmt)
        mask_assign = loop_stmt.body[0]
        self.assertIsInstance(mask_assign, SemanticAssignStmt)
        self.assertIsInstance(mask_assign.value, SemanticCallExpr)
        self.assertEqual(mask_assign.value.name, "make_mask")
        self.assertIsInstance(mask_assign.targets[0].type, SemanticMaskType)
        self.assertIsInstance(loop_stmt.body[-1], SemanticVectorStoreStmt)

        text = specialized.mlir_text()
        self.assertIn('%mask_7 = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>', text)
        self.assertIn("%vec_8 = pto.vlds %src_0[%lane_6] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>", text)
        self.assertIn(
            "%biased_9 = pto.vadds %vec_8, %factor_2, %mask_7 : !pto.vreg<64xf32>, f32, !pto.mask<b32> -> !pto.vreg<64xf32>",
            text,
        )
        self.assertIn(
            "%summed_10 = pto.vadd %biased_9, %vec_8, %mask_7 : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>",
            text,
        )
        self.assertIn(
            "%activated_11 = pto.vrelu %summed_10, %mask_7 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>",
            text,
        )
        self.assertIn(
            "pto.vsts %activated_11, %dst_1[%lane_6], %mask_7 : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>",
            text,
        )

    def test_tail_make_mask_lowers_to_typed_plt_and_updates_remaining(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f32, pto.i32)])
        def kernel(inp: pto.TensorView, tile: pto.Tile, remaining: pto.i32):
            pto.dma_load(inp[0:16, 0:16], tile)
            with pto.strict_vecscope(tile, tile, remaining, 0, 64, 64) as (src, dst, rem_in, lb, ub, step):
                mask, next_remaining = pto.make_mask(pto.f32, rem_in)
                vec = pto.vlds(src, lb)
                pto.vsts(vec, dst, lb, mask)
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(16, 16),
                memory_space=pto.MemorySpace.UB,
            )
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        vecscope = semantic_kernel.body[1]
        self.assertIsInstance(vecscope, SemanticStrictVecscopeStmt)
        mask_assign = vecscope.body[0]
        self.assertIsInstance(mask_assign, SemanticAssignStmt)
        self.assertEqual(mask_assign.value.name, "make_mask")
        self.assertEqual(len(mask_assign.targets), 2)
        self.assertIsInstance(mask_assign.targets[0].type, SemanticMaskType)
        self.assertIsInstance(mask_assign.targets[1].type, SemanticScalarType)
        self.assertEqual(mask_assign.targets[1].type.dtype, pto.i32)

        text = specialized.mlir_text()
        self.assertRegex(
            text,
            r"%mask_\d+, %next_remaining_\d+ = pto\.plt_b32 %rem_in_\d+ : i32 -> !pto\.mask<b32>, i32",
        )
        self.assertIn(
            "pto.vsts %vec_",
            text,
        )

    def test_nested_index_arithmetic_lowers_before_vector_accesses(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f32, pto.f32, pto.f32, pto.f32, pto.f32)])
        def kernel(
            lhs_gm: pto.TensorView,
            rhs_gm: pto.TensorView,
            out_gm: pto.TensorView,
            lhs_tile: pto.Tile,
            rhs_tile: pto.Tile,
            dst_tile: pto.Tile,
        ):
            rows = lhs_gm.shape[0]
            cols = lhs_gm.shape[1]
            row_stride = lhs_tile.shape[1]

            pto.dma_load(lhs_gm[0:rows, 0:cols], lhs_tile)
            pto.dma_load(rhs_gm[0:rows, 0:cols], rhs_tile)
            with pto.strict_vecscope(
                lhs_tile,
                rhs_tile,
                dst_tile,
                rows,
                cols,
                row_stride,
                0,
                rows,
                1,
            ) as (lhs, rhs, dst, valid_rows, valid_cols, stride, row_lb, row_ub, row_step):
                for row in range(row_lb, row_ub, row_step):
                    for lane in range(0, valid_cols, 64):
                        offset = row * stride + lane
                        mask, next_remaining = pto.make_mask(pto.f32, valid_cols - lane)
                        summed = pto.vadd(pto.vlds(lhs, offset), pto.vlds(rhs, offset), mask)
                        pto.vsts(summed, dst, offset, mask)
            pto.dma_store(dst_tile, out_gm[0:rows, 0:cols])
            return None

        specialized = kernel.specialize(
            lhs_tile=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            rhs_tile=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            dst_tile=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertRegex(text, r"%tmp_\d+ = arith\.muli %row_\d+, %stride_\d+ : index")
        self.assertRegex(text, r"%offset_\d+ = arith\.addi %tmp_\d+, %lane_\d+ : index")
        self.assertRegex(text, r"%tmp_\d+ = arith\.subi %valid_cols_\d+, %lane_\d+ : index")
        self.assertRegex(text, r"%tmp_\d+ = arith\.index_cast %tmp_\d+ : index to i32")
        self.assertIn("pto.plt_b32", text)
        self.assertIn("pto.vadd", text)

    def test_advanced_mode_infers_vecscope_and_lowers_tile_vector_sugar(self) -> None:
        @pto.vkernel(op="tadd", dtypes=[(pto.f32, pto.f32, pto.f32)], advanced=True)
        def kernel(dst: pto.Tile, src0: pto.Tile, src1: pto.Tile):
            dtype = dst.element_type
            rows, cols = dst.valid_shape
            all_mask = pto.make_mask(dtype, pto.PAT.ALL)
            for row in range(0, rows, 1):
                for col in range(0, cols, pto.get_lanes(dtype)):
                    lhs = pto.vlds(src0[row, col:])
                    rhs = pto.vlds(src1[row, col:])
                    summed = pto.vadd(lhs, rhs, all_mask)
                    pto.vsts(summed, dst[row, col:], all_mask)
            return None

        self.assertTrue(kernel.advanced_enabled)

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            src0=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            src1=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        vecscope_stmts = [stmt for stmt in semantic_kernel.body if isinstance(stmt, SemanticVecscopeStmt)]
        self.assertEqual(len(vecscope_stmts), 1)
        vecscope = vecscope_stmts[0]
        self.assertIsInstance(vecscope, SemanticVecscopeStmt)
        outer_loop = next(stmt for stmt in vecscope.body if isinstance(stmt, SemanticForStmt))
        self.assertIsInstance(outer_loop, SemanticForStmt)
        inner_loop = outer_loop.body[0]
        self.assertIsInstance(inner_loop, SemanticForStmt)
        self.assertTrue(inner_loop.body)

        text = specialized.mlir_text()
        self.assertIn("// tilelang.advanced = True", text)
        self.assertIn("pto.vecscope {", text)
        self.assertNotIn("pto.strict_vecscope(", text)
        self.assertRegex(text, r"pto\.vecscope \{\n(?:.|\n)*scf\.for %row_")
        self.assertEqual(text.count("pto.vecscope {"), 1)
        self.assertLess(text.index("%rows_1 = arith.constant 8 : index"), text.index("pto.vecscope {"))
        self.assertLess(text.index("%cols_2 = arith.constant 64 : index"), text.index("pto.vecscope {"))
        self.assertRegex(text, r"%tmp_\d+ = arith\.muli %row_\d+, %c64 : index")
        self.assertRegex(text, r"%tmp_\d+ = arith\.addi %tmp_\d+, %col_\d+ : index")
        self.assertIn("pto.vlds %arg1[", text)
        self.assertIn("pto.vlds %arg2[", text)
        self.assertIn("pto.vsts %summed_", text)

    def test_element_type_valid_shape_and_get_lanes_surface_lower_in_advanced_mode(self) -> None:
        @pto.vkernel(op="tadd", dtypes=[(pto.f32, pto.f32, pto.f32)], advanced=True)
        def kernel(dst: pto.Tile, src0: pto.Tile, src1: pto.Tile):
            dtype = dst.element_type
            valid_rows, valid_cols = dst.valid_shape
            remained = valid_cols
            for row in range(0, valid_rows, 1):
                for col in range(0, valid_cols, pto.get_lanes(dtype)):
                    mask, remained = pto.make_mask(dtype, remained)
                    summed = pto.vadd(pto.vlds(src0[row, col:]), pto.vlds(src1[row, col:]), mask)
                    pto.vsts(summed, dst[row, col:], mask)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            src0=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            src1=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertIn("step %c64", text)
        self.assertRegex(text, r"%mask_\d+, %remained_\d+ = pto\.plt_b32 %remained_iter_\d+ : i32 -> !pto\.mask<b32>, i32")
        self.assertIn("pto.vadd", text)
        self.assertIn("pto.vsts", text)

    def test_advanced_mode_scalar_boundary_cuts_inferred_vecscope(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f32)], advanced=True)
        def kernel(src: pto.Tile, dst: pto.Tile):
            dtype = src.element_type
            first_mask = pto.make_mask(dtype, pto.PAT.ALL)
            first = pto.vlds(src[0, 0:])
            pto.vsts(first, dst[0, 0:], first_mask)
            boundary = 1
            second_mask = pto.make_mask(dtype, pto.PAT.ALL)
            second = pto.vlds(src[1, 0:])
            pto.vsts(second, dst[1, 0:], second_mask)
            return None

        specialized = kernel.specialize(
            src=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertEqual(text.count("pto.vecscope {"), 2)
        self.assertLess(text.index("%boundary_"), text.rindex("pto.vecscope {"))

    def test_advanced_mode_control_flow_boundary_cuts_inferred_vecscope(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f32, pto.i32)], advanced=True)
        def kernel(src: pto.Tile, dst: pto.Tile, flag: pto.i32):
            dtype = src.element_type
            all_mask = pto.make_mask(dtype, pto.PAT.ALL)
            if flag:
                first = pto.vlds(src[0, 0:])
                pto.vsts(first, dst[0, 0:], all_mask)
            else:
                second = pto.vlds(src[1, 0:])
                pto.vsts(second, dst[1, 0:], all_mask)
            return None

        specialized = kernel.specialize(
            src=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertIn("scf.if", text)
        self.assertEqual(text.count("pto.vecscope {"), 2)
        self.assertLess(text.index("scf.if"), text.index("pto.vecscope {"))

    def test_advanced_mode_keeps_strict_vecscope_as_hard_boundary(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f32)], advanced=True)
        def kernel(src: pto.Tile, dst: pto.Tile):
            all_mask = pto.make_mask(pto.f32, pto.PAT.ALL)
            rows = src.shape[0]
            for row in range(0, rows, 1):
                vec = pto.vlds(src[row, 0:])
                pto.vsts(vec, dst[row, 0:], all_mask)
            with pto.strict_vecscope(src, dst, all_mask, 0, 64, 64) as (vin, vout, mask, lb, ub, step):
                for lane in range(lb, ub, step):
                    scoped = pto.vlds(vin, lane)
                    pto.vsts(scoped, vout, lane, mask)
            return None

        specialized = kernel.specialize(
            src=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertEqual(text.count("pto.vecscope {"), 1)
        self.assertEqual(text.count("pto.strict_vecscope("), 1)

    def test_advanced_mode_lowers_raw_pointer_and_low_level_dma_surface(self) -> None:
        @pto.vkernel(op="ptr_dma", dtypes=[(pto.f32, pto.f32, pto.i64)], advanced=True)
        def kernel(
            src_gm: pto.ptr(pto.f32, pto.MemorySpace.GM),
            dst_gm: pto.ptr(pto.f32, pto.MemorySpace.GM),
            addr: pto.i64,
        ):
            ub_src = pto.castptr(addr, pto.ptr(pto.f32, pto.MemorySpace.UB))
            ub_dst = pto.addptr(ub_src, 64)
            mask = pto.make_mask(pto.f32, pto.PAT.ALL)
            vec = pto.vlds(ub_src, 0)
            pto.vsts(vec, ub_dst, 0, mask)

            src_bytes = pto.castptr(src_gm, pto.ptr(pto.i8, pto.MemorySpace.GM))
            dst_bytes = pto.castptr(dst_gm, pto.ptr(pto.i8, pto.MemorySpace.GM))
            src_offset = pto.addptr(src_bytes, 0)
            dst_offset = pto.addptr(dst_bytes, 0)
            typed_src = pto.castptr(src_offset, pto.ptr(pto.f32, pto.MemorySpace.GM))
            typed_dst = pto.castptr(dst_offset, pto.ptr(pto.f32, pto.MemorySpace.GM))

            pto.set_loop2_stride_outtoub(4096, 4096)
            pto.set_loop1_stride_outtoub(4096, 4096)
            pto.set_loop_size_outtoub(1, 1)
            pto.copy_gm_to_ubuf(typed_src, ub_src, 0, 32, 128, 0, 0, False, 0, 128, 128)

            pto.set_loop2_stride_ubtoout(4096, 4096)
            pto.set_loop1_stride_ubtoout(4096, 4096)
            pto.set_loop_size_ubtoout(1, 1)
            pto.copy_ubuf_to_ubuf(ub_src, ub_dst, 0, 32, 128, 128, 128)
            pto.copy_ubuf_to_gm(ub_dst, typed_dst, 0, 32, 128, 0, 128, 128)
            return None

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(kernel))
        self.assertIsInstance(semantic_kernel.parameters[0].type, SemanticPtrType)
        self.assertEqual(semantic_kernel.parameters[0].type.memory_space, "gm")
        self.assertIsInstance(semantic_kernel.parameters[1].type, SemanticPtrType)
        self.assertEqual(semantic_kernel.parameters[1].type.memory_space, "gm")
        self.assertTrue(any(isinstance(stmt, SemanticVecscopeStmt) for stmt in semantic_kernel.body))
        self.assertTrue(any(isinstance(stmt, SemanticDmaConfigStmt) for stmt in semantic_kernel.body))
        self.assertTrue(any(isinstance(stmt, SemanticLowLevelCopyStmt) for stmt in semantic_kernel.body))

        text = kernel.mlir_text()
        self.assertIn(
            "func.func @kernel(%arg0: !pto.ptr<f32, gm>, %arg1: !pto.ptr<f32, gm>, %arg2: i64) {",
            text,
        )
        self.assertRegex(
            text,
            r"%ub_src_\d+ = pto\.castptr %arg2 : i64 -> !pto\.ptr<f32, ub>",
        )
        self.assertRegex(
            text,
            r"%ub_dst_\d+ = pto\.addptr %ub_src_\d+, %c64 : !pto\.ptr<f32, ub> -> !pto\.ptr<f32, ub>",
        )
        self.assertIn("pto.vecscope {", text)
        self.assertRegex(
            text,
            r"%vec_\d+ = pto\.vlds %ub_src_\d+\[%c0\] : !pto\.ptr<f32, ub> -> !pto\.vreg<64xf32>",
        )
        self.assertRegex(
            text,
            r"pto\.vsts %vec_\d+, %ub_dst_\d+\[%c0\], %mask_\d+ : !pto\.vreg<64xf32>, !pto\.ptr<f32, ub>, !pto\.mask<b32>",
        )
        self.assertRegex(
            text,
            r"%src_bytes_\d+ = pto\.castptr %arg0 : !pto\.ptr<f32, gm> -> !pto\.ptr<i8, gm>",
        )
        self.assertRegex(
            text,
            r"%dst_bytes_\d+ = pto\.castptr %arg1 : !pto\.ptr<f32, gm> -> !pto\.ptr<i8, gm>",
        )
        self.assertRegex(
            text,
            r"%src_offset_\d+ = pto\.addptr %src_bytes_\d+, %c0 : !pto\.ptr<i8, gm> -> !pto\.ptr<i8, gm>",
        )
        self.assertRegex(
            text,
            r"%dst_offset_\d+ = pto\.addptr %dst_bytes_\d+, %c0 : !pto\.ptr<i8, gm> -> !pto\.ptr<i8, gm>",
        )
        self.assertRegex(
            text,
            r"pto\.set_loop2_stride_outtoub %tmp_\d+, %tmp_\d+ : i64, i64",
        )
        self.assertRegex(
            text,
            r"pto\.set_loop1_stride_outtoub %tmp_\d+, %tmp_\d+ : i64, i64",
        )
        self.assertRegex(
            text,
            r"pto\.set_loop_size_outtoub %tmp_\d+, %tmp_\d+ : i64, i64",
        )
        self.assertRegex(
            text,
            r"pto\.copy_gm_to_ubuf %typed_src_\d+, %ub_src_\d+, %tmp_\d+, %tmp_\d+, %tmp_\d+, %tmp_\d+, %tmp_\d+, %false, %tmp_\d+, %tmp_\d+, %tmp_\d+",
        )
        self.assertIn(
            ": !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64, i1, i64, i64, i64",
            text,
        )
        self.assertRegex(
            text,
            r"pto\.set_loop2_stride_ubtoout %tmp_\d+, %tmp_\d+ : i64, i64",
        )
        self.assertRegex(
            text,
            r"pto\.set_loop1_stride_ubtoout %tmp_\d+, %tmp_\d+ : i64, i64",
        )
        self.assertRegex(
            text,
            r"pto\.set_loop_size_ubtoout %tmp_\d+, %tmp_\d+ : i64, i64",
        )
        self.assertRegex(
            text,
            r"pto\.copy_ubuf_to_ubuf %ub_src_\d+, %ub_dst_\d+, %tmp_\d+, %tmp_\d+, %tmp_\d+, %tmp_\d+, %tmp_\d+",
        )
        self.assertIn(
            ": !pto.ptr<f32, ub>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64",
            text,
        )
        self.assertRegex(
            text,
            r"pto\.copy_ubuf_to_gm %ub_dst_\d+, %typed_dst_\d+, %tmp_\d+, %tmp_\d+, %tmp_\d+, %tmp_\d+, %tmp_\d+, %tmp_\d+",
        )

    def test_advanced_mode_lowers_compare_predicate_carry_and_rearrangement_families(self) -> None:
        @pto.vkernel(op="advanced_family", dtypes=[(pto.i32, pto.i32, pto.i32, pto.i32)], advanced=True)
        def kernel(dst: pto.Tile, src0: pto.Tile, src1: pto.Tile, scalar: pto.i32):
            all_mask = pto.make_mask(pto.i32, pto.PAT.ALL)
            lhs = pto.vlds(src0[0, 0:])
            rhs = pto.vlds(src1[0, 0:])
            cmp_mask = pto.vcmp(lhs, rhs, all_mask, "lt")
            cmp_scalar_mask = pto.vcmps(lhs, scalar, all_mask, "gt")
            negated = pto.pnot(cmp_mask, all_mask)
            picked = pto.psel(cmp_mask, negated, cmp_scalar_mask)
            packed = pto.ppack(picked, "PART_EVEN")
            unpacked = pto.punpack(packed, "PART_ODD")
            sum_vec, carry_mask = pto.vaddc(lhs, rhs, all_mask)
            diff_vec, borrow_mask = pto.vsubc(lhs, rhs, all_mask)
            sum_with_carry, carry_mask2 = pto.vaddcs(sum_vec, diff_vec, carry_mask, all_mask)
            diff_with_borrow, borrow_mask2 = pto.vsubcs(sum_with_carry, diff_vec, borrow_mask, all_mask)
            low, high = pto.vintlv(sum_with_carry, diff_with_borrow)
            dlow, dhigh = pto.vdintlv(low, high)
            even = pto.vintlvv2(dlow, dhigh, "PART_EVEN")
            odd = pto.vdintlvv2(dlow, dhigh, "PART_ODD")
            selected = pto.vsel(even, odd, unpacked)
            selected_r = pto.vselr(selected, sum_with_carry)
            final = pto.vselrv2(selected_r, diff_with_borrow)
            pto.vsts(final, dst[0, 0:], all_mask)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            src0=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            src1=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        vecscope_stmts = [stmt for stmt in semantic_kernel.body if isinstance(stmt, SemanticVecscopeStmt)]
        self.assertEqual(len(vecscope_stmts), 1)

        text = specialized.mlir_text()
        self.assertIn("pto.vecscope {", text)
        self.assertIn('pto.vcmp ', text)
        self.assertIn(', "lt" : !pto.vreg<64xi32>, !pto.vreg<64xi32>, !pto.mask<b32> -> !pto.mask<b32>', text)
        self.assertIn('pto.vcmps ', text)
        self.assertIn(', "gt" : !pto.vreg<64xi32>, i32, !pto.mask<b32> -> !pto.mask<b32>', text)
        self.assertIn(" = pto.pnot ", text)
        self.assertIn(" = pto.psel ", text)
        self.assertIn(' = pto.ppack ', text)
        self.assertIn('"PART_EVEN"', text)
        self.assertIn(' = pto.punpack ', text)
        self.assertIn('"PART_ODD"', text)
        self.assertRegex(
            text,
            r"%sum_vec_\d+, %carry_mask_\d+ = pto\.vaddc %lhs_\d+, %rhs_\d+, %all_mask_\d+ : !pto\.vreg<64xi32>, !pto\.vreg<64xi32>, !pto\.mask<b32> -> !pto\.vreg<64xi32>, !pto\.mask<b32>",
        )
        self.assertRegex(
            text,
            r"%diff_vec_\d+, %borrow_mask_\d+ = pto\.vsubc %lhs_\d+, %rhs_\d+, %all_mask_\d+ : !pto\.vreg<64xi32>, !pto\.vreg<64xi32>, !pto\.mask<b32> -> !pto\.vreg<64xi32>, !pto\.mask<b32>",
        )
        self.assertRegex(
            text,
            r"%sum_with_carry_\d+, %carry_mask2_\d+ = pto\.vaddcs %sum_vec_\d+, %diff_vec_\d+, %carry_mask_\d+, %all_mask_\d+ : !pto\.vreg<64xi32>, !pto\.vreg<64xi32>, !pto\.mask<b32>, !pto\.mask<b32> -> !pto\.vreg<64xi32>, !pto\.mask<b32>",
        )
        self.assertRegex(
            text,
            r"%diff_with_borrow_\d+, %borrow_mask2_\d+ = pto\.vsubcs %sum_with_carry_\d+, %diff_vec_\d+, %borrow_mask_\d+, %all_mask_\d+ : !pto\.vreg<64xi32>, !pto\.vreg<64xi32>, !pto\.mask<b32>, !pto\.mask<b32> -> !pto\.vreg<64xi32>, !pto\.mask<b32>",
        )
        self.assertRegex(
            text,
            r"%low_\d+, %high_\d+ = pto\.vintlv %sum_with_carry_\d+, %diff_with_borrow_\d+ : !pto\.vreg<64xi32>, !pto\.vreg<64xi32> -> !pto\.vreg<64xi32>, !pto\.vreg<64xi32>",
        )
        self.assertRegex(
            text,
            r"%dlow_\d+, %dhigh_\d+ = pto\.vdintlv %low_\d+, %high_\d+ : !pto\.vreg<64xi32>, !pto\.vreg<64xi32> -> !pto\.vreg<64xi32>, !pto\.vreg<64xi32>",
        )
        self.assertIn(" = pto.vintlvv2 ", text)
        self.assertIn(" = pto.vdintlvv2 ", text)
        self.assertIn(" = pto.vsel ", text)
        self.assertIn(" = pto.vselr ", text)
        self.assertIn(" = pto.vselrv2 ", text)
        self.assertIn("pto.vsts ", text)

    def test_elementwise_kernel_positive_regression_covers_dma_vecscope_tail_mask_and_dynamic_loop_bound(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f32, pto.f32, pto.i32)])
        def kernel(inp: pto.TensorView, out: pto.TensorView, tile: pto.Tile, remaining: pto.i32):
            rows = inp.shape[0]
            pto.dma_load(inp[0:rows, 0:16], tile)
            with pto.strict_vecscope(tile, tile, remaining, 0, rows, 64) as (
                src,
                dst,
                rem,
                lb,
                ub,
                step,
            ):
                for lane in range(lb, ub, step):
                    mask, rem = pto.make_mask(pto.f32, rem)
                    vec = pto.vlds(src, lane)
                    pto.vsts(vec, dst, lane, mask)
            pto.dma_store(tile, out[0:rows, 0:16])
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(16, 16),
                memory_space=pto.MemorySpace.UB,
            )
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        self.assertEqual(len(semantic_kernel.body), 5)
        self.assertIsInstance(semantic_kernel.body[1], SemanticDmaLoadStmt)
        self.assertIsInstance(semantic_kernel.body[2], SemanticStrictVecscopeStmt)
        self.assertIsInstance(semantic_kernel.body[3], SemanticDmaStoreStmt)

        vecscope = semantic_kernel.body[2]
        self.assertIsInstance(vecscope, SemanticStrictVecscopeStmt)
        loop_stmt = vecscope.body[0]
        self.assertIsInstance(loop_stmt, SemanticForStmt)
        self.assertEqual(len(loop_stmt.loop_carried), 1)
        self.assertEqual(loop_stmt.loop_carried[0].name, "rem")

        text = specialized.mlir_text()
        self.assertIn(
            "func.func @kernel(%arg0: !pto.ptr<f32, gm>, %arg1: !pto.ptr<f32, gm>, %arg2: !pto.ptr<f32, ub>, %arg3: i32, %arg4: index) {",
            text,
        )
        self.assertIn(
            "pto.copy_gm_to_ubuf %arg0, %arg2, %c0_i64, %c16_i64, %c64_i64",
            text,
        )
        self.assertIn(
            "pto.strict_vecscope(%arg2, %arg2, %arg3, %c0, %arg4, %c64)",
            text,
        )
        self.assertRegex(
            text,
            r"scf\.for %lane_\d+ = %lb_\d+ to %ub_\d+ step %step_\d+ iter_args\(%rem_iter_\d+ = %rem_\d+\) -> \(i32\) \{",
        )
        self.assertRegex(
            text,
            r"%mask_\d+, %rem_\d+ = pto\.plt_b32 %rem_iter_\d+ : i32 -> !pto\.mask<b32>, i32",
        )
        self.assertIn(
            "pto.copy_ubuf_to_gm %arg2, %arg1, %c0_i64, %c16_i64, %c64_i64",
            text,
        )

    def test_if_else_and_sync_ops_lower_to_scf_if_and_authoring_sync_ops(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f32, pto.i32)])
        def kernel(inp: pto.TensorView, tile: pto.Tile, flag: pto.i32):
            pto.set_flag(pto.PIPE.MTE2, pto.PIPE.V, pto.EVENT.ID0)
            pto.wait_flag(pto.PIPE.MTE2, pto.PIPE.V, pto.EVENT.ID0)
            step = 64
            if flag:
                step = 64
                pto.set_flag(pto.PIPE.V, pto.PIPE.MTE3, pto.EVENT.ID0)
            else:
                step = 128
                pto.wait_flag(pto.PIPE.V, pto.PIPE.MTE3, pto.EVENT.ID0)
            with pto.strict_vecscope(tile, tile, 0, 256, step) as (src, dst, lb, ub, vec_step):
                for lane in range(lb, ub, vec_step):
                    mask = pto.make_mask(pto.f32, pto.PAT.ALL)
                    vec = pto.vlds(src, lane)
                    pto.vsts(vec, dst, lane, mask)
            pto.pipe_barrier(pto.PIPE.ALL)
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(16, 16),
                memory_space=pto.MemorySpace.UB,
            )
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        self.assertIsInstance(semantic_kernel.body[0], SemanticSetFlagStmt)
        self.assertIsInstance(semantic_kernel.body[1], SemanticWaitFlagStmt)
        self.assertIsInstance(semantic_kernel.body[3], SemanticIfStmt)
        self.assertIsInstance(semantic_kernel.body[5], SemanticPipeBarrierStmt)

        text = specialized.mlir_text()
        self.assertIn('pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]', text)
        self.assertIn('pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]', text)
        self.assertIn("= arith.cmpi ne, %arg2, %c0_i32 : i32", text)
        self.assertIn("%step_3 = scf.if %tmp_0 -> (index) {", text)
        self.assertIn('pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]', text)
        self.assertIn('pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]', text)
        self.assertRegex(text, r"scf\.yield %step_\d+ : index")
        self.assertIn("%step_2 = arith.constant 128 : index", text)
        self.assertIn("pto.strict_vecscope(%arg1, %arg1, %c0, %c256, %step_3)", text)
        self.assertIn("scf.for %lane_", text)
        self.assertIn("pto.barrier #pto.pipe<PIPE_ALL>", text)

    def test_strict_vecscope_rejects_implicit_capture_during_semantic_analysis(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f16, pto.i32)])
        def kernel(inp: pto.TensorView, tile: pto.Tile, scale: pto.i32):
            with pto.strict_vecscope(inp, tile) as (vin, vtmp):
                leaked = scale
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(8, 16),
                memory_space=pto.MemorySpace.UB,
            )
        )

        with self.assertRaises(ValueError) as ctx:
            analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        self.assertIn("implicit capture of 'scale' is not allowed", str(ctx.exception))


class TileLangDSLDiagnosticsTests(unittest.TestCase):
    def test_matcher_feature_validation_rejects_invalid_constraints_and_priority(self) -> None:
        def kernel(x: pto.TensorView):
            return None

        with self.assertRaises(TypeError) as constraints_ctx:
            pto.vkernel(op="x", dtypes=[(pto.f32,)], constraints=[123])(kernel)
        self.assertIn("constraints[0] must be callable", str(constraints_ctx.exception))

        with self.assertRaises(TypeError) as priority_ctx:
            pto.vkernel(op="x", dtypes=[(pto.f32,)], priority=True)(kernel)
        self.assertIn("priority must be an int", str(priority_ctx.exception))

    def test_advanced_mode_keeps_vreduce_rejected_until_authoring_op_exists(self) -> None:
        with self.assertRaises(pto.TileLangFrontendError) as ctx:

            @pto.vkernel(op="x", dtypes=[(pto.i32,)], advanced=True)
            def kernel(x: pto.Tile):
                pto.vreduce(x)
                return None

        self.assertIn("advanced family surface `pto.vreduce`", str(ctx.exception))

    def test_unsupported_python_syntax_reports_source_location(self) -> None:
        with self.assertRaises(pto.TileLangFrontendError) as ctx:

            @pto.vkernel(op="x", dtypes=[(pto.f32,)])
            def kernel(x: pto.TensorView):
                while True:
                    return None

        self.assertIn("unsupported Python syntax `while`", str(ctx.exception))
        self.assertIn(f"{__file__}:", str(ctx.exception))

    def test_arbitrary_external_call_reports_source_location(self) -> None:
        def helper():
            return None

        with self.assertRaises(pto.TileLangFrontendError) as ctx:

            @pto.vkernel(op="x", dtypes=[(pto.f32,)])
            def kernel(x: pto.TensorView):
                helper()
                return None

        self.assertIn("arbitrary external call `helper`", str(ctx.exception))
        self.assertIn(f"{__file__}:", str(ctx.exception))

    def test_unsupported_pto_surface_reports_source_location(self) -> None:
        with self.assertRaises(pto.TileLangFrontendError) as ctx:

            @pto.vkernel(op="x", dtypes=[(pto.f32,)])
            def kernel(x: pto.TensorView):
                pto.vadd(x)
                return None

        self.assertIn("vector op surface `pto.vadd` requires explicit pto.strict_vecscope", str(ctx.exception))
        self.assertIn(f"{__file__}:", str(ctx.exception))

    def test_advanced_family_requires_advanced_mode(self) -> None:
        with self.assertRaises(pto.TileLangFrontendError) as ctx:

            @pto.vkernel(op="x", dtypes=[(pto.f32, pto.f32)])
            def kernel(x: pto.TensorView, tile: pto.Tile):
                with pto.strict_vecscope(tile, tile, 0, 256, 64) as (lhs, rhs, lb, ub, step):
                    mask = pto.make_mask(pto.f32, pto.PAT.ALL)
                    pto.vcmp(lhs, rhs, mask, "lt")
                return None

        self.assertIn("surface `pto.vcmp` requires advanced=True", str(ctx.exception))
        self.assertIn(f"{__file__}:", str(ctx.exception))

    def test_missing_specialization_reports_source_location(self) -> None:
        @pto.vkernel(op="x", dtypes=[(pto.f32, pto.f16)])
        def kernel(x: pto.TensorView, tile: pto.Tile):
            return None

        with self.assertRaises(pto.TileLangFrontendError) as ctx:
            kernel.mlir_text()

        self.assertIn("requires specialize() bindings for bare Tile parameters", str(ctx.exception))
        self.assertIn(f"{__file__}:", str(ctx.exception))

    def test_dynamic_shape_and_illegal_profile_report_source_location(self) -> None:
        @pto.vkernel(op="x", dtypes=[(pto.f32, pto.f16)])
        def kernel(x: pto.TensorView, tile: pto.Tile):
            return None

        with self.assertRaises(pto.TileLangFrontendError) as dynamic_ctx:
            kernel.specialize(tile={"shape": (16, "n"), "memory_space": "ub"})
        self.assertIn("dynamic physical Tile shape is not supported", str(dynamic_ctx.exception))
        self.assertIn(f"{__file__}:", str(dynamic_ctx.exception))

        with self.assertRaises(pto.TileLangFrontendError) as rank_ctx:
            kernel.specialize(tile={"shape": (4, 4, 4), "memory_space": "ub"})
        self.assertIn("v1 only supports rank-1 or rank-2 Tile shapes", str(rank_ctx.exception))
        self.assertIn(f"{__file__}:", str(rank_ctx.exception))

        with self.assertRaises(pto.TileLangFrontendError) as space_ctx:
            kernel.specialize(tile={"shape": (4, 4), "memory_space": "gm"})
        self.assertIn("v1 only supports MemorySpace.UB", str(space_ctx.exception))
        self.assertIn(f"{__file__}:", str(space_ctx.exception))


if __name__ == "__main__":
    unittest.main()

TileLang DSL local documentation lives here.

Current docs:
- `v1-surface.md`: the TileLang DSL v1 contract implemented by
  `add-tilelang-dsl-core-foundation`
- `v1-lowering.md`: the TileLang DSL v1 authoring-form VPTO lowering contract
  implemented by `add-tilelang-dsl-authoring-vpto-lowering`
- `matcher-and-advanced-surface-migration.md`: migration notes from the
  original v1 core/lowering boundary to the matcher and advanced-surface
  capability implemented by
  `extend-tilelang-dsl-matcher-and-advanced-surface`

Documentation boundary:
- `tilelang-dsl/docs/` is the local documentation source of truth for the new
  `tilelang_dsl` frontend
- repository-level docs may link here, but should not redefine this package's
  implemented v1 boundary
- `python/pto/dialects/pto.py` is not the source of truth for TileLang DSL v1

# Proposal: 扩展 TileLang DSL 的 matcher 与 advanced surface

## 概述

`add-tilelang-dsl-core-foundation` 和 `add-tilelang-dsl-authoring-vpto-lowering` 会先收敛出一个可闭合的 v1 核心子集，但 `docs/tilelang-dsl-guide.md` 还定义了更完整的方向：kernel matcher、多 signature `dtypes`、`Any*` / `TypeVar`、constraint-based selection、implicit vecscope inference、raw pointer surface 和 advanced vector family。  
本 change 作为明确的 follow-up capability，负责把这些能力从“v1 diagnostics 中被拒绝的延期项”升级为正式契约，并继续要求相关工作集中在 `tilelang-dsl/` 下实现。

## 背景与动机

v1 核心 change 有意做了三项收缩：

1. 只保留单一 monomorphic `dtypes`
2. 只接受显式 `strict_vecscope`
3. 只支持 elementwise 套餐

这些收缩让首版可实现，但也与 `docs/tilelang-dsl-guide.md` 的完整愿景存在差距：

- guide 已经定义 matcher / priority / constraints / wildcard typing
- guide 已经把 implicit vecscope inference 作为默认 authoring 体验
- guide 还覆盖 raw pointer、低层 DMA、compare/select、predicate movement、rearrangement、reduction 等更大 surface

如果不把这些延期项显式收口成 follow-up change，v1 中的 reject diagnostics 就会长期停留在“未来再说”，缺少明确的能力落点。

## 目标

- 为 TileLang DSL 建立正式的 kernel matcher capability：多 signature `dtypes`、`Any*`、`TypeVar`、`constraints`、`priority` 和 deterministic selection。
- 为 TileLang DSL 建立 advanced surface capability：implicit vecscope inference、raw pointer / low-level DMA surface、advanced vector family。
- 保持核心实现继续集中在 `tilelang-dsl/`，不把 matcher 或 advanced lowering 回填到现有其他 Python binding 入口。

## 非目标

- 不修改 v1 基础 change 中已经固定的 package/目录边界。
- 不重新设计 `verify()` 的基本验证路径；advanced change 仍以当前 repo 的 VPTO legality 契约为输出收口。
- 不在本 change 中扩展到 `a5` 之外的 target。

## 变更内容

- 新增 `tilelang-dsl-kernel-matcher` capability，定义 kernel registry、match order、wildcard/type-variable 语义、constraint evaluation 和 selection tie-breaking。
- 新增 `tilelang-dsl-advanced-surface` capability，定义 implicit vecscope inference、raw pointer/UBRef authoring、low-level DMA surface 以及 advanced vector family 的扩展 lowering 契约。
- 要求 core-foundation change 中对延期 feature 的 reject diagnostics 在本 change 落地后转为正式支持路径。

## Capabilities

### New Capabilities

- `tilelang-dsl-kernel-matcher`: 定义多 kernel 注册、target/op/type/constraint/priority 匹配、wildcard typing 和 deterministic selection 契约。
- `tilelang-dsl-advanced-surface`: 定义 implicit vecscope inference、raw pointer / low-level DMA / UBRef surface 以及 advanced vector family 的扩展 lowering 契约。

### Modified Capabilities

- 无

## 预期结果

- TileLang DSL 从 v1 的“固定单 kernel elementwise 子集”扩展到可注册、可选择、可约束的 kernel authoring 体系。
- 当用户省略显式 scope 时，frontend 能按规则推断 `pto.vecscope`，同时继续保留 `strict_vecscope` 作为硬边界。
- raw pointer / low-level DMA / advanced family 有清晰 capability，而不再只是文档愿景。

## 成功标准

- 新增 `openspec/changes/extend-tilelang-dsl-matcher-and-advanced-surface/`，包含 proposal、design、tasks。
- 新增 `specs/tilelang-dsl-kernel-matcher/spec.md` 和 `specs/tilelang-dsl-advanced-surface/spec.md`。
- proposal/design/tasks 明确写清：
  - kernel registry / selection API
  - 多 signature `dtypes`、`Any*`、`TypeVar`、constraint evaluation 和 priority 决策顺序
  - implicit vecscope inference 的默认行为和边界
  - raw pointer / low-level DMA / advanced family 的支持范围

## 影响

- 受影响目录：
  - `tilelang-dsl/python/`
  - `tilelang-dsl/tests/`
  - `tilelang-dsl/examples/`
  - `tilelang-dsl/docs/`
- 受影响 public API：
  - `@pto.vkernel(... dtypes=[...], constraints=[...], priority=...)`
  - `pto.select_kernel(...)` 或等价 registry 查询入口
  - implicit vecscope inference 相关的 compile behavior

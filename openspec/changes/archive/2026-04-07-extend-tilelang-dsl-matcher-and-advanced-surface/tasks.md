## 1. OpenSpec 契约落定

- [ ] 1.1 新增 `openspec/changes/extend-tilelang-dsl-matcher-and-advanced-surface/specs/tilelang-dsl-kernel-matcher/spec.md`，固定 registry、selection API、wildcard/type-variable、constraint 与 priority 规则。
- [ ] 1.2 新增 `openspec/changes/extend-tilelang-dsl-matcher-and-advanced-surface/specs/tilelang-dsl-advanced-surface/spec.md`，固定 implicit vecscope inference、raw pointer / low-level DMA 和 advanced family 的扩展契约。
- [ ] 1.3 在 `proposal.md` 和 `design.md` 中明确该 change 依赖 core-foundation 与 authoring-vpto-lowering 两个前置 change。

## 2. Matcher 能力

- [ ] 2.1 在 `tilelang-dsl/python/` 中实现 `KernelRegistry` 和 `pto.select_kernel(...)` 入口。
- [ ] 2.2 实现多 signature `dtypes`、`Any*`、`TypeVar` 的 matcher 语义和 deterministic selection 顺序。
- [ ] 2.3 实现 `constraints` evaluation 与 `priority` 决策；对最高优先级 tie 保持显式报错。

## 3. Advanced surface

- [ ] 3.1 实现 implicit vecscope inference，并保证 `strict_vecscope` 仍然是硬边界。
- [ ] 3.2 扩展 raw pointer / UBRef / low-level DMA / `copy_ubuf_to_ubuf` surface 到 authoring-form VPTO lowering。
- [ ] 3.3 扩展 compare/select、predicate movement、carry、rearrangement、reduction family 的 lowering 支持。

## 4. 测试与文档

- [ ] 4.1 在 `tilelang-dsl/tests/` 增加 matcher regression，覆盖 wildcard/type-variable、constraint fallback、priority tie error。
- [ ] 4.2 增加 vecscope inference regression，覆盖连续 vector chain 自动分组、scalar/control-flow 边界切断、`strict_vecscope` 边界保留。
- [ ] 4.3 增加 raw pointer / low-level DMA / advanced family regression，确认输出仍满足当前 VPTO legality contract。
- [ ] 4.4 在 `tilelang-dsl/docs/` 更新从 v1 core 到 matcher/advanced-surface 的迁移说明，并同步 `docs/tilelang-dsl-guide.md` 的已支持/延期状态。

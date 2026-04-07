# tilelang-dsl-kernel-matcher Specification

## ADDED Requirements

### Requirement: TileLang DSL MUST provide a deterministic kernel registry and selection API

当同一 `target/op` 下存在多个 `@pto.vkernel` descriptor 时，TileLang DSL MUST 将它们注册到可查询的 registry。  
系统 MUST 提供显式 selection API，用于在给定 `target`、`op`、operand type 信息和上下文属性时选择唯一 kernel。  
选择过程 MUST deterministic。

#### Scenario: selector returns the unique best kernel

- **WHEN** registry 中存在多个针对同一 `target/op` 的 kernel descriptor，且其中一个在全部匹配步骤后成为唯一最佳候选
- **THEN** `pto.select_kernel(...)` MUST 返回该 descriptor
- **AND** 返回结果 MUST 可继续走 `specialize()` / `mlir_text()` / `verify()` 流程

### Requirement: matcher MUST support concrete types, `Any*`, and `TypeVar` across multiple signatures

matcher MUST 支持：

- 多个 `dtypes` signature
- `AnyFloat`
- `AnyInt`
- `AnyType`
- `AnyMask`
- `TypeVar`

`TypeVar` 在单个 signature 内 MUST 约束所有同名位置绑定到同一最终类型。

#### Scenario: wildcard and type-variable signatures match deterministically

- **WHEN** 某个 kernel 使用多个 `dtypes` signature，并在其中混用 concrete type、`Any*` 与 `TypeVar`
- **THEN** matcher MUST 对每个 signature 独立求值
- **AND** 只有满足所有 `TypeVar` 一致性约束的 signature 才能视为匹配成功

### Requirement: constraint evaluation MUST happen after type matching and before priority resolution

对同一 `target/op` 的候选集合，matcher MUST 先完成 dtype matching，再评估 `constraints`。  
只有通过 constraint evaluation 的候选，才允许进入 `priority` 比较阶段。

#### Scenario: higher-priority kernel with failing constraint does not win

- **WHEN** 一个更高 `priority` 的 kernel 在 target/op/type 层面匹配成功，但 `constraints` 评估失败
- **THEN** 该 kernel MUST 从候选集合中移除
- **AND** selector MUST 继续在剩余候选中选择合法 kernel

### Requirement: priority ties MUST raise an explicit selection error

若在 target/op/type/constraint 全部通过后，最高 `priority` 仍对应多个候选，matcher MUST 报显式选择错误。  
系统 MUST NOT 依赖定义顺序、导入顺序或其他隐式规则做 tiebreak。

#### Scenario: equal-priority winners cause deterministic tie error

- **WHEN** 多个 kernel 在 target/op/type/constraint 匹配后拥有相同的最高 `priority`
- **THEN** selector MUST 报错
- **AND** 错误消息 MUST 指出发生 tie 的 kernel 集合
- **AND** MUST NOT 静默选择第一个已注册 kernel

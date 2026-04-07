# tilelang-dsl-advanced-surface Specification

## ADDED Requirements

### Requirement: advanced mode MUST infer `pto.vecscope` for eligible vector chains while preserving `strict_vecscope` boundaries

在 advanced mode 下，当用户省略显式 scope 且书写连续的 supported vector chain 时，frontend MUST 自动推断 dedicated `pto.vecscope`。  
scalar op、控制流边界、外部 call 和显式 `strict_vecscope` MUST 切断该推断。  
`strict_vecscope` 继续作为硬边界，inference MUST NOT 穿越其边界。  
inference 结果 MUST 继续满足当前 authoring-form VPTO legality contract，不得因为自动推断而放宽 typed-mask、capture operand、地址形态或 vecscope carrier 约束。

#### Scenario: contiguous vector chain becomes one inferred `pto.vecscope`

- **WHEN** 用户在 advanced mode 下连续书写一条由 load -> vector ALU -> store 组成的纯 vector chain，且中间没有 scalar/control-flow boundary
- **THEN** frontend MUST 将该 chain lower 为一个 dedicated `pto.vecscope`
- **AND** 该 inferred vecscope MUST 满足当前 VPTO authoring legality contract

#### Scenario: scalar or control-flow boundary cuts vecscope inference

- **WHEN** 一条候选 vector chain 中间穿插 scalar op、`if` / `for` 边界或外部 call
- **THEN** frontend MUST 在该边界处切断 inference
- **AND** MUST NOT 把边界两侧的 vector 片段合并到同一个隐式 `pto.vecscope`

#### Scenario: explicit `strict_vecscope` remains an inference barrier

- **WHEN** 用户在 advanced mode 下显式书写 `strict_vecscope`
- **THEN** frontend MUST 保留该 `strict_vecscope` 原样语义
- **AND** scope inference MUST NOT 跨越该显式边界去并合前后 vector chain

### Requirement: advanced mode MUST support raw pointer, UBRef, low-level DMA, and `copy_ubuf_to_ubuf` authoring

advanced mode MUST 将以下 surface 纳入正式契约：

- `castptr`
- `addptr`
- raw UBRef load/store authoring
- low-level DMA programming
- `copy_ubuf_to_ubuf`

这些 surface 仍 MUST lower 到当前合法的 authoring-form VPTO，不得发明另一套公开中间 IR。  
对于 raw pointer 与 UBRef 相关 surface，frontend MUST 继续遵守当前 ptr-only / buffer-like / copy-family 地址契约。

#### Scenario: low-level pointer and DMA surface lowers to legal authoring-form VPTO

- **WHEN** 用户使用 `castptr`、`addptr`、raw UBRef、低层 DMA programming 或 `copy_ubuf_to_ubuf`
- **THEN** frontend MUST 生成对应的合法 authoring-form VPTO surface
- **AND** 输出结果 MUST 继续满足当前 copy/buffer-like/ptr-only 地址契约

#### Scenario: `copy_ubuf_to_ubuf` remains inside the existing DMA and address contract

- **WHEN** 用户在 advanced mode 下书写 `copy_ubuf_to_ubuf`
- **THEN** lowering MUST 只生成当前 VPTO 已允许的 UB-to-UB copy programming 与 copy surface
- **AND** MUST NOT 通过额外的公开 helper IR 绕过现有 legality 检查

### Requirement: advanced mode MUST extend lowering to compare/select, predicate movement, carry, and rearrangement family capability sets

advanced mode MUST 将以下 family 分组纳入正式 lowering capability：

- compare/select
- predicate movement
- carry family
- rearrangement

这些 family 的 lowering MUST 继续落在当前 authoring-form VPTO contract 内，并与已有 typed-mask、vecscope、pointer/buffer legality 规则兼容。  
对未进入这些 capability set 的 family，frontend MUST 继续显式 reject。

#### Scenario: advanced family kernel lowers without leaving the authoring-form VPTO contract

- **WHEN** 用户在 advanced mode 下使用 compare/select、predicate movement、carry 或 rearrangement family 编写 kernel
- **THEN** frontend MUST 为该 family 生成合法的 authoring-form VPTO IR
- **AND** typed-mask、vecscope 和地址形态契约 MUST 与当前 VPTO legality contract 保持一致

#### Scenario: family outside the declared advanced capability set is still rejected

- **WHEN** 用户使用未纳入上述 capability set 的 family
- **THEN** frontend MUST 继续报 unsupported-feature 错误
- **AND** MUST NOT 因启用了 advanced mode 就默认放开全部 VPTO family

### Requirement: advanced mode MUST keep reduction-family authoring rejected until a public authoring-form VPTO op exists

当前 repo 尚未暴露可供 TileLang DSL 直接复用的 reduction authoring-form VPTO op。  
因此，在该 authoring 契约存在之前，frontend MUST 继续显式 reject reduction family surface，MUST NOT 通过额外公开 helper IR 或绕经 OpLib/EmitC 专用路径把 reduction 伪装成当前 capability 的一部分。

#### Scenario: reduction family remains deferred without a public authoring-form VPTO op

- **WHEN** 用户在 advanced mode 下尝试书写 reduction family surface
- **THEN** frontend MUST 报 unsupported-feature 错误
- **AND** MUST 说明该 family 仍处于 follow-up / deferred 状态

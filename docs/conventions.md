# 仓库约定

## Lore commit protocol

提交是制度知识的原子单位。格式:

```
<intent line: 为什么改,不是改了什么>

<body: 约束、方案取舍的叙述>

Constraint: <塑造决策的外部约束>
Rejected: <被否掉的替代方案> | <否决理由>
Confidence: <low|medium|high>
Scope-risk: <narrow|moderate|broad>
Directive: <给未来修改者的前置警告>
Tested: <验证过什么(unit/integration/manual)>
Not-tested: <已知的验证缺口>
```

规则:
1. 第一行写 **why**,diff 已经说明 what。
2. trailer 可选但鼓励;只写有价值的。
3. `Rejected:` 防止后来者重新踩同一个坑。
4. `Directive:` 是留给未来的"改 X 之前先查 Y"。
5. `Constraint:` 记录外部力量(API 限制、上游 bug、策略要求)。
6. `Not-tested:` 必须诚实 —— 声明缺口比假装全覆盖更有价值。
7. 全部使用 git 原生 trailer 格式(空行后 key-value)。

## 代码风格

- ruff(`line-length=100`,select E/F/I/W/UP;UP042/UP046 因 py310 兼容忽略)。
- 公开接口写中文 docstring,交代对应设计文档章节;实现细节少注释。
- 新依赖需要显式理由(默认拒绝);优先复用 `channellm` 内工具。

# GR00T 与 OpenPI 的统一结构

两边共用同一套调度逻辑，各自保留模型执行接口，不让一个后端依赖另一个后端。

```text
elegantvla/
  scheduling/        共用动作定义、教师阈值、RL 推理接口和缓存控制器
  gr00t/core/        GR00T 的 wrapper / adapter / action_head
  openpi/core/       OpenPI 的 wrapper / adapter / action_head
  openpi/legacy/     原来的实验配置与实现，显式启用才会执行
```

## 一次推理

1. `Policy` 做模型自己的输入变换，交给 `wrapper`。
2. 非 Image-Free 时，`adapter` 执行首层 probe；这一份结果同时给 CKA 和后续执行使用。
3. `rule_policy` 或 `rl_policy` 选择 VLM 五档、Action Head 三档。
4. `wrapper` 检查缓存可用性与连续复用限制，让 `adapter` 执行相应路径。
5. `action_head` 按选定的去噪调度输出动作，成功后提交缓存状态。

## 什么是相同的

| 项目 | 两边共同的规则 |
|---|---|
| VLM 五档 | 全算、首尾重算、首尾重算后再接 1/2/3 次 Image-Free |
| 默认边界 | 首层 1 层、尾层 1 层重算；中间复用 |
| Probe | 计算一次，继续用于该次推理，不重复算首层 |
| 中间缓存 | 完整刷新时更新；首尾重算不覆盖原来的中间锚点 |
| Image-Free | 使用最近一次产出的条件缓存，不执行 vision encoder 和 VLM |
| Action Head 三档 | 去噪每步全算、首尾步全算、只有首步全算 |
| 调度接口 | `build_schedule_state -> select_action -> execute_schedule_action` |

这里的 Action Head 首尾是去噪时间步，VLM 首尾是网络层，不能混为一谈。

## 什么需要各自适配

GR00T 把 VLM 输出的 token 交给独立 DiT，DiT 再投影出 cross-attention K/V。
OpenPI 的 action expert 则直接使用 VLM 各层产生的 prefix K/V。

因此 OpenPI 的首尾重算具体是：

```text
当前首层 -> 当前首层 K/V
    |
    + 缓存的中间总残差     中间各层 K/V 沿用完整刷新时的缓存
    |
当前尾层 -> 当前尾层 K/V
    |
首层新 K/V + 中间旧 K/V + 尾层新 K/V -> action expert
```

Image-Free 使用上次组装好的整份 K/V，动作输入仍然取当前状态。
Pi0.5 写入 prefix 的离散状态 token 属于被复用的条件，不额外声称它们是当前帧重算的。

## 实现与验证范围

- OpenPI 的统一调度路径支持 PyTorch；JAX baseline 和旧配置通过独立兼容路径提供。
- `legacy/` 仅在显式启用旧配置时执行，不参与统一路径的调度。
- 共用 RL 推理接口不代表训练环境或 checkpoint 可以跨模型直接使用；OpenPI 尚未提供对应的 RL 训练环境。
- GR00T 包含可选融合算子与 CUDA Graph，OpenPI 尚未接入这些专用优化，详见[四层执行架构](soft_hardware_architecture_zh.md)。
- OpenPI 已有 CPU 小模型 forward 测试；两个发布后端的全尺寸权重、GPU 延时和闭环成功率仍需单独验证。

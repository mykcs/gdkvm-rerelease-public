# Current | mykcs/gdkvm-rerelease-public

这是长期协作 Agent 的统一接手入口，遵循 [共享 Handoff Pattern](https://github.com/mykcs/.agents/blob/main/docs/agents/HANDOFF_PATTERN.md)。Current 负责导航和恢复上下文，不是实时锁，也不复制 Wish、Dev 或产品规则。

## 项目身份

- 已登记的项目职责：public GDKVM re-release source and reproducible training/evaluation tooling; scientific qualification stays in gdkvm-rerelease
- 根 [AGENTS](../../../AGENTS.md) 是项目 Agent 规则入口；任务相关事实仍应按项目既有标准读取。
- [Current STATUS](STATUS.md)：有时间和 base SHA 的当前状态快照，不能替代 GitHub 实时状态。
- [Wish](../../wish/LATEST.md) 决定目标，不在 Current 复制。
- [Dev](../../dev/LATEST.md) 决定技术/CI/实施方式，不在 Current 复制。

## 新 Agent 接手顺序

1. 读取 AGENTS、本页以及当前 STATUS 指向的规范来源，不把历史记录当作当前授权。
2. **实时**检查默认分支 HEAD、[Issues](https://github.com/mykcs/gdkvm-rerelease-public/issues)、[PRs](https://github.com/mykcs/gdkvm-rerelease-public/pulls)、相关分支与任务认领。
3. 根据现行 Issues/PR 判断是否有人正在编辑同一文件；若已有 owner，优先只读审查或补独立文件，不重建第二条相同工作线。
4. 确认目标、受控写入范围、base/head SHA 和验收证据；在独立分支做修改并通过 PR 交付。
5. 跨窗口交接在 Issue/PR 留下实际结果、文件、命令、风险、下一动作；集成者串行合并并核实远端 main。

## 权威与新鲜度

- 意图和科研/产品规则：项目 AGENTS、Wish、代码和原始来源。
- CI、构建、部署和权限：Dev/配置及实时 provider，不从 Current 快照推断。
- 任务、owner、PR head/base：**GitHub 实时数据**，不是任何 Markdown 的当前数值。
- 最新项目状态：STATUS 仅保存有证据的项目级快照；用户或实际任务改变后才由集成者更新。

未经授权不发布、不调整仓库权限、不删除原有工作线；该目录不创建自动调度器或第二套任务看板。

# 山地复发起冰风险建模与硬预算预警

本项目研究一个面向稀有复发事件的完整方法链：利用过去24小时观测，在10分钟时间步上预测未来6小时新起冰风险，再在逐站逐月误报预算约束下决定是否发出告警。主概率模型统一命名为 `local_weather_hazard`，策略和全部参数仅由2022年OOF数据选择，2023与2024年只做冻结评价。

2026年8月26日，严格事件网格全链实验、动态硬预算控制器、消融、敏感性分析、站点簇Bootstrap、电商和美国事故公共诊断，以及标准DMD/随机背包基线、嵌套代价Bootstrap和复杂度审计均已完成。本轮理论补充实验耗时45分35秒；服务器 `experiments/` 的55个文件已合并到本地，其中31个同名文件被替换、24个新文件被加入，并通过完成标记及清单内4个关键SHA-256校验。

完整结论与数值解读见 [docs/results_analysis.md](docs/results_analysis.md)。本文件只说明项目结构、评价合同、复现入口和结果位置。

## 最重要的结论

- 山地起冰适合建模为离散时间复发风险过程；本站冷湿环境和气象时间演变是最稳定的预测信息。
- 完整复发历史、邻站滞后信号和复杂时序网络没有表现出稳定的跨年增量优势，因此主概率模型采用 `local_weather_hazard`。
- 严格网格修正后，完整窗口队列覆盖2023年的84/116个事件、2024年的116/181个事件；运行现实队列分别覆盖113/116和170/181个事件。
- 新控制器 `uadhbac` 在2023与2024、2/5/10/20小时四档预算上均保持逐站逐月零超支、未结算容量可行和跨预算严格嵌套。
- 在预先指定的5小时和10小时预算上，`uadhbac` 相对 `budget_safe` 的运行现实队列提前量效用增益均为正，并通过5000次站点簇Bootstrap检验。
- 新增标准DMD基线的保留预算利用率达到90.4%至98.8%，但存在跨预算嵌套冲突；switch-over在线背包基线同时满足硬预算和嵌套，却只利用27.0%至34.3%的保留预算。`uadhbac`在四个预设“年份×5/10小时”运行点上的效用均高于这两类标准基线。
- 公共数据仅用于跨领域流程诊断，不构成外部覆冰验证；美国事故结果尤其应作为诊断性压力测试解释。

## 冻结评价合同

| 项目 | 固定设置 |
| --- | --- |
| 主概率模型 | `local_weather_hazard` |
| 历史窗口 | 24小时 |
| 决策时间步 | 10分钟 |
| 风险预测窗口 | 6小时，共36步 |
| 策略选择数据 | 2022年 pooled OOF |
| 冻结年份 | 2023、2024 |
| 主预算 | 2、5、10、20小时/站点/月 |
| 主评价队列 | 运行现实队列；完整窗口队列同步报告 |
| 硬预算定义 | 已确认误报 + 未结算告警预留容量均不超过预算 |
| 跨预算要求 | 低预算告警集合必须是高预算告警集合的子集 |
| Bootstrap | 5000次站点整簇重采样 |
| 严格事件合同 | `strict_event_grid_reset_2026-08-25` |

严格网格要求最后一个预测点严格早于事件起点且相差不超过一个10分钟步长。在线控制器和离线评价器都在已观测事件边界切分告警段，避免容量释放与严格误报审计使用不同的告警段定义。

## 项目结构

```text
ice_project/
├── README.md                       # 项目说明（本文件）
├── configs/                        # 实验合同与模型配置
├── data/                           # 原始数据和公共基准数据
├── docs/                           # 综合结果、项目规范与维护文档
├── requirements/                   # 依赖清单
├── scripts/                        # 命令入口、实验与维护脚本
├── src/rig_hazard/                 # 风险建模、控制器与评价源码
├── tests/                          # 单元与回归测试
├── visualization/                  # 论文图与集中导出的可视化结果
│   └── manuscript_figures/         # 论文主图及配套图形数据
└── results/                        # 指标、预测、检查点和运行日志
    ├── dynamic_hard_budget/        # 论文主线冻结结果
    │   ├── local_weather_hazard_trajectory/ # 主模型冻结风险轨迹
    │   ├── experiments/            # 动态控制器、基线、消融和敏感性分析
    │   └── public_recurrence_benchmarks/ # 电商与美国事故诊断
    ├── alert_governance/           # 历史告警治理与严格网格基线
    ├── recurrence_analysis/        # 事件定义、统计模型与基础比较
    ├── recurrence_modeling/        # 公平基线、空间泛化和补充实验
    ├── icing_model_experiments/    # 双尺度天气、复发门控和旧嵌套告警
    └── legacy_exploration/         # 早期探索，仅供溯源
```

`results/dynamic_hard_budget/` 是当前冻结结果入口，`visualization/manuscript_figures/`
是集中成图入口。`src/` 是源码根目录，`src/rig_hazard/` 是 Python 导入包；两层不能
合并，否则会破坏 `import rig_hazard`、`python -m rig_hazard` 和可编辑安装。其他结果
目录保留论文各实验线的原始与历史产物，不应与冻结主表混用。文件和代码命名规则见
[docs/project_conventions.md](docs/project_conventions.md)。

## 大型数据与结果下载

为避免将约31 GiB的数据与实验产物写入Git历史，`data/`和`results/`不随源码仓库
克隆。它们以小于2 GiB的7-Zip分卷发布在
[project-assets-initial](https://github.com/Xp0712/-UAD-HBAC/releases/tag/project-assets-initial)
Release 中：

```powershell
# 下载全部 data.7z.* 与 results.7z.* 分卷后，在项目根目录执行
7z x data.7z.001 -o.
7z x results.7z.001 -o.
```

解压时只需指定每组的第一个 `.001` 文件，7-Zip会自动读取其余分卷并恢复原目录。
Release 同时提供SHA-256清单，用于下载后的完整性校验。

## 你询问的五个结果文件

当前代码采用以下实际文件名：

| 预期名称 | 当前实际文件 | 说明 |
| --- | --- | --- |
| `frozen_main_comparison.csv` | `results/dynamic_hard_budget/experiments/frozen_main_results.csv` | 176行统一冻结主表，包含基线、在线控制器、硬守卫、8项消融和oracle诊断。 |
| `paired_station_bootstrap.csv` | `results/dynamic_hard_budget/experiments/paired_station_cluster_bootstrap.csv` | 48行、5000次站点整簇Bootstrap汇总。 |
| `ablation_results.csv` | `results/dynamic_hard_budget/experiments/frozen_main_results.csv` 中 `method` 以 `dynamic_` 开头的64行 | 消融被并入统一主表，没有重复保存一份同值CSV。 |
| `nestedness_audit.csv` | `results/dynamic_hard_budget/experiments/frozen_nesting_audit.csv` | 78行跨预算嵌套审计。 |
| `station_month_budget_audit.csv` | `results/dynamic_hard_budget/experiments/frozen_station_month_audit.csv.gz` | 59,752行逐方法、逐站、逐月预算审计；使用gzip压缩。 |

没有为这些名称额外复制别名文件，以免同一结果出现多个可能漂移的副本。

## 关键结果文件

| 文件 | 用途 |
| --- | --- |
| `experiments/selected_controller_2022.json` | 只基于2022 OOF选定的控制器参数和轨迹哈希。 |
| `experiments/frozen_main_results.csv` | 2023/2024全部冻结方法与预算结果。 |
| `experiments/paired_station_cluster_bootstrap.csv` | `uadhbac` 与预设比较方法的站点簇区间和P值。 |
| `experiments/frozen_event_records.csv.gz` | 26,136条严格事件匹配记录。 |
| `experiments/frozen_station_month_audit.csv.gz` | 59,752条月度误报、未结算容量和利用率记录。 |
| `experiments/frozen_nesting_audit.csv` | 跨预算告警集合嵌套性。 |
| `experiments/dense_budget_sensitivity.csv` | 1/2/3/5/7/10/15/20/30小时预算敏感性。 |
| `experiments/utility_function_sensitivity.csv` | 四种提前量效用定义的敏感性分析。 |
| `experiments/prediction_window_sensitivity.csv` | 1/3/6小时预测窗口敏感性。 |
| `experiments/hard_guard_algorithm_taxonomy.csv` | 原六类硬守卫与新增标准DMD、switch-over在线背包基线的算法谱系审计。 |
| `experiments/standard_baseline_selection_2022.csv`、`selected_standard_baselines_2022.json` | 11个2022 OOF候选运行点及冻结选中的DMD、switch-over参数。 |
| `experiments/standard_algorithm_baselines.csv` | 新增标准在线算法在2023/2024冻结年份的比较结果。 |
| `experiments/standard_baseline_station_cluster_bootstrap.csv` | `uadhbac`相对两类新增标准基线的48条配对站点簇Bootstrap。 |
| `experiments/nesting_utility_cost.csv` | 同参数耦合/非耦合控制器的逐年份、逐预算嵌套效用代价。 |
| `experiments/nesting_cost_station_cluster_bootstrap.csv` | 嵌套效用代价的完整窗口与运行现实队列站点簇Bootstrap。 |
| `experiments/controller_complexity_audit.csv` | 理论复杂度、实测运行时间、吞吐量和主要数组空间下界。 |
| `experiments/theory_supplement_manifest.json` | 理论补充实验的冻结年份、预算、总耗时、合同版本和关键SHA-256。 |
| `public_recurrence_benchmarks/*/frozen_test_metrics.csv` | 两个公共数据集的冻结指标。 |
| `experiments/frozen_run_manifest.json`、`public_recurrence_benchmarks/*/run_manifest.json` | 数据合同、模型配置、运行时间和文件哈希。 |

相对路径均以 `results/dynamic_hard_budget/` 为起点。

## 环境与测试

建议使用 Python 3.10 及以上版本：

```powershell
python -m pip install -r requirements/runtime.txt
python -m pip install -r requirements/deep_learning.txt
python -m pip install -e . --no-deps
python -m unittest discover -s tests -v
```

安装为可编辑包后，可使用 `rig-hazard --help` 或 `python -m rig_hazard --help`
进入统一命令行接口；
一次性实验仍从 `scripts/` 运行。

当前本地全量回归测试共133项并全部通过。GPU用于风险轨迹导出和支持CUDA的
XGBoost训练；控制器回放、严格匹配、月度审计和Bootstrap主要使用CPU。

## 主要复现入口

```powershell
# 导出冻结的 local_weather_hazard 36步风险轨迹
python scripts/export_selected_full_trajectory.py --device cuda

# 2022选择，随后冻结评价2023/2024，并运行消融与敏感性分析
python scripts/run_dynamic_hard_budget_experiments.py --phase all

# 只补算标准在线基线、嵌套代价Bootstrap和复杂度审计
python scripts/run_dynamic_hard_budget_experiments.py --phase theory

# 公共重复事件诊断；--resume 会复用模型和策略检查点
python scripts/run_public_recurrence_benchmarks.py `
  --datasets ecommerce us_accidents `
  --output-root results/dynamic_hard_budget/public_recurrence_benchmarks `
  --resume
```

已完成结果不应直接使用 `--overwrite` 覆盖。新的探索实验应使用新的输出目录；正式结论只从冻结主表和对应运行清单读取。

## 术语

| 术语 | 含义 |
| --- | --- |
| 风险集 | 当前无覆冰、未处于冷却期且具备合法历史输入时，可发生下一次事件的时间点集合。 |
| 完整窗口队列 | 事件前36个预测点全部合法，用于标准化比较。 |
| 运行现实队列 | 未来6小时内至少存在一个合法预测点，用于部署覆盖评估。 |
| 未结算告警 | 随访尚未结束，暂时不能判定为误报的告警。 |
| 保留容量 | 已确认误报与未结算告警共同占用的预算容量。 |
| 硬可行 | 每个站点、每个月的保留容量都不超过预算。 |
| 嵌套 | 任意低预算档位的告警均包含在更高预算档位中。 |
| 提前量效用 | 事件命中率与有效提前量共同形成的效用指标。 |

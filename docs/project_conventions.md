# 项目结构与命名规范

本项目的公开名称为 **RIG-Hazard**，主概率模型统一命名为
`local_weather_hazard`。新文件、新配置、新结果目录和用户可见报告均使用功能语义命名，
不再使用迭代序号或阶段编号。

## 目录职责

| 目录 | 职责 | 命名方式 |
| --- | --- | --- |
| `src/rig_hazard/` | 可复用的模型、控制器和评价代码 | 名词或领域概念，如 `risk_trajectory.py` |
| `configs/` | 可复现实验合同 | `rig_hazard_<用途>.json` |
| `scripts/` | 可直接运行的维护与实验脚本 | 动词开头，如 `run_`、`export_`、`inspect_` |
| `tests/` | 单元和回归测试 | 与被测模块对应的 `test_<模块>.py` |
| `requirements/` | 分层依赖清单 | 按作用域命名，如 `runtime.txt` |
| `docs/` | 长期维护文档 | 小写 `snake_case.md` |
| `results/` | 不可变实验产物 | 按模型或实验目的命名 |
| `visualization/` | 论文图与集中导出的可视化产物 | 按图件用途命名 |

顶层不得再次创建旧式源码、产物、工具或日志目录。
源码包必须位于 `src/`；实验指标、预测、检查点和日志必须位于 `results/`；集中导出的
论文图必须位于 `visualization/`。`src/rig_hazard/` 中，前者是源码根目录，后者是
Python 包目录，不得把包内模块直接平铺到 `src/`。

原始数据、冻结结果和检查点不作为源码重构对象。它们内部可能保留旧产物标识，读取兼容性
集中在 `src/rig_hazard/naming.py` 的边界常量中；新接口不得继续传播旧标识。

## 文件与标识规则

- Python、Shell、JSON 和 Markdown 文件使用小写 `snake_case`；Python 类使用 `PascalCase`，
  函数、变量和配置键使用 `snake_case`，常量使用 `UPPER_SNAKE_CASE`。
- 名称必须表达对象和用途，例如 `run_spatial_generalization_queue.sh`，不使用 `v2`、
  `v3`、`m0`、`p3`、`latest`、`new`、`final2` 等时间性或顺序性名称。
- 主模型使用 `local_weather_hazard`；主模型轨迹目录使用
  `local_weather_hazard_trajectory`。
- 合同变化使用语义合同 ID、日期和内容哈希记录在 manifest 中，不把版本号写入文件名。
- 实验重复运行使用 `YYYYMMDD_HHMMSS` 时间戳或配置哈希区分；不要复制出带递增后缀的脚本。
- 布尔值以 `is_`、`has_`、`should_`、`enable_` 开头；集合使用复数名；路径变量以
  `_path` 或 `_root` 结尾。

## 当前语义映射

| 旧名称 | 统一名称 |
| --- | --- |
| Python 源码目录 | `src/rig_hazard/` |
| 可执行脚本目录 | `scripts/` |
| 实验产物与日志 | `results/` |
| 集中可视化结果 | `visualization/` |
| 默认配置 | `configs/rig_hazard_preprocessing.json` |
| 基础依赖 | `requirements/runtime.txt` |
| 深度学习依赖 | `requirements/deep_learning.txt` |
| 局地天气危害率基线 | `results/local_weather_hazard_baselines/` |
| 深度模型合同审计 | `results/deep_model_experiments/model_contract_audit/` |
| 主模型轨迹目录 | `results/dynamic_hard_budget/local_weather_hazard_trajectory/` |
| 空间泛化排队脚本 | `scripts/run_spatial_generalization_queue.sh` |
| 主实验恢复脚本 | `scripts/resume_main_experiments.sh` |

## 代码可读性基线

- 模块文档字符串放在 `from __future__` 之前；公开函数说明输入、输出和关键不变量。
- 优先使用 `pathlib.Path`、类型标注、数据类和具名常量，避免散落的路径、模型名和魔法数字。
- 函数只承担一个可描述的职责；长流程拆成加载、校验、计算、写出四类步骤。
- 修改配置路径或脚本名时，同步更新 CLI 默认值、Shell 调用、远程上传清单、README 和测试。
- 提交前至少执行语法编译、命名规范测试和全量 `unittest`。

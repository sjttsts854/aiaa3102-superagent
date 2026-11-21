# AIAA3102 SuperAgent v0.3.0

**v0.3.0 是当前唯一维护版本。** 活跃源码位于 `superagent/`，推荐入口是 `llm-code` 配合 `config/hkust-deepseek-flash-code-json.json`。版本身份、源码指纹及检查结果见 [正式版本元数据](releases/v0.3.0/release.json)，成绩及其来源见 [结果登记表](docs/results_registry.json)。

SuperAgent 接收分类任务，自动完成：通用基线 → LLM 生成完整 Python → 训练与验证 → 根据分数或错误修改代码 → 按验证分数选模 → 重训并保存模型。LLM 可以设计新特征、组合估计器和自定义分类器，候选不受预置算法目录限制。程序不按数据集名称选择专用解法。

支持 CSV 表格、单列文本、有顺序的双文本、独立 NPY 特征／标签，以及 CSV 清单引用的 RGB 图像。新数据需满足任务配置和输入接口；这些能力不保证任意未见任务都能成功。

## 安装与配置

在仓库根目录运行以下 PowerShell 命令。已在 Python 3.13.5 验证依赖，建议使用独立虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
Copy-Item .env.example .env.local
```

仅首次配置时复制 `.env.example`。在本机 `.env.local` 的 `AIGC_API_KEY=` 后填入自己的学校 API Key，或设置同名环境变量。密钥不写入代码、日志或 Git。学校配置须能访问校园内网接口，默认绕开系统代理并使用配置中的校园地址。

先做不发送请求的配置检查，再运行离线测试：

```powershell
.\.venv\Scripts\python.exe -m superagent check-config --llm-config config/hkust-deepseek-flash-code-json.json
.\.venv\Scripts\python.exe -m pytest -q
```

可选的真实连通检查只发送一次收费请求，不读取数据或训练模型；每次使用新的输出目录：

```powershell
.\.venv\Scripts\python.exe -m superagent check-api --llm-config config/hkust-deepseek-flash-code-json.json --max-cost-rmb 0.10 --output runs/api_v030_001
```

仓库不包含 API Key、课程数据、虚拟环境、原始运行日志或训练模型。下载源码后，需自行安装依赖、配置密钥并准备数据。

## 运行、预测与最终评分

首次使用可生成独立合成任务，检查完整流程。生成命令不会覆盖已有数据；合成成绩不代表课程任务成绩：

```powershell
.\.venv\Scripts\python.exe -m superagent make-code-demo --output examples/code_demo
.\.venv\Scripts\python.exe -m superagent campaign --task examples/code_demo/tabular/task.json --planner llm-code --llm-configs config/hkust-deepseek-flash-code-json.json --budget-rmb 0.25 --output runs/demo_v030_001
.\.venv\Scripts\python.exe -m superagent predict --model runs/demo_v030_001/planner_01/model.joblib --input examples/code_demo/tabular/predict.csv --output runs/demo_v030_001/reloaded_predictions.csv
```

演示任务也可换为 `examples/code_demo/text/task.json` 或 `examples/code_demo/image/task.json`。`predict` 只重载已保存模型进行预测，不调用 LLM，也不重新训练。

课程任务配置为 `tasks/coupon.json` 和 `tasks/headlines.json`。数据路径相对于任务 JSON 解析，请先按配置放置课程数据；两份配置只含官方训练／验证集，不含测试路径。以下为**首次运行**示例，每个任务最多8次候选实验，包含通用基线：

```powershell
.\.venv\Scripts\python.exe -m superagent campaign --task tasks/coupon.json --planner llm-code --llm-configs config/hkust-deepseek-flash-code-json.json --budget-rmb 1.0 --output runs/coupon_v030_001
.\.venv\Scripts\python.exe -m superagent campaign --task tasks/headlines.json --planner llm-code --llm-configs config/hkust-deepseek-flash-code-json.json --budget-rmb 1.0 --output runs/headlines_v030_001
```

继续同一任务时，使用新输出目录，并加上 `--previous` 指向实际上一轮的 `campaign.json`，将历史费用计入累计预算。已有未知用量时，见下文的开发预留规则；新目录或新版本都不代表费用清零。训练和重训超时分别由 `--experiment-timeout`、`--refit-timeout` 控制，与 API 请求截止时间独立。

只有在方案和源码冻结后，才进行独立最终评分。它只读取保存模型预测，不训练、不把测试成绩反馈给规划器：

```powershell
.\.venv\Scripts\python.exe -m superagent evaluate --model runs/coupon_v030_001/planner_01/model.joblib --input ../final_project_tasks/final_project_tasks/in_vehicle_coupon_recommendation/C_test.npy --labels ../final_project_tasks/final_project_tasks/in_vehicle_coupon_recommendation/y_test.npy --output runs/coupon_v030_final
.\.venv\Scripts\python.exe -m superagent evaluate --model runs/headlines_v030_001/planner_01/model.joblib --input ../final_project_tasks/final_project_tasks/headlines_ctr_clean/test.csv --output runs/headlines_v030_final
```

以上最终评分命令仅在相应模型已生成并正式冻结时运行。不要用正式测试集选模、调参或验证新修改。支持的指标为 `accuracy`、`balanced_accuracy`、`f1_macro` 和二分类 `roc_auc`，均按越高越好选择；课程任务使用 accuracy。

## 默认三档请求策略

推荐的 Flash code-json 配置启用 `request_policy="three_tier"`，只能用于 `llm-code` 实验。学校请求用 `chat_template_kwargs.enable_thinking` 控制思考；本地配置字段为 `enable_thinking`，`reasoning_effort` 保持 `null`。

| 档位 | 思考 | 实际输出上限 | 总请求截止时间 | 触发条件 |
| --- | --- | ---: | ---: | --- |
| `default` | 关闭 | 8192 token | 240秒 | 正常生成；首次代码执行错误后先在此档修复 |
| `long_code` | 关闭 | 16384 token | 360秒 | 默认档首次返回 `finish_reason="length"` 后，下一请求升档；每轮最多一次 |
| `difficult` | 开启 | 16384 token | 600秒 | 连续两份不同的生成代码执行失败，或至少三份不同的成功生成候选均未超过基线；每轮最多一次 |

基线与 JSON／协议错误不计作生成代码执行失败。困难档发送后回到默认档。长代码档或困难档仍截断时停止；用过长代码档后默认档再次截断也停止。网络失败不自动重试，不触发思考救援。`check-api` 始终只用默认档检查一次。

240／360／600秒限制的是**整个 API 请求**，通过独立请求进程实施，不只是 HTTP 各阶段等待时间。本地终止请求不证明服务端取消或未计费。配置外层 `max_output_tokens=16384` 是每次费用检查使用的保守预留上限；实际档位、原因、输出上限、思考开关及截止时间记录在 `llm_calls.jsonl` 的 `request_parameters` 中。预算不足、请求未发送时不消耗一次性升档额度。

此策略目前只推荐给已测试的 DeepSeek Flash code-json。旧 `*-code.json`、目录规划 `llm` 和 `offline` 保留兼容用途，不作为当前推荐入口；Pro／Qwen 未因本次发布获得三档在线验证。

## 费用与运行记录

Flash 配置沿用学校2026-11-14记录价格：输入 ¥0.50、输出 ¥2.00／百万 token，`pricing_basis="provider_current"`。配置每任务上限为 ¥20；示例中的 ¥0.25／¥1 是额外设置的开发预算，**不是课程规定**。模型和价格来源保存在配置中。

有返回用量的请求按记录的 token 计费，包括无效 JSON、截断和无法执行的代码。学校计价计算值不是平台账单；超时且无用量的费用保持未知，原预留金额继续保留，不能填写为0。

严格模式在账目不明时停止。显式 `--development-reserves` 仅允许开发阶段保留未知请求及其预留、继续后续规划器或新一轮；它不重试原请求，也不证明正式费用合规。只有取得真实平台证据后，才可用 `reconcile-usage` 记录对账结果。课程要求的历史官方标准价格与正式计费证明仍需补齐。

每轮输出包括：

| 文件 | 内容 |
| --- | --- |
| `campaign.json` | 任务累计费用、各规划器运行状态和最终选择 |
| `planner_01/summary.json`、`planner_01/selection.json` | 运行结论、验证成绩、选中候选及警告 |
| `planner_01/llm_calls.jsonl` | 完整请求／响应、实际参数和费用记录 |
| `planner_01/experiments.jsonl`、`planner_01/experiments/trial_*` | 所有候选、源码、执行结果和错误 |
| `planner_01/manifest.json`、`planner_01/source_snapshot/` | 数据／源码指纹及本轮源码快照 |
| `planner_01/model.joblib` | 自动选中的完整预测模型 |

`runs/`、模型及原始日志只保存在本地，不随公共仓库上传。它们是实验与费用证据，不能因更新版本而重置或混用。

## 成绩归属与历史归档

**v0.3.0 尚未完成两个已发布任务在本版本上的最终自主运行及最终测试。** V1 的冻结测试成绩、旧 V2 的验证结果，均不计作 v0.3.0 成绩。公开 [结果登记表](docs/results_registry.json) 按版本、源码指纹、数据划分和证据位置区分结果；每个分数须能追溯到产生它的模型与源码。

运行 `python -m superagent --version` 查看版本，`python scripts/verify_release.py` 核验正式源码、配置及成绩归属。每次新训练的 manifest、summary 和模型均记录 `project_version` 与唯一 `run_id`；摘要还保存模型与 LLM 配置文件指纹。最终评估分别记录训练版本、训练运行编号和评估版本，旧模型缺少版本字段时保留为空，不自动归入新版本。

三档实验快照曾在双标题官方验证集达到69.47%，9次请求均为默认档，0超时、0截断；这是发布前实验的证据。长代码档和困难档没有在该轮真实触发，仅有离线分支检查。旧 V2 Qwen 的69.79%来自不同模型，不能据此单独归因请求策略优劣。上一轮整合检查为521项通过、1项跳过；本次正式版本的检查以 [版本元数据](releases/v0.3.0/release.json) 为准。

旧源码、旧 README 和历史清单在本地 `archives/pre-v0.3.0/` 只读归档；不再作为并行维护版本。归档／校验／恢复入口为 `scripts/manage_source_archive.py`。历史运行记录仍沿用原 `runs/` 路径，避免破坏账目引用。若旧源码快照已压缩，先按归档清单定位压缩包、验证指纹并恢复到原目录，再从相应源码快照复现历史运行；不要使用当前代码替代旧快照后宣称精确复现。

## 能力边界与待完成事项

生成代码使用已安装的 NumPy、pandas、SciPy、sklearn、CatBoost、Pillow 和有限标准库，源码上限10,000 UTF-8字节。生成的估计器接收匿名列和训练标签，LLM 只接收任务摘要与实验反馈；验证标签由主程序保管和评分。显式验证集不进入最终重训数据，执行契约要求预测阶段不得拟合或更新模型。

执行使用独立进程、源码检查、访问限制和时间上限；这些措施不是操作系统级安全隔离，也没有硬内存限额。当前面向 CPU 分类任务，不支持自行安装依赖、联网下载权重或任意文件访问。只加载可信的本地 joblib／对象 NPY 文件；模型重载要求相同 Python 完整版本、依赖版本和对应运行时源码。

提交前仍需完成：本版本的两任务最终运行和冻结评估、更广泛的未见任务验证、项目报告、AI 使用声明、展示材料，以及官方价格与费用证据整理。修复历史回复的执行回放可以验证执行器，但不能替代一次完整自主在线实验。

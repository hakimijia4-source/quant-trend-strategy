# 单股单边趋势：点时数据与离线层级强化学习项目

这个项目把以下研究流程做成了可执行代码：

1. 从合规的美国股票 SIP 行情接口收集目标股票与市场上下文。
2. 从 SEC、公司 IR、FRED/ALFRED 和 GDELT 元数据收集公开事件。
3. 把人工认知写成带创建时间、有效期、确认条件和证伪条件的假设账本。
4. 生成量价、VWAP稳定性、板块轮动、事件衰竭、议息确认和期权到期日交互特征。
5. 创建不偷看未来的离线行为轨迹。
6. 训练层级 Decision Transformer 与外生市场 World Model。
7. 通过严格的走步验证选择置信阈值，并输出研究信号。

当前版本**没有券商交易权限、没有下单代码，也不授权任何自动执行**。训练产物只输出 `ResearchSignal`；风控、仓位、订单路由和实盘熔断在后续执行项目中实现。

> [!IMPORTANT]
> 本仓库是研究原型，不构成投资建议，也没有任何收益保证。历史合成演示的模型选择了全程观望，不能作为策略有效或可盈利的证据。详见 [合成演示记录](docs/DEMO_RESULTS.md)。

## 模型结构

### 层级强化学习

上层 option：

- `observe`：观望；
- `trend_follow`：量价确认后的趋势跟随；
- `event_momentum`：公司事件或板块轮动；
- `squeeze_reversal`：逼空衰竭及量价背离；
- `macro_relief`：议息或宏观预期差得到跨资产确认。

下层动作：

- `flat`
- `long`
- `short`

网络同时学习上层 option、option 内动作、option 终止概率和回报价值。设计参考 [Option-Critic](https://arxiv.org/abs/1609.05140) 和 [MAXQ](https://arxiv.org/abs/cs/9905014)，但实现针对离线金融数据做了约束，并不声称复现论文全部算法。

### Hierarchical Decision Transformer

模型使用因果 Transformer，根据以下序列预测 option 与动作：

```text
目标剩余回报 + 当时可见市场状态 + 前一动作 + 前一option + 日内时间
```

它参考 [Decision Transformer](https://arxiv.org/abs/2106.01345) 的回报条件序列建模思想，并加入层级 option 头。

### World Model

世界模型学习：

```text
当前市场状态 -> 下一市场状态
当前状态 + 选择的仓位动作 + 下一状态 -> 组合收益
```

对于普通交易者，买卖动作不会显著改变 SIP 市场状态，因此市场转移不以个人动作为因果输入；动作只进入收益头。该实现受 [DreamerV3](https://arxiv.org/abs/2301.04104) 启发，是轻量的随机潜变量世界模型，不是完整 DreamerV3 复现。

## 安装

支持 Python 3.10 或更高版本，建议使用 Python 3.11 或更高版本：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[train]"
```

如果需要Parquet：

```powershell
pip install -e ".[train,parquet]"
```

### 已在WSL安装PyTorch

不要再次安装 `.[train]`，否则 pip 可能替换现有的 CUDA 版 PyTorch。先进入装有
PyTorch 的 WSL Conda/venv 环境，再从 Windows 项目的 WSL 映射路径运行检查脚本：

```bash
# 如使用Conda，先激活实际装有PyTorch的环境
# conda activate your_torch_env

# 在 WSL 中进入本仓库；把路径替换为你自己的 clone 位置
cd /mnt/c/path/to/quant-trend-strategy
bash scripts/setup_wsl.sh
```

脚本检查核心依赖后使用本地 `.pth` 文件注册项目源码，不启动pip隔离构建、
不访问PyPI，也不会主动重装PyTorch；同时会检查Python 3.10+、CUDA可用性、
项目导入和单元测试。如果PyTorch使用的解释器不是 `python3`，可显式指定：

```bash
PYTHON_BIN=/path/to/python bash scripts/setup_wsl.sh
```

如果不知道PyTorch安装在哪个Conda/venv环境，可以让项目自动搜索用户主目录，
优先选择支持CUDA的解释器，并完成安装、测试及合成训练：

```bash
# 在 WSL 中进入本仓库；把路径替换为你自己的 clone 位置
cd /mnt/c/path/to/quant-trend-strategy
bash scripts/bootstrap_wsl.sh --with-demo
```

只配置和测试、不立即训练时，去掉 `--with-demo`。

## API环境变量

使用 Alpaca SIP：

```powershell
$env:ALPACA_API_KEY="..."
$env:ALPACA_API_SECRET="..."
```

或者使用 Massive SIP：

```powershell
$env:MASSIVE_API_KEY="..."
```

公开事件接口：

```powershell
$env:SEC_USER_AGENT="Your Name your-email@example.com"
$env:FRED_API_KEY="..."
```

不要把密钥写入 TOML、CSV、源码或提交记录。

## 修改目标股票

编辑 [config/strategy.toml](config/strategy.toml)：

```toml
[market]
target_symbol = "AAPL"
context_symbols = ["SPY", "QQQ", "SOXX", "TLT", "IEF", "GLD", "USO", "UUP"]

[company]
cik = "0000320193"
name = "Apple Inc."
aliases = ["Apple", "AAPL", "iPhone"]
```

目标可以是一只股票，但上下文资产不能删除，否则模型难以区分个股趋势与市场共同波动。

## 运行

### 1. 无网络演示

验证数据、特征、标签和轨迹生成：

```powershell
quant-trend --config config/demo.toml demo --sessions 180 --skip-train
```

安装PyTorch后运行一次完整训练演示：

```powershell
quant-trend --config config/demo.toml demo --sessions 180
```

### 2. 收集行情

```powershell
quant-trend --config config/strategy.toml collect-market
```

指定Massive：

```powershell
quant-trend --config config/strategy.toml collect-market --provider massive
```

Alpaca免费IEX流不适合本项目的成交量训练。配置应保持 `feed = "sip"`。

### 3. 收集公开事件

```powershell
quant-trend --config config/strategy.toml collect-events
```

默认只保存SEC申报、IR公告或媒体发现元数据，不抓取彭博、路透等受版权保护的完整文章。

### 4. 加入实时形成的个人认知

把 [examples/thesis_template.json](examples/thesis_template.json) 复制为 `data/theses.jsonl`，每行写一个JSON对象。

规则：

- `created_at` 必须是观点真正形成的时间；
- `valid_from` 不能早于 `created_at`；
- 必须设置 `expires_at`；
- 必须写证伪条件；
- 复盘后形成的观点设置 `is_retrospective=true`，系统不会把它加入历史特征。

用户提出的半导体轮动、议息轧空、康宁GlassBridge及期权到期案例已经保存在 [examples/cognitive_cases.jsonl](examples/cognitive_cases.jsonl)，并被标记为回顾案例，防止污染2026年7月回测。

### 5. 构建训练数据

```powershell
quant-trend --config config/strategy.toml validate
quant-trend --config config/strategy.toml build-dataset
```

产物：

```text
data/raw/market/*.csv
data/raw/events/*.csv
data/processed/features.csv
data/processed/labeled.csv
data/processed/trajectories.csv
```

### 6. 训练与走步回测

先训练最新一个折叠进行检查：

```powershell
quant-trend --config config/strategy.toml train --max-folds 1
```

运行全部走步窗口：

```powershell
quant-trend --config config/strategy.toml train
```

每个折叠会生成：

```text
reports/fold_*/checkpoint.pt
reports/fold_*/scaler.npz
reports/fold_*/feature_spec.json
reports/fold_*/metrics.json
reports/fold_*/test_predictions.csv
reports/walk_forward_summary.json
```

阈值只能在验证区间选择；测试区间不会用于调参。优化目标优先使用成功率的Wilson置信区间下界，并同时要求交易数量和扣费后收益。

## 认知规则如何执行

### 半导体向苹果轮动

只有当SOXX放量走弱、目标股票产生正异常收益、目标量能不弱且保持在VWAP上方时，`rotation_score` 才会升高。仅仅“苹果少跌”不会被当作资金流入。

### 议息后轧空

不加息本身不是做多信号。项目要求事件预期差、债券代理表现、SOXX/QQQ重新站上VWAP等确认。已有“不加息但市场仍下跌”的样本会作为高权重困难反例。

### GlassBridge与逼空衰竭

项目计算第二日量能衰减、价格创新高但RVOL下降、尾盘VWAP丢失、收盘位置恶化和事件商业化差距。确认反转仍需要价格跌破，而不是事后把第三天直接标成顶部。

### 期权到期

当前公开数据层只加入周五、月度到期日和趋势衰竭交互。没有OPRA授权数据时，不使用未平仓量、Gamma或“最大痛点”作为训练特征，也不会把星期五自动解释为看跌。

## 奖励函数

每一步的研究奖励为：

```text
仓位 × 下一周期对数收益
- 仓位变化成本
- 持仓成本
- 不利波动惩罚
```

离线轨迹由观望、趋势、事件、反转和探索行为策略生成。行为动作只使用当前及过去状态；下一周期收益仅用于奖励和监督目标。

## 测试

```powershell
python -m unittest discover -s tests -v
```

测试覆盖：

- 认知观点不能在创建时间前生效；
- 回顾案例不会进入历史特征；
- 新闻首次获得时间不能早于事件时间；
- FRED/ALFRED 初次发布值只能从下一交易日使用；
- 趋势标签、成本奖励和回报倒推；
- 训练、验证、测试日期不重叠。

## 数据源说明

- Massive提供CTA/UTP SIP聚合行情、逐笔成交与NBBO，具体访问范围取决于订阅与授权：[官方文档](https://massive.com/docs/rest/stocks)
- Alpaca SIP覆盖美国综合市场，IEX仅为单一交易所：[官方文档](https://docs.alpaca.markets/us/docs/historical-stock-data-1)
- SEC公司申报API：[官方文档](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)
- FRED/ALFRED点时数据：采集器使用 `output_type=4` 保存初次发布值，并从下一交易日才允许模型使用：[官方文档](https://fred.stlouisfed.org/docs/api/fred/series_observations.html)
- GDELT仅作媒体发现：[官方文档](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/)

## 当前边界

- 这不是收益保证，也不能产生“确定性”交易。
- 合成演示结果不能代表真实收益。
- 单只股票的分钟样本高度相关，独立样本单位主要仍是交易日。
- 需要使用真实SIP、点时事件和足够长的样本外区间后，才能评价模型。
- `OfflinePolicyRuntime` 只输出 `execution_authorized=false` 的研究信号；项目内不存在下单端点。

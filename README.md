# ModelTrainer — 期货 Tick 级因子排序学习训练框架

面向**金融期货超高频/Tick 因子数据**的 Learning-to-Rank (LTR) 训练框架。输入 **因子列表 (X)** 与 **标签列表 (Y)**
，自动完成数据加载、分组标准化、多模型训练、排序指标评估、可视化输出，并支持超参搜索 (HPO) 与 SLURM 集群批量提交。

---

## 目录

- [一、根本功能与解决的问题](#一根本功能与解决的问题)
- [二、项目结构](#二项目结构)
- [三、数据格式](#三数据格式)
- [四、快速开始](#四快速开始)
- [五、配置文件详解 (train_configs.yaml)](#五配置文件详解-train_configsyaml)
- [六、如何修改参数](#六如何修改参数)
- [七、如何上传到 SLURM 运行](#七如何上传到-slurm-运行)
- [八、模型说明](#八模型说明)
- [九、损失函数](#九损失函数)
- [十、评估指标](#十评估指标)
- [十一、超参搜索 (HPO)](#十一超参搜索-hpo)
- [十二、输出结果](#十二输出结果)
- [十三、标准化方式](#十三标准化方式)
- [十四、编程接口](#十四编程接口)
- [十五、常见问题](#十五常见问题)

---

## 一、根本功能与解决的问题

### 要解决的核心问题

在期货市场的每一个交易时刻 `(date, time_sec, code)`，同一合约会同时存在**多个 abtype**。本框架要解决的问题是：

> **在每一个时刻、每一个合约的多个 abtype 候选中，预测哪个 abtype 的未来收益 (label) 最高，从而选出最优的交易标的。**

这是一个典型的 **Learning-to-Rank (排序学习)** 问题：

- **Query (查询组)**：`(date, time_sec, code)` —— 某天某秒某合约下的所有 abtype 候选。
- **Candidate (候选)**：每个 abtype 对应一行因子特征。
- **Label (标签)**：该 abtype 的未来收益 (如 `period_return_HIGH_weight1`)。
- **目标**：对每个组内的候选按预测分数排序，使 Top-K 命中真实最优 abtype 的概率最大化。

### 框架提供的核心能力

| 能力             | 说明                                                          |
|----------------|-------------------------------------------------------------|
| **多模型排序训练** | MLP / Set Transformer / LambdaMART(LightGBM)，支持组内排序 |
| **多种排序损失**     | Pearson 相关损失、ListNet、ListMLE 三种 Learning-to-Rank 损失         |
| **组感知评估**      | Top-K 命中率、超额命中率、高于均值率、对比先验 abtype 胜率                        |
| **超参搜索 (HPO)** | 随机搜索 / 网格搜索 / Optuna 贝叶斯优化，数据只加载一次                          |
| **多实验批量调度**    | 一个 YAML 定义多组实验，按 GPU/CPU + Train/HPO 智能分组                   |
| **SLURM 集群集成** | `submit_all.sh` 自动解析配置、分组提交 array 任务                        |
| **HDFS 数据读写**  | 从 HDFS 拉取数据、上传预测结果与模型产物                                     |
| **滚动/分组标准化**   | 无前瞻偏差的 rolling 标准化、组内 z-score 标准化                           |
| **自动可视化**      | 训练曲线、abtype 分布对比、特征重要性                                      |

---

## 二、项目结构

```
ModelTrainer/
├── train.py                         # 单实验 CLI 入口 (传统 YAML 配置)
├── train_runner.py                  # 多实验统一入口 (读 train_configs.yaml)
├── train_configs.yaml               # 多实验配置文件 (核心配置)
├── generate_runners.py              # 生成独立 python_train_runner_XXXXXX.py 脚本
├── generate_task_list.py            # 生成 HDFS 日期目录任务列表
├── python_train_runner.py           # 编程接口调用示例
├── submit.sh                        # SLURM 单任务执行脚本
├── submit_all.sh                    # SLURM 智能分组提交脚本
├── main_contract.json               # 主力合约映射 (日期 -> 合约代码列表)
├── README.md
└── model_trainer/                   # 核心包
    ├── __init__.py                  # 导出 Trainer / Config / HyperparamTuner
    ├── config.py                    # 配置数据类 (DataConfig/TrainingConfig/...)
    ├── data.py                      # 数据加载 + 标准化 + HDFS + resample
    ├── dataset.py                   # PyTorch Dataset (Tabular/Grouped/Temporal)
    ├── metrics.py                   # 排序评估指标 (Top-K 命中率等)
    ├── losses.py                    # Pearson / ListNet / ListMLE 损失
    ├── plots.py                     # 可视化 (训练曲线/abtype分布/特征重要性)
    ├── trainer.py                   # 训练编排器 (数据→训练→评估→绘图→HDFS)
    ├── tuner.py                     # 超参搜索 (random/grid/optuna)
    ├── utils.py                     # segment_softmax 等分组工具
    └── models/
        ├── __init__.py              # 模型注册表 + build_model 工厂
        ├── base.py                 # BaseModel 抽象基类
        ├── mlp.py                   # MLP 排序模型 (组内 softmax)
        ├── transformer.py           # Set Transformer 排序模型 (置换等变)
        └── lambdamart.py            # LambdaMART (LightGBM lambdarank)
```

> 注：[model_trainer/models/__init__.py](model_trainer/models/__init__.py) 的注册表还引用了 `lgbm` / `gru` / `dlinear` / `lasso`
> ，但当前仓库中仅有 `mlp` / `transformer` / `lambdamart` 三个实现文件。

---

## 三、数据格式

### 目录布局

```
{data_root}/
└── {YYYYMMDD}/
    ├── {code}_{abtype}.parquet      # 如 12502_3018.parquet
    ├── 12502_3232.parquet
    └── ...
```

- 文件命名：`{合约代码}_{abtype}.parquet`
- 训练时通过 `main_contract.json` 可筛选每日主力合约。

### Parquet 列结构

| 列名         | 类型      | 说明                             |
|------------|---------|--------------------------------|
| `date`     | int64   | 交易日期 (如 20240506)              |
| `code`     | int64   | 合约代码 (如 12502)                 |
| `abtype`   | int64   | abtype (如 3018, 3232)               |
| `time_sec` | float64 | 当天秒数 (0.5s 粒度)                 |
| 因子列 (X)    | float   | 以 `signal_*` 或 `p_*` 命名的特征列    |
| 标签列 (Y)    | float   | 如 `period_return_HIGH_weight1` |

### 排序分组定义

每个**查询组 (query group)** 由 `(date, time_sec, code)` 唯一确定：同一时刻、同一合约下的所有 `abtype` 候选构成一个待排序的
list。模型的目标是在每个组内排出最优 abtype。

---

## 四、快速开始

### 1. 安装依赖

```bash
python -m pip install torch lightgbm polars pyarrow pyyaml scipy tqdm matplotlib optuna
```

| 包                   | 用途                    |
|---------------------|-----------------------|
| `torch`             | MLP / Transformer 训练  |
| `lightgbm`          | LambdaMART 训练         |
| `polars` / `pandas` | 高性能数据加载与标准化           |
| `pyarrow`           | Parquet 读写 + HDFS 访问  |
| `scipy`             | rank_gaussian 中的分位数计算 |
| `optuna`            | 贝叶斯超参搜索 (可选)          |
| `matplotlib`        | 绘图 (Agg 后端，无需 GUI)    |
| `tqdm` / `pyyaml`   | 进度条 / 配置解析            |

### 2. 修改配置

编辑 [train_configs.yaml](train_configs.yaml)，修改 `shared_data` 中的 `data_root`、`date_range`、`y_list` 等（详见第五节）。

### 3. 运行训练

**方式 A — 单实验入口 (train.py，传统单 YAML)：**

```bash
python train.py --config train_configs.yaml
```

**方式 B — 多实验统一入口 (推荐)：**

```bash
python train_runner.py --id 1          # 运行第 1 个实验
python train_runner.py --id 1 2 3       # 顺序运行多个实验
python train_runner.py                  # 交互式选择实验
SLURM_ARRAY_TASK_ID=1 python train_runner.py   # SLURM array 模式
```

---

## 五、配置文件详解 (train_configs.yaml)

完整配置见 [train_configs.yaml](train_configs.yaml)。配置分为三大块：`shared_data`、`shared_hpo`、`experiments`。

### 5.1 `shared_data` — 共享数据配置

所有实验共用的数据源与预处理设置：

```yaml
shared_data:
  data_root: "../ResampleData"        # parquet 数据根目录
  date_range: "20240506-20240530"     # 总日期范围 (自动按比例切分)
  train_ratio: 0.5                    # 训练集占比
  val_ratio: 0.25                     # 验证集占比 (剩余为测试集)
  factor_list: null                   # 因子列; null=自动检测全部 signal_* 列
  y_list: # 标签列 (每个独立训练)
    - "period_return_HIGH_weight1"
  normalize: "group"                  # "rolling" | "global" | "group" | "none"
  winsorize_std: 0                    # z-score 截断阈值, 0=不截断
  # abtypes: [3018, 3232]             # 可选: 只用指定 abtype
  # codes:   [12502, 22502]          # 可选: 只用指定合约
```

| 参数            | 说明                                                           |
|---------------|--------------------------------------------------------------|
| `date_range`  | 总日期范围，框架会按 `train_ratio`/`val_ratio` 自动切分为 train/val/test 三段 |
| `factor_list` | `null` 时自动扫描数据文件检测所有因子列 (排除元数据与含 `LOW/MID/HIGH/weight` 的列)   |
| `normalize`   | 见 [十三、标准化方式](#十三标准化方式)                                       |

### 5.2 `shared_hpo` — 共享超参搜索配置

控制所有实验是否启用 HPO：

```yaml
shared_hpo:
  enabled: true                  # true=所有实验都搜参; false=直接训练
  method: "optuna"               # "random" | "grid" | "optuna"
  n_trials: 30                   # 搜索次数 (grid 模式忽略)
  metric: "auto"                 # "auto" 按模型类型自动选择, 或指定如 "val_top1_above_mean_rate"
  direction: "maximize"           # "maximize" | "minimize"
  seed: 42
  save_dir: "hpo_output"
  disable_torch_compile: true     # 搜参时关闭 torch.compile 降低开销
  optuna_startup_trials: 1       # Optuna TPE 启动前随机试验次数
  optuna_gamma: 0.3              # TPE 探索/开发比, (0,1], 不填用默认 0.25
```

- `enabled: false` → 直接用 `experiments` 中写死的参数训练。
- `enabled: true` → 使用默认搜索空间 (定义在 [tuner.py](model_trainer/tuner.py) 的 `get_default_search_space()`) 进行搜参。

### 5.3 `experiments` — 实验列表

每个实验对应一个 SLURM array task：

```yaml
experiments:
  - name: "listmle"               # 实验名
    use_gpu: true                 # true=DL模型申请GPU; false=非DL用CPU
    models: # 模型定义 (当前框架每实验训练1个模型)
      - type: "mlp"               # 模型类型: mlp|transformer|lambdamart
        name: "mlp"               # 显示名
        params: # 模型专属超参
          hidden_sizes: [ 256, 128, 64 ]
          dropout: 0.1
    training: # 训练超参 (DL模型共用)
      epochs: 100
      batch_size: 4096
      early_stopping_patience: 20
      learning_rate: 1.0e-4
      device: "auto"              # "auto"|"cpu"|"cuda"|"mps"
      min_delta: 0
      loss_type: "listmle"        # "pearson"|"listnet"|"listmle"
      min_epochs: 10
      rank_before_softmax: false  # listnet: 是否先 rank 再 softmax
      topk: 1                     # top-k 命中率与硬标签的 K 值
      use_hard_label: false       # listnet: true=top-k硬标签; false=软标签
    output:
      save_dir: "output"
      experiment_name: "listmle"
      upload_to_hdfs: true        # 是否上传结果到 HDFS
```

`use_gpu` 决定 SLURM 提交时的分组：

- `true` → 深度学习模型，使用 `commoncpu,commongpu` 分区并申请 GPU
- `false` → 非深度学习模型，使用 `commoncpu,new_conf` 分区，不申请 GPU

---

## 六、如何修改参数

### 6.1 修改数据与切分

编辑 `shared_data`：

```yaml
shared_data:
  data_root: "/your/data/path"        # 改数据路径
  date_range: "20240101-20240630"      # 改日期范围
  train_ratio: 0.6                     # 改训练占比
  val_ratio: 0.2                       # 改验证占比 (test = 1 - train - val)
  y_list: [ "your_label_column" ]        # 改标签
```

### 6.2 修改模型超参

在对应实验的 `models[0].params` 中修改，例如改 MLP 结构：

```yaml
models:
  - type: "mlp"
    name: "mlp_wider"
    params:
      hidden_sizes: [ 512, 256, 128 ]   # 加宽
      dropout: 0.2                     # 加大 dropout
```

### 6.3 修改训练超参

在 `training` 块修改，例如换损失函数：

```yaml
training:
  loss_type: "listnet"            # pearson | listnet | listmle
  epochs: 200                     # 加大轮数
  batch_size: 8192                # 加大 batch
  learning_rate: 5.0e-5
  early_stopping_patience: 30
  rank_before_softmax: true       # listnet 专用
  use_hard_label: true            # listnet 专用: top-k 硬标签
  topk: 1                         # 命中率计算的 K
```

### 6.4 命令行覆盖 (仅 train.py)

`train.py` 支持部分参数命令行覆盖：

```bash
python train.py --config train_configs.yaml \
    --output_dir my_output \
    --experiment my_exp \
    --device cuda \
    --normalize rolling
```

### 6.5 新增实验

在 `experiments` 列表中追加一项即可。例如新增一个 Transformer 实验：

```yaml
experiments:
  - name: "transformer_listnet"
    use_gpu: true
    models:
      - type: "transformer"
        name: "transformer"
    training:
      epochs: 100
      batch_size: 4096
      loss_type: "listnet"
      rank_before_softmax: true
      topk: 1
    output:
      save_dir: "output"
      experiment_name: "transformer_listnet"
      upload_to_hdfs: true
```

### 6.6 切换 HPO 开关

```yaml
shared_hpo:
  enabled: false    # 直接训练 experiments 中写死的参数
  # enabled: true   # 启用搜参, 自动用默认搜索空间
```

---

## 七、如何上传到 SLURM 运行

框架提供两套脚本：`submit_all.sh` (智能分组提交) 与 `submit.sh` (单任务执行)。

### 7.1 智能分组提交 (推荐)

```bash
./submit_all.sh                     # 默认 SLURM 模式, 自动分组提交
```

`submit_all.sh` 会读取 `train_configs.yaml`，按 `use_gpu` 与 `hpo_enabled` 将实验分为四组并分别提交 SLURM array 任务：

| 分组          | 分区                    | GPU | 时限 | 适用        |
|-------------|-----------------------|-----|----|-----------|
| GPU + Train | `commoncpu,commongpu` | 1   | 标准 | DL 模型直接训练 |
| GPU + HPO   | `commoncpu,commongpu` | 1   | 更长 | DL 模型超参搜索 |
| CPU + Train | `commoncpu,new_conf`  | 无   | 标准 | 非DL直接训练   |
| CPU + HPO   | `commoncpu,new_conf`  | 无   | 更长 | 非DL超参搜索   |

#### 资源配置 (环境变量覆盖)

```bash
GPU_GPUS=2 GPU_CPUS=16 GPU_MEM=200G ./submit_all.sh        # 调整 GPU 任务资源
CPU_CPUS=200 CPU_MEM=500G ./submit_all.sh                  # 调整 CPU 任务资源
GPU_HPO_TIME=10-0:0:0 ./submit_all.sh                      # 调整 HPO 时限
```

可用变量：`GPU_PARTITION` `GPU_GPUS` `GPU_CPUS` `GPU_MEM` `GPU_TRAIN_TIME` `GPU_HPO_TIME` `CPU_PARTITION` `CPU_CPUS`
`CPU_THREADS_PER_CORE` `CPU_MEM` `CPU_TRAIN_TIME` `CPU_HPO_TIME`。

### 7.2 本地调试模式

无需 SLURM，直接在本机运行 `train_runner.py`：

```bash
TARGET=local ./submit_all.sh                  # 本地跑所有实验
TARGET=local ID=3 ./submit_all.sh             # 本地只跑实验 3
TARGET=local ID="1 3 5" ./submit_all.sh       # 本地跑实验 1,3,5
DRY_RUN=1 ./submit_all.sh                     # 仅打印命令不执行
```

### 7.3 standalone 模式 (生成独立脚本)

先生成自包含的运行脚本，再逐个执行 (适合 SLURM array 中各任务完全独立)：

```bash
MODE=standalone ./submit_all.sh                # SLURM 模式下走 standalone
python generate_runners.py                     # 手动生成 python_train_runner_000001.py 等
```

生成的脚本把日期切分、配置、HPO 逻辑全部内联，可直接 `python python_train_runner_000001.py` 运行。

### 7.4 submit.sh 内部流程 (供参考)

[submit.sh](submit.sh) 是 SLURM 实际执行的脚本，流程：

1. 从 HDFS 下载 conda 环境 `modeltrainer.tar.gz` 并解压
2. 下载 `ResampleData` 数据与 `modeltrainer` 项目代码
3. 配置 Hadoop / Java 环境变量
4. 按 `MODE` 调用 `train_runner.py` (unified) 或 `generate_runners.py` (standalone)

> 直接 `sbatch submit.sh` 不可用，必须通过 `submit_all.sh` 传入分区/GPU/array 等参数。

---

## 八、模型说明

所有模型继承 [BaseModel](model_trainer/models/base.py)，实现 `fit` / `predict` / `save` 接口。模型按
`(date, time_sec, code)` 分组进行排序训练。

### 8.1 MLP (`mlp`)

- **架构**：`Linear → LayerNorm → ReLU → Dropout` × N 层 → `Linear(1)`
- **特性**：支持组内 softmax (ListNet 模式下对每个组单独 softmax)
- **训练**：Adam + CosineAnnealingLR，梯度裁剪 max_norm=1.0
- **Early Stopping**：基于验证集 `top1_above_mean_rate`
- **保存格式**：`.pth` (state_dict)

| 参数             | 默认              | 说明            |
|----------------|-----------------|---------------|
| `hidden_sizes` | `[256,256,128]` | 各隐藏层宽度        |
| `dropout`      | 0.1             | Dropout 比例    |
| `batch_norm`   | true            | 是否用 LayerNorm |

### 8.2 Set Transformer (`transformer`)

- **架构**：`Linear(F→d_model)` → N × `TransformerEncoderLayer` (Pre-LN) → `LayerNorm` → `Linear(d_model→1)`
- **特性**：**无位置编码 → 置换等变**，组内候选通过 self-attention 互相参考，适合相对排序
- **padding**：变长组通过 padding + attention mask 处理，防止跨组信息泄漏
- **Early Stopping**：基于验证集 `top1_above_mean_rate`
- **保存格式**：`.pth` (state_dict)

| 参数           | 默认  | 说明                            |
|--------------|-----|-------------------------------|
| `d_model`    | 256 | Transformer 隐藏维度              |
| `nhead`      | 8   | 注意力头数 (须整除 d_model，否则自动下调)    |
| `num_layers` | 2   | Encoder 层数                    |
| `dim_ff`     | 0   | Feed-forward 维度 (0=d_model×4) |
| `dropout`    | 0.1 | Dropout 比例                    |

### 8.3 LambdaMART (`lambdamart`)

- **架构**：基于 LightGBM 的 `lambdarank` 目标
- **特性**：自定义评估指标 (topk_hit_rate / excess_hit_rate / avg_pred_rank / topk_above_mean_rate / topk_vs_prior_rate)
  ，自定义 early stopping
- **数据准备**：按 `(date, time_sec, code)` 分组，标签做 dense rank，提供 group_sizes
- **保存格式**：`.txt`

| 参数                            | 默认   | 说明              |
|-------------------------------|------|-----------------|
| `num_leaves`                  | 63   | 叶节点数            |
| `learning_rate`               | 0.05 | 学习率             |
| `feature_fraction`            | 0.8  | 特征采样比例          |
| `bagging_fraction`            | 0.8  | 样本采样比例          |
| `lambda_l1` / `lambda_l2`     | 0.1  | 正则化             |
| `min_data_in_leaf`            | 20   | 叶节点最小样本数        |
| `lambdarank_truncation_level` | 30   | LambdaRank 截断级别 |

### 添加新模型

1. 在 [model_trainer/models/](model_trainer/models/) 下新建 `my_model.py`，继承 `BaseModel`，实现 `fit()` / `predict()` /
   `save()`。
2. 在 [model_trainer/models/__init__.py](model_trainer/models/__init__.py) 的 `_REGISTRY` 注册：`"my_model": MyModel`。
3. (可选) 在 [tuner.py](model_trainer/tuner.py) 的 `_DEFAULT_SEARCH_SPACES` 添加搜索空间。
4. 配置文件中 `type: my_model` 即可使用。

---

## 九、损失函数

损失函数定义在 [losses.py](model_trainer/losses.py)，通过 `training.loss_type` 选择：

| `loss_type` | 损失                | 说明                                                                |
|-------------|-------------------|-------------------------------------------------------------------|
| `pearson`   | `1 - Pearson相关系数` | 最大化预测与标签的相关性，点对点                                                  |
| `listnet`   | ListNet 交叉熵       | 组内 softmax 后与目标分布做交叉熵，可配 `rank_before_softmax` / `use_hard_label` |
| `listmle`   | ListMLE 负对数似然     | 最大化真实排序的似然，对排序顺序更敏感                                               |

### ListNet 标签构造选项

- `rank_before_softmax: true`：先对标签做 rank 变换再 softmax (推荐，鲁棒)
- `use_hard_label: true`：top-k 项赋 `1/k`，其余 0 (硬标签)
- `use_hard_label: false`：使用 rank + softmax 的软标签
- `label_smoothing`：标签平滑系数 [0,1]，0 关闭

---

## 十、评估指标

指标定义在 [metrics.py](model_trainer/metrics.py)，**全部在 `(date, time_sec, code)` 组内计算**：

| 指标                                    | 定义                                  | 说明                    |
|---------------------------------------|-------------------------------------|-----------------------|
| **topk_hit_rate**                     | 预测 top-K 与真实 top-K 的交集 / K          | 核心指标，越高越好             |
| **excess_hit_rate**                   | topk_hit_rate - K/组大小               | 相对随机基线的超额命中率          |
| **avg_pred_rank**                     | 预测 top-1 在真实排序中的归一化排名               | [0,1]，越高越好            |
| **topK_above_mean_rate** (K=1,3,5,10) | 预测 top-K 均值 ≥ 组均值的比例                | 模型选出的候选是否优于平均         |
| **topK_vs_prior_rate** (K=1,3,5,10)   | 预测 top-K 均值 ≥ 先验 top-K abtype 均值的比例 | 是否超越「历史最常最优 abtype」基线 |

- `prior_topk_abtypes`：由 train+val 数据统计出「最常成为真实 top-1」的 abtype 频率，作为对比基线。
- `TOPK_VALUES = [1, 3, 5, 10]`。

### 评估输出

每个 split (train/val/test) 都会打印并保存到 `metrics.json`：

```
[test] n_samples=... | Top-1 Hit Rate=0.xxxx | Excess Hit Rate=0.xxxx | Avg Pred Rank=0.xxxx | Top1 Above Mean=0.xxxx | Top1 vs Prior=0.xxxx ...
```

---

## 十一、超参搜索 (HPO)

HPO 模块定义在 [tuner.py](model_trainer/tuner.py)，由 `shared_hpo.enabled` 开关。

### 三种搜索方法

| 方法       | 说明             |
|----------|----------------|
| `random` | 均匀随机采样超参       |
| `grid`   | 穷举所有组合         |
| `optuna` | TPE 贝叶斯优化 (推荐) |

### 默认搜索空间

每种模型在 `get_default_search_space()` 中预定义搜索空间，例如 MLP：

```python
"mlp": {
    "training": {
        "learning_rate": {"type": "float", "low": 1e-6, "high": 1e-4, "log": True},
        "batch_size": {"type": "categorical", "choices": [2048, 4096, 8192]},
    },
    "model_params": {
        "hidden_sizes": {"type": "categorical", "choices": [[256, 128, 64], [512, 256, 128], ...]},
        "dropout": {"type": "float", "low": 0.0, "high": 0.5},
    },
}
```

### 优化目标

- `metric: "auto"` 时按模型类型自动选择：
    - `mlp` / `transformer` / `lambdamart` → `val_top1_above_mean_rate`
    - `gru` / `dlinear` / `lgbm` → `val_ic`
- `direction: "maximize"` (默认)

### HPO 流程

1. **数据只加载一次**，所有 trial 复用 `df_train` / `df_val`。
2. 每个 trial 用采样的超参构建 Config → 训练 → 提取最佳 metric。
3. 搜索完成后在 **test 集**上评估 best model (不重新训练)。
4. 保存 `tuning_results.csv`、`best_config.json`、`best_config.yaml`、`best_model.pth`。

### 输出

```
hpo_output/{experiment_name}/
├── tuning_results.csv        # 所有 trial 结果
├── best_config.json          # 最佳超参 + metric
├── best_config.yaml          # 可直接复用的 YAML 配置
├── best_model.pth            # 最佳模型权重
└── test_predictions.csv      # test 集预测
```

---

## 十二、输出结果

### 目录结构

```
{save_dir}/{experiment_name}/
├── {experiment_name}_hit_rate_summary.csv      # 跨 split 的命中率汇总
├── {experiment_name}_above_mean_summary.csv     # 跨 split 的 above_mean 汇总
├── {experiment_name}_vs_prior_summary.csv      # 跨 split 的 vs_prior 汇总
└── {y_col}/
    └── {model_name}/
        ├── model.{pth|txt}                      # 模型权重
        ├── metrics.json                         # train/val/test 完整指标
        ├── training_curve.png                   # 训练曲线 (loss+各指标)
        ├── feature_importance.png               # 特征重要性 (lambdamart)
        ├── abtype_distribution_train.png        # 训练集 abtype 分布对比
        ├── abtype_distribution_val.png
        └── abtype_distribution_test.png
```

### test 集预测上传 HDFS

当 `output.upload_to_hdfs: true` 时，test 集预测会按 `(date, code, abtype)` 切片，每片存为一个 parquet 并多线程上传到
HDFS：

```
{hdfs_remote_root}/{experiment_name}/{y_col}/{model_name}/predictions/{date}/{code}_{abtype}.parquet
```

### summary CSV 示例

`{experiment_name}_hit_rate_summary.csv`：

```
y_col                            model  split  topk_hit_rate  excess_hit_rate  avg_pred_rank
period_return_HIGH_weight1       mlp    train  0.4123          0.1234           0.65
period_return_HIGH_weight1       mlp    val    0.3891          0.1102           0.62
period_return_HIGH_weight1       mlp    test   0.3756          0.0987           0.60
```

---

## 十三、标准化方式

通过 `shared_data.normalize` 选择 (实现见 [data.py](model_trainer/data.py))：

| 方式        | 说明                                         | 适用场景                           |
|-----------|--------------------------------------------|--------------------------------|
| `rolling` | 用**前一天**的 mean/std 标准化当天 (无前瞻偏差)           | 多日数据，生产环境                      |
| `global`  | 用**训练集整体** mean/std 标准化全部数据                | 单日测试，研究阶段                      |
| `group`   | 在 `(date, time_sec, code)` **组内**做 z-score | 消除组间量纲差异，推荐用于排序                |
| `none`    | 不标准化                                       | 原始值有意义时 (如 LambdaMART 对标准化不敏感) |

**注意事项**：

- `rolling` 模式下，每个 split 的第一天因无前一日统计量会被置 NaN 并丢弃；若 `train_dates` 仅 1 天会得到空训练集，请改用
  `global` 或 `group`。
- `winsorize_std > 0` 时，标准化后值会被截断到 `[-winsorize_std, winsorize_std]`；`0` 关闭。
- `group` 模式会显式处理 `std=0` 的常数列 (z-score 置 0)。

---

## 十四、编程接口

除 CLI 外，可在 Python 中直接调用：

### 直接训练

```python
from model_trainer.config import Config
from model_trainer.trainer import Trainer

cfg = Config.from_dict({
    "data": {
        "data_root": "../ResampleData",
        "train_dates": "20240506-20240518",
        "val_dates": "20240519-20240524",
        "test_dates": "20240525-20240530",
        "factor_list": ["p_288", "p_291"],
        "y_list": ["period_return_HIGH_weight1"],
        "normalize": "group",
    },
    "models": [
        {"type": "mlp", "name": "mlp", "params": {"hidden_sizes": [256, 128, 64]}},
    ],
    "training": {"epochs": 100, "batch_size": 4096, "loss_type": "listmle", "topk": 1},
    "output": {"save_dir": "output", "experiment_name": "demo"},
})

trainer = Trainer(cfg)
summary_df = trainer.run()
```

### HPO 搜参

```python
from model_trainer.tuner import HyperparamTuner, get_default_search_space

search_space = get_default_search_space("mlp")
tuner = HyperparamTuner(
    base_config=cfg,
    search_space=search_space,
    method="optuna",
    n_trials=30,
    metric="auto",
    direction="maximize",
    save_dir="hpo_output/demo",
)
result = tuner.run()
print(result["best_metric"], result["best_params"])
```

### 单独使用模块

```python
# 只加载数据
from model_trainer.data import load_all_splits

df_train, df_val, df_test, factor_list, norm_stats = load_all_splits(
    data_root="../ResampleData",
    train_dates="20240506-20240518",
    val_dates="20240519-20240524",
    test_dates="20240525-20240530",
    factor_list=["p_288"],
    y_list=["period_return_HIGH_weight1"],
    normalize="group",
)

# 计算排序指标
from model_trainer.metrics import evaluate, build_prior_topk_abtypes

prior = build_prior_topk_abtypes(df=df_train, y_col="period_return_HIGH_weight1")
metrics = evaluate(y_pred=pred, y_true=df_test["period_return_HIGH_weight1"].values,
                   split_name="test", df=df_test, topk=1, prior_topk_abtypes=prior)
```

---

## 十五、常见问题

**Q: 训练集为空 (0 行)**

`normalize: "rolling"` 且 `train_dates` 仅 1 天时，首日无前一日统计量会被丢弃。解决：改用 `group` / `global` / `none`
，或增加日期。

**Q: 想要新增对比实验**

在 `train_configs.yaml` 的 `experiments` 列表追加一项即可，`submit_all.sh` 会自动纳入分组提交。

**Q: HPO 太慢**

减小 `n_trials`、改 `method: "random"`、确保 `disable_torch_compile: true`。HPO 数据只加载一次，trial 间复用。

**Q: 如何关闭 HDFS 上传**

将实验的 `output.upload_to_hdfs` 设为 `false`。

**Q: SLURM 任务如何只跑指定实验**

```bash
TARGET=local ID=2 ./submit_all.sh        # 本地只跑实验 2
# 或 SLURM 模式下修改 experiments 列表只保留目标实验
```

**Q: 组内样本数不一致怎么办**

- MLP / LambdaMART：按组扁平化处理，天然支持变长组。
- Transformer：通过 padding + attention mask 处理变长组，padding 位置不参与计算。
- 组内仅 1 个样本时，部分排序指标 (如 hit_rate) 退化为 0，属于正常现象。

**Q: LambdaMART 与神经网络损失的区别**

LambdaMART 是树模型，直接优化 `lambdarank` 目标；MLP/Transformer 通过 `loss_type` 选择 `pearson`/`listnet`/`listmle`
。三者都是组内排序损失，但 ListMLE 对排序顺序更敏感，ListNet 更关注分布相似度。

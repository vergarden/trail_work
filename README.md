# Lipophilicity logD 预测

## 1. 日志

- 2026.10.18：推倒重来，重新梳理任务和数据结构。
- 2026.10.20：用 pandas 做初步数据检查，确认数据集不能直接使用。
- 2026.10.22：尝试编写 SMILES → 分子对象的处理函数，遇到 RDKit 的 DLL 报错。
- 2026.10.23：在 AI 辅助下补全处理函数，重构 `dataclean.py`，开始考虑模块化接口。
- 2026.10.25：再次重构 `dataclean.py`，拆成可复用函数；编写 `split.py`，采用 Bemis-Murcko 骨架划分。
- 2026.11.01：完成 `split.py` 的通用化与职责拆分——清洗归 `dataclean.py`，`split.py` 只负责骨架划分和泄漏检查。
- 2026.11.03：完成 `features.py`，生成 RDKit 描述符 + Morgan 指纹特征缓存 `features.npz`。
- 2026.11.03：完成 `train_baseline.py`，跑通 Ridge、RandomForest、HistGradientBoosting 三个 baseline，最佳 scaffold test RMSE 为 0.7144。
- 2026.11.03：记录“分子图像 + CNN”预测的探索想法，目前仅完成可行性分析，尚未实现。
- 2026.11.04: 倒查数据集ing
---

## 2. 数据集检查与预处理
ing
### 2.1 数据集基本情况

`Lipophilicity.csv` 共 4200 行、3 列：

```
CMPD_CHEMBLID, exp, smiles
```

数据集的 `exp` 是 pH 7.4 下的 logD 标签。检查后发现它**不能直接用于训练**，主要问题如下：

| 问题 | 数量 | 说明 |
|---|---:|---|
| 完全重复记录 | 410 | 数据集末尾追加了一整块重复行 |
| 关键字段缺失 | 2 | `CHEMBL_BLANK_EXP`（exp 空）、`CHEMBL_BLANK_SMILES`（SMILES 空） |
| 标签非数字 | 1 | `CHEMBL_BAD_EXP_TEXT` = `definitely_wrong` |
| 标签极端值 | 1 | `CHEMBL_BAD_EXP_EXTREME` = `999999` |
| 超长 SMILES | 2 | 单条约 6000 字符；RDKit 规范化时可能栈溢出 |
| 非法 SMILES | 2 | `not_a_smiles`、`C1(CC` |
| 多组分分子 | 1 | 水合物/盐，含多个 fragment |
| 非法 ID | 1 | `ljiohohoihoa` |
| 同分子不同标签 | 2 组 | canonical SMILES 相同但 exp 冲突 |

此外，大量 `exp` 写成 `.05`、`-.37` 这种省略前导零的形式；`pd.to_numeric` 可以正确解析，但按字符串处理会出错。

### 2.2 清洗规则

清洗逻辑全部在 `dataclean.py` 中，按顺序执行：

1. 删除完全重复记录（按 `CMPD_CHEMBLID / exp / smiles` 三列判断）；
2. 删除关键字段缺失的记录；
3. 把 `exp` 转为数值，移除非数字、NaN、Inf；
4. 移出 `exp` 超出 `[-10, 10]` 的记录（本项目设定的核查范围）；
5. 检查 SMILES：
   - 长度超过 1000 字符的，在 RDKit 解析前直接移出；
   - RDKit 解析失败；
   - 重原子数超过 200；
   - 多组分结构；
6. 检查 `CMPD_CHEMBLID` 格式（`CHEMBL[0-9]+`）；
7. 同一 canonical SMILES 对应不同 `exp` 的，整组隔离；
8. 同一 canonical SMILES 且标签相同的，保留第一条。

`dataclean.py` 已从顶层脚本重构为可复用模块：

```text
load_raw()
check_smiles()
clean_dataframe() -> CleanResult
clean_file()
main()  # 命令行入口
```

既可以直接运行，也可以被后续脚本 import。

### 2.3 清洗结果

```text
原始 4200 条，保留 3778 条，移出 422 条
```

移出原因分布：

| reason | 数量 |
|---|---:|
| `duplicate_record` | 410 |
| `structure_needs_review` | 5 |
| `missing_value` | 2 |
| `conflicting_labels` | 2 |
| `invalid_exp` | 1 |
| `exp_needs_review` | 1 |
| `invalid_id` | 1 |

清洗后：

- `Lipophilicity_clean.csv`：3778 行 × 5 列；
- `Lipophilicity_review.csv`：422 行 × 8 列，逐行记录移出原因；
- `exp` 范围：`-1.5 ~ 4.5`；
- `exp` 均值 2.183，标准差 1.204；
- 无重复 ID、无重复 canonical SMILES、无缺失值。

输出列：

```
CMPD_CHEMBLID, exp, smiles, canonical_smiles, source_row
```

---

## 3. 分子表示与输入处理

### 3.1 输入表示

模型输入使用 `canonical_smiles`，而不是原始 `smiles`：

- 原始 SMILES 可能有不同的书写方式；
- canonical SMILES 保证同一分子只有一种表达；
- 骨架计算、去重、划分都以 canonical SMILES 为准。

### 3.2 特征方案

`features.py` 把每个分子转换为 2265 维特征：

| 特征块 | 维度 | 说明 |
|---|---:|---|
| RDKit 描述符 | 217 | `Descriptors.CalcMolDescriptors`，包含分子量、LogP、HBA/HBD、环数、可旋转键等 |
| Morgan 指纹 | 2048 | radius=2、2048 bit，使用 `rdFingerprintGenerator` |

合计：

```
217 + 2048 = 2265
```

特征缓存：

```
features.npz
  X      (3778, 2265) float32
  ids    (3778,)      CMPD_CHEMBLID
  names  (2265,)      特征名
```

### 3.3 CPU / GPU 与速度

- 特征计算完全在 **CPU** 上完成；
- 3778 个分子只需要计算一次，结果缓存为 `features.npz`；
- 训练 baseline 时直接读缓存，不重复做 RDKit 解析；
- 描述符里可能出现的 NaN/Inf 用 `np.nan_to_num` 替换为 0，避免后续模型报错；
- 这部分是本项目的主要 CPU 开销之一，但没有使用 GPU。

---

## 4. Baseline 模型

### 4.1 三个 baseline

| 模型 | 作用 |
|---|---|
| Ridge | 线性 baseline，检验特征是否具有基本线性信号 |
| RandomForest | 非线性树模型 baseline，对描述符尺度不敏感 |
| HistGradientBoosting | 梯度提升树，通常在表格特征上表现较强 |

选择原因：

- 三者覆盖“线性 → Bagging → Boosting”三种典型思路；
- 都能直接处理 2265 维稠密特征；
- 训练成本可控，适合作为 GNN/图像模型之前的对照。

### 4.2 预处理

- 缺失值：`SimpleImputer(strategy="median")`，只在训练集上 fit；
- Ridge：
  - RDKit 描述符中有 `Ipc` 这类重尾特征，直接用 `StandardScaler` 会让测试集极端值外推爆炸；
  - 改为 `QuantileTransformer(output_distribution="uniform")`，把每个特征压到 `[0, 1]`；
  - 该变换只在训练集上 fit，不使用验证集/测试集统计量；
  - Ridge `alpha=100.0`。
- 树模型：只做中位数填充，不做标准化。

### 4.3 结果

在 scaffold split 的 valid/test 上：

| 模型 | valid RMSE | valid MAE | valid R² | test RMSE | test MAE | test R² |
|---|---:|---:|---:|---:|---:|---:|
| Ridge | 0.7819 | 0.6050 | 0.5519 | 0.8377 | 0.6548 | 0.5570 |
| RandomForest | 0.8609 | 0.6715 | 0.4569 | 0.9251 | 0.7333 | 0.4597 |
| **HistGradientBoosting** | **0.7495** | **0.5476** | **0.5883** | **0.7144** | **0.5459** | **0.6777** |

模型选择规则：**只看 valid RMSE**，选 valid 最低的模型；test 只用于最终报告。

最佳模型：**HistGradientBoosting**

```
valid RMSE = 0.7495
test  RMSE = 0.7144
test  MAE  = 0.5459
test  R²   = 0.6777
```

### 4.4 RandomForest 专项实验

`train_rf.py` 是独立的 RandomForest 方案：复用同一份特征和 `split.csv`，只训练 RF，并保存模型、预测、特征重要性和 parity/residual 图。

默认 RF（`max_features="sqrt"`、`min_samples_leaf=2`）明显过拟合：

| 配置 | train RMSE | valid RMSE | test RMSE | test MAE | test R² |
|---|---:|---:|---:|---:|---:|
| RF 默认 | 0.3848 | 0.8609 | 0.9251 | 0.7333 | 0.4597 |
| RF 正则化后 | 0.4565 | 0.8386 | 0.8745 | 0.6905 | 0.5172 |

正则化配置（只在 valid 上选择）：

```
max_features = 0.1
min_samples_leaf = 5
max_depth = 20
n_estimators = 500
```

结论：

- 正则化后 RF 比默认 RF 好，但仍明显差于 HistGradientBoosting 的 test RMSE 0.7144；
- RF 的 train RMSE 0.4565 仍远优于 valid/test，说明高维特征上仍有过拟合；
- 按 logD 区间看，RF 在低 logD 端高估、高 logD 端低估，预测向均值收缩；这是 RF 在该特征空间上的主要误差来源；
- 因此最终 baseline 仍选 HistGradientBoosting，RF 作为对照实验保留。

---

## 5. 文献启发与模型设计

### 5.1 当前状态

目前完成的是**描述符/指纹 + 树模型**的 baseline，还没有实现图神经网络或图像模型。因此本节既记录已经采用的思路，也记录尚未实现的探索方向，避免把“想法”写成“结果”。

### 5.2 已参考的划分思路：MoleculeNet

- 文献：Wu et al., *MoleculeNet: A Benchmark for Molecular Machine Learning*, 2018。
- 借鉴点：Lipophilicity 属于 MoleculeNet 中的回归任务，推荐使用 **scaffold split**，而不是随机划分。
- 落地方式：
  - 用 RDKit 计算 Bemis-Murcko 骨架；
  - 同一骨架的分子只进入一个集合；
  - 80/10/10，seed=42；
  - 校验 canonical SMILES、骨架、group_key 三重零重叠。

### 5.3 探索想法：把分子渲染成图像做预测（尚未实现）

这是一个**尚未实现**的探索方向，不作为当前 baseline 的结果。

基本流程：

```
SMILES
  → RDKit 2D 坐标（固定朝向）
  → 渲染成固定尺寸 PNG
  → CNN / ViT
  → logD
```

可以使用的 RDKit 接口：

```python
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

mol = Chem.MolFromSmiles(smiles)
rdDepictor.Compute2DCoords(mol, canonOrient=True)

drawer = rdMolDraw2D.MolDraw2DCairo(224, 224)
opts = drawer.drawOptions()
opts.bondLineWidth = 2
opts.minFontSize = 12
opts.maxFontSize = 16
opts.padding = 0.05
opts.clearBackground = True

drawer.DrawMolecule(mol)
drawer.FinishDrawing()
png = drawer.GetDrawingText()
```

模型侧可以有两种做法：

1. **图像嵌入 + 浅层回归**
   - 用 ImageNet 预训练的 ResNet18/EfficientNet 提取图像嵌入；
   - 在训练集上拟合 Ridge/RandomForest；
   - 先判断图像里有没有可用信号，再决定是否微调。
2. **端到端微调**
   - 冻结 backbone，只训练回归头；
   - 再解冻最后一层，用小学习率微调；
   - 用验证集早停，测试集只看一次。

相关文献：

- Goh et al., *Chemception: A Deep Neural Network with Minimal Chemistry Knowledge Matches the Performance of Expert-developed QSAR/QSPR Models*, 2017；
- Zeng et al., *ImageMol: A deep learning model for molecular property prediction based on molecular images*, 2022。

需要注意的问题：

- 图像是**有损表示**：原子类型、电荷、同位素、立体化学、3D 构象都可能丢失；
- 3778 条数据对从零训练 CNN 来说偏小，容易过拟合；
- 模型可能学到渲染风格（字体、键长、背景）而不是化学；
- 必须固定渲染参数，训练/验证/测试使用同一套设置；
- 不做水平翻转（会改变立体化学），不做强烈颜色抖动（原子颜色有语义）；
- 预期图像模型不一定优于描述符/树模型，更适合作为**对照实验或集成成员**。

因此，图像预测目前只作为后续可选的探索方向，没有写入任何实验结果。

### 5.4 下一步的模型方向

1. **图神经网络**：GIN / MPNN 或简化的 D-MPNN，直接在原子-键图上传递消息；
2. **图像模型**：先做 ResNet18 嵌入 + Ridge 的最小实验，再决定是否微调；
3. **集成**：如果图像模型与树模型误差相关性低，可以尝试简单平均或 stacking。

---

## 6. AI 辅助过程

### 6.1 使用的工具

- AI 编程助手：Codex / GPT（用于代码生成、重构建议、报错分析）；
- 本会话中的 DeepSeek Harness 助手（用于数据检查、脚本重构、baseline 实现和结果核对）。

### 6.2 AI 参与的内容

- 把 `dataclean.py` 从顶层脚本重构为可复用函数，增加 `CleanResult` 和 CLI；
- 实现 `split.py` 的 Bemis-Murcko 骨架划分、通用列名支持和泄漏检查；
- 编写 `train_baseline.py`，完成 Ridge / RandomForest / HistGBR 的训练与评估；
- 协助定位环境问题，例如 `libglib` 覆盖 `intl-8.dll` 导致 RDKit 绘图模块加载失败。

### 6.3 prompt 设计

给 AI 的请求尽量包含明确约束，例如：

- 输入是什么、输出是什么、哪些列不能改；
- 哪些逻辑属于 `dataclean.py`，哪些属于 `split.py`；
- 不允许重新划分、不允许用 test 调参；
- 要求保留原有输出格式，并给出可运行的验证步骤；
- 要求先备份，再修改。

### 6.4 如何核查 AI 生成内容

- **数据清洗**：用旧版脚本和新版脚本跑同一份数据，比较 clean CSV 的 SHA256；清洗结果逐字节一致；review 行内容逐列一致。
- **数据划分**：检查 `split.csv` 行数、各集合数量、canonical SMILES / 骨架 / group_key 三重零重叠；与旧版划分逐分子对比，3778/3778 一致。
- **baseline**：独立运行 `train_baseline.py`，检查 `baseline_metrics.json` 中的 valid/test 指标和 `baseline_predictions.csv` 的行数、NaN、split 分布。
- **代码审阅**：逐段阅读 AI 生成的代码，确认没有把清洗、划分、训练混在一起；发现 Ridge 数值不稳定后，回到 valid 集重新选择预处理，而不是用 test 调参。

AI 生成的内容只作为参考，最终代码和结果由本人负责。

---

## 7. 实验设置与评价方式

### 7.1 数据划分

- 方式：Bemis-Murcko scaffold split；
- 骨架计算：`MurckoScaffoldSmiles(mol=mol, includeChirality=False)`；
- 无环分子：骨架为空，每个分子单独成组；
- 比例：80% / 10% / 10%；
- seed：42；
- 结果：

| split | 分子数 | 非空骨架数 |
|---|---:|---:|
| train | 3022 | 1460 |
| valid | 378 | 376 |
| test | 378 | 377 |
| 合计 | 3778 | 2213 |

- 5 个无环分子，单独成组；
- 骨架组总数约 2218（2213 个非空骨架 + 5 个无环分子）；
- `split.py` 支持 `--id-col / --smiles-col / --target-col`，默认自动识别常见列名；
- `split.py` 只做校验，不删除/修复/去重；缺失、非法、重复或超长 SMILES 会直接报错并提示先运行 `dataclean.py`。

泄漏检查：

```
smiles    train-valid/train-test/valid-test = 0/0/0
scaffold  train-valid/train-test/valid-test = 0/0/0
group     train-valid/train-test/valid-test = 0/0/0
```

标签分布：

| split | count | mean | std | min | max |
|---|---:|---:|---:|---:|---:|
| train | 3022 | 2.1655 | 1.2010 | -1.50 | 4.50 |
| valid | 378 | 2.2579 | 1.1697 | -1.33 | 4.49 |
| test | 378 | 2.2517 | 1.2602 | -1.42 | 4.40 |

### 7.2 特征与训练

- 特征：217 个 RDKit 描述符 + 2048 bit Morgan 指纹，共 2265 维；
- 模型：
  - Ridge：`alpha=100.0` + `QuantileTransformer(uniform)`；
  - RandomForest：500 棵树，`max_features="sqrt"`，`min_samples_leaf=2`；
  - HistGradientBoosting：`max_iter=500`，`learning_rate=0.05`，`max_leaf_nodes=31`，`l2_regularization=1.0`；
- 随机种子：42；
- 运行环境：
  - Python 3.12.15；
  - rdkit 2026.03.1；
  - pandas 3.0.6；
  - scikit-learn 1.9.1；
  - scipy 1.18.1；
  - numpy 2.5.3；
  - matplotlib 3.11.2（pip wheel，自带 raqm/harfbuzz，避免 conda 的 `intl-8.dll` 冲突）；
  - PyTorch 2.14.1+cu132（baseline 未使用，GPU 已确认可用）。

### 7.3 评价指标

- **RMSE**：主指标，和任务要求一致；
- MAE：平均绝对误差，对异常值不如 RMSE 敏感；
- R²：解释方差比例，便于和随机猜测比较；
- 模型选择只看 valid RMSE；
- test 不参与模型选择或调参，只用于最终报告。

---

## 8. 实验结果与分析

### 8.1 baseline 结果

| 模型 | valid RMSE | test RMSE | test MAE | test R² |
|---|---:|---:|---:|---:|
| Ridge | 0.7819 | 0.8377 | 0.6548 | 0.5570 |
| RandomForest | 0.8609 | 0.9251 | 0.7333 | 0.4597 |
| **HistGradientBoosting** | **0.7495** | **0.7144** | **0.5459** | **0.6777** |

### 8.2 分析

- HistGradientBoosting 在 valid 和 test 上都最好，说明在 2265 维描述符/指纹特征上，Boosting 比 Bagging 和线性模型更适合；
- RandomForest 的 valid/test 都明显差于 HistGBR，说明在这个高维描述符/指纹空间上，Bagging 的偏差较大；
- 对 RF 做了只在 valid 上的小范围正则化（`max_features=0.1`、`min_samples_leaf=5`、`max_depth=20`），valid/test RMSE 从 0.8609/0.9251 改善到 0.8386/0.8745，但仍不如 HistGBR；按 logD 区间看，RF 在低 logD 端高估、高 logD 端低估，预测向均值收缩；
- Ridge 最初使用 `StandardScaler` 时 test RMSE 异常大，原因是描述符重尾特征导致外推爆炸；改用只在训练集上 fit 的 `QuantileTransformer(uniform)` 后，Ridge 稳定在 test RMSE 0.838；
- HistGBR 的 test RMSE 略低于 valid，但这只是单次 scaffold 划分的结果，不能据此认为 test 更容易；样本量较小，指标存在波动。

### 8.3 当前主要问题

- 只做了一次 80/10/10 划分，没有 5 折 scaffold 交叉验证，指标稳定性有限；
- 模型超参数是固定值，没有系统调参；
- 只使用 2D 描述符/指纹，没有图结构信息；
- 没有误差分析（按骨架、分子量、logD 区间分组）；
- 图像预测和图神经网络都还没有实现；
- 还没有 `test.csv` 推理接口和最终报告生成脚本。

---

## 9. 复现说明

### 9.1 主要文件

| 文件 | 作用 |
|---|---|
| `dataclean.py` | 数据清洗，生成 clean/review 数据 |
| `split.py` | Bemis-Murcko 骨架划分，生成 split.csv/json |
| `features.py` | RDKit 描述符 + Morgan 指纹，生成 features.npz |
| `train_baseline.py` | Ridge / RandomForest / HistGBR 训练与评估 |
| `env.yml` | conda 环境 |
| `Lipophilicity_clean.csv` | 清洗后的数据 |
| `Lipophilicity_review.csv` | 被移出的记录及原因 |
| `split.csv` / `split.json` | 划分结果与统计 |
| `features.npz` | 特征缓存 |
| `baseline_metrics.json` | baseline 指标 |
| `baseline_best.joblib` | 最佳 baseline 模型 |
| `baseline_predictions.csv` | 最佳模型在全部样本上的预测 |
| `train_rf.py` | RandomForest 专项训练、预测、特征重要性和可视化 |
| `rf_metrics.json` | RF 的 train/valid/test 指标 |
| `rf_model.joblib` | RF 模型 |
| `rf_predictions.csv` | RF 在全部样本上的预测 |
| `rf_feature_importance.csv` | RF 特征重要性前 50 |
| `rf_parity_test.png` / `rf_residuals_test.png` | RF 测试集 parity 图与残差图 |

### 9.2 运行顺序

```bash
cd C:\kf\code\trail_work

# 1. 创建环境（已有 myenv 可跳过）
conda env create -f env.yml
conda activate myenv

# 2. 数据清洗
python dataclean.py

# 3. 骨架划分
python split.py

# 4. 特征
python features.py

# 5. baseline
python train_baseline.py

# 6. RandomForest 专项（可选）
python train_rf.py
```

### 9.3 当前输出

- `dataclean.py` → `Lipophilicity_clean.csv`、`Lipophilicity_review.csv`
- `split.py` → `split.csv`、`split.json`
- `features.py` → `features.npz`
- `train_baseline.py` → `baseline_metrics.json`、`baseline_best.joblib`、`baseline_predictions.csv`

### 9.4 test.csv 推理接口

**目前尚未实现。**

任务要求提供一个脚本，在拿到带标签的 `test.csv` 后完成推理、与真实标签对比并输出检验报告。当前代码只完成了训练侧流程，还没有 `predict.py`。

计划中的接口：

```bash
python predict.py --input test.csv --model baseline_best.joblib --out reports/
```

需要保证：

- 复用 `features.py` 的特征逻辑，训练和推理特征列完全一致；
- 不重新拟合任何预处理，直接使用模型内的 imputer / transformer；
- 对非法/缺失 SMILES 不崩溃，单独记录；
- 输出预测结果、RMSE/MAE/R²、失败行和误差最大的样本。

---

## 10. 当前未完成 / 下一步

1. **`predict.py`**：完成 `test.csv` 推理与报告生成（交付要求）；
2. **误差分析**：parity plot、残差分布、按骨架/分子量/logD 区间分组；
3. **5 折 scaffold 交叉验证**：给 baseline 一个均值 ± 标准差，而不是单次划分；
4. **改进模型**：GIN / MPNN 或简化的 D-MPNN，与 HistGBR 在同一 split 上对比；
5. **图像预测实验（可选）**：先做 ResNet18 嵌入 + Ridge 的最小实验，再决定是否微调 CNN；
6. **README 第 11 节手写承诺与签名**：必须由本人完成，不能由 AI 代写；
7. 最终提交前检查 `env.yml`、模型文件、`test.csv` 推理脚本和复现命令。

---

## 11. 手写内容与真实性声明：**本部分必须由本人独立撰写，不可直接复制 AI 输出。**

请至少写一段，内容可以包括：

- 本次任务中你遇到的最大问题，以及你如何处理
- 你从文章中借鉴的思路为什么值得保留
- 你认为当前模型最薄弱的地方
- 如果继续做下去，你最想改进什么

并请在这里手写承诺你的代码完全符合你声称的功能，并真实得到了你所呈现的结果，并为此负责

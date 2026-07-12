"""
Baseline 模型：RDKit 描述符 + Morgan 指纹 → 随机森林 / 线性回归
以 RMSE 为主要评估指标，对比两个模型建立性能参照线
"""
import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / 'Lipophilicity_features.csv'
MODEL_PATH = ROOT / 'model_rf.pkl'
FEATURE_LIST_PATH = ROOT / 'feature_list.json'

# 元数据列：不作为特征输入
META_COLS = ['CMPD_CHEMBLID', 'exp', 'smiles']

# 模型配置
RF_PARAMS = {'n_estimators': 200, 'max_depth': 20, 'random_state': 42, 'n_jobs': -1}
RIDGE_PARAMS = {'alpha': 1.0}
GBR_PARAMS = {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05,
              'subsample': 0.8, 'random_state': 42}


def load_data():
    """读取特征表，分离特征矩阵 X 和目标向量 y。"""
    df = pd.read_csv(INPUT)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feature_cols].values.astype(np.float32)
    y = df['exp'].values.astype(np.float32)
    return df, X, y, feature_cols


def scaffold_split(df, test_size=0.3, val_size=0.15, random_state=42):
    """用 RingCount + LogP 分箱模拟骨架划分：环数相同+LogP 区间相同 = 同类。"""
    ring = df['RingCount'].astype(int)
    logp_bin = pd.cut(df['LogP'], bins=8, labels=False).fillna(-1).astype(int)
    # 组合键: "环数_LogP区间"
    groups = (ring.astype(str) + '_' + logp_bin.astype(str)).values

    scaffolds = defaultdict(list)
    for i, g in enumerate(groups):
        scaffolds[g].append(i)

    np.random.seed(random_state)
    keys = list(scaffolds.keys())
    np.random.shuffle(keys)

    total = len(df)
    idx_test, idx_val, idx_train = [], [], []
    for key in keys:
        ids = scaffolds[key]
        if len(idx_test) / total < test_size:
            idx_test.extend(ids)
        elif len(idx_val) / total < val_size:
            idx_val.extend(ids)
        else:
            idx_train.extend(ids)

    idx_train = np.array(idx_train); idx_val = np.array(idx_val); idx_test = np.array(idx_test)
    print(f'  伪骨架种类: {len(scaffolds)}')
    print(f'  训练集: {len(idx_train)} | 验证集: {len(idx_val)} | 测试集: {len(idx_test)}')
    return idx_train, idx_val, idx_test, len(scaffolds)

def evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test, name):
    """训练模型并在训练集、验证集、测试集上报告 RMSE / MAE / R²。"""
    model.fit(X_train, y_train)

    results = {}
    for split, Xs, ys in [('训练集', X_train, y_train),
                           ('验证集', X_val,  y_val),
                           ('测试集', X_test, y_test)]:
        preds = model.predict(Xs)
        rmse = np.sqrt(mean_squared_error(ys, preds))
        mae  = mean_absolute_error(ys, preds)
        r2   = r2_score(ys, preds)
        results[split] = {'rmse': rmse, 'mae': mae, 'r2': r2}

    print(f'\n{"="*10} {name} {"="*10}')
    for split, m in results.items():
        print(f'{split}: RMSE={m["rmse"]:.4f}, MAE={m["mae"]:.4f}, R²={m["r2"]:.4f}')

    return results


def main():
    df, X, y, feature_cols = load_data()
    print(f'特征表: {df.shape[0]} 行, {len(feature_cols)} 个特征')
    print(f'目标 (logD): mean={y.mean():.2f}, std={y.std():.2f}, '
          f'min={y.min():.2f}, max={y.max():.2f}')

    # 划分：70% 训练 / 15% 验证 / 15% 测试
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42)
    print(f'训练集: {len(X_train)} | 验证集: {len(X_val)} | 测试集: {len(X_test)}')

    # 模型 1: 随机森林
    rf = RandomForestRegressor(**RF_PARAMS)
    rf_results = evaluate(rf, X_train, y_train, X_val, y_val, X_test, y_test, '随机森林')

    # 模型 2: 岭回归
    ridge = Ridge(**RIDGE_PARAMS)
    ridge_results = evaluate(ridge, X_train, y_train, X_val, y_val, X_test, y_test, '岭回归')

    # 模型 3: 梯度提升回归 (GBR，XGBoost 的 sklearn 等价版)
    gbr = GradientBoostingRegressor(**GBR_PARAMS)
    gbr_results = evaluate(gbr, X_train, y_train, X_val, y_val, X_test, y_test, 'GBR')

    # 特征重要性（随机森林）
    print(f'\n{"="*10} 特征重要性 Top 10 {"="*10}')
    importance = rf.feature_importances_
    top_idx = np.argsort(importance)[::-1][:10]
    for rank, idx in enumerate(top_idx, 1):
        print(f'  {rank:2d}. {feature_cols[idx]:25s}  {importance[idx]:.4f}')

    # 汇总对比（三模型）
    sep = "=" * 10
    print(f'\n{sep} 三模型对比 (随机划分) {sep}')
    print(f'{"模型":<12} {"测试集 RMSE":>12} {"测试集 R²":>10}')
    for name, res in [('GBR', gbr_results), ('随机森林', rf_results), ('岭回归', ridge_results)]:
        print(f'{name:<12} {res["测试集"]["rmse"]:>12.4f} {res["测试集"]["r2"]:>10.4f}')

    # 伪骨架划分对比 (仅随机森林)
    print(f'\n{sep} 骨架隔离评估 (仅随机森林) {sep}')
    idx_train, idx_val, idx_test, n_scaffolds = scaffold_split(df)
    rf2 = RandomForestRegressor(**RF_PARAMS)
    rf_scaf = evaluate(rf2,
                       X[idx_train], y[idx_train],
                       X[idx_val],   y[idx_val],
                       X[idx_test],  y[idx_test],
                       'RF (骨架隔离)')
    rf_random_rmse = rf_results['测试集']['rmse']
    rf_scaf_rmse = rf_scaf['测试集']['rmse']
    print(f'\n{sep} 划分方式对比 (RF) {sep}')
    print(f'  随机划分:    RMSE={rf_random_rmse:.4f}  R²={rf_results["测试集"]["r2"]:.4f}')
    print(f'  骨架隔离:    RMSE={rf_scaf_rmse:.4f}  R²={rf_scaf["测试集"]["r2"]:.4f}')
    if rf_random_rmse > 0:
        print(f'  骨架隔离 RMSE 上升: {(rf_scaf_rmse - rf_random_rmse) / rf_random_rmse * 100:.1f}%')

    # 保存最佳模型（GBR）供推理使用
    joblib.dump(gbr, MODEL_PATH)
    with open(FEATURE_LIST_PATH, 'w') as f:
        json.dump(feature_cols, f)
    print(f'\n模型已保存 (GBR): {MODEL_PATH}')
    print(f'特征列表已保存: {FEATURE_LIST_PATH}')


if __name__ == '__main__':
    main()

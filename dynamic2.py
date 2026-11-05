import argparse
from datetime import datetime
from collections import Counter
from itertools import combinations
from typing import NamedTuple
import math
import os
import tempfile
from time import perf_counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, GridSearchCV, ParameterGrid, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from threadpoolctl import threadpool_limits
import joblib

from catboost import CatBoostRegressor

import torch
import torch.nn as nn
import torch.optim as optim


DATA_PATH = r'C:\Users\X\Desktop\机器学习（新）\ADA811.csv'
OUTPUT_DIR = r'C:\Users\X\Desktop\机器学习（新）\dynamic_fusion_cat_svr_rf_gridsearchcv10'
FONT_PATH = r'C:\Users\X\Desktop\机器学习（新）\timesbd.ttf'
FEATURES = ['CT', 'AZ', 'DC', 'AC', 'RP', 'IS', 'SSA']
TARGET = 'ICE'
BASE_MODELS = ('CAT', 'SVR', 'RF')
FUSION_MODEL = 'DynamicFusion_CAT_SVR_RF'
COMPLEX_FUSION_HIDDEN_DIMS = (64, 32, 16)
COMPLEX_GRID_UPDATES = {
    'CAT': {
        'n_estimators': [1300, 2200, 3000],
        'depth': [4, 5, 6, 7],
    },
    'SVR': {
        'C': [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0],
        'gamma': [0.003, 0.005, 0.01, 0.02, 0.03, 0.1, 0.3],
        'epsilon': [0.01, 0.03, 0.04, 0.05, 0.06, 0.07],
    },
    'RF': {
        'n_estimators': [300, 400, 800, 1200],
        'max_depth': [9, 11, 15, 19],
        'max_features': [0.7, 1.0],
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description='动态融合预测模型的多次独立划分与稳定性评估')
    parser.add_argument('--min-minutes', type=float, default=60, help='调参与训练累计分钟数下限，默认 60')
    parser.add_argument('--min-repeats', type=int, default=5, help='完整评估轮数下限，默认 5')
    parser.add_argument('--complexity-repeats', type=int, default=2,
                        help='从已完成的正式评估中选取几轮做复杂模型对照，默认 2，设为 0 跳过')
    parser.add_argument('--research-mode', choices=('full', 'quick', 'off'), default='full',
                        help='独立研究流程，默认 full，quick 使用较小网格，off 跳过')
    parser.add_argument('--output-dir', default=OUTPUT_DIR, help='本次实验结果的父目录')
    args = parser.parse_args()
    if not math.isfinite(args.min_minutes) or args.min_minutes < 0:
        parser.error('--min-minutes 必须是有限的非负数')
    if args.min_repeats < 1:
        parser.error('--min-repeats 必须至少为 1')
    if args.complexity_repeats < 0:
        parser.error('--complexity-repeats 必须是非负整数')
    return args


def huber_loss_np(y_true, y_pred, delta=1.0):
    err = np.asarray(y_true) - np.asarray(y_pred)
    abs_err = np.abs(err)
    quadratic = np.minimum(abs_err, delta)
    return np.mean(0.5 * quadratic**2 + delta * (abs_err - quadratic))


def calc_metrics(y_true, y_pred):
    return {
        'R2': r2_score(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred),
        'Huber': huber_loss_np(y_true, y_pred),
    }


def print_metrics(title, metrics):
    print(
        f"{title} -> R2: {metrics['R2']:.6f}  "
        f"RMSE: {metrics['RMSE']:.6f}  "
        f"MAE: {metrics['MAE']:.6f}  "
        f"MAPE: {metrics['MAPE']:.6f}  "
        f"Huber: {metrics['Huber']:.6f}"
    )


def fix_heatmap_border(ax):
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)


def finalize_heatmap_figure(fig, ax, left=0.16, right=0.98, bottom=0.18, top=0.92):
    fix_heatmap_border(ax)
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)


def plot_feature_correlation(X_train, y_train, save_dir):
    voc_corr = pd.concat([X_train, y_train], axis=1).corr().round(3)
    colors = ['#0000ff', '#ffffff', '#ff0000']
    custom_cmap = LinearSegmentedColormap.from_list('blue_red', colors)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    sns.heatmap(
        voc_corr,
        annot=True,
        vmin=-1,
        vmax=1,
        fmt='.3f',
        square=True,
        cmap=custom_cmap,
        annot_kws={'weight': 'bold', 'fontsize': 10},
        cbar_kws={'shrink': 0.8},
        ax=ax
    )
    ax.set_title('Feature Correlation Heatmap', fontsize=14, pad=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    finalize_heatmap_figure(fig, ax, left=0.16, right=0.98, bottom=0.22, top=0.92)
    plt.show(block=True)
    fig.savefig(os.path.join(save_dir, 'feature_correlation_heatmap.png'),
                dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)


def plot_fit_combined(y_train_true, y_train_pred, y_test_true, y_test_pred, title, save_path):
    y_train_true = np.asarray(y_train_true)
    y_train_pred = np.asarray(y_train_pred)
    y_test_true = np.asarray(y_test_true)
    y_test_pred = np.asarray(y_test_pred)

    all_true = np.concatenate([y_train_true, y_test_true, ])
    all_pred = np.concatenate([y_train_pred, y_test_pred, ])

    min_val = min(all_true.min(), all_pred.min())
    max_val = max(all_true.max(), all_pred.max())
    padding = 0.01 * (max_val - min_val) if max_val > min_val else 0.01

    line_x = np.linspace(min_val - padding, max_val + padding, 200)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)

    ax.scatter(
        y_train_true, y_train_pred,
        s=55, alpha=0.75,
        c='#1f77b4', edgecolors='k', linewidths=0.5,
        label='Train'
    )
    ax.scatter(
        y_test_true, y_test_pred,
        s=55, alpha=0.75,
        c='#ff7f0e', edgecolors='k', linewidths=0.5,
        label='Test'
    )

    ax.plot(line_x, line_x, 'r--', linewidth=1.8, label='Ideal')

    ax.set_xlabel('True Value', fontweight='bold')
    ax.set_ylabel('Predicted Value', fontweight='bold')
    ax.set_title(title, fontweight='bold')

    ax.set_xlim(min_val - padding, max_val + padding)
    ax.set_ylim(min_val - padding, max_val + padding)

    leg = ax.legend(frameon=True)
    for text in leg.get_texts():
        text.set_fontweight('bold')

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    ax.tick_params(axis='both', width=1.2, length=5)
    fig.subplots_adjust(left=0.15, right=0.96, bottom=0.14, top=0.90)

    plt.show(block=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)


class DynamicFusionNet(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=16, extra_hidden_dims=()):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.ReLU()
        layers = []
        previous_width = hidden_dim
        for width in extra_hidden_dims:
            layers.extend([nn.Linear(previous_width, width), nn.ReLU(), nn.Dropout(p=0.1)])
            previous_width = width
        self.hidden_layers = nn.Sequential(*layers)
        self.fc2 = nn.Linear(previous_width, in_dim)

    def forward(self, preds):
        h = self.act(self.fc1(preds))
        scores = self.fc2(self.hidden_layers(h))
        weights = torch.softmax(scores, dim=1)
        fused = (weights * preds).sum(dim=1)
        return fused, weights


def fit_base_models(X_train, X_test, y_train, seed, n_jobs, *, complex_models=False):
    cv10 = KFold(n_splits=10, shuffle=True, random_state=seed)
    model_configs = [
        ('CAT', CatBoostRegressor(
            loss_function='RMSE', random_state=seed, thread_count=n_jobs,
            verbose=0, allow_writing_files=False,
        ), {
            'n_estimators': [1300],
            'depth': [4, 5, 6],
            'learning_rate': [0.01, 0.03],
        }),
        ('SVR', SVR(kernel='rbf'), {
            'C': [0.1, 0.3, 1.0, 3.0, 10.0],
            'gamma': [0.003, 0.005, 0.01, 0.02, 0.03],
            'epsilon': [0.03, 0.04, 0.05, 0.06, 0.07],
        }),
        ('RF', RandomForestRegressor(random_state=seed, n_jobs=n_jobs), {
            'n_estimators': [300, 400],
            'max_depth': [9, 11],
            'min_samples_leaf': [1, 2],
        }),
    ]
    searches, pred_oof, pred_train, pred_test = {}, {}, {}, {}
    for name, estimator, param_grid in model_configs:
        if complex_models:
            param_grid = {**param_grid, **COMPLEX_GRID_UPDATES[name]}
        candidates = len(ParameterGrid(param_grid))
        print(f'  {name}：{candidates} 组参数 × 10 折，随后生成训练集折外预测', flush=True)
        # 每折各自拟合补缺与标准化，且输入中不含目标列 ICE。
        pipeline = Pipeline([
            ('imputer', KNNImputer(n_neighbors=1)),
            ('scaler', StandardScaler()),
            ('model', estimator),
        ])
        search = GridSearchCV(
            estimator=pipeline,
            param_grid={f'model__{key}': values for key, values in param_grid.items()},
            scoring='r2', cv=cv10, n_jobs=n_jobs, refit=True,
            return_train_score=True, error_score='raise',
        )
        search.fit(X_train, y_train)
        best_pipeline = search.best_estimator_
        searches[name] = search
        pred_oof[name] = cross_val_predict(best_pipeline, X_train, y_train, cv=cv10, n_jobs=n_jobs)
        pred_train[name] = best_pipeline.predict(X_train)
        pred_test[name] = best_pipeline.predict(X_test)

    return searches, pred_oof, pred_train, pred_test


def fusion_input(predictions):
    return torch.tensor(np.column_stack([predictions[name] for name in BASE_MODELS]), dtype=torch.float32)


def fit_fusion(pred_oof, y_train, seed, *, complex_models=False):
    torch.manual_seed(seed)
    hidden_dims = COMPLEX_FUSION_HIDDEN_DIMS if complex_models else (16,)
    epochs = 2000 if complex_models else 500
    fusion_net = DynamicFusionNet(
        in_dim=len(BASE_MODELS),
        hidden_dim=hidden_dims[0],
        extra_hidden_dims=hidden_dims[1:],
    )
    if complex_models:
        print(f'  动态融合：隐藏层宽度 {hidden_dims}，训练 {epochs} 个 epoch', flush=True)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(fusion_net.parameters(), lr=1e-3, weight_decay=1e-4)
    inputs = fusion_input(pred_oof)
    targets = torch.tensor(np.asarray(y_train), dtype=torch.float32)

    fusion_net.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        fused_pred, _ = fusion_net(inputs)
        loss = criterion(fused_pred, targets)
        loss.backward()
        optimizer.step()
    return fusion_net.eval()


def save_reference_models(searches, fusion_net, save_dir):
    file_stems = {'CAT': 'catboost', 'SVR': 'svr', 'RF': 'rf'}
    best_params = []
    for name, search in searches.items():
        stem = file_stems[name]
        grid_results = pd.DataFrame(search.cv_results_)
        grid_results.to_csv(os.path.join(save_dir, f'{stem}_gridsearch_results.csv'),
                            index=False, encoding='utf-8-sig')
        best_params.append({
            'Model': 'CatBoost' if name == 'CAT' else name,
            'Best_Params': str({key.split('__', 1)[1]: value for key, value in search.best_params_.items()}),
            'Best_CV_R2': search.best_score_,
        })
        joblib.dump(search.best_estimator_.named_steps['model'],
                    os.path.join(save_dir, f'{stem}_best_model.pkl'))

    pd.DataFrame(best_params).to_csv(os.path.join(save_dir, 'best_params_summary.csv'),
                                     index=False, encoding='utf-8-sig')
    # 三个基模型的最终拟合使用同一训练集和相同的预处理步骤。
    preprocessing = searches['CAT'].best_estimator_.named_steps
    joblib.dump(preprocessing['imputer'], os.path.join(save_dir, 'knn_imputer.pkl'))
    joblib.dump(preprocessing['scaler'], os.path.join(save_dir, 'standard_scaler.pkl'))
    torch.save(fusion_net.state_dict(), os.path.join(save_dir, 'dynamic_fusion_net.pth'))


def run_round(X, y, round_number, n_jobs, save_dir, *, complex_models=False):
    split_seed = 7 + round_number
    model_seed = 4 + round_number
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=split_seed)
    label = '复杂模型对照' if complex_models else '正式评估'
    print(f'{label}第 {round_number} 轮 | 划分种子 {split_seed} | 训练种子 {model_seed}', flush=True)

    reference_round = round_number == 1 and not complex_models
    if reference_round:
        print('请关闭相关性热图窗口以开始训练。', flush=True)
        plot_feature_correlation(X_train, y_train, save_dir)

    started_at = perf_counter()
    searches, pred_oof, pred_train, pred_test = fit_base_models(
        X_train, X_test, y_train, model_seed, n_jobs, complex_models=complex_models,
    )
    # 融合网络只用外层训练集中各样本的折外基模型预测来训练。
    fusion_net = fit_fusion(pred_oof, y_train, model_seed, complex_models=complex_models)
    training_seconds = perf_counter() - started_at

    with torch.no_grad():
        fused_train, train_weights = fusion_net(fusion_input(pred_train))
        fused_test, test_weights = fusion_net(fusion_input(pred_test))
    pred_train[FUSION_MODEL] = fused_train.numpy()
    pred_test[FUSION_MODEL] = fused_test.numpy()

    results = []
    for name in (*BASE_MODELS, FUSION_MODEL):
        train_metrics = calc_metrics(y_train, pred_train[name])
        test_metrics = calc_metrics(y_test, pred_test[name])
        results.append({
            'Model': name,
            **{f'Train_{metric}': value for metric, value in train_metrics.items()},
            **{f'Test_{metric}': value for metric, value in test_metrics.items()},
        })
        print_metrics(f'  {name} Test', test_metrics)

    if reference_round:
        pd.DataFrame(results[:-1]).to_csv(os.path.join(save_dir, 'base_model_results.csv'),
                                           index=False, encoding='utf-8-sig')
        pd.DataFrame(results[-1:]).to_csv(os.path.join(save_dir, 'dynamic_fusion_summary.csv'),
                                          index=False, encoding='utf-8-sig')
        weight_df = pd.DataFrame({
            'Model': BASE_MODELS,
            'Avg_Train_Weight': train_weights.mean(dim=0).numpy(),
            'Avg_Test_Weight': test_weights.mean(dim=0).numpy(),
        })
        weight_df.to_csv(os.path.join(save_dir, 'dynamic_fusion_weights.csv'), index=False, encoding='utf-8-sig')
        for split, truth, predictions in [('train', y_train, pred_train), ('test', y_test, pred_test)]:
            pd.DataFrame({
                f'y_{split}_true': np.asarray(truth),
                f'y_{split}_cat': predictions['CAT'],
                f'y_{split}_svr': predictions['SVR'],
                f'y_{split}_rf': predictions['RF'],
                f'y_{split}_fused': predictions[FUSION_MODEL],
            }).to_csv(os.path.join(save_dir, f'{split}_predictions.csv'), index=False, encoding='utf-8-sig')
        save_reference_models(searches, fusion_net, save_dir)
        print('请关闭拟合图窗口以继续。', flush=True)
        plot_fit_combined(
            y_train, pred_train[FUSION_MODEL], y_test, pred_test[FUSION_MODEL],
            'Dynamic Fusion Fit (Train + Test)', os.path.join(save_dir, 'dynamic_fusion_fit_train_test.png'),
        )

    return [dict(Round=round_number, Split_Seed=split_seed, Model_Seed=model_seed,
                 Round_Training_Seconds=training_seconds, **row) for row in results], training_seconds


def save_stability_reports(round_results, save_dir):
    results = pd.DataFrame(round_results)
    grouped = results.groupby('Model', sort=False)
    metrics = [column for column in results if column.startswith('Test_')]
    summary = grouped[metrics].agg(['mean', 'std', 'min', 'max'])
    summary.columns = [f'{metric}_{stat}' for metric, stat in summary.columns]
    summary.insert(0, 'Completed_Rounds', grouped.size())
    for filename, frame in [('stability_runs.csv', results), ('stability_summary.csv', summary.reset_index())]:
        path = os.path.join(save_dir, filename)
        temporary_path = path + '.tmp'
        frame.to_csv(temporary_path, index=False, encoding='utf-8-sig')
        os.replace(temporary_path, path)


def main():
    args = parse_args()
    # Windows 下避免默认启动大量 GridSearchCV 子进程。
    try:
        n_jobs = max(1, int(os.environ.get('ML_N_JOBS', '1')))
    except ValueError:
        n_jobs = 1
    torch.set_num_threads(n_jobs)

    df = pd.read_csv(DATA_PATH)
    # 缺失的目标值不参与训练或评估，特征缺失值留到各折内补全。
    df = df.dropna(subset=[TARGET])
    X, y = df[FEATURES], df[TARGET]

    if os.path.exists(FONT_PATH):
        font_prop = fm.FontProperties(fname=FONT_PATH, size=10, weight='bold')
        matplotlib.rcParams['font.family'] = font_prop.get_name()
        matplotlib.rcParams['font.size'] = font_prop.get_size()

    os.makedirs(args.output_dir, exist_ok=True)
    prefix = datetime.now().strftime('stability_%Y%m%d_%H%M%S_')
    save_dir = tempfile.mkdtemp(prefix=prefix, dir=os.path.abspath(args.output_dir))
    print(f'输出目录：{save_dir}', flush=True)
    print(f'并行数 {n_jobs} | 最少 {args.min_repeats} 轮 | 调参与训练累计至少 {args.min_minutes:g} 分钟',
          flush=True)

    completed_rounds = 0
    training_seconds = 0.0
    round_results = []
    while completed_rounds < args.min_repeats or training_seconds < args.min_minutes * 60:
        rows, duration = run_round(X, y, completed_rounds + 1, n_jobs, save_dir)
        round_results.extend(rows)
        training_seconds += duration
        completed_rounds += 1
        save_stability_reports(round_results, save_dir)
        print(f'已完成 {completed_rounds} 轮 | 调参与训练累计 {training_seconds / 60:.2f} 分钟', flush=True)

    # 正式结果已保存。对照复用同一组划分种子，只在终端报告指标。
    baseline_fusion = {row['Round']: row for row in round_results if row['Model'] == FUSION_MODEL}
    comparison_rounds = min(args.complexity_repeats, completed_rounds)
    for round_number in range(1, comparison_rounds + 1):
        rows, duration = run_round(X, y, round_number, n_jobs, save_dir=None, complex_models=True)
        comparison = next(row for row in rows if row['Model'] == FUSION_MODEL)
        baseline = baseline_fusion[round_number]
        print(f"对照耗时 {duration / 60:.2f} 分钟 | 融合模型测试 R² "
              f"正式 {baseline['Test_R2']:.6f} / 对照 {comparison['Test_R2']:.6f} | 测试 RMSE "
              f"正式 {baseline['Test_RMSE']:.6f} / 对照 {comparison['Test_RMSE']:.6f}", flush=True)

    if args.research_mode != 'off':
        run_research_study(df, args.research_mode, n_jobs, completed_rounds)

    print(f'稳定性评估完成，逐轮记录和汇总已保存到 {save_dir}', flush=True)


# 独立研究流程。这里的拟合、排序和诊断只在终端报告。
# 正式评估、模型保存和两处交互式绘图由上面的入口负责。


class ResearchProfile(NamedTuple):
    row_repeats: int
    group_folds: int
    inner_folds: int
    stacking_folds: int
    gate_epochs: int
    bootstrap_draws: int
    perturbation_repeats: int
    curve_repeats: int
    null_repeats: int
    reduced_grid: bool


class ResearchSample(NamedTuple):
    features: pd.DataFrame
    targets: np.ndarray
    groups: np.ndarray


class ResearchSplit(NamedTuple):
    label: str
    protocol: str
    seed: int
    grouped: bool
    train: ResearchSample
    test: ResearchSample


class ResearchFamily(NamedTuple):
    name: str
    estimator: object
    grid: dict


class ResearchFit(NamedTuple):
    name: str
    model: object
    oof: np.ndarray
    oof_fold_rmse: tuple
    nested_params: tuple
    best_params: dict
    validation_rmse: float
    candidate_count: int
    seconds: float


RESEARCH_GROUP_COLUMN = '文献序号'
RESEARCH_COVERAGE_LEVELS = (0.80, 0.90, 0.95)
RESEARCH_CALIBRATION_FAMILIES = ('Ridge', 'ExtraTrees', 'CatBoost')
RESEARCH_PROFILES = {
    'full': ResearchProfile(
        row_repeats=2,
        group_folds=3,
        inner_folds=5,
        stacking_folds=4,
        gate_epochs=600,
        bootstrap_draws=400,
        perturbation_repeats=10,
        curve_repeats=3,
        null_repeats=8,
        reduced_grid=False,
    ),
    'quick': ResearchProfile(
        row_repeats=1,
        group_folds=2,
        inner_folds=3,
        stacking_folds=3,
        gate_epochs=60,
        bootstrap_draws=80,
        perturbation_repeats=3,
        curve_repeats=1,
        null_repeats=2,
        reduced_grid=True,
    ),
}


def research_rmse(truth, prediction):
    return float(np.sqrt(np.mean(np.square(np.asarray(truth) - prediction))))


def research_pearson(x, y):
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def research_spearman(x, y):
    x_ranks = pd.Series(x).rank(method='average').values
    y_ranks = pd.Series(y).rank(method='average').values
    return research_pearson(x_ranks, y_ranks)


def research_metric_row(truth, prediction):
    errors = np.asarray(prediction) - truth
    abs_errors = np.abs(errors)
    return {
        'R2': r2_score(truth, prediction) if len(truth) > 1 and np.ptp(truth) > 0 else float('nan'),
        'RMSE': research_rmse(truth, prediction),
        'MAE': float(abs_errors.mean()),
        'MedAE': float(np.median(abs_errors)),
        'Bias': float(errors.mean()),
        'P90AbsError': float(np.quantile(abs_errors, 0.90)),
        'Spearman': research_spearman(truth, prediction),
    }


def research_table(title, rows, *, columns=None, sort_by=None, limit=None):
    if not rows:
        return
    frame = pd.DataFrame(rows, columns=columns)
    if sort_by is not None:
        frame = frame.sort_values(sort_by, kind='mergesort')
    if limit is not None:
        frame = frame.head(limit)
    print(f'\n{title}', flush=True)
    print(frame.to_string(index=False, float_format=lambda value: f'{value:.5f}', na_rep='—'), flush=True)


def research_sample(df):
    features = df.loc[:, FEATURES].apply(pd.to_numeric, errors='coerce')
    features = features.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    target = pd.to_numeric(df[TARGET], errors='coerce').values.astype(float)
    if RESEARCH_GROUP_COLUMN in df:
        raw_groups = df[RESEARCH_GROUP_COLUMN]
        groups = raw_groups.fillna('').astype(str).str.strip().values
    else:
        groups = np.full(len(df), '', dtype=object)
    valid = np.flatnonzero(np.isfinite(target))
    return ResearchSample(features.iloc[valid].reset_index(drop=True), target[valid], groups[valid])


def research_subset(sample, rows):
    return ResearchSample(
        sample.features.iloc[rows].reset_index(drop=True),
        sample.targets[rows],
        sample.groups[rows],
    )


def research_profile_audit(sample):
    known_groups = sample.groups[sample.groups != '']
    unique_groups = np.unique(known_groups)
    duplicated_inputs = sample.features.duplicated(keep=False).sum()
    combined = sample.features.assign(_target=sample.targets)
    duplicated_records = combined.duplicated(keep=False).sum()
    print(f'研究样本 {len(sample.targets)} 行 | {len(sample.features.columns)} 个特征 | '
          f'{len(unique_groups)} 个已知文献组 | 未知文献组 {len(sample.targets) - len(known_groups)} 行',
          flush=True)
    print(f'存在完全相同特征的行 {duplicated_inputs} 个，其中同特征且同目标的行 {duplicated_records} 个',
          flush=True)

    target = sample.targets
    cuts = np.quantile(target, [0.05, 0.25, 0.50, 0.75, 0.95])
    research_table('目标 ICE 的分布', [{
        'Mean': float(target.mean()),
        'SD': float(target.std(ddof=1)) if len(target) > 1 else float('nan'),
        'P05': cuts[0],
        'P25': cuts[1],
        'Median': cuts[2],
        'P75': cuts[3],
        'P95': cuts[4],
    }])

    feature_rows = []
    for feature in sample.features:
        values = sample.features[feature].values
        valid = np.isfinite(values)
        present = values[valid]
        absent = target[~valid]
        target_gap = target[valid].mean() - absent.mean() if len(present) and len(absent) else float('nan')
        feature_rows.append({
            'Feature': feature,
            'Observed': int(valid.sum()),
            'MissingPct': float((~valid).mean() * 100),
            'Distinct': len(np.unique(present)),
            'Median': float(np.median(present)) if len(present) else float('nan'),
            'SpearmanWithICE': research_spearman(present, target[valid]),
            'ICEGapPresentMinusMissing': target_gap,
        })
    research_table('特征分布、缺失率与目标关联', feature_rows)

    patterns = sample.features.isna().apply(
        lambda row: ', '.join(row.index[row]) or '完整', axis=1,
    )
    pattern_rows = []
    for pattern, positions in patterns.groupby(patterns, sort=False).groups.items():
        group_targets = target[np.asarray(positions)]
        pattern_rows.append({
            'MissingFeatures': pattern,
            'Rows': len(positions),
            'TargetMean': float(group_targets.mean()),
            'TargetMedian': float(np.median(group_targets)),
        })
    pattern_rows.sort(key=lambda row: -row['Rows'])
    research_table('最常见的联合缺失模式', pattern_rows, limit=8)

    if len(unique_groups):
        group_sizes = pd.Series(known_groups).value_counts().values
        research_table('文献组大小', [{
            'Groups': len(group_sizes),
            'Singletons': int((group_sizes == 1).sum()),
            'MedianRows': float(np.median(group_sizes)),
            'LargestRows': int(group_sizes.max()),
        }])


def research_feature_overlap(train, test):
    train_hashes = pd.util.hash_pandas_object(train.features, index=False).values
    test_hashes = pd.util.hash_pandas_object(test.features, index=False).values
    return int(np.isin(test_hashes, train_hashes).sum())


def research_split_audit(split):
    train = split.train
    test = split.test
    train_known = set(train.groups[train.groups != ''])
    test_known = set(test.groups[test.groups != ''])
    shared_groups = len(train_known & test_known)
    if split.grouped and shared_groups:
        raise ValueError(f'{split.label} 的训练集和测试集出现相同文献组')

    duplicates = research_feature_overlap(train, test)
    print(f'{split.label} | 训练 {len(train.targets)} 行 | 测试 {len(test.targets)} 行 | '
          f'重叠文献组 {shared_groups} 个 | 与训练集同特征的测试行 {duplicates} 个', flush=True)

    rows = []
    for name in train.features:
        train_column = train.features[name].dropna().values
        test_column = test.features[name].dropna().values
        shift = float('nan')
        support = float('nan')
        if len(train_column) and len(test_column):
            spread = np.std(train_column)
            if spread > 0:
                shift = float((np.mean(test_column) - np.mean(train_column)) / spread)
            support = float(np.mean((test_column < train_column.min()) | (test_column > train_column.max())))
        rows.append({
            'Feature': name,
            'TrainMissing': 1 - len(train_column) / len(train.targets),
            'TestMissing': 1 - len(test_column) / len(test.targets),
            'MeanShiftInTrainSD': shift,
            'OutsideTrainRange': support,
        })
    research_table(f'{split.label} 的训练与测试分布', rows)
    return shared_groups, duplicates


def research_outer_splits(sample, profile, completed_rounds):
    positions = np.arange(len(sample.targets))
    for repeat in range(min(profile.row_repeats, completed_rounds)):
        seed = 8 + repeat
        train_index, test_index = train_test_split(positions, test_size=0.2, random_state=seed)
        yield ResearchSplit(
            label=f'随机划分 {repeat + 1}',
            protocol='Random80/20',
            seed=seed,
            grouped=False,
            train=research_subset(sample, train_index),
            test=research_subset(sample, test_index),
        )

    known = np.flatnonzero(sample.groups != '')
    labelled = research_subset(sample, known)
    known_groups = len(np.unique(labelled.groups))
    if known_groups < 6:
        print('可用文献组少于 6 个，结束文献分组研究。', flush=True)
        return

    splitter = GroupKFold(n_splits=min(profile.group_folds, known_groups))
    folds = splitter.split(labelled.features, labelled.targets, labelled.groups)
    for number, (train_index, test_index) in enumerate(folds, 1):
        yield ResearchSplit(
            label=f'文献留出 {number}',
            protocol='LiteratureGroup',
            seed=108 + number,
            grouped=True,
            train=research_subset(labelled, train_index),
            test=research_subset(labelled, test_index),
        )


def research_cv(sample, requested_folds, seed, grouped):
    if grouped:
        fold_count = min(requested_folds, len(np.unique(sample.groups)))
        if fold_count < 2:
            return []
        splitter = GroupKFold(n_splits=fold_count)
        return list(splitter.split(sample.features, sample.targets, sample.groups))

    fold_count = min(requested_folds, len(sample.targets) // 2)
    if fold_count < 2:
        return []
    splitter = KFold(n_splits=fold_count, shuffle=True, random_state=seed)
    return list(splitter.split(sample.features, sample.targets))


def research_nested_plan(split, profile):
    train = split.train
    if len(train.targets) < 24 or len(split.test.targets) < 4:
        return []
    if not np.isfinite(train.features.values).any():
        return []

    stacking_folds = research_cv(train, profile.stacking_folds, split.seed, split.grouped)
    plan = []
    for number, (fit_index, predict_index) in enumerate(stacking_folds, 1):
        fold_train = research_subset(train, fit_index)
        inner = research_cv(fold_train, profile.inner_folds, split.seed + number, split.grouped)
        if not inner or min(len(index) for index, _ in inner) < 3:
            return []
        if any(not np.isfinite(fold_train.features.iloc[index].values).any() for index, _ in inner):
            return []
        plan.append((fold_train, predict_index, inner))

    if plan:
        covered = np.concatenate([predict_index for _, predict_index, _ in plan])
        if not np.array_equal(np.sort(covered), np.arange(len(train.targets))):
            raise ValueError(f'{split.label} 的嵌套折外预测应恰好覆盖每个训练样本一次')
    return plan


# 每个堆叠折都有自己的内层搜索，折外预测的样本不参与相应的参数选择。


def research_pipeline(estimator):
    return Pipeline([
        ('imputer', KNNImputer(n_neighbors=1)),
        ('scaler', StandardScaler()),
        ('model', estimator),
    ])


def research_families(seed):
    polynomial_ridge = Pipeline([
        ('polynomial', PolynomialFeatures(include_bias=False)),
        ('ridge', Ridge(solver='cholesky')),
    ])
    return [
        ResearchFamily(
            'Ridge',
            Ridge(solver='cholesky'),
            {'alpha': [0.05, 0.5, 5.0, 50.0]},
        ),
        ResearchFamily(
            'PolynomialRidge',
            polynomial_ridge,
            {
                'polynomial__degree': [2, 3],
                'ridge__alpha': [0.5, 5.0, 50.0],
            },
        ),
        ResearchFamily(
            'KNeighbors',
            KNeighborsRegressor(n_jobs=1, algorithm='brute'),
            {
                'n_neighbors': [3, 7, 12],
                'weights': ['uniform', 'distance'],
                'p': [1, 2],
            },
        ),
        ResearchFamily(
            'KernelRidge',
            KernelRidge(),
            {
                'kernel': ['rbf', 'laplacian'],
                'alpha': [0.03, 0.3, 3.0],
                'gamma': [0.04, 0.16, 0.5],
            },
        ),
        ResearchFamily(
            'SVR',
            SVR(kernel='rbf'),
            {
                'C': [0.3, 3.0, 30.0],
                'gamma': [0.01, 0.06, 0.2],
                'epsilon': [0.02, 0.05],
            },
        ),
        ResearchFamily(
            'RandomForest',
            RandomForestRegressor(random_state=seed, n_jobs=1),
            {
                'n_estimators': [180, 360],
                'max_depth': [8, None],
                'min_samples_leaf': [1, 3],
                'max_features': [0.7, 1.0],
            },
        ),
        ResearchFamily(
            'ExtraTrees',
            ExtraTreesRegressor(random_state=seed, n_jobs=1),
            {
                'n_estimators': [160, 320],
                'max_features': [0.7, 1.0],
                'min_samples_leaf': [1, 3],
            },
        ),
        ResearchFamily(
            'GradientBoosting',
            GradientBoostingRegressor(loss='huber', random_state=seed),
            {
                'n_estimators': [120, 240],
                'max_depth': [2, 3],
                'min_samples_leaf': [2, 5],
                'subsample': [0.75, 1.0],
            },
        ),
        ResearchFamily(
            'CatBoost',
            CatBoostRegressor(
                loss_function='RMSE',
                random_state=seed,
                thread_count=1,
                verbose=0,
                allow_writing_files=False,
            ),
            {
                'n_estimators': [300, 650],
                'depth': [4, 6],
                'learning_rate': [0.02, 0.06],
            },
        ),
    ]


def research_family_grid(family, folds, profile):
    grid = family.grid.copy()
    if 'n_neighbors' in grid:
        smallest_fit = min(len(index) for index, _ in folds)
        grid['n_neighbors'] = [value for value in grid['n_neighbors'] if value <= smallest_fit]

    if profile.reduced_grid:
        grid = {key: values[:1] for key, values in grid.items()}
        if 'n_estimators' in grid:
            grid['n_estimators'] = [30]

    return {f'model__{key}': values for key, values in grid.items()}


def research_search(family, train, folds, profile, n_jobs):
    grid = research_family_grid(family, folds, profile)
    search = GridSearchCV(
        estimator=research_pipeline(family.estimator),
        param_grid=grid,
        cv=folds,
        scoring='neg_mean_squared_error',
        n_jobs=n_jobs,
        refit=True,
        error_score='raise',
    )
    return search.fit(train.features, train.targets)


def research_params(search):
    return {key.split('__', 1)[1]: value for key, value in search.best_params_.items()}


def research_fit_family(family, split, nested_plan, refit_folds, profile, n_jobs):
    started = perf_counter()
    train = split.train
    oof = np.full(len(train.targets), np.nan)
    fold_errors = []
    nested_params = []
    candidate_count = 0

    for fold_train, prediction_index, inner_folds in nested_plan:
        search = research_search(family, fold_train, inner_folds, profile, n_jobs)
        candidate_count += len(search.cv_results_['params']) * len(inner_folds)
        fold_prediction = search.predict(train.features.iloc[prediction_index])
        oof[prediction_index] = fold_prediction
        fold_errors.append(research_rmse(train.targets[prediction_index], fold_prediction))
        nested_params.append(research_params(search))

    if not np.isfinite(oof).all():
        raise ValueError(f'{split.label} / {family.name} 的嵌套折外预测有缺失')

    final_search = research_search(family, train, refit_folds, profile, n_jobs)
    candidate_count += len(final_search.cv_results_['params']) * len(refit_folds)
    return ResearchFit(
        name=family.name,
        model=final_search.best_estimator_,
        oof=oof,
        oof_fold_rmse=tuple(fold_errors),
        nested_params=tuple(nested_params),
        best_params=research_params(final_search),
        validation_rmse=float(np.sqrt(-final_search.best_score_)),
        candidate_count=candidate_count,
        seconds=perf_counter() - started,
    )


def research_base_models(split, nested_plan, profile, n_jobs):
    refit_folds = research_cv(split.train, profile.inner_folds, split.seed + 71, split.grouped)
    fitted = []
    for family in research_families(split.seed):
        result = research_fit_family(family, split, nested_plan, refit_folds, profile, n_jobs)
        fitted.append(result)
        print(f'  {family.name:<18} | {result.candidate_count:>4} 次内层拟合 | '
              f'嵌套 OOF RMSE {research_rmse(split.train.targets, result.oof):.5f} | '
              f'{result.seconds:.1f} 秒', flush=True)
    return fitted


def research_parameter_stability(fitted, truth):
    rows = []
    for result in fitted:
        choices = Counter(tuple(sorted(params.items())) for params in result.nested_params)
        _, consensus = choices.most_common(1)[0]
        rows.append({
            'Family': result.name,
            'OOFRMSE': research_rmse(truth, result.oof),
            'FoldRMSE_SD': float(np.std(result.oof_fold_rmse, ddof=1)),
            'SelectedConfigCount': len(choices),
            'MostCommonConfigShare': consensus / len(result.nested_params),
            'RefitCV_RMSE': result.validation_rmse,
            'RefitParams': str(result.best_params),
        })
    research_table('各模型在嵌套折之间的超参数稳定性', rows, sort_by='OOFRMSE')


def research_error_diversity(fitted, truth):
    rows = []
    for left, right in combinations(fitted, 2):
        error_left = left.oof - truth
        error_right = right.oof - truth
        rows.append({
            'First': left.name,
            'Second': right.name,
            'OOFResidualCorrelation': research_pearson(error_left, error_right),
            'PredictionDisagreement': float(np.mean(np.abs(left.oof - right.oof))),
        })
    research_table('折外误差相关性最低的模型组合', rows, sort_by='OOFResidualCorrelation', limit=8)


def research_simplex_projection(values):
    descending = np.sort(values)[::-1]
    cumulative = np.cumsum(descending) - 1.0
    positions = np.arange(1, len(values) + 1)
    active = np.flatnonzero(descending > cumulative / positions)
    threshold = cumulative[active[-1]] / positions[active[-1]]
    projected = np.maximum(values - threshold, 0.0)
    return projected / projected.sum()


def research_simplex_weights(oof, truth, ridge_strength):
    member_count = oof.shape[1]
    equal = np.full(member_count, 1.0 / member_count)
    centered = oof - oof.mean(axis=1, keepdims=True)
    target = truth - oof.mean(axis=1)
    gram = centered.T.dot(centered) / len(truth)
    cross = centered.T.dot(target) / len(truth)
    # 凸组合权重和为 1，因此按行中心化保留优化目标，并改善条件数。
    lipschitz = float(np.linalg.eigvalsh(gram)[-1]) + ridge_strength
    weights = equal.copy()
    for _ in range(1600):
        gradient = gram.dot(weights) - cross + ridge_strength * (weights - equal)
        updated = research_simplex_projection(weights - gradient / lipschitz)
        if np.max(np.abs(updated - weights)) < 1e-9:
            return updated
        weights = updated
    return weights


class ResearchContextGate(nn.Module):
    def __init__(self, in_features, members):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, 48),
            nn.LayerNorm(48),
            nn.SiLU(),
            nn.Dropout(0.12),
            nn.Linear(48, 24),
            nn.SiLU(),
            nn.Linear(24, members),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, context, predictions):
        weights = torch.softmax(self.network(context), dim=1)
        return (weights * predictions).sum(dim=1), weights


class ResearchGateFit:
    def __init__(self, features, oof, truth, seed, epochs):
        self.feature_transform = Pipeline([
            ('imputer', KNNImputer(n_neighbors=1)),
            ('scaler', StandardScaler()),
        ]).fit(features)
        self.prediction_scaler = StandardScaler().fit(oof)
        context = self.context_tensor(features, oof)
        predictions = torch.tensor(oof, dtype=torch.float32)
        targets = torch.tensor(truth, dtype=torch.float32)

        # 保存外部的随机状态，让该研究网络的训练独立于此前的正式评估。
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.model = ResearchContextGate(context.shape[1], oof.shape[1])
            self.train_network(context, predictions, targets, epochs)
        self.model.eval()

    def context_tensor(self, features, base_predictions):
        transformed = self.feature_transform.transform(features)
        missingness = features.isna().values.astype(np.float32)
        scaled_predictions = self.prediction_scaler.transform(base_predictions)
        context = np.column_stack([transformed, missingness, scaled_predictions])
        return torch.tensor(context, dtype=torch.float32)

    def train_network(self, context, predictions, targets, epochs):
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=2e-4)
        batch_size = min(32, len(targets))
        for _ in range(epochs):
            indices = torch.randperm(len(targets))
            for batch in indices.split(batch_size):
                optimizer.zero_grad()
                prediction, weights = self.model(context[batch], predictions[batch])
                fit_loss = nn.functional.smooth_l1_loss(prediction, targets[batch], beta=0.08)
                # 对趋向单模型的权重加轻微约束，避免小样本上过早塌缩。
                concentration = torch.square(weights).sum(dim=1).mean()
                loss = fit_loss + 0.002 * concentration
                loss.backward()
                optimizer.step()

    def predict(self, features, base_predictions):
        context = self.context_tensor(features, base_predictions)
        values = torch.tensor(base_predictions, dtype=torch.float32)
        with torch.no_grad():
            prediction, weights = self.model(context, values)
        return prediction.numpy().astype(float), weights.numpy()


class ResearchEnsemble:
    def __init__(self, fitted, split, profile):
        self.fitted = fitted
        self.names = [member.name for member in fitted]
        self.oof = np.column_stack([member.oof for member in fitted])
        truth = split.train.targets
        oof_errors = np.sqrt(np.mean(np.square(self.oof - truth[:, None]), axis=0))
        inverse = 1.0 / np.maximum(oof_errors, 1e-8)
        self.inverse_weights = inverse / inverse.sum()
        penalty = max(float(np.var(truth)) * 0.025, 1e-8)
        self.simplex_weights = research_simplex_weights(self.oof, truth, penalty)

        folds = research_cv(split.train, profile.inner_folds, split.seed + 117, split.grouped)
        ridge_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(solver='cholesky')),
        ])
        self.ridge = GridSearchCV(
            ridge_pipeline,
            {'ridge__alpha': [0.1, 1.0, 10.0, 50.0, 100.0]},
            cv=folds,
            scoring='neg_mean_squared_error',
            error_score='raise',
        ).fit(self.oof, truth)

        simplex_oof = self.oof.dot(self.simplex_weights)
        self.calibrator = IsotonicRegression(out_of_bounds='clip').fit(simplex_oof, truth)
        self.gate = ResearchGateFit(
            split.train.features,
            self.oof,
            truth,
            split.seed + 233,
            profile.gate_epochs,
        )

    def predict_base(self, features):
        return np.column_stack([member.model.predict(features) for member in self.fitted])

    def predict_all(self, features, base):
        simplex = base.dot(self.simplex_weights)
        gate, _ = self.gate.predict(features, base)
        predictions = {name: base[:, i] for i, name in enumerate(self.names)}
        predictions.update({
            'EqualBlend': base.mean(axis=1),
            'InverseErrorBlend': base.dot(self.inverse_weights),
            'SimplexBlend': simplex,
            'RidgeStack': self.ridge.predict(base),
            'IsotonicBlend': self.calibrator.predict(simplex),
            'ContextGate': gate,
        })
        return predictions

    def predict_focus(self, features):
        base = self.predict_base(features)
        return self.gate.predict(features, base)[0]

    def weight_diagnostics(self, features, base):
        _, gate_weights = self.gate.predict(features, base)
        rows = []
        for index, name in enumerate(self.names):
            column = gate_weights[:, index]
            rows.append({
                'Member': name,
                'InverseError': self.inverse_weights[index],
                'Simplex': self.simplex_weights[index],
                'GateMean': float(column.mean()),
                'GateSD': float(column.std()),
                'GateP10': float(np.quantile(column, 0.10)),
                'GateP90': float(np.quantile(column, 0.90)),
                'TopGateShare': float(np.mean(np.argmax(gate_weights, axis=1) == index)),
            })
        research_table('集成权重与上下文门控的测试样本权重分布', rows)
        ridge_alpha = self.ridge.best_params_['ridge__alpha']
        print(f'岭堆叠在训练侧选择的 alpha 为 {ridge_alpha:g}', flush=True)


def research_eval_predictions(split, predictions):
    rows = []
    for name, prediction in predictions.items():
        if prediction.shape != split.test.targets.shape or not np.isfinite(prediction).all():
            raise ValueError(f'{split.label} / {name} 的测试预测长度或数值无效')
        metrics = research_metric_row(split.test.targets, prediction)
        rows.append({'Split': split.label, 'Protocol': split.protocol, 'Model': name, **metrics})
    research_table(
        '同一外层测试集的研究模型比较', rows,
        columns=['Model', 'R2', 'RMSE', 'MAE', 'MedAE', 'Bias', 'P90AbsError', 'Spearman'],
        sort_by='RMSE',
    )
    return rows


def research_error_slice(label, truth, prediction, selector):
    actual = truth[selector]
    fitted = prediction[selector]
    errors = fitted - actual
    return {
        'Slice': label,
        'Rows': len(actual),
        'TargetMean': float(actual.mean()),
        'RMSE': research_rmse(actual, fitted),
        'MAE': float(np.abs(errors).mean()),
        'Bias': float(errors.mean()),
    }


def research_quantile_slices(split, prediction):
    train_target = split.train.targets
    test_target = split.test.targets
    edges = np.unique(np.quantile(train_target, [0.25, 0.50, 0.75]))
    membership = np.searchsorted(edges, test_target, side='right')
    labels = np.concatenate([[-np.inf], edges, [np.inf]])
    rows = []
    for index in range(len(labels) - 1):
        selector = membership == index
        if selector.any():
            left = labels[index]
            right = labels[index + 1]
            name = f'[{left:.4f}, {right:.4f})'
            rows.append(research_error_slice(name, test_target, prediction, selector))
    research_table('按训练目标分位数定义的测试误差切片', rows)


def research_input_slices(split, prediction):
    test = split.test
    missing_counts = test.features.isna().sum(axis=1).values
    rows = []
    for label, selector in [
        ('没有缺失特征', missing_counts == 0),
        ('缺失 1 个特征', missing_counts == 1),
        ('缺失 2 个及以上特征', missing_counts >= 2),
    ]:
        if selector.any():
            rows.append(research_error_slice(label, test.targets, prediction, selector))
    research_table('按缺失特征数量划分的误差', rows)

    train_min = split.train.features.min(axis=0)
    train_max = split.train.features.max(axis=0)
    outside = (test.features.lt(train_min, axis=1) | test.features.gt(train_max, axis=1)).any(axis=1).values
    comparable = test.features.notna().any(axis=1).values
    range_rows = []
    for label, selector in [
        ('所有已观测特征都在训练范围内', comparable & ~outside),
        ('至少一个特征超出训练范围', outside),
        ('全部特征缺失', ~comparable),
    ]:
        if selector.any():
            range_rows.append(research_error_slice(label, test.targets, prediction, selector))
    research_table('训练支持范围内外的测试误差', range_rows)


def research_group_error(split, prediction):
    sample = split.test
    known_groups = np.unique(sample.groups[sample.groups != ''])
    if not len(known_groups):
        return
    rows = []
    for group in known_groups:
        selector = sample.groups == group
        actual = sample.targets[selector]
        fitted = prediction[selector]
        rows.append({
            'Literature': group,
            'Rows': len(actual),
            'RMSE': research_rmse(actual, fitted),
            'MAE': float(np.mean(np.abs(fitted - actual))),
            'Bias': float(np.mean(fitted - actual)),
        })
    known_rows = sample.groups != ''
    group_mae = float(np.mean([row['MAE'] for row in rows]))
    sample_mae = float(np.mean(np.abs(prediction[known_rows] - sample.targets[known_rows])))
    print(f'已知文献组等权 MAE {group_mae:.5f} | 对应样本等权 MAE {sample_mae:.5f}', flush=True)
    rows.sort(key=lambda row: -row['RMSE'])
    research_table('误差较大的测试文献组', rows, limit=8)


def research_calibration_bins(split, prediction):
    values = np.unique(np.quantile(prediction, [0.2, 0.4, 0.6, 0.8]))
    bins = np.searchsorted(values, prediction, side='right')
    rows = []
    weighted_gap = 0.0
    for group in np.unique(bins):
        selected = bins == group
        expected = prediction[selected].mean()
        observed = split.test.targets[selected].mean()
        count = int(selected.sum())
        weighted_gap += count * abs(expected - observed)
        rows.append({
            'PredictionBin': int(group) + 1,
            'Rows': count,
            'PredictedMean': float(expected),
            'ObservedMean': float(observed),
            'MeanGap': float(expected - observed),
        })
    research_table('上下文门控的预测分箱校准', rows)
    print(f'分箱加权平均绝对校准差 {weighted_gap / len(prediction):.5f}', flush=True)


def research_bootstrap_indices(sample, grouped, draws, seed):
    rng = np.random.RandomState(seed)
    if grouped:
        groups = np.unique(sample.groups)
        blocks = [np.flatnonzero(sample.groups == group) for group in groups]
        for _ in range(draws):
            selected = rng.randint(len(blocks), size=len(blocks))
            yield np.concatenate([blocks[index] for index in selected])
    else:
        size = len(sample.targets)
        for _ in range(draws):
            yield rng.randint(size, size=size)


def research_bootstrap(split, predictions, ensemble, profile):
    selected_member = min(ensemble.fitted, key=lambda fit: research_rmse(split.train.targets, fit.oof))
    contrasts = [
        ('ContextGate', 'EqualBlend'),
        ('SimplexBlend', 'EqualBlend'),
        ('ContextGate', selected_member.name),
    ]
    positions = list(research_bootstrap_indices(
        split.test, split.grouped, profile.bootstrap_draws, split.seed + 359,
    ))
    rows = []
    for method, reference in contrasts:
        method_errors = predictions[method] - split.test.targets
        reference_errors = predictions[reference] - split.test.targets
        deltas = {'RMSE': [], 'MAE': []}
        for selected in positions:
            first = method_errors[selected]
            second = reference_errors[selected]
            deltas['RMSE'].append(np.sqrt(np.mean(first ** 2)) - np.sqrt(np.mean(second ** 2)))
            deltas['MAE'].append(np.mean(np.abs(first)) - np.mean(np.abs(second)))

        point_estimates = {
            'RMSE': np.sqrt(np.mean(method_errors ** 2)) - np.sqrt(np.mean(reference_errors ** 2)),
            'MAE': np.mean(np.abs(method_errors)) - np.mean(np.abs(reference_errors)),
        }
        for metric, samples in deltas.items():
            lower, upper = np.quantile(samples, [0.025, 0.975])
            rows.append({
                'Model': method,
                'Reference': reference,
                'Metric': metric,
                'ObservedDelta': float(point_estimates[metric]),
                'P025': lower,
                'P975': upper,
                'LowerErrorShare': float(np.mean(np.asarray(samples) < 0)),
            })
    unit = '文献组' if split.grouped else '测试样本'
    research_table(f'单次测试划分的配对 Bootstrap 差值，抽样单位为{unit}', rows)


def research_error_changes(truth, predictions, baseline_rmse, baseline_mae):
    rmse = np.asarray([research_rmse(truth, prediction) for prediction in predictions])
    mae = np.asarray([np.mean(np.abs(prediction - truth)) for prediction in predictions])
    delta = rmse - baseline_rmse
    return {
        'RMSE': float(rmse.mean()),
        'RMSEDelta': float(delta.mean()),
        'RMSEDeltaP10': float(np.quantile(delta, 0.10)),
        'RMSEDeltaP90': float(np.quantile(delta, 0.90)),
        'MAEDelta': float(mae.mean() - baseline_mae),
    }


def research_random_masking(split, ensemble, prediction, profile):
    test = split.test
    baseline_rmse = research_rmse(test.targets, prediction)
    baseline_mae = float(np.mean(np.abs(prediction - test.targets)))
    rng = np.random.RandomState(split.seed + 401)
    observed = test.features.notna().values
    # 每次重复用同一张随机矩阵生成各缺失强度，便于逐级比较。
    masks = [rng.rand(*observed.shape) for _ in range(profile.perturbation_repeats)]
    rows = []
    for fraction in (0.10, 0.25, 0.40):
        predictions = []
        removed = []
        for mask in masks:
            selected = (mask < fraction) & observed
            altered = test.features.mask(selected)
            predictions.append(ensemble.predict_focus(altered))
            removed.append(int(selected.sum()))
        rows.append({
            'MaskFraction': fraction,
            'MeanCellsRemoved': float(np.mean(removed)),
            **research_error_changes(test.targets, predictions, baseline_rmse, baseline_mae),
        })
    research_table('额外随机缺失的鲁棒性，仅扰动测试输入', rows)


def research_feature_perturbation(split, ensemble, prediction, profile):
    test = split.test
    rng = np.random.RandomState(split.seed + 479)
    baseline_rmse = research_rmse(test.targets, prediction)
    baseline_mae = float(np.mean(np.abs(prediction - test.targets)))
    permutation_rows = []
    removal_rows = []
    for feature in test.features:
        column = test.features[feature].values
        observed = np.flatnonzero(np.isfinite(column))
        if not len(observed):
            continue

        fully_missing = test.features.copy()
        fully_missing[feature] = np.nan
        missing_prediction = ensemble.predict_focus(fully_missing)
        removal_rows.append({
            'Feature': feature,
            'MaskedRows': len(observed),
            **research_error_changes(test.targets, [missing_prediction], baseline_rmse, baseline_mae),
        })

        if len(observed) < 2 or np.ptp(column[observed]) == 0:
            continue
        predictions = []
        for _ in range(profile.perturbation_repeats):
            altered = test.features.copy()
            reordered = column.copy()
            reordered[observed] = rng.permutation(column[observed])
            altered[feature] = reordered
            predictions.append(ensemble.predict_focus(altered))
        permutation_rows.append({
            'Feature': feature,
            'ShuffledRows': len(observed),
            **research_error_changes(test.targets, predictions, baseline_rmse, baseline_mae),
        })

    removal_rows.sort(key=lambda row: -row['RMSEDelta'])
    permutation_rows.sort(key=lambda row: -row['RMSEDelta'])
    research_table('单个输入全列缺失时的误差变化', removal_rows)
    research_table('保持原缺失位置的单特征置换敏感性', permutation_rows)


class ResearchCalibrationSplit(NamedTuple):
    proper: ResearchSample
    calibration: ResearchSample
    folds: list


def research_calibration_split(split, profile):
    train = split.train
    seed = split.seed + 563
    if split.grouped:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        fit_index, cal_index = next(splitter.split(train.features, train.targets, train.groups))
    else:
        fit_index, cal_index = train_test_split(
            np.arange(len(train.targets)), test_size=0.25, random_state=seed,
        )
    proper = research_subset(train, fit_index)
    calibration = research_subset(train, cal_index)
    folds = research_cv(proper, profile.inner_folds, seed + 1, split.grouped)
    if not folds or min(len(index) for index, _ in folds) < 3:
        return None
    if any(not np.isfinite(proper.features.iloc[index].values).any() for index, _ in folds):
        return None
    if split.grouped and set(proper.groups) & set(calibration.groups):
        raise ValueError(f'{split.label} 的区间拟合与校准文献组发生重叠')
    return ResearchCalibrationSplit(proper, calibration, folds)


def research_interval_models(split, calibration_split, profile, n_jobs):
    proper = calibration_split.proper
    calibration = calibration_split.calibration
    calibration_columns = []
    test_columns = []
    for family in research_families(split.seed + 601):
        if family.name not in RESEARCH_CALIBRATION_FAMILIES:
            continue
        model = research_search(family, proper, calibration_split.folds, profile, n_jobs)
        calibration_columns.append(model.predict(calibration.features))
        test_columns.append(model.predict(split.test.features))
    return np.column_stack(calibration_columns), np.column_stack(test_columns)


def research_group_max_scores(scores, groups):
    return np.asarray([scores[groups == group].max() for group in np.unique(groups)])


def research_conformal_cutoff(scores, coverage):
    # 分离校准的有限样本次序统计量。需要的秩超出校准集时，区间为无穷宽。
    rank = int(math.ceil((len(scores) + 1) * coverage))
    if rank > len(scores):
        return float('inf')
    return float(np.partition(scores, rank - 1)[rank - 1])


def research_interval_report(split, center, half_width, level):
    test = split.test
    residual = np.abs(test.targets - center)
    covered = residual <= half_width
    excess = np.maximum(residual - half_width, 0.0)
    interval_score = 2 * half_width + (2.0 / (1 - level)) * excess
    group_coverage = float('nan')
    if split.grouped:
        group_coverage = float(np.mean([
            covered[test.groups == group].all() for group in np.unique(test.groups)
        ]))
    return {
        'RowCoverage': float(covered.mean()),
        'AllRowsGroupCoverage': group_coverage,
        'MeanWidth': float(np.mean(2 * half_width)),
        'MedianWidth': float(np.median(2 * half_width)),
        'MeanIntervalScore': float(interval_score.mean()),
    }


def research_interval_slices(split, method, center, half_width):
    mask_counts = split.test.features.isna().sum(axis=1).values
    errors = np.abs(split.test.targets - center)
    rows = []
    for label, selected in [
        ('没有缺失特征', mask_counts == 0),
        ('存在缺失特征', mask_counts > 0),
    ]:
        if selected.any():
            rows.append({
                'Method': method,
                'Slice': label,
                'Rows': int(selected.sum()),
                'Coverage90': float(np.mean(errors[selected] <= half_width[selected])),
                'MeanWidth90': float(np.mean(2 * half_width[selected])),
            })
    return rows


def research_prediction_intervals(split, profile, n_jobs):
    calibration_split = research_calibration_split(split, profile)
    if calibration_split is None:
        print('独立校准划分的可用训练样本不足，本划分的预测区间研究结束。', flush=True)
        return []

    calibration, test = research_interval_models(split, calibration_split, profile, n_jobs)
    calibration_center = calibration.mean(axis=1)
    test_center = test.mean(axis=1)
    calibration_residual = np.abs(calibration_split.calibration.targets - calibration_center)
    floor = max(0.1 * float(np.std(calibration_split.proper.targets)), 1e-6)
    calibration_scale = np.maximum(calibration.std(axis=1), floor)
    test_scale = np.maximum(test.std(axis=1), floor)
    configurations = [
        ('Absolute', calibration_residual, np.ones(len(test_center))),
        ('DisagreementScaled', calibration_residual / calibration_scale, test_scale),
    ]

    rows = []
    sliced = []
    for method, scores, scales in configurations:
        if split.grouped:
            scores = research_group_max_scores(scores, calibration_split.calibration.groups)
        for level in RESEARCH_COVERAGE_LEVELS:
            cutoff = research_conformal_cutoff(scores, level)
            half_width = cutoff * scales
            rows.append({
                'Split': split.label,
                'Protocol': split.protocol,
                'Method': method,
                'TargetCoverage': level,
                'CalibrationUnits': len(scores),
                'FiniteWidth': bool(np.isfinite(cutoff)),
                **research_interval_report(split, test_center, half_width, level),
            })
            if level == 0.90 and np.isfinite(cutoff):
                sliced.extend(research_interval_slices(split, method, test_center, half_width))

    unit = '文献组最大误差' if split.grouped else '单样本误差'
    print(f'分离校准 | 拟合 {len(calibration_split.proper.targets)} 行 | '
          f'校准 {len(calibration_split.calibration.targets)} 行 | 校准单位为{unit}', flush=True)
    research_table('独立校准的测试预测区间', rows, columns=[
        'Method', 'TargetCoverage', 'CalibrationUnits', 'FiniteWidth', 'RowCoverage',
        'AllRowsGroupCoverage', 'MeanWidth', 'MedianWidth', 'MeanIntervalScore',
    ])
    research_table('90% 预测区间在缺失输入子集上的覆盖率', sliced)
    return rows


def research_weight_ablation(split, ensemble, base_predictions):
    truth = split.train.targets
    holdout_truth = split.test.targets
    reference = base_predictions.dot(ensemble.simplex_weights)
    reference_rmse = research_rmse(holdout_truth, reference)
    ridge_strength = max(float(np.var(truth)) * 0.025, 1e-8)
    rows = []
    for index, name in enumerate(ensemble.names):
        included = np.arange(len(ensemble.names)) != index
        weights = research_simplex_weights(ensemble.oof[:, included], truth, ridge_strength)
        prediction = base_predictions[:, included].dot(weights)
        error = research_rmse(holdout_truth, prediction)
        rows.append({
            'Omitted': name,
            'OriginalWeight': ensemble.simplex_weights[index],
            'AblatedOOFRMSE': research_rmse(truth, ensemble.oof[:, included].dot(weights)),
            'AblatedTestRMSE': error,
            'TestRMSEDelta': error - reference_rmse,
            'RemainingActiveWeights': int((weights > 1e-5).sum()),
        })
    rows.sort(key=lambda row: -row['TestRMSEDelta'])
    research_table('逐个移除基模型并在训练折外预测上重新拟合权重', rows)


def research_curve_models(seed, reduced):
    # 学习曲线各尺寸都用预先固定的参数，以免用到该尺寸未纳入的训练样本。
    tree_count = 40 if reduced else 180
    return [
        ('Ridge', Ridge(alpha=1.0, solver='cholesky')),
        ('ExtraTrees', ExtraTreesRegressor(
            n_estimators=tree_count,
            max_features=0.8,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=1,
        )),
        ('CatBoost', CatBoostRegressor(
            n_estimators=100 if reduced else 420,
            depth=5,
            learning_rate=0.035,
            loss_function='RMSE',
            random_state=seed,
            thread_count=1,
            verbose=0,
            allow_writing_files=False,
        )),
    ]


def research_curve_subsets(sample, fractions, grouped, seed):
    rng = np.random.RandomState(seed)
    if grouped:
        ordered = rng.permutation(np.unique(sample.groups))
    else:
        ordered = rng.permutation(len(sample.targets))
    for fraction in fractions:
        count = min(len(ordered), max(3, int(math.ceil(fraction * len(ordered)))))
        chosen = ordered[:count]
        if grouped:
            rows = np.flatnonzero(np.isin(sample.groups, chosen))
        else:
            rows = np.sort(chosen)
        yield fraction, rows


def research_curve_summary(results):
    if not results:
        return
    table = pd.DataFrame(results)
    summary = []
    for (name, fraction), rows in table.groupby(['Model', 'Fraction'], sort=False):
        summary.append({
            'Model': name,
            'Fraction': fraction,
            'Repeats': len(rows),
            'MedianRows': rows['Rows'].median(),
            'MeanTrainRMSE': rows['TrainRMSE'].mean(),
            'MeanTestRMSE': rows['TestRMSE'].mean(),
            'TestRMSE_SD': rows['TestRMSE'].std(),
            'MeanGeneralizationGap': (rows['TestRMSE'] - rows['TrainRMSE']).mean(),
            'MeanFitSeconds': rows['FitSeconds'].mean(),
        })
    research_table('固定外层测试集上的多训练样本量学习曲线', summary, sort_by=['Model', 'Fraction'])


def research_learning_curve(split, profile):
    results = []
    sample = split.train
    for repeat in range(profile.curve_repeats):
        seed = split.seed + 661 + repeat
        selections = research_curve_subsets(sample, (0.30, 0.60, 1.00), split.grouped, seed)
        for fraction, indices in selections:
            subset = research_subset(sample, indices)
            if not np.isfinite(subset.features.values).any() or np.ptp(subset.targets) == 0:
                continue
            for name, model in research_curve_models(seed, profile.reduced_grid):
                started = perf_counter()
                pipeline = research_pipeline(model).fit(subset.features, subset.targets)
                elapsed = perf_counter() - started
                train_prediction = pipeline.predict(subset.features)
                test_prediction = pipeline.predict(split.test.features)
                results.append({
                    'Model': name,
                    'Fraction': fraction,
                    'Rows': len(indices),
                    'TrainRMSE': research_rmse(subset.targets, train_prediction),
                    'TestRMSE': research_rmse(split.test.targets, test_prediction),
                    'FitSeconds': elapsed,
                })
    research_curve_summary(results)


def research_target_shuffle(split, profile):
    # 随机划分上的目标置乱作为实现检查。测试标签始终固定。
    train = split.train
    test = split.test
    rng = np.random.RandomState(split.seed + 739)
    permutations = [rng.permutation(train.targets) for _ in range(profile.null_repeats)]
    rows = []
    for name, estimator in research_curve_models(split.seed, profile.reduced_grid):
        real = research_pipeline(clone(estimator)).fit(train.features, train.targets)
        observed_rmse = research_rmse(test.targets, real.predict(test.features))
        permuted_rmse = []
        for targets in permutations:
            shuffled = research_pipeline(clone(estimator)).fit(train.features, targets)
            permuted_rmse.append(research_rmse(test.targets, shuffled.predict(test.features)))
        low, median, high = np.quantile(permuted_rmse, [0.10, 0.50, 0.90])
        rows.append({
            'Model': name,
            'ObservedRMSE': observed_rmse,
            'ShuffledP10': low,
            'ShuffledMedian': median,
            'ShuffledP90': high,
            'ShuffledMedianDelta': median - observed_rmse,
            'Shuffles': len(permuted_rmse),
        })
    research_table('训练标签置乱的对照分布', rows)


def research_feature_scale(training):
    quartiles = training.quantile([0.25, 0.75])
    spread = quartiles.loc[0.75] - quartiles.loc[0.25]
    fallback = training.std(ddof=0)
    scale = spread.where(spread > 0, fallback)
    return scale.where(np.isfinite(scale) & (scale > 0), 1.0)


def research_analog_distances(split):
    train = split.train.features
    test = split.test.features
    scale = research_feature_scale(train).values
    train_values = train.values / scale
    test_values = test.values / scale
    shared = np.isfinite(test_values[:, None, :]) & np.isfinite(train_values[None, :, :])
    differences = test_values[:, None, :] - train_values[None, :, :]
    shared_counts = shared.sum(axis=2)
    # 在共同观测维度上计算均方距离。过少的共同观测不参与近邻比较。
    squared = np.where(shared, np.square(differences), 0.0).sum(axis=2)
    distances = np.full(shared_counts.shape, np.inf)
    eligible = shared_counts >= min(3, train.shape[1])
    np.divide(squared, shared_counts, out=distances, where=eligible)
    nearest = np.argmin(distances, axis=1)
    rows = np.arange(len(test))
    return np.sqrt(distances[rows, nearest]), shared_counts[rows, nearest]


def research_analog_error(split, prediction):
    distances, shared_counts = research_analog_distances(split)
    eligible = np.isfinite(distances)
    error = np.abs(prediction - split.test.targets)
    association = research_spearman(distances[eligible], error[eligible])
    print(f'可找到训练近邻的测试行 {eligible.sum()} / {len(eligible)} | '
          f'近邻距离与绝对误差的 Spearman {association:.5f}', flush=True)

    rows = []
    if eligible.any():
        cuts = np.unique(np.quantile(distances[eligible], [1 / 3, 2 / 3]))
        memberships = np.searchsorted(cuts, distances, side='right')
        for bucket in np.unique(memberships[eligible]):
            selected = eligible & (memberships == bucket)
            row = research_error_slice(f'近邻距离组 {bucket + 1}', split.test.targets, prediction, selected)
            row['MedianDistance'] = float(np.median(distances[selected]))
            row['MeanSharedFeatures'] = float(np.mean(shared_counts[selected]))
            rows.append(row)
    if (~eligible).any():
        rows.append(research_error_slice('共同观测特征不足', split.test.targets, prediction, ~eligible))
    research_table('与训练样本相似程度对应的误差', rows)


def research_local_response(split, ensemble):
    train = split.train.features
    test = split.test.features
    scales = research_feature_scale(train)
    rows = []
    for feature in train:
        column = test[feature].values
        lower_bound = train[feature].min()
        upper_bound = train[feature].max()
        if not np.isfinite(lower_bound) or not np.isfinite(upper_bound) or lower_bound == upper_bound:
            continue
        step = 0.10 * scales[feature]
        selected = np.isfinite(column) & (column >= lower_bound + step) & (column <= upper_bound - step)
        if not selected.any():
            continue
        below = test.loc[selected].copy()
        above = test.loc[selected].copy()
        below[feature] -= step
        above[feature] += step
        response = ensemble.predict_focus(above) - ensemble.predict_focus(below)
        normalized = response / (2 * step) * scales[feature]
        rows.append({
            'Feature': feature,
            'Rows': int(selected.sum()),
            'Step': float(step),
            'MedianResponsePerScale': float(np.median(normalized)),
            'MedianAbsoluteResponse': float(np.median(np.abs(response))),
            'P90AbsoluteResponse': float(np.quantile(np.abs(response), 0.90)),
            'PositiveResponseShare': float(np.mean(response > 0)),
        })
    rows.sort(key=lambda row: -row['MedianAbsoluteResponse'])
    research_table('训练特征范围内的双侧小扰动响应', rows)


def research_baseline_predictions(split):
    size = len(split.test.targets)
    return {
        'TrainingMean': np.full(size, split.train.targets.mean()),
        'TrainingMedian': np.full(size, np.median(split.train.targets)),
    }


def research_gate_regimes(split, ensemble, base):
    _, weights = ensemble.gate.predict(split.test.features, base)
    uncertainty = -np.sum(weights * np.log(np.maximum(weights, 1e-12)), axis=1)
    effective_members = np.exp(uncertainty)
    raw_disagreement = np.std(base, axis=1)
    missing = split.test.features.isna().sum(axis=1).values
    rows = []
    for label, selected in [
        ('无特征缺失', missing == 0),
        ('有特征缺失', missing > 0),
        ('基模型分歧低于中位数', raw_disagreement < np.median(raw_disagreement)),
        ('基模型分歧达到中位数', raw_disagreement >= np.median(raw_disagreement)),
    ]:
        if selected.any():
            rows.append({
                'Slice': label,
                'Rows': int(selected.sum()),
                'MeanEffectiveMembers': float(np.mean(effective_members[selected])),
                'MeanMaxWeight': float(np.mean(weights[selected].max(axis=1))),
                'MeanBaseDisagreement': float(np.mean(raw_disagreement[selected])),
            })
    research_table('门控在不同输入和预测分歧下的有效模型数', rows)


class ResearchSplitReport(NamedTuple):
    split: ResearchSplit
    metrics: list
    intervals: list
    predictions: dict
    fit_count: int
    seconds: float


def research_run_split(split, nested_plan, profile, n_jobs):
    started = perf_counter()
    print(f'\n========== 研究评估 · {split.label} ==========', flush=True)
    research_split_audit(split)
    print(f'内层搜索嵌套在 {len(nested_plan)} 个堆叠折中。', flush=True)

    fitted = research_base_models(split, nested_plan, profile, n_jobs)
    research_parameter_stability(fitted, split.train.targets)
    research_error_diversity(fitted, split.train.targets)

    ensemble = ResearchEnsemble(fitted, split, profile)
    base = ensemble.predict_base(split.test.features)
    predictions = research_baseline_predictions(split)
    predictions.update(ensemble.predict_all(split.test.features, base))
    metrics = research_eval_predictions(split, predictions)
    ensemble.weight_diagnostics(split.test.features, base)
    research_gate_regimes(split, ensemble, base)
    research_weight_ablation(split, ensemble, base)

    focus = predictions['ContextGate']
    research_quantile_slices(split, focus)
    research_input_slices(split, focus)
    research_group_error(split, focus)
    research_calibration_bins(split, focus)
    research_analog_error(split, focus)
    research_bootstrap(split, predictions, ensemble, profile)
    research_random_masking(split, ensemble, focus, profile)
    research_feature_perturbation(split, ensemble, focus, profile)
    research_local_response(split, ensemble)

    intervals = research_prediction_intervals(split, profile, n_jobs)
    research_learning_curve(split, profile)
    if not split.grouped:
        research_target_shuffle(split, profile)

    duration = perf_counter() - started
    print(f'{split.label} 的全部研究分析用时 {duration / 60:.2f} 分钟', flush=True)
    return ResearchSplitReport(
        split=split,
        metrics=metrics,
        intervals=intervals,
        predictions=predictions,
        fit_count=sum(member.candidate_count for member in fitted),
        seconds=duration,
    )


def research_protocol_summary(reports):
    scores = pd.DataFrame([row for report in reports for row in report.metrics])
    summaries = []
    for (protocol, model), rows in scores.groupby(['Protocol', 'Model'], sort=False):
        summaries.append({
            'Protocol': protocol,
            'Model': model,
            'TestSplits': len(rows),
            'RMSE_Mean': rows['RMSE'].mean(),
            'RMSE_SD': rows['RMSE'].std(),
            'RMSE_Worst': rows['RMSE'].max(),
            'MAE_Mean': rows['MAE'].mean(),
            'R2_Mean': rows['R2'].mean(),
            'Bias_Mean': rows['Bias'].mean(),
            'RankCorrelationMean': rows['Spearman'].mean(),
        })
    research_table('研究模型的跨划分描述性汇总', summaries, sort_by=['Protocol', 'RMSE_Mean'])


def research_pooled_groups(reports):
    grouped_reports = [report for report in reports if report.split.grouped]
    if not grouped_reports:
        return
    groups = np.concatenate([report.split.test.groups for report in grouped_reports])
    group_splits = np.concatenate([
        np.full(len(report.split.test.targets), report.split.label) for report in grouped_reports
    ])
    group_ids = np.unique(groups)
    blocks = [np.flatnonzero(groups == group) for group in group_ids]
    if any(len(np.unique(group_splits[index])) != 1 for index in blocks):
        raise ValueError('跨折文献汇总要求各文献组只出现在一个测试折')

    truth = np.concatenate([report.split.test.targets for report in grouped_reports])
    rows = []
    for model in grouped_reports[0].predictions:
        prediction = np.concatenate([report.predictions[model] for report in grouped_reports])
        errors = np.abs(truth - prediction)
        group_errors = np.asarray([np.mean(errors[index]) for index in blocks])
        rows.append({
            'Model': model,
            'UnseenGroups': len(group_ids),
            'TestRows': len(truth),
            'PooledRMSE': research_rmse(truth, prediction),
            'MicroMAE': float(errors.mean()),
            'MacroMAE': float(group_errors.mean()),
            'P90GroupMAE': float(np.quantile(group_errors, 0.90)),
        })
    research_table('合并文献留出预测的样本级和文献级误差', rows, sort_by='MacroMAE')


def research_interval_summary(reports):
    intervals = [row for report in reports for row in report.intervals]
    if not intervals:
        return
    data = pd.DataFrame(intervals)
    rows = []
    keys = ['Protocol', 'Method', 'TargetCoverage']
    for (protocol, method, level), subset in data.groupby(keys, sort=False):
        finite = subset.loc[subset['FiniteWidth']]
        rows.append({
            'Protocol': protocol,
            'Method': method,
            'TargetCoverage': level,
            'TotalSplits': len(subset),
            'FiniteSplits': len(finite),
            'MeanFiniteRowCoverage': finite['RowCoverage'].mean(),
            'MeanFiniteGroupCoverage': finite['AllRowsGroupCoverage'].mean(),
            'MeanFiniteWidth': finite['MeanWidth'].mean(),
            'MeanFiniteIntervalScore': finite['MeanIntervalScore'].mean(),
        })
    research_table('分离校准区间的跨划分描述性汇总', rows)


def research_workload_summary(reports):
    rows = []
    for report in reports:
        rows.append({
            'Protocol': report.split.protocol,
            'Split': report.split.label,
            'TrainRows': len(report.split.train.targets),
            'TestRows': len(report.split.test.targets),
            'NestedCandidateFits': report.fit_count,
            'StudyMinutes': report.seconds / 60,
        })
    research_table('各划分实际完成的内层候选拟合与研究耗时', rows)


def run_research_study(df, mode, n_jobs, completed_rounds):
    profile = RESEARCH_PROFILES[mode]
    started = perf_counter()
    print(f'\n========== 独立研究流程 · {mode} ==========', flush=True)
    sample = research_sample(df)
    if len(sample.targets) < 32:
        print('可用目标值少于 32 行，结束独立研究流程。', flush=True)
        return
    research_profile_audit(sample)
    reports = []
    # 树搜索可沿用配置的进程数，单进程中的线性代数限制为一线程。
    with threadpool_limits(limits=1):
        for split in research_outer_splits(sample, profile, completed_rounds):
            nested_plan = research_nested_plan(split, profile)
            if not nested_plan:
                print(f'{split.label} 的嵌套验证可用训练数据不足，结束该划分的研究。', flush=True)
                continue
            reports.append(research_run_split(split, nested_plan, profile, n_jobs))
    if reports:
        research_protocol_summary(reports)
        research_pooled_groups(reports)
        research_interval_summary(reports)
        research_workload_summary(reports)
    print(f'独立研究完成 | {len(reports)} 个外层划分 | 总用时 {(perf_counter() - started) / 60:.2f} 分钟',
          flush=True)


if __name__ == '__main__':
    main()

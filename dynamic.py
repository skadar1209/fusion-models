import argparse
from datetime import datetime
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

    print(f'稳定性评估完成，逐轮记录和汇总已保存到 {save_dir}', flush=True)


if __name__ == '__main__':
    main()

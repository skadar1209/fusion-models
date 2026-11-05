# -*- coding: utf-8 -*-
"""ElasticNet：复制到PyCharm后可直接运行的完整单模型代码。"""

import json
import os
import platform
import warnings

import joblib
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import sklearn
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from sklearn.impute import KNNImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import ElasticNet

warnings.filterwarnings("ignore")

# =========================
# 0. 固定数据、特征、目标、随机种子和保存路径
# =========================
model_name = "ElasticNet"
data_path = r"D:\daima\全数据不删除_四模型调参\MP3_original_unchanged.csv"
save_dir = r"D:\daima\工作二\单模型复现输出\ElasticNet"
os.makedirs(save_dir, exist_ok=True)

features = ["CT", "AZ", "AC", "RP", "SSA"]
target = "MP7"
test_size = 0.2
split_seed = 162470
huber_threshold = 0.0001


# =========================
# 0.1 Huber损失与评价指标
# =========================
def huber_loss(y_true, y_pred, delta=1.0):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    error = y_true - y_pred
    absolute_error = np.abs(error)
    quadratic = np.minimum(absolute_error, delta)
    return float(
        np.mean(0.5 * quadratic**2 + delta * (absolute_error - quadratic))
    )


def calc_metrics(y_true, y_pred):
    pearson_result = stats.pearsonr(y_true, y_pred)
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": float(mean_absolute_percentage_error(y_true, y_pred)),
        "Huber": huber_loss(y_true, y_pred, delta=1.0),
        "Pearson_r": float(pearson_result[0]),
        "Pearson_p": float(pearson_result[1]),
    }




# =========================
# 1. 读取原始数据（171条记录，不删除任何记录）
# =========================
if not os.path.exists(data_path):
    raise FileNotFoundError("找不到数据文件：" + data_path)

df_original = pd.read_csv(data_path)
print("原始数据记录数：", len(df_original))
print("删除记录数：0")


# =========================
# 2. 分离非数值列并按原流程KNN填补
# =========================
non_numeric_columns = df_original.select_dtypes(exclude=[np.number]).columns
numeric_df = df_original.drop(columns=non_numeric_columns)
numeric_df = numeric_df.dropna(axis=1, how="all")

imputer = KNNImputer(n_neighbors=1)
numeric_imputed = imputer.fit_transform(numeric_df)
numeric_df = pd.DataFrame(
    numeric_imputed,
    columns=numeric_df.columns,
    index=df_original.index,
)

df = pd.concat(
    [numeric_df, df_original[non_numeric_columns].reset_index(drop=True)],
    axis=1,
)

missing_columns = [column for column in features + [target] if column not in df.columns]
if missing_columns:
    raise KeyError("数据缺少必要列：" + ", ".join(missing_columns))


# =========================
# 3. 特征和输出变量
# =========================
X = df[features].copy()
y = df[target].copy()


# =========================
# 4. 相关性矩阵和热图
# =========================
correlation_data = pd.concat([X, y], axis=1)
correlation_matrix = correlation_data.corr().round(3)
correlation_matrix.to_csv(
    os.path.join(save_dir, "feature_correlation_matrix.csv"),
    encoding="utf-8-sig",
)

custom_cmap = LinearSegmentedColormap.from_list(
    "blue_white_red", ["#0000ff", "#ffffff", "#ff0000"]
)
figure, axis = plt.subplots(figsize=(8.5, 7.2))
font_path = r"C:\Users\X\Desktop\机器学习（新）\timesbd.ttf"
if os.path.exists(font_path):
    font_property = fm.FontProperties(fname=font_path, size=10, weight="bold")
    matplotlib.rcParams["font.family"] = font_property.get_name()
    matplotlib.rcParams["font.size"] = font_property.get_size()

sns.heatmap(
    correlation_matrix,
    annot=True,
    vmin=-1,
    vmax=1,
    fmt=".3f",
    square=True,
    cmap=custom_cmap,
    annot_kws={"weight": "bold", "fontsize": 10},
    cbar_kws={"shrink": 0.8},
    ax=axis,
)
axis.set_ylim(correlation_matrix.shape[0], 0)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(
    os.path.join(save_dir, "feature_correlation_heatmap.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.savefig(
    os.path.join(save_dir, "feature_correlation_heatmap.pdf"),
    bbox_inches="tight",
)
plt.close()


# =========================
# 5. 训练集/测试集拆分，同时保留原始CSV行号
# =========================
row_indices = np.arange(len(df))
train_indices, test_indices = train_test_split(
    row_indices,
    test_size=test_size,
    random_state=split_seed,
)

X_train = X.iloc[train_indices]
X_test = X.iloc[test_indices]
y_train = y.iloc[train_indices]
y_test = y.iloc[test_indices]


# =========================
# 6. 数据标准化：只在训练集拟合
# =========================
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)


# =========================
# 7. 建立ElasticNet模型（全部锁定超参数直接写出）
# =========================
model = ElasticNet(
    alpha=2.0304506664238584e-05,
    l1_ratio=0.32894604561901036,
    max_iter=200000,
    selection='cyclic',
)


# =========================
# 8. 训练和预测
# =========================
model.fit(X_train_std, y_train)
y_train_hat = model.predict(X_train_std)
y_test_hat = model.predict(X_test_std)


# =========================
# 9. 计算全部评价指标
# =========================
train_metrics = calc_metrics(y_train, y_train_hat)
test_metrics = calc_metrics(y_test, y_test_hat)

metrics_df = pd.DataFrame([
    {"Model": model_name, "Dataset": "Train", **train_metrics},
    {"Model": model_name, "Dataset": "Test", **test_metrics},
])
metrics_df.to_csv(
    os.path.join(save_dir, "metrics.csv"),
    index=False,
    encoding="utf-8-sig",
)

print("\n测试集Huber损失：{:.6f}".format(test_metrics["Huber"]))
print("Huber阈值：      {:.6f}".format(huber_threshold))
if test_metrics["Huber"] <= huber_threshold:
    print("模型达到设定的Huber阈值。")
else:
    print("模型未达到设定的Huber阈值，但不改变数据或指标。")


# =========================
# 10. 训练集/测试集预测散点图
# =========================
all_values = np.concatenate([
    np.asarray(y_train),
    np.asarray(y_train_hat),
    np.asarray(y_test),
    np.asarray(y_test_hat),
])
minimum_value = float(np.min(all_values))
maximum_value = float(np.max(all_values))

plt.figure(figsize=(9, 8))
plt.style.use("default")
train_points = plt.scatter(
    y_train, y_train_hat, s=28, c="#b2df8a", alpha=0.78
)
test_points = plt.scatter(
    y_test, y_test_hat, s=34, c="#1f78b4", alpha=0.90
)
plt.plot(
    [minimum_value, maximum_value],
    [minimum_value, maximum_value],
    "k:",
    linewidth=1.5,
)
plt.xlabel("Observed MP7", fontsize=12)
plt.ylabel("Predicted MP7", fontsize=12)
plt.title(
    "{} - Train R²: {:.3f}, Test R²: {:.3f}".format(
        model_name, train_metrics["R2"], test_metrics["R2"]
    ),
    fontsize=12,
)
plt.legend((train_points, test_points), ("Train", "Test"), fontsize=12)
plt.tight_layout()
plt.savefig(
    os.path.join(save_dir, "train_test_scatter.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.savefig(
    os.path.join(save_dir, "train_test_scatter.pdf"),
    bbox_inches="tight",
)
plt.close()


# =========================
# 11. 残差诊断图
# =========================
train_residual = np.asarray(y_train) - np.asarray(y_train_hat)
test_residual = np.asarray(y_test) - np.asarray(y_test_hat)
figure, axes = plt.subplots(1, 2, figsize=(10.6, 4.5))
axes[0].scatter(
    y_train_hat, train_residual, color="#b2df8a", s=28, alpha=0.75, label="Train"
)
axes[0].scatter(
    y_test_hat, test_residual, color="#1f78b4", s=34, alpha=0.88, label="Test"
)
axes[0].axhline(0, color="black", linestyle=":", linewidth=1.2)
axes[0].set_xlabel("Predicted MP7")
axes[0].set_ylabel("Residual")
axes[0].set_title("Residual versus prediction")
axes[0].legend()
axes[1].hist(train_residual, bins=14, color="#b2df8a", alpha=0.65, label="Train")
axes[1].hist(test_residual, bins=10, color="#1f78b4", alpha=0.65, label="Test")
axes[1].set_xlabel("Residual")
axes[1].set_ylabel("Count")
axes[1].set_title("Residual distribution")
axes[1].legend()
figure.suptitle(model_name + " residual diagnostics")
figure.tight_layout()
figure.savefig(
    os.path.join(save_dir, "residual_diagnostics.png"),
    dpi=300,
    bbox_inches="tight",
)
figure.savefig(
    os.path.join(save_dir, "residual_diagnostics.pdf"),
    bbox_inches="tight",
)
plt.close(figure)


# =========================
# 12. 保存训练集、测试集逐行预测
# =========================
train_save_df = pd.DataFrame({
    "OriginalDataFrameIndex_0Based": train_indices,
    "OriginalCSVRow_WithHeader": train_indices + 2,
    "y_train_true": np.asarray(y_train),
    "y_train_pred": np.asarray(y_train_hat),
    "train_residual": train_residual,
    "train_abs_error": np.abs(train_residual),
})
test_save_df = pd.DataFrame({
    "OriginalDataFrameIndex_0Based": test_indices,
    "OriginalCSVRow_WithHeader": test_indices + 2,
    "y_test_true": np.asarray(y_test),
    "y_test_pred": np.asarray(y_test_hat),
    "test_residual": test_residual,
    "test_abs_error": np.abs(test_residual),
})
train_save_df.to_csv(
    os.path.join(save_dir, "train_predictions.csv"),
    index=False,
    encoding="utf-8-sig",
)
test_save_df.to_csv(
    os.path.join(save_dir, "test_predictions.csv"),
    index=False,
    encoding="utf-8-sig",
)
np.savetxt(
    os.path.join(save_dir, "train_true_pred.csv"),
    np.c_[y_train, y_train_hat],
    delimiter=",",
    header="y_train_true,y_train_pred",
    comments="",
)
np.savetxt(
    os.path.join(save_dir, "test_true_pred.csv"),
    np.c_[y_test, y_test_hat],
    delimiter=",",
    header="y_test_true,y_test_pred",
    comments="",
)


# =========================
# 13. 保存特征重要性（严格沿用原项目判断方式）
# =========================
if hasattr(model, "feature_importances_"):
    importance_values = np.asarray(model.feature_importances_)
    importance_method = "model_built_in_importance"
else:
    permutation = permutation_importance(
        model,
        X_test_std,
        y_test,
        n_repeats=30,
        random_state=20260728,
        scoring="r2",
    )
    importance_values = permutation.importances_mean
    importance_method = "test_permutation_importance"

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance_values,
    "Method": importance_method,
}).sort_values("Importance", ascending=False)
importance_df.to_csv(
    os.path.join(save_dir, "feature_importance.csv"),
    index=False,
    encoding="utf-8-sig",
)


# =========================
# 14. 保存模型、标准化器和KNN填补器
# =========================
joblib.dump(model, os.path.join(save_dir, "model.pkl"))
joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))
joblib.dump(imputer, os.path.join(save_dir, "knn_imputer.pkl"))


# =========================
# 15. 保存全部参数、种子、环境和方法说明
# =========================
metadata = {
    "model": model_name,
    "data_path": data_path,
    "save_dir": save_dir,
    "source_rows": int(len(df_original)),
    "rows_deleted": 0,
    "rows_used": int(len(df)),
    "features": features,
    "target": target,
    "split_seed": split_seed,
    "test_size": test_size,
    "train_rows": int(len(train_indices)),
    "test_rows": int(len(test_indices)),
    "model_parameters": model.get_params(deep=False),
    "train_metrics": train_metrics,
    "test_metrics": test_metrics,
    "preprocessing": (
        "KNNImputer(n_neighbors=1)在拆分前对全部数值表拟合；"
        "StandardScaler只在训练集拟合；不删除任何记录。"
    ),
    "selection_note": (
        "划分种子和超参数直接使用测试集R2筛选，以复刻原四模型流程；"
        "测试分数属于探索性乐观结果。"
    ),
}
with open(os.path.join(save_dir, "run_metadata.json"), "w", encoding="utf-8") as file:
    json.dump(metadata, file, ensure_ascii=False, indent=2, default=str)

environment = {
    "python": platform.python_version(),
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scipy": scipy.__version__,
    "scikit_learn": sklearn.__version__,
    "seaborn": sns.__version__,
}
with open(os.path.join(save_dir, "environment_versions.json"), "w", encoding="utf-8") as file:
    json.dump(environment, file, ensure_ascii=False, indent=2)


# =========================
# 16. 打印评价指标和保存内容
# =========================
print("\n=== {} 单模型评价指标 ===".format(model_name))
print("训练集Pearson r：", stats.pearsonr(y_train, y_train_hat))
print("测试集Pearson r：", stats.pearsonr(y_test, y_test_hat))
print("训练集R2：   {:.6f}".format(train_metrics["R2"]))
print("训练集RMSE： {:.6f}".format(train_metrics["RMSE"]))
print("训练集MAE：  {:.6f}".format(train_metrics["MAE"]))
print("训练集MAPE： {:.6f}".format(train_metrics["MAPE"]))
print("测试集R2：   {:.6f}".format(test_metrics["R2"]))
print("测试集RMSE： {:.6f}".format(test_metrics["RMSE"]))
print("测试集MAE：  {:.6f}".format(test_metrics["MAE"]))
print("测试集MAPE： {:.6f}".format(test_metrics["MAPE"]))
print("\n================ 程序运行完成 ================")
print("所有结果已保存到：", save_dir)
print("保存内容包括预测表、指标表、相关性图、散点图、残差图、")
print("特征重要性、模型、StandardScaler、KNNImputer和元数据。")


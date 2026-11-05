import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.impute import KNNImputer

# =========================
# 0. 路径设置
# =========================
csv_path = r'C:\Users\X\Desktop\机器学习（新）\RC84.csv'
font_path = r"C:\Users\X\Desktop\机器学习（新）\timesbd.ttf"

# 输出图片路径（你可以改文件名/后缀：png / tif / pdf 等）
out_img_path = r"C:\Users\X\Desktop\机器学习（新）\pearson_heatmap.png"

# =========================
# 1. 读取数据
# =========================
df = pd.read_csv(csv_path)

# =========================
# 2. 分离非数值列 + 处理缺失（仅用于相关系数计算）
# =========================
non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
numeric_df = df.drop(columns=non_numeric_columns)

# 移除全空列
numeric_df = numeric_df.dropna(axis=1, how='all')

# KNN 填补缺失值
imputer = KNNImputer(n_neighbors=1)
numeric_df_imputed = imputer.fit_transform(numeric_df)
numeric_df = pd.DataFrame(numeric_df_imputed, columns=numeric_df.columns)

# 将非数值列加回来（虽然热图只用数值列，但保留原结构）
df = pd.concat(
    [numeric_df, df[non_numeric_columns].reset_index(drop=True)],
    axis=1
)

# =========================
# 3. 选择要计算 Pearson 的特征与目标
# =========================
features = ["N","SP","TS","EC","FA","3D","AS","CS", "CT","AZ", "DC", "AC", "RP", "IS", "SSA", "IC"]
X = df[features]
y = df["ICE"]

# =========================
# 4. 相关性矩阵与热图（Pearson）
# =========================
data_corr = pd.concat([X, y], axis=1)
corr_mat = data_corr.corr(method="pearson").round(3)

# 自定义蓝-白-红 colormap
colors = ["#0000ff", "#ffffff", "#ff0000"]
custom_cmap = LinearSegmentedColormap.from_list("blue_red", colors)

# 字体设置（Times）
font_prop = fm.FontProperties(fname=font_path, size=10, weight='bold')
matplotlib.rcParams['font.family'] = font_prop.get_name()
matplotlib.rcParams['font.size'] = font_prop.get_size()

fig, ax = plt.subplots(figsize=(10, 8))
ax = sns.heatmap(
    corr_mat,
    annot=True,
    vmin=-1,
    vmax=1,
    fmt='.3f',
    square=True,
    cmap=custom_cmap,
    annot_kws={'weight': 'bold', "fontsize": 10},
    cbar_kws={'shrink': 0.8}
)

# 防止第一行/最后一行被裁掉
ax.set_ylim(corr_mat.shape[0], 0)

plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# =========================
# 5. 保存图片（关键新增）
# =========================
plt.savefig(out_img_path, dpi=300, bbox_inches='tight')  # ⭐ 保存高清图，避免边缘被裁
plt.show()

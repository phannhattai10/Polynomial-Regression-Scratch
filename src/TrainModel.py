import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from ClassPolynomial import Polynomial, compute_metrics, get_poly_feature_names
from preprocessing import transform


def find_project_root(start=None, required_dirs=('src', 'data_processed'), max_levels=6):
    """Walk up the directory tree to find the project root (containing both 'src/' and 'data_processed/'),
    mirroring the approach used in the notebooks so the script works regardless of where it is invoked from."""
    current = os.path.abspath(start or os.getcwd())
    for _ in range(max_levels):
        if all(os.path.isdir(os.path.join(current, d)) for d in required_dirs):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return None


PROJECT_ROOT = find_project_root(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT is None:
    raise RuntimeError(
        "Cannot find project root (must contain both 'src/' and 'data_processed/'). "
        "Make sure you have run the pipeline notebook 05_export_pipeline.ipynb first."
    )

DATA_DIR = os.path.join(PROJECT_ROOT, 'data_processed')
REPORT_DIR = os.path.join(PROJECT_ROOT, 'report')
os.makedirs(REPORT_DIR, exist_ok=True)


plt.rcParams.update({
    'figure.figsize': (8, 5.5),
    'figure.dpi': 110,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'axes.edgecolor': '#333333',
    'axes.grid': True,
    'grid.color': '#dddddd',
    'grid.linewidth': 0.6,
    'legend.frameon': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLOR_PRIMARY = '#2563eb'   # blue
COLOR_SECONDARY = '#f97316'  # orange
COLOR_ACCENT = '#16a34a'    # green
COLOR_REF_LINE = '#9ca3af'  # gray dashed reference line


def save_fig(fig, filename):
    """Save a figure into the report/ directory with a consistent style."""
    path = os.path.join(REPORT_DIR, filename)
    fig.savefig(path)
    print(f"Saved figure: {filename}")


def split_train_val(X, y, val_ratio=0.2, seed=42):
    """Tách tập validation từ tập train bằng NumPy/Pandas (không dùng scikit-learn)
    chỉ phục vụ riêng cho bước đánh giá & chọn siêu tham số (bậc đa thức)."""
    np.random.seed(seed)
    n = len(X)
    shuffled_idx = np.random.permutation(n)
    n_val = int(np.floor(n * val_ratio))
    val_idx = shuffled_idx[:n_val]
    train_idx = shuffled_idx[n_val:]

    if isinstance(X, pd.DataFrame):
        X_tr = X.iloc[train_idx].reset_index(drop=True)
        X_v = X.iloc[val_idx].reset_index(drop=True)
    else:
        X_tr = X[train_idx]
        X_v = X[val_idx]

    if isinstance(y, (pd.Series, pd.DataFrame)):
        y_tr = y.iloc[train_idx].reset_index(drop=True)
        y_v = y.iloc[val_idx].reset_index(drop=True)
    else:
        y_tr = y[train_idx]
        y_v = y[val_idx]

    return X_tr, y_tr, X_v, y_v


# Load preprocessed training data (fitted solely on train_raw)
X_train_df = pd.read_csv(os.path.join(DATA_DIR, 'X_train.csv'))
y_train_series = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv'))['mpg']

print(f'Full Train data loaded: X_train={X_train_df.shape}, y_train={y_train_series.shape}')

# Split a validation subset from train specifically for hyperparameter selection
X_sub_train_df, y_sub_train_series, X_val_df, y_val_series = split_train_val(
    X_train_df, y_train_series, val_ratio=0.2, seed=42
)

feature_names = list(X_train_df.columns)  # fixed column order, used to cross-reference W later
X_sub_train_np = X_sub_train_df.to_numpy()
y_sub_train_np = y_sub_train_series.to_numpy()
X_val_np = X_val_df.to_numpy()
y_val_np = y_val_series.to_numpy()

print(f'Sub-train for tuning : {X_sub_train_np.shape}   y: {y_sub_train_np.shape}')
print(f'Validation set       : {X_val_np.shape}   y: {y_val_np.shape}')

model = Polynomial(2)  # initial degree doesn't matter; will be overwritten in the loop below

results = []  # store (degree, metrics_train, metrics_val) for comparison / best-degree selection

MAX_DEGREE = 4

for i in range(1, MAX_DEGREE + 1):
    print(f"\n===== Polynomial degree = {i} =====")

    X_poly = model.TransformMultiFeature(X_sub_train_np, i)
    X_val_poly = model.TransformMultiFeature(X_val_np, i)
    print(f"Number of columns in Xbar (including bias): {X_poly.shape[1]}  |  train rows: {X_poly.shape[0]}")

    try:
        model.Fit(X_poly, y_sub_train_np)
    except ValueError as e:
        print(f"SKIPPING degree {i}: {e}")
        continue

    y_train_pred = model.Predict(X_poly)
    y_val_pred = model.Predict(X_val_poly)

    metrics_train = compute_metrics(y_sub_train_np, y_train_pred)
    metrics_val = compute_metrics(y_val_np, y_val_pred)

    print("TRAIN (Sub-train):", metrics_train)
    print("VALIDATION       :", metrics_val)

    results.append({'degree': i, 'train': metrics_train, 'val': metrics_val})


if results:
    print("\n===== SUMMARY (selecting degree by lowest RMSE on VALIDATION) =====")
    for r in results:
        print(f"degree={r['degree']:>2}  RMSE_train={r['train']['RMSE']:.4f}  "
              f"RMSE_val={r['val']['RMSE']:.4f}  R2_val={r['val']['R2']:.4f}")
    best = min(results, key=lambda r: r['val']['RMSE'])
    print(f"\n=> Best degree by RMSE_val: {best['degree']}")

    # Cross-reference feature column names with W coefficients for the best degree (for reference / debug)
    xbar_col_names = get_poly_feature_names(feature_names, best['degree'])
    print(f"Number of generated column names: {len(xbar_col_names)} (must equal number of Xbar columns above)")


X_poly = model.TransformMultiFeature(X_sub_train_np, best['degree'])
model.Fit(X_poly, y_sub_train_np)

X_val_poly = model.TransformMultiFeature(X_val_np, best['degree'])
y_pred = model.Predict(X_val_poly)

fig, ax = plt.subplots()
ax.scatter(y_pred, y_val_np, s=45, color=COLOR_PRIMARY, alpha=0.75,
           edgecolors='white', linewidths=0.6, label='Validation samples')

min_value = min(y_pred.min(), y_val_np.min())
max_value = max(y_pred.max(), y_val_np.max())
ax.plot([min_value, max_value], [min_value, max_value],
        linestyle='--', color=COLOR_REF_LINE, linewidth=1.8, label='Perfect prediction')

ax.set_xlabel("Predicted MPG")
ax.set_ylabel("Actual MPG")
ax.set_title(f"Predicted vs Actual — Polynomial Degree {best['degree']}")
ax.legend(loc='upper left')
fig.tight_layout()
save_fig(fig, "11_predicted_vs_actual.png")
plt.show()


residual = y_val_np - y_pred
fig, ax = plt.subplots()
ax.scatter(y_pred, residual, s=45, color=COLOR_SECONDARY, alpha=0.75,
           edgecolors='white', linewidths=0.6)
ax.axhline(y=0, linestyle='--', color=COLOR_REF_LINE, linewidth=1.8)
ax.set_xlabel("Predicted MPG")
ax.set_ylabel("Residual (Actual − Predicted)")
ax.set_title("Residual Plot")
fig.tight_layout()
save_fig(fig, "12_residual_plot.png")
plt.show()


degrees = [r['degree'] for r in results]
rmse_train = [r['train']['RMSE'] for r in results]
rmse_val = [r['val']['RMSE'] for r in results]

fig, ax = plt.subplots()
ax.plot(degrees, rmse_train, marker='o', markersize=7, linewidth=2.2,
        color=COLOR_PRIMARY, label='Train RMSE')
ax.plot(degrees, rmse_val, marker='o', markersize=7, linewidth=2.2,
        color=COLOR_SECONDARY, label='Validation RMSE')
best_idx = degrees.index(best['degree'])
ax.scatter([best['degree']], [rmse_val[best_idx]], s=140, facecolors='none',
           edgecolors=COLOR_ACCENT, linewidths=2.2, zorder=5,
           label=f"Best degree = {best['degree']}")

ax.set_xlabel('Polynomial Degree')
ax.set_ylabel('RMSE')
ax.set_title('Learning Curve by Polynomial Degree')
ax.set_xticks(degrees)
ax.legend(loc='best')
fig.tight_layout()
save_fig(fig, "13_learning_curve_rmse.png")
plt.show()


fig, ax = plt.subplots()
ax.hist(residual, bins=20, color=COLOR_ACCENT, alpha=0.85, edgecolor='white', linewidth=0.8)
ax.axvline(x=0, linestyle='--', color=COLOR_REF_LINE, linewidth=1.8)
ax.set_xlabel("Residual")
ax.set_ylabel("Frequency")
ax.set_title("Residual Distribution")
fig.tight_layout()
save_fig(fig, "14_residual_distribution.png")
plt.show()


# -------------------------------------------------------------
# Final Model: Retrain on the ENTIRE training set (100% of Train)
# and evaluate on the independent, raw test set (test_raw.csv).
# -------------------------------------------------------------
with open(os.path.join(DATA_DIR, 'preprocessing_params.pkl'), 'rb') as f:
    params = pickle.load(f)

test_raw = pd.read_csv(os.path.join(DATA_DIR, 'test_raw.csv'), dtype=str, keep_default_na=False, na_values=[])

# Transform test set using the parameters fitted ONLY on train_raw
X_test, y_test = transform(test_raw, params)
X_test_np = X_test.to_numpy()
y_test_np = y_test.to_numpy()

# Fit final model on the entire X_train
X_full_train_np = X_train_df.to_numpy()
y_full_train_np = y_train_series.to_numpy()

FinalModel = Polynomial(best['degree'])
X_full_train_poly = FinalModel.TransformMultiFeature(X_full_train_np, best['degree'])
FinalModel.Fit(X_full_train_poly, y_full_train_np)

# Predict on test
X_test_poly = FinalModel.TransformMultiFeature(X_test_np, best['degree'])
y_test_pred = FinalModel.Predict(X_test_poly)
metric_test = compute_metrics(y_test_np, y_test_pred)
print(f"\n===== FINAL RESULT ON RAW TEST SET (Degree {best['degree']}) =====")
print(metric_test)


fig, ax = plt.subplots()
ax.scatter(y_test_pred, y_test_np, s=45, color=COLOR_PRIMARY, alpha=0.75,
           edgecolors='white', linewidths=0.6, label='Test samples')
min_v = min(y_test_pred.min(), y_test_np.min())
max_v = max(y_test_pred.max(), y_test_np.max())
ax.plot([min_v, max_v], [min_v, max_v], linestyle='--', color=COLOR_REF_LINE,
        linewidth=1.8, label='Perfect prediction')
ax.set_xlabel("Predicted MPG")
ax.set_ylabel("Actual MPG")
ax.set_title(f"Final Model on Test Set — Degree {best['degree']}\n"
             f"RMSE={metric_test['RMSE']:.3f} | R²={metric_test['R2']:.3f}")
ax.legend(loc='upper left')
fig.tight_layout()
save_fig(fig, "15_final_test_predicted_vs_actual.png")
plt.show()

print(f"\nAll charts saved to: {REPORT_DIR}")
import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from ClassPolynomial import compute_metrics, get_poly_feature_names


def find_project_root(start=None, required_dirs=('src', 'data_processed'), max_levels=6):
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
        "Cannot find project root. Make sure you have run notebooks/05_export_pipeline.ipynb first."
    )

DATA_DIR = os.path.join(PROJECT_ROOT, 'data_processed')

def split_train_val(X, y, val_ratio=0.2, seed=42):
    """Tách tập validation từ tập train bằng NumPy/Pandas (không dùng scikit-learn)
    đồng bộ với TrainModel.py để đối chiếu kết quả."""
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


# Load preprocessed data (fitted solely on train_raw)
X_train_df = pd.read_csv(os.path.join(DATA_DIR, 'X_train.csv'))
y_train_series = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv'))['mpg']

X_sub_train, y_sub_train, X_val, y_val = split_train_val(
    X_train_df, y_train_series, val_ratio=0.2, seed=42
)

print(f"Full Train data: X={X_train_df.shape}, y={y_train_series.shape}")
print(f"Sub-train      : X={X_sub_train.shape}, y={y_sub_train.shape}")
print(f"Validation     : X={X_val.shape}, y={y_val.shape}")

DEGREE = 1

poly = PolynomialFeatures(degree=DEGREE)
poly.fit(X_sub_train)

X_poly_train = poly.transform(X_sub_train)
X_poly_val = poly.transform(X_val)

model = LinearRegression(fit_intercept=False)  # disable separate intercept; PolynomialFeatures already adds a bias column
model.fit(X_poly_train, y_sub_train)

y_train_pred = model.predict(X_poly_train)
y_val_pred = model.predict(X_poly_val)

print(f"\n===== sklearn - Polynomial degree = {DEGREE} (for cross-reference only) =====")
print("W (bias at index 0) =", model.coef_)

print("\nTRAIN (Sub-train):", compute_metrics(y_sub_train, y_train_pred))
print("VALIDATION       :", compute_metrics(y_val, y_val_pred))

# Cross-reference column order between get_poly_feature_names() and sklearn.PolynomialFeatures
col_names_custom = get_poly_feature_names(list(X_train_df.columns), DEGREE)
col_names_sklearn = list(poly.get_feature_names_out(list(X_train_df.columns)))
print(f"\nColumn count: custom = {len(col_names_custom)}, sklearn = {len(col_names_sklearn)}")
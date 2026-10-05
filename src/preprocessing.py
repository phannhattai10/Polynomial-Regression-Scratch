from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

# Tokens representing missing values found in the raw data (see 01_data_overview.ipynb)
MISSING_TOKENS: List[str] = [
    '?', 'NA', 'N/A', 'na', 'n/a', 'unknown', 'Unknown', 'UNKNOWN',
    'nan', 'NaN', 'NAN', 'null', 'NULL', 'None', 'none', '-', '--', '',
]

# cylinders written as words -> integers (see 03_eda.ipynb section 1c)
MAPPING_CYLINDERS_WORD_TO_INT: Dict[str, int] = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
    'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
}

# origin written as words -> numeric code 1=USA, 2=Europe, 3=Japan (see 03_eda.ipynb section 1d)
MAPPING_ORIGIN_WORD_TO_INT: Dict[str, int] = {
    'usa': 1, 'us': 1, 'america': 1,
    'europe': 2, 'eu': 2,
    'japan': 3, 'jp': 3,
}

# Feature columns fed into the model - decided from EDA conclusions (03_eda.ipynb section 7):
# keeping all variables with clear physical meaning relevant to fuel consumption.
# 'origin' is a categorical code (1/2/3) but is still a valid numeric variable; degree=1 when using Polynomial.
FEATURE_COLUMNS: List[str] = [
    'cylinders', 'displacement', 'horsepower', 'weight',
    'acceleration', 'model year', 'origin',
]
TARGET_COLUMN: str = 'mpg'

# Columns checked / clipped for outliers using IQR computed on train (per project requirements)
OUTLIER_COLUMNS: List[str] = ['horsepower', 'weight', 'mpg']

# Columns to apply log-transform (log1p) before scaling - decided based on
# skewness measured on cleaned train_raw (see notebook 04, section 2):
# ONLY 'horsepower' has |skew| > 1 clearly (skew ~ 6.9 before clipping outliers,
# still > 1 after clipping) -> the most justified candidate for log-transform.
# 'weight' (|skew| ~ 0.78) and 'displacement' (|skew| ~ 0.68) do not exceed
# threshold 1 on this dataset -> NOT log-transformed to avoid unnecessary distortion.
LOG_TRANSFORM_COLUMNS: List[str] = ['horsepower']

# Type alias for the "params" dict returned by fit_preprocessing() (for reference / code readability;
# not enforced at runtime because params is a "dynamic" dict containing various sub-types).
PreprocessingParams = Dict[str, Any]

def _normalize_cylinders(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    s = str(value).strip().lower()
    if s in MAPPING_CYLINDERS_WORD_TO_INT:
        return MAPPING_CYLINDERS_WORD_TO_INT[s]
    try:
        return float(s)
    except ValueError:
        return np.nan


def _normalize_origin(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    s = str(value).strip().lower()
    if s in MAPPING_ORIGIN_WORD_TO_INT:
        return MAPPING_ORIGIN_WORD_TO_INT[s]
    try:
        num = float(s)
        return num if num in (1.0, 2.0, 3.0) else np.nan
    except ValueError:
        return np.nan


def _normalize_model_year(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    try:
        n = float(value)
    except ValueError:
        return np.nan
    n = 1900 + n if n < 100 else n
    if n < 1960 or n > 2030:
        return np.nan
    return n


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # In case data was not read as strings (e.g. already numeric), cast to str first for consistency
    df = df.apply(lambda col: col.astype(str).str.strip())

    # 1) Normalize various missing-value representations to actual NaN
    df = df.replace({tok: np.nan for tok in MISSING_TOKENS})

    # 2) weight: may contain the unit suffix 'lbs' -> strip letters, cast to numeric
    df['weight'] = (
        df['weight'].astype(str)
        .str.replace('lbs', '', case=False, regex=False)
        .str.strip()
    )
    df['weight'] = pd.to_numeric(df['weight'], errors='coerce')

    # Negative weight (e.g. -1945) is physically impossible -> NaN.
    # Identified in 03_eda.ipynb (section 1.1): clean_raw() previously only handled the 'lbs'
    # suffix but did not remove negative values, inconsistent with how horsepower/model year were handled.
    df.loc[df['weight'] < 0, 'weight'] = np.nan

    # 3) acceleration: European decimal comma -> replace ',' with '.'
    df['acceleration'] = df['acceleration'].astype(str).str.replace(',', '.', regex=False)
    df['acceleration'] = pd.to_numeric(df['acceleration'], errors='coerce')

    # 4) horsepower: cast to numeric; negative or unrealistic values (>= 1000) -> NaN
    df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
    df.loc[(df['horsepower'] < 0) | (df['horsepower'] >= 1000), 'horsepower'] = np.nan

    # 5) cylinders: word -> integer
    df['cylinders'] = df['cylinders'].apply(_normalize_cylinders)

    # 6) origin: mixed word/number -> unified numeric code 1/2/3
    df['origin'] = df['origin'].apply(_normalize_origin)

    # 7) model year: 2-digit -> 4-digit, remove unrealistic years
    df['model year'] = df['model year'].apply(_normalize_model_year)

    # 8) mpg, displacement: cast to numeric; unrealistic mpg (e.g. 9999) -> NaN
    for c in ['mpg', 'displacement']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df.loc[df['mpg'] > 200, 'mpg'] = np.nan

    # 9) Final cleanup (precaution in case val/test has duplicate or all-NaN rows not
    #    removed at split time): drop fully-duplicate rows and all-NaN rows
    df = df.drop_duplicates(keep='first')
    df = df.dropna(how='all')

    df = df.reset_index(drop=True)
    return df

def _compute_iqr_bounds(series: pd.Series) -> Tuple[float, float]:
    s = series.dropna()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def fit_preprocessing(train_df: pd.DataFrame) -> PreprocessingParams:
    df = train_df.copy()

    all_numeric_cols: List[str] = FEATURE_COLUMNS + [TARGET_COLUMN]

    # --- (a) Imputation values: median of TRAIN for each numeric column (computed BEFORE imputing) ---
    impute_values: Dict[str, float] = {col: float(df[col].median()) for col in all_numeric_cols}

    # --- (b) Temporarily impute to compute stable outlier bounds / mean / std (avoids NaN bias) ---
    df_temp = df.copy()
    for col in all_numeric_cols:
        df_temp[col] = df_temp[col].fillna(impute_values[col])

    # --- (c) Outlier clipping bounds via IQR OF TRAIN (only for horsepower/weight/mpg) ---
    outlier_bounds: Dict[str, Tuple[float, float]] = {}
    for col in OUTLIER_COLUMNS:
        lower, upper = _compute_iqr_bounds(df_temp[col])
        outlier_bounds[col] = (float(lower), float(upper))
        df_temp[col] = df_temp[col].clip(lower=lower, upper=upper)

    # --- (d) Log-transform (log1p) for columns identified in EDA (features only) ---
    for col in LOG_TRANSFORM_COLUMNS:
        df_temp[col] = np.log1p(df_temp[col])

    # --- (e) Mean/std of TRAIN (AFTER impute + clip + log-transform) for each feature ---
    scale_mean: Dict[str, float] = {}
    scale_std: Dict[str, float] = {}
    for col in FEATURE_COLUMNS:
        mean_ = float(df_temp[col].mean())
        std_ = float(df_temp[col].std(ddof=0))
        if std_ == 0:
            std_ = 1.0  # avoid division by zero if a column has zero variance
        scale_mean[col] = mean_
        scale_std[col] = std_

    params: PreprocessingParams = {
        'features': list(FEATURE_COLUMNS),
        'target': TARGET_COLUMN,
        'log_transform_cols': list(LOG_TRANSFORM_COLUMNS),
        'impute_values': impute_values,
        'outlier_bounds': outlier_bounds,
        'scale_mean': scale_mean,
        'scale_std': scale_std,
    }
    return params

def transform(
    raw_df: pd.DataFrame,
    params: PreprocessingParams,
) -> Tuple[pd.DataFrame, pd.Series]:
    df = clean_raw(raw_df)

    features: List[str] = params['features']
    target: str = params['target']
    log_cols: List[str] = params['log_transform_cols']
    impute_values: Dict[str, float] = params['impute_values']
    outlier_bounds: Dict[str, Tuple[float, float]] = params['outlier_bounds']
    scale_mean: Dict[str, float] = params['scale_mean']
    scale_std: Dict[str, float] = params['scale_std']

    # (1) Drop rows with missing target (mpg) - do NOT impute target to avoid label fabrication;
    #     imputation is applied only to input features.
    df = df.loc[df[target].notna()].reset_index(drop=True)

    # (2) Impute feature columns using values learned from TRAIN (median of train)
    for col in features:
        df[col] = df[col].fillna(impute_values[col])

    # (3) Clip outliers using IQR bounds learned from TRAIN - applied to both features and target
    for col, (lower, upper) in outlier_bounds.items():
        df[col] = df[col].clip(lower=lower, upper=upper)

    # (4) Log-transform (log1p) for columns identified at fit time (features only)
    for col in log_cols:
        df[col] = np.log1p(df[col])

    # (5) Standardize: (x - mean) / std using mean/std learned from TRAIN
    X = pd.DataFrame(index=df.index)
    for col in features:
        X[col] = (df[col] - scale_mean[col]) / scale_std[col]

    y = df[target].copy()
    y.name = target

    return X, y
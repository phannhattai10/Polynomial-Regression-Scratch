import pandas as pd
import numpy as np
import itertools


# Matrix inverse via Gauss-Jordan elimination
def GaussJordan(x):
    n = x.shape[0]
    m = x.shape[1]
    if n != m:
        raise ValueError("Only accepts square matrices")

    I = np.eye(n)  # create identity matrix
    X_aug = np.concatenate((x, I), axis=1)

    for i in range(n):
        pivot = X_aug[i][i]

        if pivot == 0:
            for k in range(i + 1, n):
                if X_aug[k][i] != 0:
                    temp = X_aug[i].copy()
                    X_aug[i] = X_aug[k]
                    X_aug[k] = temp
                    pivot = X_aug[i][i]
                    break
        if pivot == 0:
            raise ValueError("Cannot find inverse (singular / non-invertible matrix)")

        X_aug[i] = X_aug[i] / pivot
        for j in range(n):
            if j == i:
                continue
            factor = X_aug[j][i]
            X_aug[j] = X_aug[j] - (factor * X_aug[i])

    return X_aug[:, n:]

def MSE(y_test, y_pred):
    y_test = np.asarray(y_test, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean((y_test - y_pred) ** 2)


def R_2(y_test, y_pred):
    y_test = np.asarray(y_test, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    return 1 - (ss_res / ss_tot)


def MAE(y_test, y_pred):
    y_test = np.asarray(y_test, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean(np.abs(y_test - y_pred))


def compute_metrics(y_test, y_pred):
    mse = MSE(y_test, y_pred)
    return {
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": MAE(y_test, y_pred),
        "R2": R_2(y_test, y_pred),
    }


class Polynomial():
    def __init__(self, degree,method ="normal",learning_rate = 0.1,epochs=1000):
        self.degree = degree
        self.method = method
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.W = None

    def Transform(self, x, degree):
        """Polynomial feature expansion for a single feature (1D)."""
        n = x.shape[0]

        Xbar = np.ones((n, degree + 1))
        for i in range(1, degree + 1):
            Xbar[:, i] = x[:, 0] ** i
        return Xbar

    def TransformMultiFeature(self, x, degree):
        """Polynomial feature expansion for multiple features using combinations_with_replacement."""
        n = x.shape[0]
        num_features = x.shape[1]

        Xbar = [np.ones(n)]
        feature_indices = list(range(num_features))

        # iterate from degree 1 to max degree to build all combinations
        for d in range(1, degree + 1):
            for combo in itertools.combinations_with_replacement(feature_indices, d):
                term = np.ones(n)

                for feature_index in combo:
                    term = term * x[:, feature_index]
                Xbar.append(term)
        return np.column_stack(Xbar)

    def Fit(self, Xbar, y):
        if self.method == "normal":
            a = GaussJordan(Xbar.T @ Xbar)
            b = Xbar.T @ y
            self.W = a @ b
        elif self.method == "gradient":
            n = len(Xbar)
            self.W = np.zeros(Xbar.shape[1])
            for loop in range(self.epochs):
                y_pred = Xbar @ self.W
                error = y_pred-y

                gradient = (2 / n) * (Xbar.T @ error)

                # Update W
                self.W = (self.W- self.learning_rate * gradient)
        else:
            raise ValueError("method must be 'normal' or 'gradient'")

    def Predict(self, Xbar):
        if self.W is None:
            raise ValueError("Model has no data")
        result = Xbar @ self.W
        return result

def get_poly_feature_names(feature_names, degree):
    """Return the list of polynomial feature name strings matching TransformMultiFeature order."""
    col_names = ["1"]
    num_features = len(feature_names)
    feature_indices = list(range(num_features))

    for d in range(1, degree + 1):
        for combo in itertools.combinations_with_replacement(feature_indices, d):
            counts = {}
            for idx in combo:
                counts[idx] = counts.get(idx, 0) + 1
            parts = []
            for idx in sorted(counts.keys()):
                if counts[idx] == 1:
                    parts.append(feature_names[idx])
                else:
                    parts.append(f"{feature_names[idx]}^{counts[idx]}")
            col_names.append("*".join(parts))

    return col_names
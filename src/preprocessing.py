"""Kat-içi (fold-local) ön işleme: imputation, clipping, ölçekleme, one-hot.

Tüm parametreler (median, en sık kategori, kategori sözlüğü, scaler istatistikleri,
clipping eşikleri) yalnızca .fit() çağrılan alt kümeden öğrenilir; diğer alt kümelere
yalnızca .transform() uygulanır.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler

from .config import CATEGORICAL, ENGINEERED_BINARY, NUMERIC

SCALERS = {
    "standard": StandardScaler,  # z-skoru (standardization): (x - ortalama) / SD
    "minmax": MinMaxScaler,      # aralık ölçekleme (normalization): (x - min) / (max - min)
    "robust": RobustScaler,      # (x - median) / IQR, aykırı değerlere dayanıklı
}


class PercentileClipper(BaseEstimator, TransformerMixin):
    """Eğitim alt kümesinde öğrenilen yüzdelik sınırlarla kırpma (winsorizing)."""

    def __init__(self, lower=1.0, upper=99.0):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lower_ = np.nanpercentile(X, self.lower, axis=0)
        self.upper_ = np.nanpercentile(X, self.upper, axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def unknown_to_nan(X):
    X = pd.DataFrame(X).astype(object)
    return X.where(X != "unknown", np.nan)


def build_preprocessor(scaler="standard", clipping=False, cat_unknown="category",
                       clip_lower=1.0, clip_upper=99.0):
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if clipping:
        num_steps.append(("clip", PercentileClipper(clip_lower, clip_upper)))
    num_steps.append(("scale", SCALERS[scaler]()))

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if cat_unknown == "category":
        cat = Pipeline([("onehot", ohe)])  # "unknown" ayrı bir kategori olarak kalır
    elif cat_unknown == "mode":
        cat = Pipeline([
            ("unknown_to_nan", FunctionTransformer(unknown_to_nan, feature_names_out="one-to-one")),
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", ohe),
        ])
    else:
        raise ValueError(cat_unknown)

    return ColumnTransformer(
        [
            ("num", Pipeline(num_steps), NUMERIC),
            ("cat", cat, CATEGORICAL),
            ("bin", "passthrough", ENGINEERED_BINARY),
        ],
        remainder="drop",
    )


def preprocessor_for(cfg):
    return build_preprocessor(
        scaler=cfg["scaler"],
        clipping=bool(cfg.get("clipping", False)),
        cat_unknown=cfg.get("cat_unknown", "category"),
    )


def learned_statistics(pre):
    """Fit edilmiş ColumnTransformer'dan öğrenilmiş istatistikleri JSON'a uygun biçimde çıkarır."""
    out = {"feature_names_out": pre.get_feature_names_out().tolist()}
    num = pre.named_transformers_["num"]
    out["numeric_columns"] = list(NUMERIC)
    out["numeric_median"] = num.named_steps["impute"].statistics_.tolist()
    if "clip" in num.named_steps:
        out["clip_lower"] = num.named_steps["clip"].lower_.tolist()
        out["clip_upper"] = num.named_steps["clip"].upper_.tolist()
    sc = num.named_steps["scale"]
    for attr in ("mean_", "scale_", "var_", "center_", "data_min_", "data_max_"):
        if hasattr(sc, attr) and getattr(sc, attr) is not None:
            out[f"scaler_{attr.rstrip('_')}"] = np.asarray(getattr(sc, attr)).tolist()
    cat = pre.named_transformers_["cat"]
    out["categories"] = {c: list(map(str, cats)) for c, cats in zip(CATEGORICAL, cat.named_steps["onehot"].categories_)}
    if "impute" in cat.named_steps:
        out["categorical_mode"] = dict(zip(CATEGORICAL, map(str, cat.named_steps["impute"].statistics_)))
    return out

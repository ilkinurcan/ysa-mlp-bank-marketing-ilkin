"""Veri indirme, doğrulama, özellik hazırlama, kopya grupları ve bölünmeler."""
from __future__ import annotations

import io
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split

from .config import (
    CATEGORICAL, DATA_DIR, DROP_COLUMNS, ENGINEERED_BINARY, EXPECTED_COLUMNS,
    EXPECTED_N_ROWS, NUMERIC, RESULTS_DIR, TARGET,
)

UCI_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
RAW_CSV = DATA_DIR / "bank-additional.csv"
SPLIT_CSV = RESULTS_DIR / "dataset_split.csv"
SPLIT_SUMMARY_CSV = RESULTS_DIR / "dataset_split_summary.csv"


# ---------------------------------------------------------------- indirme
def _find_in_zip(zbytes, target="bank-additional.csv", depth=0):
    with zipfile.ZipFile(io.BytesIO(zbytes)) as z:
        names = z.namelist()
        for name in names:
            if name.split("/")[-1] == target and "__MACOSX" not in name:
                return z.read(name)
        if depth < 3:
            for name in names:
                if name.lower().endswith(".zip") and "__MACOSX" not in name:
                    found = _find_in_zip(z.read(name), target, depth + 1)
                    if found is not None:
                        return found
    return None


def download_dataset(force=False):
    """UCI arşivinden yalnızca bank-additional.csv dosyasını çıkarır (full sürüm kullanılmaz)."""
    if RAW_CSV.exists() and not force:
        return RAW_CSV
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(UCI_URL, timeout=120) as r:
            outer = r.read()
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            f"İndirme başarısız ({e}). {UCI_URL} adresinden zip'i indirip içindeki "
            f"bank-additional/bank-additional.csv dosyasını {RAW_CSV} yoluna koy."
        ) from e
    content = _find_in_zip(outer)
    if content is None:
        raise RuntimeError("Zip içinde bank-additional.csv bulunamadı.")
    RAW_CSV.write_bytes(content)
    return RAW_CSV


# ---------------------------------------------------------------- yükleme ve doğrulama
def load_raw():
    df = pd.read_csv(RAW_CSV, sep=";")
    df.index.name = "original_index"  # dosyadaki 0-tabanlı satır sırası
    return df


def validate_raw(df):
    checks = {
        "n_rows": len(df),
        "n_rows_expected": EXPECTED_N_ROWS,
        "n_columns": df.shape[1],
        "columns_match": list(df.columns) == EXPECTED_COLUMNS,
        "target_labels": sorted(df[TARGET].unique().tolist()),
        "numeric_dtypes_ok": all(pd.api.types.is_numeric_dtype(df[c]) for c in NUMERIC + ["duration"]),
        "categorical_dtypes_ok": all(not pd.api.types.is_numeric_dtype(df[c]) for c in CATEGORICAL),
        "n_nan_cells": int(df.isna().sum().sum()),
    }
    ok = (
        checks["n_rows"] == EXPECTED_N_ROWS
        and checks["columns_match"]
        and checks["target_labels"] == ["no", "yes"]
        and checks["numeric_dtypes_ok"]
        and checks["categorical_dtypes_ok"]
    )
    checks["all_ok"] = bool(ok)
    return checks


def encode_target(df):
    return (df[TARGET] == "yes").astype(int).rename("y")


def prepare_features(df):
    """Hedeften bağımsız, satır-bazlı (fit gerektirmeyen) sabit kurallar.

    - duration çıkarılır (önceden alınmış uygulama kuralı)
    - pdays == 999 -> 'önceden temas yok' göstergesi; pdays değeri NaN yapılır ve
      kat-içinde (fold-local) median ile doldurulur. Satır silinmez.
    """
    X = df.drop(columns=[TARGET] + DROP_COLUMNS).copy()
    X["pdays_no_prev_contact"] = (X["pdays"] == 999).astype(int)
    X["pdays"] = X["pdays"].where(X["pdays"] != 999, np.nan).astype(float)
    return X[NUMERIC + CATEGORICAL + ENGINEERED_BINARY]


# ---------------------------------------------------------------- kopyalar
def duplicate_report(df):
    feats_wo_duration = [c for c in df.columns if c not in DROP_COLUMNS + [TARGET]]
    key = df[feats_wo_duration].astype(str).agg("|".join, axis=1)
    grp = pd.DataFrame({"key": key, "y": df[TARGET]})
    n_label_per_key = grp.groupby("key")["y"].nunique()
    return {
        "exact_duplicate_rows_all_columns": int(df.duplicated().sum()),
        "duplicate_rows_features_wo_duration": int(df[feats_wo_duration].duplicated().sum()),
        "duplicate_rows_features_wo_duration_plus_y": int(df[feats_wo_duration + [TARGET]].duplicated().sum()),
        "groups_with_conflicting_labels": int((n_label_per_key > 1).sum()),
        "n_unique_feature_groups": int(key.nunique()),
    }


def duplicate_groups(df):
    """duration hariç tüm girdi özellikleri aynı olan satırlar aynı grup kimliğini alır."""
    feats_wo_duration = [c for c in df.columns if c not in DROP_COLUMNS + [TARGET]]
    key = df[feats_wo_duration].astype(str).agg("|".join, axis=1)
    codes, _ = pd.factorize(key)
    return pd.Series(codes, index=df.index, name="duplicate_group")


# ---------------------------------------------------------------- bölünmeler
def stratified_group_holdout(y, groups, test_size, seed):
    """Katmanlı hold-out; aynı kopya grubundaki satırlar aynı tarafta kalır.

    Kopya yoksa (tüm gruplar tekil) sonuç sıradan stratified train_test_split ile aynıdır.
    """
    y = np.asarray(y)
    groups = np.asarray(groups)
    g = pd.DataFrame({"g": groups, "y": y}).groupby("g")["y"].mean()
    g_ids = g.index.to_numpy()
    g_y = (g.to_numpy() >= 0.5).astype(int)
    g_tr, g_te = train_test_split(g_ids, test_size=test_size, stratify=g_y, random_state=seed)
    te_mask = np.isin(groups, g_te)
    return np.flatnonzero(~te_mask), np.flatnonzero(te_mask)


def make_kfold_splits(y, groups, n_splits, seed):
    """Kopya yoksa StratifiedKFold; kopya varsa aynı gruplar tek kata düşsün diye StratifiedGroupKFold."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    dummy_X = np.zeros(len(y))
    if len(np.unique(groups)) == len(groups):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return [(tr, va) for tr, va in cv.split(dummy_X, y)]
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(tr, va) for tr, va in cv.split(dummy_X, y, groups)]


def create_split(seeds, test_size=0.20):
    df = load_raw()
    y = encode_target(df)
    groups = duplicate_groups(df)
    dev_pos, test_pos = stratified_group_holdout(y.values, groups.values, test_size, seeds["split"])
    split = pd.Series("development", index=df.index, name="split")
    split.iloc[test_pos] = "test"
    out = pd.DataFrame({"original_index": df.index, "split": split.values, "duplicate_group": groups.values})
    if SPLIT_CSV.exists():
        old = pd.read_csv(SPLIT_CSV)
        if not old[["original_index", "split"]].equals(out[["original_index", "split"]]):
            raise RuntimeError("Mevcut dataset_split.csv yeniden üretilen bölünmeyle uyuşmuyor (seed değişti mi?).")
    out.to_csv(SPLIT_CSV, index=False)
    summary = (
        pd.DataFrame({"split": split.values, "y": y.values})
        .groupby("split")["y"].agg(n="size", n_positive="sum")
        .assign(n_negative=lambda d: d["n"] - d["n_positive"])
        .assign(positive_rate=lambda d: d["n_positive"] / d["n"])
        .assign(share_of_total=lambda d: d["n"] / len(df))
        .reset_index()
    )
    summary.loc[len(summary)] = ["all", len(df), int(y.sum()), int(len(df) - y.sum()), y.mean(), 1.0]
    summary["split_seed"] = seeds["split"]
    summary.to_csv(SPLIT_SUMMARY_CSV, index=False)
    return out, summary


def _split_frame():
    if not SPLIT_CSV.exists():
        raise RuntimeError("Önce 01_eda.ipynb çalıştırılarak dataset_split.csv üretilmeli.")
    return pd.read_csv(SPLIT_CSV)


def load_development():
    """Development X, y, kopya grupları (orijinal indeksler korunur)."""
    df = load_raw()
    sp = _split_frame()
    idx = sp.loc[sp["split"] == "development", "original_index"].to_numpy()
    sub = df.loc[idx]
    X = prepare_features(sub)
    y = encode_target(sub)
    groups = sp.set_index("original_index").loc[idx, "duplicate_group"]
    return X, y, groups


def load_development_raw():
    df = load_raw()
    sp = _split_frame()
    idx = sp.loc[sp["split"] == "development", "original_index"].to_numpy()
    return df.loc[idx]


def load_test_features():
    """Test girdileri; hedef sütunu daha okunurken atılır (etiketler 03 notebook'una kadar kullanılmaz)."""
    df = load_raw()
    sp = _split_frame()
    idx = sp.loc[sp["split"] == "test", "original_index"].to_numpy()
    sub = df.loc[idx].drop(columns=[TARGET])
    sub[TARGET] = "no"  # prepare_features imzası için yer tutucu; hemen atılır
    return prepare_features(sub)


def load_test_labels():
    """YALNIZCA 03_final_test.ipynb içinde, kilitli tahmin commit'inden sonra çağrılır."""
    df = load_raw()
    sp = _split_frame()
    idx = sp.loc[sp["split"] == "test", "original_index"].to_numpy()
    return encode_target(df.loc[idx]), df.loc[idx]

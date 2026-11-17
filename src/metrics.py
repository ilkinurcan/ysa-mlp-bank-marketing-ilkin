"""Sınıflandırma metrikleri, Mean ± SD özetleri ve bootstrap güven aralıkları."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, auc, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_recall_curve, roc_auc_score,
)

MAIN_METRICS = ["AP", "ROC_AUC", "F1", "Sensitivity", "Specificity", "Balanced_Accuracy",
                "Accuracy", "Precision"]


def classification_metrics(y, proba, threshold=0.5):
    y = np.asarray(y).astype(int)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    p_curve, r_curve, _ = precision_recall_curve(y, proba)
    return {
        # AP: adım fonksiyonlu ortalama hassasiyet (sklearn average_precision_score)
        "AP": average_precision_score(y, proba),
        # PR_AUC_trapz: PR eğrisi altında trapez alanı; AP ile AYNI şey değildir
        "PR_AUC_trapz": auc(r_curve, p_curve),
        "ROC_AUC": roc_auc_score(y, proba),
        "Accuracy": accuracy_score(y, pred),
        "Precision": prec,
        "Sensitivity": sens,
        "Specificity": spec,
        "F1": f1_score(y, pred, zero_division=0),
        "Balanced_Accuracy": (sens + spec) / 2,
        "Brier": brier_score_loss(y, proba),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "threshold": threshold,
    }


def mean_sd(df, group_cols, metrics=MAIN_METRICS, digits=4):
    """Kat sonuçlarını ortalama ± örnek SD (ddof=1) olarak özetler."""
    g = df.groupby(group_cols)[metrics]
    mean, sd, n = g.mean(), g.std(ddof=1), g.size()
    out = pd.DataFrame(index=mean.index)
    for m in metrics:
        out[f"{m}_mean"] = mean[m]
        out[f"{m}_sd"] = sd[m]
        out[m] = [f"{a:.{digits}f} ± {b:.{digits}f}" for a, b in zip(mean[m], sd[m])]
    out["n_folds"] = n
    return out.reset_index()


def bootstrap_ci(y, proba, n_boot=1000, seed=0, threshold=0.5, metrics=("ROC_AUC", "AP", "F1"), alpha=0.05):
    """Yüzdelik bootstrap %95 GA. İki sınıfı içermeyen yeniden örneklemler atılıp yeniden çekilir."""
    y = np.asarray(y).astype(int)
    proba = np.asarray(proba, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = {m: [] for m in metrics}
    n_redraw = 0
    while len(vals[metrics[0]]) < n_boot:
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():
            n_redraw += 1
            continue
        m = classification_metrics(yb, proba[idx], threshold)
        for k in metrics:
            vals[k].append(m[k])
    point = classification_metrics(y, proba, threshold)
    rows = []
    for k in metrics:
        v = np.asarray(vals[k])
        rows.append({
            "metric": k, "point_estimate": point[k],
            "ci_lower": np.percentile(v, 100 * alpha / 2), "ci_upper": np.percentile(v, 100 * (1 - alpha / 2)),
            "bootstrap_mean": v.mean(), "bootstrap_sd": v.std(ddof=1),
            "n_boot": n_boot, "n_redrawn_single_class": n_redraw, "seed": seed,
        })
    return pd.DataFrame(rows), vals

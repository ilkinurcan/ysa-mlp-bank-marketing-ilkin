"""Rapordaki tüm şekiller kodla üretilir (matplotlib)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from .config import FIG_DIR

C_POS, C_NEG = "#c2410c", "#1d4ed8"
PALETTE = ["#1d4ed8", "#c2410c", "#15803d", "#7e22ce", "#b45309", "#0f766e"]
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 160, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9})


def _save(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, bbox_inches="tight")
    return path


# ---------------------------------------------------------------- akış şeması
def flow_diagram(name="fig01_nested_cv_flow.png"):
    fig, ax = plt.subplots(figsize=(11, 7.2))
    ax.set_xlim(0, 100), ax.set_ylim(0, 100), ax.axis("off")

    def box(x, y, w, h, text, fc="#eef2ff", ec="#1d4ed8", fs=8.5, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", wrap=True)

    def arrow(x1, y1, x2, y2, txt=None):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.1, color="#334155"))
        if txt:
            ax.text((x1 + x2) / 2 + 1, (y1 + y2) / 2, txt, fontsize=7.5, color="#334155")

    box(30, 91, 40, 6, "bank-additional.csv (4.119 satır) · 'duration' çıkarıldı", bold=True)
    arrow(50, 91, 50, 87)
    box(30, 81, 40, 6, "Stratified split (seed=split)", fc="#f1f5f9", ec="#475569")
    arrow(42, 81, 22, 76)
    arrow(58, 81, 80, 76)
    box(4, 70, 36, 6, "Development %80 → EDA, nested CV, nihai eğitim", fc="#ecfdf5", ec="#15803d", bold=True)
    box(64, 70, 32, 6, "Bağımsız Test %20 (kilitli, saklı)", fc="#fff7ed", ec="#c2410c", bold=True)

    ax.add_patch(FancyBboxPatch((2, 22), 60, 45, boxstyle="round,pad=0.4", fc="none", ec="#15803d", ls="--"))
    ax.text(4, 64.5, "Outer StratifiedKFold (3 kat) — her kat için:", fontsize=8.5, color="#15803d", fontweight="bold")
    box(4, 56, 26, 6, "Outer Train (2/3)", fc="#ecfdf5", ec="#15803d")
    box(34, 56, 26, 6, "Outer Evaluation (1/3)\nyalnızca ölçüm", fc="#fefce8", ec="#a16207")
    ax.add_patch(FancyBboxPatch((4, 25), 56, 28, boxstyle="round,pad=0.3", fc="none", ec="#1d4ed8", ls=":"))
    ax.text(6, 50.5, "Inner StratifiedKFold (3 kat) · 6 manuel (M1–M6) + 6 otomatik (R1–R6)", fontsize=8, color="#1d4ed8")
    box(6, 41, 16, 7, "Inner Train\n→ fit %85\n(ön işleme + ağırlık)", fs=7.5)
    box(24, 41, 16, 7, "Inner Train\n→ ES-monitor %15\n(BCE: ES + LR drop)", fs=7.5)
    box(42, 41, 16, 7, "Inner Validation\n→ AP (aday skoru)", fs=7.5)
    box(6, 28, 52, 9, "Her aramada en yüksek iç CV mean AP → seçilen aday outer-train'de yeniden eğitilir\n"
                      "→ outer-eval'da bir kez ölçülür (Dummy ve LogisticRegression da aynı katlarda)", fs=7.5)
    arrow(17, 56, 17, 53)

    box(66, 50, 30, 10, "Outer kat sonuçları\nMean ± SD (n=3)\nmanuel / otomatik / referans", fc="#f8fafc", ec="#475569")
    arrow(62, 40, 66, 52)
    box(66, 33, 30, 12, "Tam development'ta yeniden\ninner CV (3 × 12 aday)\n→ nihai yöntem + konfigürasyon", fc="#ecfdf5", ec="#15803d")
    box(66, 18, 30, 11, "Nihai eğitim (3 seed)\n→ test_predictions_locked.csv\n(etiketsiz, ayrı commit)", fc="#fff7ed", ec="#c2410c")
    arrow(81, 33, 81, 29)
    box(66, 4, 30, 10, "Test metrikleri, ROC/PR,\nconfusion matrix, bootstrap %95 GA", fc="#fff7ed", ec="#c2410c", bold=True)
    arrow(81, 18, 81, 14)
    ax.text(2, 12, "Fit işlemleri (imputer, clipper, scaler, encoder) yalnızca ilgili 'fit' alt kümesinde yapılır;\n"
                   "outer-eval ve test hiçbir seçim/fit için kullanılmaz.", fontsize=8, color="#7f1d1d")
    return _save(fig, name)


def mlp_architecture(n_in, hidden, activation, dropout, n_params, name="fig03_mlp_architecture.png"):
    sizes = [n_in, *hidden, 1]
    labels = [f"Girdi\n{n_in}"] + [f"Gizli {i+1}\n{h} ({activation})" + (f"\nDropout {dropout}" if dropout else "")
                                   for i, h in enumerate(hidden)] + ["Çıkış\n1 logit\n→ sigmoid"]
    fig, ax = plt.subplots(figsize=(2.6 * len(sizes) + 1, 4))
    ax.axis("off")
    xs = np.arange(len(sizes)) * 2.5
    maxn = 12
    for li, (x, n) in enumerate(zip(xs, sizes)):
        k = min(n, maxn)
        ys = np.linspace(-1, 1, k) if k > 1 else np.array([0.0])
        for y in ys:
            ax.add_patch(plt.Circle((x, y), 0.07, color=PALETTE[min(li, 2)], zorder=3))
        if n > maxn:
            ax.text(x + 0.15, 0, "⋮", fontsize=14, va="center")
        ax.text(x, -1.45, labels[li], ha="center", va="top", fontsize=8)
        if li < len(sizes) - 1:
            k2 = min(sizes[li + 1], maxn)
            ys2 = np.linspace(-1, 1, k2) if k2 > 1 else np.array([0.0])
            for y1 in ys:
                for y2 in ys2:
                    ax.plot([x, xs[li + 1]], [y1, y2], color="#94a3b8", lw=0.25, zorder=1)
    ax.set_xlim(-0.8, xs[-1] + 0.8), ax.set_ylim(-2.4, 1.4)
    ax.set_title(f"MLP mimarisi — öğrenilebilir parametre sayısı: {n_params:,}".replace(",", "."), fontsize=10)
    return _save(fig, name)


# ---------------------------------------------------------------- EDA
def class_distribution(y, name="fig02a_class_distribution.png"):
    counts = pd.Series(y).value_counts().reindex([0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    bars = ax.bar(["no (0)", "yes (1)"], counts.values, color=[C_NEG, C_POS])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v}\n(%{100 * v / counts.sum():.1f})", ha="center", va="bottom")
    ax.set_ylabel("Gözlem sayısı"), ax.set_title("Development: sınıf dağılımı")
    ax.set_ylim(0, counts.max() * 1.2)
    return _save(fig, name)


def numeric_distributions(df, cols, y, name="fig02b_numeric_hist_box.png"):
    n = len(cols)
    fig, axes = plt.subplots(2, n, figsize=(2.3 * n, 4.8))
    for i, c in enumerate(cols):
        v = df[c].dropna()
        axes[0, i].hist(v, bins=30, color="#64748b")
        axes[0, i].set_title(c, fontsize=8.5)
        data = [df.loc[y == 0, c].dropna(), df.loc[y == 1, c].dropna()]
        try:
            bp = axes[1, i].boxplot(data, tick_labels=["no", "yes"], patch_artist=True, widths=0.6,
                                    flierprops=dict(markersize=2))
        except TypeError:  # matplotlib < 3.9
            bp = axes[1, i].boxplot(data, labels=["no", "yes"], patch_artist=True, widths=0.6,
                                    flierprops=dict(markersize=2))
        for patch, col in zip(bp["boxes"], [C_NEG, C_POS]):
            patch.set_facecolor(col), patch.set_alpha(0.45)
    axes[0, 0].set_ylabel("Frekans"), axes[1, 0].set_ylabel("Değer (hedefe göre)")
    fig.suptitle("Development: sayısal özellikler — histogram (üst) ve hedefe göre boxplot (alt)", fontsize=10)
    fig.tight_layout()
    return _save(fig, name)


def categorical_bars(df, cols, y, name="fig02c_categorical_bars.png"):
    n = len(cols)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3.1 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, cols):
        vc = df[c].value_counts()
        rate = pd.Series(y.values, index=df.index).groupby(df[c]).mean().reindex(vc.index)
        cols_ = ["#a16207" if k == "unknown" else "#64748b" for k in vc.index]
        ax.bar(vc.index.astype(str), vc.values, color=cols_)
        ax.set_title(f"{c} (sarı = 'unknown')", fontsize=9)
        ax.tick_params(axis="x", rotation=45, labelsize=7.5)
        ax2 = ax.twinx()
        ax2.plot(vc.index.astype(str), rate.values * 100, "o-", color=C_POS, ms=3, lw=1)
        ax2.set_ylabel("% yes", color=C_POS, fontsize=7.5)
        ax2.spines["top"].set_visible(False)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Development: kategorik frekanslar (çubuk) ve kategori içi pozitif oranı (çizgi, betimsel)", fontsize=10)
    fig.tight_layout()
    return _save(fig, name)


def correlation_matrix(df, cols, name="fig02d_correlation.png", method="pearson"):
    corr = df[cols].corr(method=method)
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if abs(v) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"Development: sayısal özellik korelasyonu ({method})")
    fig.tight_layout()
    return _save(fig, name), corr


# ---------------------------------------------------------------- eğitim eğrileri
def learning_curves(histories, name="fig04_learning_curves.png", title="Eğitim ve iç early-stop kaybı; epoch bazlı LR"):
    """histories: {etiket: DataFrame(epoch, train_loss, val_loss, lr)}"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for i, (lab, h) in enumerate(histories.items()):
        col = PALETTE[i % len(PALETTE)]
        axes[0].plot(h.epoch, h.train_loss, color=col, lw=1, ls="--", alpha=0.8)
        axes[0].plot(h.epoch, h.val_loss, color=col, lw=1.4, label=lab)
        best = h.val_loss.idxmin()
        axes[0].scatter(h.epoch[best], h.val_loss[best], color=col, s=18, zorder=3)
        axes[1].step(h.epoch, h.lr, where="post", color=col, lw=1.3, label=lab)
    axes[0].set_xlabel("Epoch"), axes[0].set_ylabel("BCE kaybı")
    axes[0].set_title("Kesikli: eğitim · düz: iç ES-monitor · nokta: en iyi epoch", fontsize=8.5)
    axes[0].legend(fontsize=7.5)
    axes[1].set_yscale("log"), axes[1].set_xlabel("Epoch"), axes[1].set_ylabel("Öğrenme oranı (log)")
    axes[1].set_title("ReduceLROnPlateau (factor=0.5, patience=5, min_lr=1e-5)", fontsize=8.5)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return _save(fig, name)


# ---------------------------------------------------------------- arama karşılaştırma
def search_comparison(inner_summary, outer, name="fig05_manual_vs_random.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    cands = list(dict.fromkeys(inner_summary["candidate"]))
    folds = sorted(inner_summary["outer_fold"].unique())
    w = 0.8 / len(folds)
    for fi, f in enumerate(folds):
        s = inner_summary[inner_summary.outer_fold == f].set_index("candidate").reindex(cands)
        x = np.arange(len(cands)) + (fi - (len(folds) - 1) / 2) * w
        ax.bar(x, s.inner_AP_mean, w, yerr=s.inner_AP_sd, capsize=2, color=PALETTE[fi], alpha=0.8,
               label=f"outer kat {f}", error_kw=dict(lw=0.7))
        sel = s.selected.fillna(False).astype(bool).to_numpy()
        ax.scatter(x[sel], s.inner_AP_mean[sel] + s.inner_AP_sd[sel] + 0.01, marker="v", color="k", s=14)
    ax.axvline(5.5, color="#94a3b8", ls="--")
    ax.set_xticks(range(len(cands)), cands)
    ax.set_ylabel("İç CV mean AP ± SD"), ax.set_title("Adayların iç CV AP değeri (▼ = o katta seçilen)")
    lo = np.nanmin(inner_summary.inner_AP_mean - inner_summary.inner_AP_sd)
    ax.set_ylim(max(0, lo - 0.05), None)
    ax.legend(fontsize=7.5, ncol=3)

    ax = axes[1]
    methods = ["MLP_manual_search", "MLP_random_search", "LogisticRegression", "Dummy_most_frequent"]
    for i, m in enumerate(methods):
        v = outer.loc[outer.method == m, "AP"].to_numpy()
        ax.scatter(np.full(len(v), i), v, color=PALETTE[i], s=22, zorder=3)
        ax.errorbar(i + 0.15, v.mean(), yerr=v.std(ddof=1), fmt="s", color="k", capsize=3, ms=4)
    ax.set_xticks(range(len(methods)), ["Manuel", "Otomatik", "LogReg", "Dummy"])
    ax.set_ylabel("Outer evaluation AP"), ax.set_title("Dış kat AP (nokta: kat, kare: mean ± SD)")
    fig.tight_layout()
    return _save(fig, name)


# ---------------------------------------------------------------- test şekilleri
def confusion_matrices(results, name="fig06_confusion_matrices.png"):
    """results: {etiket: (y_true, y_pred)}"""
    fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 3.6))
    axes = np.atleast_1d(axes)
    for ax, (lab, (yt, yp)) in zip(axes, results.items()):
        cm = confusion_matrix(yt, yp, labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n(%{100 * cm[i, j] / cm[i].sum():.1f} satır)", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8.5)
        ax.set_xticks([0, 1], ["Tahmin: no", "Tahmin: yes"]), ax.set_yticks([0, 1], ["Gerçek: no", "Gerçek: yes"])
        ax.set_title(lab, fontsize=9.5)
    fig.suptitle("Bağımsız test — confusion matrix (eşik = 0.50)", fontsize=10)
    fig.tight_layout()
    return _save(fig, name)


def roc_pr_curves(y, probas, name="fig07_roc_pr.png", ap_values=None, auc_values=None):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for i, (lab, p) in enumerate(probas.items()):
        col = PALETTE[i % len(PALETTE)]
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        a = f" (AUC={auc_values[lab]:.3f})" if auc_values else ""
        b = f" (AP={ap_values[lab]:.3f})" if ap_values else ""
        axes[0].plot(fpr, tpr, color=col, lw=1.5, label=lab + a)
        axes[1].step(rec, prec, where="post", color=col, lw=1.5, label=lab + b)
    axes[0].plot([0, 1], [0, 1], ls="--", color="#94a3b8", lw=1)
    prev = np.mean(y)
    axes[1].axhline(prev, ls="--", color="#94a3b8", lw=1, label=f"Pozitif oranı = {prev:.3f}")
    axes[0].set_xlabel("1 − Specificity (FPR)"), axes[0].set_ylabel("Sensitivity (TPR)"), axes[0].set_title("ROC eğrisi")
    axes[1].set_xlabel("Recall"), axes[1].set_ylabel("Precision"), axes[1].set_title("Precision–Recall eğrisi")
    for ax in axes:
        ax.legend(fontsize=7.5, loc="lower right" if ax is axes[0] else "upper right")
    fig.suptitle("Bağımsız test — ROC ve PR eğrileri", fontsize=10)
    fig.tight_layout()
    return _save(fig, name)


def calibration_plot(y, probas, name="fig08_calibration.png", n_bins=10):
    from sklearn.calibration import calibration_curve

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for i, (lab, p) in enumerate(probas.items()):
        frac, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
        ax.plot(mean_pred, frac, "o-", ms=3, color=PALETTE[i], label=lab)
    ax.plot([0, 1], [0, 1], ls="--", color="#94a3b8")
    ax.set_xlabel("Ortalama tahmin olasılığı"), ax.set_ylabel("Gözlenen pozitif oranı")
    ax.set_title("Kalibrasyon eğrisi (test, betimsel)"), ax.legend(fontsize=7.5)
    return _save(fig, name)


def seed_stability_plot(df, name="fig09_seed_stability.png", metrics=("AP", "ROC_AUC", "F1")):
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for i, m in enumerate(metrics):
        ax.scatter(np.full(len(df), i), df[m], color=PALETTE[i], s=26)
        ax.errorbar(i + 0.15, df[m].mean(), yerr=df[m].std(ddof=1), fmt="s", color="k", capsize=3, ms=4)
    ax.set_xticks(range(len(metrics)), metrics), ax.set_title("Üç başlangıç seed'inin test metrikleri")
    return _save(fig, name)

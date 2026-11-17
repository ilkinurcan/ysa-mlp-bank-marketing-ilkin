"""Aday konfigürasyonlar, inner CV değerlendirmesi, nested CV, referans modeller ve nihai eğitim."""
from __future__ import annotations

import hashlib
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline

from .config import CACHE_DIR, CONFIG_DIR, HIST_DIR, MODELS_DIR, RESULTS_DIR, load_json, save_json
from .data import make_kfold_splits, stratified_group_holdout
from .metrics import classification_metrics
from .model import parameter_formula, predict_proba, save_model, train_mlp
from .preprocessing import build_preprocessor, learned_statistics, preprocessor_for

HP_KEYS = ["optimizer", "hidden_layers", "activation", "lr", "weight_decay", "batch_size",
           "scaler", "dropout", "clipping", "scheduler", "cat_unknown"]


# ================================================================ adaylar
def load_manual_candidates():
    return load_json(CONFIG_DIR / "manual_candidates.json")["candidates"]


def generate_random_candidates(seed):
    """Optimizer başına tam 2 aday; LR log-ölçekte optimizer'a özgü aralıktan örneklenir."""
    space = load_json(CONFIG_DIR / "search_space.json")
    cands, k = [], 1
    for oi, opt in enumerate(space["optimizers"]):
        lo, hi = space["lr_range"][opt]
        dist = {
            "hidden_layers": space["hidden_layers"],
            "activation": space["activation"],
            "lr": loguniform(lo, hi),
            "weight_decay": space["weight_decay"],
            "batch_size": space["batch_size"],
            "scaler": space["scaler"],
            "dropout": space["dropout"],
            "clipping": space["clipping"],
        }
        for params in ParameterSampler(dist, n_iter=space["n_per_optimizer"], random_state=seed + oi):
            cands.append({
                "id": f"R{k}",
                "optimizer": opt,
                "hidden_layers": list(params["hidden_layers"]),
                "activation": params["activation"],
                "lr": float(f"{params['lr']:.6g}"),
                "weight_decay": params["weight_decay"],
                "batch_size": int(params["batch_size"]),
                "scaler": params["scaler"],
                "dropout": float(params["dropout"]),
                "clipping": bool(params["clipping"]),
                "scheduler": space["fixed"]["scheduler"],
                "cat_unknown": space["fixed"]["cat_unknown"],
                "purpose": f"Random Search ({opt}, örnek {len([c for c in cands if c['optimizer'] == opt]) + 1})",
            })
            k += 1
    return cands


def preregister_random_candidates(seeds, overwrite=False):
    """Adaylar nested CV başlamadan önce sabitlenip kaydedilir; dosya varsa yeniden üretilmez."""
    path = CONFIG_DIR / "random_candidates.json"
    if path.exists() and not overwrite:
        saved = load_json(path)
        if saved.get("random_search_seed") != seeds["random_search"]:
            raise RuntimeError("random_candidates.json farklı bir seed ile üretilmiş.")
        return saved["candidates"]
    cands = generate_random_candidates(seeds["random_search"])
    save_json({
        "random_search_seed": seeds["random_search"],
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sampler": "sklearn.model_selection.ParameterSampler (+ scipy.stats.loguniform for lr)",
        "candidates": cands,
    }, path)
    return cands


def candidates_table(cands):
    rows = []
    for c in cands:
        r = {k: c.get(k) for k in ["id"] + HP_KEYS}
        r["hidden_layers"] = str(tuple(c["hidden_layers"]))
        r["purpose"] = c.get("purpose", "")
        rows.append(r)
    return pd.DataFrame(rows)


def config_hash(cfg):
    s = json.dumps({k: cfg.get(k) for k in HP_KEYS}, sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()[:10]


# ================================================================ tek eğitim
def fit_mlp_candidate(cfg, X_train, y_train, groups_train, X_eval, seeds, train_cfg, run_key=None):
    """Eğitim bölümünü 'fit' (%85) ve 'early-stop monitor' (%15) alt kümelerine ayırır;
    ön işlemeyi yalnızca 'fit' alt kümesinde öğrenir, MLP'yi eğitir ve X_eval için olasılık döndürür."""
    y_train = np.asarray(y_train)
    fit_pos, es_pos = stratified_group_holdout(
        y_train, np.asarray(groups_train), train_cfg["early_stop_fraction"], seeds["early_stop_split"]
    )
    pre = preprocessor_for(cfg)
    Xf = pre.fit_transform(X_train.iloc[fit_pos])
    Xe = pre.transform(X_train.iloc[es_pos])
    Xv = pre.transform(X_eval)
    model, info, history = train_mlp(Xf, y_train[fit_pos], Xe, y_train[es_pos], cfg, seeds["torch"], train_cfg)
    proba = predict_proba(model, Xv)
    info["n_inputs"] = int(Xf.shape[1])
    if run_key is not None:
        pd.DataFrame(history).to_csv(HIST_DIR / f"{run_key}.csv", index=False)
    return proba, info, model, pre, history


def _cache_get(key):
    p = CACHE_DIR / f"{key}.json"
    return load_json(p) if p.exists() else None


def _cache_put(key, obj):
    save_json(obj, CACHE_DIR / f"{key}.json")


# ================================================================ inner CV
def inner_cv_evaluate(cands, X, y, groups, inner_splits, context, search_name, seeds, train_cfg, verbose=True):
    """Her aday için 3 inner fold'da eğitim; inner validation AP ile skor."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    fold_rows = []
    for c in cands:
        for j, (tr, va) in enumerate(inner_splits):
            key = f"{context}__{c['id']}__inner{j}"
            rec = _cache_get(key)
            if rec is None or rec.get("config_hash") != config_hash(c):
                t0 = time.perf_counter()
                proba, info, *_ = fit_mlp_candidate(
                    c, X.iloc[tr], y[tr], groups[tr], X.iloc[va], seeds, train_cfg, run_key=key
                )
                rec = {
                    "config_hash": config_hash(c),
                    "inner_val_AP": float(average_precision_score(y[va], proba)),
                    "wall_time_s": time.perf_counter() - t0,
                    **info,
                }
                _cache_put(key, rec)
            fold_rows.append({"context": context, "search": search_name, "candidate": c["id"],
                              "inner_fold": j, "n_inner_train": len(tr), "n_inner_val": len(va),
                              **{k: v for k, v in rec.items() if k != "config_hash"}})
        if verbose:
            aps = [r["inner_val_AP"] for r in fold_rows if r["candidate"] == c["id"]]
            print(f"  [{context}] {c['id']:>3} ({c['optimizer']:5s}) inner AP = "
                  f"{np.mean(aps):.4f} ± {np.std(aps, ddof=1):.4f}")
    folds = pd.DataFrame(fold_rows)
    summary = (
        folds.groupby(["context", "search", "candidate"], sort=False)
        .agg(inner_AP_mean=("inner_val_AP", "mean"), inner_AP_sd=("inner_val_AP", lambda s: s.std(ddof=1)),
             total_time_s=("wall_time_s", "sum"), mean_best_epoch=("best_epoch", "mean"),
             best_epochs=("best_epoch", lambda s: ",".join(map(str, s))),
             stopped_epochs=("stopped_epoch", lambda s: ",".join(map(str, s))),
             n_params=("n_params", "first"),
             warnings=("warnings", lambda s: "|".join(sorted({w for w in s if w}))))
        .reset_index()
    )
    hp = candidates_table(cands).rename(columns={"id": "candidate"})
    summary = summary.merge(hp, on="candidate", how="left")
    summary["inner_AP"] = [f"{m:.4f} ± {s:.4f}" for m, s in zip(summary.inner_AP_mean, summary.inner_AP_sd)]
    return folds, summary


def select_best(summary, cands):
    """Önceden tanımlı kural: en yüksek inner mean AP; eşitlikte düşük SD; sonra liste sırası."""
    order = {c["id"]: i for i, c in enumerate(cands)}
    s = summary.assign(_o=summary["candidate"].map(order))
    s = s.sort_values(["inner_AP_mean", "inner_AP_sd", "_o"], ascending=[False, True, True])
    best_id = s.iloc[0]["candidate"]
    return next(c for c in cands if c["id"] == best_id)


# ================================================================ referans modeller
def make_logreg(seed, cat_unknown="category"):
    return Pipeline([
        ("pre", build_preprocessor(scaler="standard", clipping=False, cat_unknown=cat_unknown)),
        ("clf", LogisticRegression(max_iter=5000, C=1.0, random_state=seed)),
    ])


def make_dummy():
    return Pipeline([
        ("pre", build_preprocessor(scaler="standard")),
        ("clf", DummyClassifier(strategy="most_frequent")),
    ])


def fit_reference(model, X_train, y_train, X_eval):
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X_train, y_train)
    proba = model.predict_proba(X_eval)[:, 1]
    warn = ";".join(sorted({type(x.message).__name__ for x in w if issubclass(x.category, ConvergenceWarning)}))
    return proba, {"train_time_s": time.perf_counter() - t0, "warnings": warn}


def logreg_unknown_selection(X, y, groups, inner_splits, seeds):
    """Ek analiz: 'unknown'u ayrı kategori mi tutmalı, en sık sınıfla mı doldurmalı?
    Karar her outer-train içinde inner CV ile (LR üzerinde) yeniden alınır."""
    y = np.asarray(y)
    rows = []
    for opt in ("category", "mode"):
        for j, (tr, va) in enumerate(inner_splits):
            p, _ = fit_reference(make_logreg(seeds["logreg"], opt), X.iloc[tr], y[tr], X.iloc[va])
            rows.append({"cat_unknown": opt, "inner_fold": j, "inner_val_AP": average_precision_score(y[va], p)})
    df = pd.DataFrame(rows)
    s = df.groupby("cat_unknown")["inner_val_AP"].agg(["mean", lambda v: v.std(ddof=1)])
    s.columns = ["inner_AP_mean", "inner_AP_sd"]
    choice = s.sort_values(["inner_AP_mean", "inner_AP_sd"], ascending=[False, True]).index[0]
    return choice, s.reset_index()


# ================================================================ nested CV
def run_nested_cv(X_dev, y_dev, groups_dev, manual, random_, seeds, train_cfg, n_outer=3, n_inner=3):
    y_dev = np.asarray(y_dev)
    groups_dev = np.asarray(groups_dev)
    outer_splits = make_kfold_splits(y_dev, groups_dev, n_outer, seeds["outer_cv"])
    all_folds, all_summ, outer_rows, pred_rows, unk_rows, split_rows = [], [], [], [], [], []

    for k, (otr, oev) in enumerate(outer_splits):
        print(f"=== Outer fold {k}: outer-train={len(otr)}, outer-eval={len(oev)}, "
              f"pozitif oranı train={y_dev[otr].mean():.3f} eval={y_dev[oev].mean():.3f}")
        Xtr, ytr, gtr = X_dev.iloc[otr], y_dev[otr], groups_dev[otr]
        Xev, yev = X_dev.iloc[oev], y_dev[oev]
        inner_splits = make_kfold_splits(ytr, gtr, n_inner, seeds["inner_cv"])
        for j, (itr, iva) in enumerate(inner_splits):
            split_rows.append({"outer_fold": k, "inner_fold": j, "n_inner_train": len(itr), "n_inner_val": len(iva),
                               "inner_val_pos_rate": ytr[iva].mean()})
        ctx = f"outer{k}"
        f_m, s_m = inner_cv_evaluate(manual, Xtr, ytr, gtr, inner_splits, ctx, "manual", seeds, train_cfg)
        f_r, s_r = inner_cv_evaluate(random_, Xtr, ytr, gtr, inner_splits, ctx, "random", seeds, train_cfg)
        best_m, best_r = select_best(s_m, manual), select_best(s_r, random_)
        s_m["selected"] = s_m["candidate"] == best_m["id"]
        s_r["selected"] = s_r["candidate"] == best_r["id"]
        all_folds += [f_m, f_r]
        all_summ += [s_m.assign(outer_fold=k), s_r.assign(outer_fold=k)]

        # seçilen adayların outer-train üzerinde yeniden eğitimi, outer-eval üzerinde TEK ölçüm
        for method, cand, summ in (("MLP_manual_search", best_m, s_m), ("MLP_random_search", best_r, s_r)):
            key = f"{ctx}__refit_{method}__{cand['id']}"
            t0 = time.perf_counter()
            proba, info, *_ = fit_mlp_candidate(cand, Xtr, ytr, gtr, Xev, seeds, train_cfg, run_key=key)
            m = classification_metrics(yev, proba, train_cfg["threshold"])
            inner_ap = float(summ.loc[summ.candidate == cand["id"], "inner_AP_mean"].iloc[0])
            outer_rows.append({"outer_fold": k, "method": method, "selected_candidate": cand["id"],
                               "inner_AP_of_selected": inner_ap, **m,
                               "train_time_s": info["train_time_s"], "best_epoch": info["best_epoch"],
                               "stopped_epoch": info["stopped_epoch"], "warnings": info["warnings"]})
            pred_rows.append(pd.DataFrame({"original_index": Xev.index, "outer_fold": k, "method": method,
                                           "y_true": yev, "proba": proba}))
            print(f"  -> {method}: {cand['id']} | outer AP={m['AP']:.4f} ROC-AUC={m['ROC_AUC']:.4f} F1={m['F1']:.4f}")

        # referans modeller: yalnızca outer-train ile eğitilir, outer-eval üzerinde ölçülür
        for method, mdl in (("Dummy_most_frequent", make_dummy()), ("LogisticRegression", make_logreg(seeds["logreg"]))):
            proba, info = fit_reference(mdl, Xtr, ytr, Xev)
            m = classification_metrics(yev, proba, train_cfg["threshold"])
            outer_rows.append({"outer_fold": k, "method": method, "selected_candidate": "-",
                               "inner_AP_of_selected": np.nan, **m, **info})
            pred_rows.append(pd.DataFrame({"original_index": Xev.index, "outer_fold": k, "method": method,
                                           "y_true": yev, "proba": proba}))

        # ek analiz: 'unknown' kodlaması kararının inner CV ile seçilmesi (LR üzerinde)
        choice, s_unk = logreg_unknown_selection(Xtr, ytr, gtr, inner_splits, seeds)
        proba, info = fit_reference(make_logreg(seeds["logreg"], choice), Xtr, ytr, Xev)
        m = classification_metrics(yev, proba, train_cfg["threshold"])
        outer_rows.append({"outer_fold": k, "method": "LogisticRegression_unknown_selected",
                           "selected_candidate": f"cat_unknown={choice}", "inner_AP_of_selected":
                           float(s_unk.loc[s_unk.cat_unknown == choice, "inner_AP_mean"].iloc[0]), **m, **info})
        unk_rows.append(s_unk.assign(outer_fold=k, selected=lambda d: d.cat_unknown == choice))

    res = {
        "inner_folds": pd.concat(all_folds, ignore_index=True),
        "inner_summary": pd.concat(all_summ, ignore_index=True),
        "outer": pd.DataFrame(outer_rows),
        "outer_predictions": pd.concat(pred_rows, ignore_index=True),
        "unknown_selection": pd.concat(unk_rows, ignore_index=True),
        "split_sizes": pd.DataFrame(split_rows),
        "outer_split_sizes": pd.DataFrame([{"outer_fold": k, "n_outer_train": len(a), "n_outer_eval": len(b),
                                            "outer_train_pos_rate": y_dev[a].mean(), "outer_eval_pos_rate": y_dev[b].mean()}
                                           for k, (a, b) in enumerate(outer_splits)]),
    }
    return res


# ================================================================ development üzerinde yeniden seçim
def development_reselection(X_dev, y_dev, groups_dev, manual, random_, seeds, train_cfg, n_inner=3):
    y_dev = np.asarray(y_dev)
    inner_splits = make_kfold_splits(y_dev, np.asarray(groups_dev), n_inner, seeds["inner_cv"])
    f_m, s_m = inner_cv_evaluate(manual, X_dev, y_dev, groups_dev, inner_splits, "dev", "manual", seeds, train_cfg)
    f_r, s_r = inner_cv_evaluate(random_, X_dev, y_dev, groups_dev, inner_splits, "dev", "random", seeds, train_cfg)
    best_m, best_r = select_best(s_m, manual), select_best(s_r, random_)
    s_m["selected_within_search"] = s_m.candidate == best_m["id"]
    s_r["selected_within_search"] = s_r.candidate == best_r["id"]
    ap_m = float(s_m.loc[s_m.selected_within_search, "inner_AP_mean"].iloc[0])
    sd_m = float(s_m.loc[s_m.selected_within_search, "inner_AP_sd"].iloc[0])
    ap_r = float(s_r.loc[s_r.selected_within_search, "inner_AP_mean"].iloc[0])
    sd_r = float(s_r.loc[s_r.selected_within_search, "inner_AP_sd"].iloc[0])
    # önceden tanımlı kural: yüksek dev inner mean AP; eşitlikte düşük SD; yine eşitse manuel
    if (ap_r > ap_m) or (ap_r == ap_m and sd_r < sd_m):
        method, final = "random_search", best_r
    else:
        method, final = "manual_search", best_m
    decision = {
        "rule": "Tam development üzerinde 3-fold inner CV; en yüksek mean AP (eşitlikte düşük SD, sonra manuel).",
        "manual_best": best_m["id"], "manual_best_inner_AP_mean": ap_m, "manual_best_inner_AP_sd": sd_m,
        "random_best": best_r["id"], "random_best_inner_AP_mean": ap_r, "random_best_inner_AP_sd": sd_r,
        "final_method": method, "final_candidate": final["id"],
    }
    return pd.concat([f_m, f_r], ignore_index=True), pd.concat([s_m, s_r], ignore_index=True), final, decision


# ================================================================ nihai eğitim ve kilitli tahminler
def final_training_and_lock(final_cfg, decision, X_dev, y_dev, groups_dev, X_test, seeds, train_cfg):
    """Nihai konfigürasyonu 3 seed ile eğitir, test olasılıklarını etiketsiz olarak kilitler."""
    y_dev = np.asarray(y_dev)
    fit_pos, es_pos = stratified_group_holdout(y_dev, np.asarray(groups_dev),
                                               train_cfg["early_stop_fraction"], seeds["early_stop_split"])
    locked = pd.DataFrame({"original_index": X_test.index})
    seed_rows, histories = [], {}
    thr = train_cfg["threshold"]
    for s in seeds["stability_seeds"]:
        pre = preprocessor_for(final_cfg)
        Xf = pre.fit_transform(X_dev.iloc[fit_pos])
        Xe = pre.transform(X_dev.iloc[es_pos])
        Xt = pre.transform(X_test)
        model, info, hist = train_mlp(Xf, y_dev[fit_pos], Xe, y_dev[es_pos], final_cfg, s, train_cfg)
        proba = predict_proba(model, Xt)
        tag = f"seed{s}"
        save_model(model, MODELS_DIR / f"final_mlp_{tag}.pt")
        joblib.dump(pre, MODELS_DIR / f"final_preprocessor_{tag}.joblib")
        pd.DataFrame(hist).to_csv(HIST_DIR / f"final__{final_cfg['id']}__{tag}.csv", index=False)
        histories[s] = pd.DataFrame(hist)
        locked[f"mlp_{tag}_proba"] = proba
        locked[f"mlp_{tag}_pred"] = (proba >= thr).astype(int)
        seed_rows.append({"seed": s, "is_primary": s == seeds["primary_seed"], **info})
        if s == seeds["primary_seed"]:
            save_json(learned_statistics(pre), RESULTS_DIR / "final_preprocessor_stats.json")
            n_inputs = Xf.shape[1]

    # referans modeller: tam development üzerinde eğitilir
    lr = make_logreg(seeds["logreg"])
    p_lr, info_lr = fit_reference(lr, X_dev, y_dev, X_test)
    joblib.dump(lr, MODELS_DIR / "final_logreg.joblib")
    dm = make_dummy()
    p_dm, _ = fit_reference(dm, X_dev, y_dev, X_test)
    locked["logreg_proba"], locked["logreg_pred"] = p_lr, (p_lr >= thr).astype(int)
    locked["dummy_proba"], locked["dummy_pred"] = p_dm, (p_dm >= thr).astype(int)

    path = RESULTS_DIR / "test_predictions_locked.csv"
    locked.to_csv(path, index=False)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    (RESULTS_DIR / "test_predictions_locked.sha256").write_text(sha + "\n")

    parts, total = parameter_formula(n_inputs, final_cfg["hidden_layers"])
    final_info = {
        "decision": decision,
        "final_config": final_cfg,
        "primary_model": f"mlp_seed{seeds['primary_seed']}",
        "primary_seed": seeds["primary_seed"],
        "stability_seeds": seeds["stability_seeds"],
        "threshold": thr,
        "threshold_policy": "Sabit 0.50 (test veya outer sonuçlarına göre ayarlanmadı).",
        "n_inputs_after_preprocessing": int(n_inputs),
        "n_trainable_params": total,
        "param_breakdown": [{"in": a, "out": b, "params": p} for a, b, p in parts],
        "early_stop_split": {"n_fit": len(fit_pos), "n_es": len(es_pos), "seed": seeds["early_stop_split"]},
        "training_config": train_cfg,
        "seeds": seeds,
        "per_seed_training": seed_rows,
        "logreg_reference": {"C": 1.0, "max_iter": 5000, "preprocessing": "median + StandardScaler + OHE(unknown=kategori)", **info_lr},
        "locked_predictions_sha256": sha,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_json(final_info, RESULTS_DIR / "final_hyperparameters.json")
    return locked, final_info, histories, pd.DataFrame(seed_rows)

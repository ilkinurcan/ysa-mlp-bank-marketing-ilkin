"""PyTorch MLP, eğitim döngüsü (early stopping + ReduceLROnPlateau) ve tahmin."""
from __future__ import annotations

import copy
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn

ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh}
DEVICE = torch.device("cpu")  # küçük veri; CPU determinizm sağlar


class MLP(nn.Module):
    """Gizli katmanlar: Linear -> aktivasyon -> (Dropout). Çıkış: tek logit."""

    def __init__(self, n_in, hidden_layers=(32,), activation="relu", dropout=0.0):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(ACTIVATIONS[activation]())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)  # logit; olasılık için sigmoid uygulanır


def count_parameters(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def parameter_formula(n_in, hidden_layers):
    """Elle hesap: her Linear katman için (girdi * çıktı) ağırlık + çıktı kadar bias."""
    sizes = [n_in, *hidden_layers, 1]
    parts = [(a, b, a * b + b) for a, b in zip(sizes[:-1], sizes[1:])]
    return parts, int(sum(p[2] for p in parts))


def set_torch_seed(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_optimizer(name, params, lr, weight_decay, momentum=0.9):
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)  # L2 (gradyana eklenir)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)  # ayrıştırılmış (decoupled) WD
    raise ValueError(name)


def train_mlp(X_fit, y_fit, X_es, y_es, cfg, seed, train_cfg):
    """MLP'yi 'fit' alt kümesinde eğitir; 'early-stop monitor' alt kümesindeki BCE kaybını izler.

    Dönüş: (en iyi ağırlıkları yüklenmiş model, bilgi sözlüğü, epoch geçmişi listesi)
    """
    set_torch_seed(seed)
    Xf = torch.as_tensor(np.asarray(X_fit, dtype=np.float32), device=DEVICE)
    yf = torch.as_tensor(np.asarray(y_fit, dtype=np.float32), device=DEVICE)
    Xe = torch.as_tensor(np.asarray(X_es, dtype=np.float32), device=DEVICE)
    ye = torch.as_tensor(np.asarray(y_es, dtype=np.float32), device=DEVICE)

    model = MLP(Xf.shape[1], tuple(cfg["hidden_layers"]), cfg["activation"], cfg["dropout"]).to(DEVICE)
    opt = make_optimizer(cfg["optimizer"], model.parameters(), cfg["lr"], cfg["weight_decay"],
                         train_cfg.get("sgd_momentum", 0.9))
    sched = None
    if cfg.get("scheduler", True):
        sc = train_cfg["scheduler"]
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=sc["factor"], patience=sc["patience"], min_lr=sc["min_lr"]
        )
    loss_fn = nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(seed)

    max_epochs = train_cfg["max_epochs"]
    patience = train_cfg["early_stopping_patience"]
    min_delta = train_cfg.get("early_stopping_min_delta", 0.0)
    n, bs = len(Xf), int(cfg["batch_size"])

    best_loss, best_epoch, best_state, wait = math.inf, 0, None, 0
    history, warn = [], []
    stop_reason = "max_epochs"
    t0 = time.perf_counter()
    epoch = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=gen)
        running = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(Xf[idx]), yf[idx])
            loss.backward()
            opt.step()
            running += loss.item() * len(idx)
        train_loss = running / n
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xe), ye).item()
        lr_used = opt.param_groups[0]["lr"]  # bu epoch'ta kullanılan öğrenme oranı
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr_used})

        if not (math.isfinite(train_loss) and math.isfinite(val_loss)):
            warn.append(f"non_finite_loss_at_epoch_{epoch}")
            stop_reason = "non_finite_loss"
            break
        if val_loss < best_loss - min_delta:
            best_loss, best_epoch, wait = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
        if sched is not None:
            sched.step(val_loss)  # her epoch sonunda iç early-stop kaybıyla güncellenir
        if wait >= patience:
            stop_reason = "early_stopping"
            break

    if stop_reason == "max_epochs":
        warn.append("max_epochs_reached_without_early_stop")
    if best_state is not None:
        model.load_state_dict(best_state)  # best weights restore
    else:
        warn.append("no_best_state")
    info = {
        "best_epoch": best_epoch,
        "stopped_epoch": epoch,
        "best_val_bce": best_loss,
        "stop_reason": stop_reason,
        "final_lr": history[-1]["lr"] if history else cfg["lr"],
        "n_lr_reductions": int(sum(1 for a, b in zip(history[:-1], history[1:]) if b["lr"] < a["lr"])),
        "train_time_s": time.perf_counter() - t0,
        "n_params": count_parameters(model),
        "n_fit": n,
        "n_es": len(Xe),
        "warnings": ";".join(warn),
    }
    return model, info, history


@torch.no_grad()
def predict_proba(model, X):
    model.eval()
    Xt = torch.as_tensor(np.asarray(X, dtype=np.float32), device=DEVICE)
    return torch.sigmoid(model(Xt)).cpu().numpy().astype(float)


def save_model(model, path):
    torch.save(model.state_dict(), path)

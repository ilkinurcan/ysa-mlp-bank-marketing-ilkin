"""Yol tanımları, sabitler, seed türetme ve JSON yardımcıları."""
from __future__ import annotations

import json
import os
import platform
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
DATA_DIR = ROOT / "data"
MODELS_DIR = RESULTS_DIR / "models"
HIST_DIR = RESULTS_DIR / "histories"
CACHE_DIR = RESULTS_DIR / "cache"

for _d in (RESULTS_DIR, FIG_DIR, DATA_DIR, MODELS_DIR, HIST_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- veri sözlüğü
TARGET = "y"
DROP_COLUMNS = ["duration"]  # görüşme öncesi bilinmiyor -> tüm modellerden çıkarılır
NUMERIC = [
    "age", "campaign", "pdays", "previous",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed",
]
CATEGORICAL = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "poutcome",
]
ENGINEERED_BINARY = ["pdays_no_prev_contact"]  # pdays == 999 göstergesi
EXPECTED_COLUMNS = [
    "age", "job", "marital", "education", "default", "housing", "loan", "contact",
    "month", "day_of_week", "duration", "campaign", "pdays", "previous", "poutcome",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed", "y",
]
EXPECTED_N_ROWS = 4119


# ---------------------------------------------------------------- JSON
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"JSON'a çevrilemiyor: {type(o)}")


def project_config():
    return load_json(CONFIG_DIR / "project.json")


def training_config():
    return load_json(CONFIG_DIR / "training.json")


# ---------------------------------------------------------------- seed
SEED_OFFSETS = {
    "python": 0,
    "numpy": 1,
    "torch": 2,
    "split": 3,
    "outer_cv": 4,
    "inner_cv": 5,
    "random_search": 6,
    "early_stop_split": 7,
    "bootstrap": 8,
    "logreg": 9,
}


def get_seeds():
    """Temel seed = öğrenci numarasının son 5 hanesi + 2026; alt seed'ler sabit ofsetlerle türetilir."""
    last5 = str(project_config()["student_id_last5"]).strip()
    if not (len(last5) == 5 and last5.isdigit()):
        raise ValueError(
            "configs/project.json içindeki 'student_id_last5' alanına öğrenci numaranın "
            f"son 5 hanesini yaz (şu an: {last5!r})."
        )
    base = int(last5) + 2026
    seeds = {"student_id_last5": last5, "base_seed": base}
    seeds.update({k: base + v for k, v in SEED_OFFSETS.items()})
    # nihai kararlılık kontrolü: birincil torch seed'i + önceden ilan edilmiş iki seed
    seeds["stability_seeds"] = [seeds["torch"], base + 101, base + 202]
    seeds["primary_seed"] = seeds["torch"]
    return seeds


def set_global_seeds(seeds):
    random.seed(seeds["python"])
    np.random.seed(seeds["numpy"])
    os.environ["PYTHONHASHSEED"] = str(seeds["python"])
    try:
        import torch

        torch.manual_seed(seeds["torch"])
    except ImportError:
        pass


def environment_info():
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
    }
    for mod in ("numpy", "pandas", "sklearn", "scipy", "matplotlib", "torch", "joblib"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except ImportError:
            info[mod] = None
    try:
        import torch

        info["torch_cuda_available"] = bool(torch.cuda.is_available())
        info["torch_num_threads"] = torch.get_num_threads()
        info["device_used"] = "cpu"  # determinizm için tüm eğitimler CPU'da
    except ImportError:
        pass
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    try:
        with open("/proc/meminfo") as f:
            info["mem_total"] = f.readline().split(":", 1)[1].strip()
    except OSError:
        pass
    return info


def write_environment_files():
    info = environment_info()
    save_json(info, RESULTS_DIR / "environment.json")
    lines = []
    pip_names = {"sklearn": "scikit-learn"}
    for mod in ("numpy", "pandas", "sklearn", "scipy", "matplotlib", "torch", "joblib"):
        if info.get(mod):
            lines.append(f"{pip_names.get(mod, mod)}=={info[mod].split('+')[0]}")
    (RESULTS_DIR / "requirements-lock.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return info

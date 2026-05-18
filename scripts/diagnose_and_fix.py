"""
diagnose_and_fix.py
===================
Policy collapse tespitinde çalışır.
1. Son checkpoint'te fire_head dağılımını analiz eder
2. Kök nedeni tespit eder
3. config.yaml'ı günceller
4. Raporu logs/collapse_report.txt'ye yazar
"""

import sys
import os
import torch
import numpy as np
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from training.train_mappo import MAPPOActor

PROJ   = Path(__file__).parent.parent
CONFIG = PROJ / "configs" / "config.yaml"
CKPT_DIR = PROJ / "checkpoints"
REPORT = PROJ / "logs" / "collapse_report.txt"


def latest_checkpoint():
    ckpts = sorted(CKPT_DIR.glob("mappo_ep*.pt"),
                   key=lambda p: int(p.stem.replace("mappo_ep", "")))
    return ckpts[-1] if ckpts else None


def analyze_fire_head(ckpt_path):
    ckpt  = torch.load(str(ckpt_path), map_location="cpu")
    actor = MAPPOActor(obs_dim=49, action_dim=5, hidden=256)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    torch.manual_seed(0)
    obs = torch.randn(200, 49)
    with torch.no_grad():
        mean, log_std, fire_logit = actor.forward(obs)
        fire_probs = torch.sigmoid(fire_logit).squeeze(-1)
        std_vals   = log_std.exp()

    return {
        "fire_mean":     float(fire_probs.mean()),
        "fire_std":      float(fire_probs.std()),
        "fire_gt09":     int((fire_probs > 0.9).sum()),
        "fire_lt01":     int((fire_probs < 0.1).sum()),
        "ctrl_std_mean": float(std_vals.mean()),
        "ctrl_std_min":  float(std_vals.min()),
        "entropy_dead":  bool(float(std_vals.mean()) < 0.05),
        "fire_saturated":bool(int((fire_probs > 0.9).sum()) > 150 or
                              int((fire_probs < 0.1).sum()) > 150),
    }


def load_config():
    with open(CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(cfg):
    with open(CONFIG, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def apply_fixes(diag, cfg):
    fixes = []
    tr  = cfg.setdefault("training", {})
    rw  = cfg.setdefault("reward", {})
    op  = cfg.setdefault("opponent_pool", {})

    # Fix 1: Entropy tamamen ölmüşse daha agresif artır
    ec = float(tr.get("entropy_coeff", 0.10))
    if diag["entropy_dead"] and ec < 0.20:
        new_ec = min(ec + 0.05, 0.20)
        tr["entropy_coeff"] = new_ec
        fixes.append(f"entropy_coeff: {ec:.2f} → {new_ec:.2f} (ctrl std çok düşük)")

    # Fix 2: Fire head satüre olduysa fire logit init daha nötr olmalı
    # — config üzerinden yapılamaz, not olarak raporla
    if diag["fire_saturated"]:
        fixes.append("UYARI: fire_head satüre — train_mappo.py'de fire logit init -0.85'e döndürülmeli")

    # Fix 3: w_kill yetersizse artır
    wk = float(rw.get("w_kill", 25.0))
    if wk < 30.0:
        rw["w_kill"] = 30.0
        fixes.append(f"w_kill: {wk:.1f} → 30.0")

    # Fix 4: Pool sıfırla (snapshot'ları sil) + update interval artır
    op["pool_update_interval"] = 300
    fixes.append("pool_update_interval: → 300 (daha seyrek snapshot)")

    return fixes


def clear_pool_snapshots():
    removed = []
    for p in CKPT_DIR.glob("pool_actor_*.pt"):
        p.unlink()
        removed.append(p.name)
    return removed


def write_report(diag, fixes, removed, ckpt_name):
    lines = [
        "=" * 60,
        "POLICY COLLAPSE RAPORU",
        "=" * 60,
        f"Checkpoint  : {ckpt_name}",
        "",
        "--- Fire Head ---",
        f"  mean prob  : {diag['fire_mean']:.4f}",
        f"  std        : {diag['fire_std']:.4f}",
        f"  p>0.9      : {diag['fire_gt09']} / 200",
        f"  p<0.1      : {diag['fire_lt01']} / 200",
        f"  satüre     : {diag['fire_saturated']}",
        "",
        "--- Ctrl Head ---",
        f"  std mean   : {diag['ctrl_std_mean']:.4f}",
        f"  std min    : {diag['ctrl_std_min']:.4f}",
        f"  entropy ölü: {diag['entropy_dead']}",
        "",
        "--- Uygulanan Düzeltmeler ---",
    ] + [f"  * {f}" for f in fixes] + [
        "",
        "--- Silinen Pool Snapshot'ları ---",
    ] + [f"  - {r}" for r in removed] + [
        "",
        "Eğitim sıfırdan yeniden başlatılacak.",
        "=" * 60,
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    ckpt = latest_checkpoint()
    if ckpt is None:
        print("Checkpoint bulunamadı.")
        sys.exit(1)

    print(f"Analiz ediliyor: {ckpt.name}")
    diag  = analyze_fire_head(ckpt)
    cfg   = load_config()
    fixes = apply_fixes(diag, cfg)
    save_config(cfg)
    removed = clear_pool_snapshots()
    write_report(diag, fixes, removed, ckpt.name)

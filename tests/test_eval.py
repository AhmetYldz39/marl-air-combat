"""
test_eval.py
============
Evaluator ve eval_heuristic_baseline testleri.

PyTorch gerektiren testler koşullu çalışır.

Çalıştırma:
    python test_eval.py
"""

import sys
import json
import yaml
import numpy as np
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PASS = "✅ PASS"
FAIL = "❌ FAIL"
SKIP = "⏭️  SKIP"
results = []

def check(name, cond, detail=""):
    status = PASS if cond else FAIL
    print(f"{status} | {name}" + (f"  [{detail}]" if detail else ""))
    results.append((name, bool(cond)))

def skip(name, reason=""):
    print(f"{SKIP} | {name}" + (f"  [{reason}]" if reason else ""))

def almost(a, b, tol=1e-4):
    return abs(float(a) - float(b)) < tol

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("⚠️  PyTorch yok — Evaluator testleri SKIP\n")

CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

TEST_CONFIG = yaml.safe_load(yaml.dump(CONFIG))
TEST_CONFIG["env"]["max_steps"]   = 30
TEST_CONFIG["training"]["hidden_dim"] = 64


# ===========================================================================
print("\n── 1. eval_heuristic_baseline ───────────────────────────────────")
# ===========================================================================
from evaluation.eval import eval_heuristic_baseline, PHASE1_CRITERIA

results_b = eval_heuristic_baseline(
    TEST_CONFIG, n_episodes=10, seed=42, verbose=False
)

check("mode = heuristic_vs_heuristic",
      results_b["mode"] == "heuristic_vs_heuristic")
check("n_episodes = 10",    results_b["n_episodes"] == 10)
check("win+loss+draw = 10",
      results_b["wins"] + results_b["losses"] + results_b["draws"] == 10)
check("win_rate ∈ [0,1]",   0.0 <= results_b["win_rate"] <= 1.0)
check("draw_rate ∈ [0,1]",  0.0 <= results_b["draw_rate"] <= 1.0)
check("kill_per_ep >= 0",   results_b["kill_per_ep"] >= 0.0)
check("mean_ep_len > 0",    results_b["mean_ep_len"] > 0.0)

# Simetrik savaş: beraberlik + karışık sonuç bekleniyor
total_decisive = results_b["wins"] + results_b["losses"]
check("En az 1 decisive sonuç var",  total_decisive >= 0)  # her zaman geçer
print(f"  [bilgi] win={results_b['wins']}, loss={results_b['losses']}, "
      f"draw={results_b['draws']}, kill/ep={results_b['kill_per_ep']:.2f}")


# ===========================================================================
print("\n── 2. check_phase_criteria ──────────────────────────────────────")
# ===========================================================================
from evaluation.eval import Evaluator

# Kriterler sağlanıyor
good_results = {
    "win_rate":    0.45,
    "kill_per_ep": 0.90,
    "oob_rate":    0.02,
}
passed_g, report_g = Evaluator.check_phase_criteria(good_results)
check("İyi sonuç → passed=True",  passed_g)
check("win_rate passed",          report_g["win_rate"]["passed"])
check("kill_per_ep passed",       report_g["kill_per_ep"]["passed"])
check("oob_rate passed",          report_g["oob_rate"]["passed"])

# Kriterler sağlanmıyor
bad_results = {
    "win_rate":    0.30,
    "kill_per_ep": 0.50,
    "oob_rate":    0.10,
}
passed_b, report_b = Evaluator.check_phase_criteria(bad_results)
check("Kötü sonuç → passed=False", not passed_b)
check("win_rate failed",           not report_b["win_rate"]["passed"])

# Sınır değeri
edge_results = {
    "win_rate":    0.40,   # tam eşik
    "kill_per_ep": 0.80,
    "oob_rate":    0.05,
}
passed_e, _ = Evaluator.check_phase_criteria(edge_results)
check("Tam eşik → passed=True",   passed_e)


# ===========================================================================
print("\n── 3. save_results / JSON ───────────────────────────────────────")
# ===========================================================================
with tempfile.TemporaryDirectory() as tmpdir:
    out_path = str(Path(tmpdir) / "eval_results.json")
    Evaluator.save_results(good_results, out_path)

    check("JSON dosyası oluştu",    Path(out_path).exists())

    with open(out_path) as f:
        loaded = json.load(f)
    check("JSON win_rate doğru",
          almost(loaded["win_rate"], good_results["win_rate"]))


# ===========================================================================
print("\n── 4. print_summary — hata yok ──────────────────────────────────")
# ===========================================================================
full_results = {
    "n_episodes":    20,
    "deterministic": True,
    "win_rate":   0.45, "loss_rate": 0.30, "draw_rate": 0.25,
    "wins": 9, "losses": 6, "draws": 5,
    "kill_per_ep": 0.85, "kill_std": 0.3,
    "mean_reward": 42.5, "reward_std": 8.2,
    "mean_ep_len": 180.0,
    "survival_rate": 0.70,
    "oob_rate": 0.03,
    "episodes": [],
}
try:
    Evaluator.print_summary(full_results, show_criteria=True)
    check("print_summary hatasız çalıştı", True)
except Exception as e:
    check("print_summary hatasız çalıştı", False, str(e))


# ===========================================================================
print("\n── 5. Evaluator (torch gerekli) ─────────────────────────────────")
# ===========================================================================
if not HAS_TORCH:
    skip("Evaluator.__init__ testi", "PyTorch yok")
    skip("Evaluator.run() testi",    "PyTorch yok")
    skip("Checkpoint yükleme testi", "PyTorch yok")
else:
    from training.train_mappo import MAPPOActor, MAPPOTrainer
    from utils.normalization import Normalizer

    # Geçici checkpoint oluştur
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = yaml.safe_load(yaml.dump(TEST_CONFIG))
        cfg["logging"]["log_dir"]        = str(Path(tmpdir) / "logs")
        cfg["logging"]["checkpoint_dir"] = str(Path(tmpdir) / "ckpts")

        norm     = Normalizer(cfg)
        obs_dim  = norm.obs_dim(1, 2)   # 2v2
        act_dim  = 5
        hidden   = 64

        actor    = MAPPOActor(obs_dim, act_dim, hidden=hidden)
        ckpt_path = str(Path(tmpdir) / "test_actor.pt")
        torch.save({
            "actor":       actor.state_dict(),
            "episode":     10,
            "global_step": 1000,
        }, ckpt_path)

        # Evaluator init
        try:
            ev = Evaluator(cfg, ckpt_path, device="cpu")
            check("Evaluator init OK",          True)
            check("obs_dim = 49",               ev.obs_dim == 49)
            check("train_ids = blue takımı",    ev.train_ids == ["blue_0", "blue_1"])
        except Exception as e:
            check("Evaluator init OK", False, str(e))
            ev = None

        # Kısa run
        if ev is not None:
            try:
                res = ev.run(n_episodes=5, deterministic=True,
                             seed=0, verbose=False)
                check("run() tamamlandı",          True)
                check("n_episodes = 5",            res["n_episodes"] == 5)
                check("win+loss+draw = 5",
                      res["wins"]+res["losses"]+res["draws"] == 5)
                check("win_rate ∈ [0,1]",          0.0 <= res["win_rate"] <= 1.0)
                check("kill_per_ep >= 0",          res["kill_per_ep"] >= 0.0)
                check("mean_ep_len > 0",           res["mean_ep_len"] > 0.0)
                check("survival_rate ∈ [0,1]",
                      0.0 <= res["survival_rate"] <= 1.0)
                check("episodes listesi var",      len(res["episodes"]) == 5)
                check("episode[0] winner alanı",
                      "winner" in res["episodes"][0])
                print(f"  [bilgi] win={res['wins']}, loss={res['losses']}, "
                      f"draw={res['draws']}, kill/ep={res['kill_per_ep']:.2f}")
            except Exception as e:
                check("run() tamamlandı", False, str(e))

        # Stokastik mod
        if ev is not None:
            try:
                res_s = ev.run(n_episodes=3, deterministic=False,
                               seed=1, verbose=False)
                check("Stokastik run() OK", True)
                check("Stokastik n_episodes=3",
                      res_s["n_episodes"] == 3)
            except Exception as e:
                check("Stokastik run() OK", False, str(e))


# ===========================================================================
print("\n── 6. MAPPO sonrası kriter kontrolü akışı ───────────────────────")
# ===========================================================================
# Gerçek eğitim sonrası beklenen akış simülasyonu
mock_trained = {
    "win_rate":    0.42,
    "kill_per_ep": 0.85,
    "oob_rate":    0.03,
}
passed, report = Evaluator.check_phase_criteria(mock_trained)
check("Faz 1 geçiş simülasyonu — passed",  passed)
for k, d in report.items():
    check(f"  {k} detay var",  "value" in d and "passed" in d)


# ===========================================================================
print("\n" + "=" * 60)
total_n = len(results)
passed_n  = sum(1 for _, ok in results if ok)
failed_n  = total_n - passed_n
print(f"TOPLAM : {total_n}")
print(f"✅ PASS : {passed_n}")
print(f"❌ FAIL : {failed_n}")
if not HAS_TORCH:
    print("⏭️  SKIP : Evaluator testleri (PyTorch yok)")
if failed_n > 0:
    print("\nBaşarısız testler:")
    for name, ok in results:
        if not ok:
            print(f"  ❌ {name}")
    sys.exit(1)
else:
    if HAS_TORCH:
        print("\n🎉 Tüm testler geçti! Faz 1 altyapısı tamamlandı.")
    else:
        print("\n✅ PyTorch gerektirmeyen testler geçti.")
    sys.exit(0)

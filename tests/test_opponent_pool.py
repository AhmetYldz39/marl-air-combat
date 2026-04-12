"""
test_opponent_pool.py
=====================
OpponentPool (fictitious self-play) testleri.

Kapsam:
    1. Pool başlatma ve ring buffer yönetimi
    2. reset() — boş pool → heuristic fallback
    3. reset() — geçerli checkpoint → actor yükleme
    4. act()   — fallback modu
    5. act()   — neural policy modu
    6. Bozuk checkpoint → heuristic fallback'e geç
    7. Eğitim entegrasyonu — pool snapshot otomatik oluşturma

Çalıştırma:
    python tests/test_opponent_pool.py
"""

import sys
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


try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("⚠️  PyTorch bulunamadı — testler SKIP edilecek.\n")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

TEST_CONFIG = yaml.safe_load(yaml.dump(CONFIG))
TEST_CONFIG["env"]["max_steps"]           = 20
TEST_CONFIG["training"]["hidden_dim"]     = 64
TEST_CONFIG["training"]["n_steps"]        = 16
TEST_CONFIG["training"]["n_epochs"]       = 2
TEST_CONFIG["training"]["minibatch_size"] = 8

OBS_DIM    = 49
ACTION_DIM = 5


# ===========================================================================
print("\n── 1. Pool başlatma ve ring buffer yönetimi ─────────────────────")
# ===========================================================================
if not HAS_TORCH:
    skip("Pool testleri", "PyTorch yok")
else:
    from agents.opponent_pool import OpponentPool
    from agents.heuristic_agent import MultiHeuristicPolicy
    from envs.dogfight_env import DogfightEnv

    env = DogfightEnv(TEST_CONFIG)
    team_map = {aid: ("blue" if "blue" in aid else "red")
                for aid in env.agent_ids}
    fallback = MultiHeuristicPolicy(TEST_CONFIG, env.agent_ids, team_map)

    pool = OpponentPool(
        config        = TEST_CONFIG,
        red_ids       = env.red_ids,
        obs_dim       = OBS_DIM,
        action_dim    = ACTION_DIM,
        device        = torch.device("cpu"),
        fallback      = fallback,
        max_pool_size = 5,
    )

    check("Pool başlangıçta boş",         pool.size == 0)
    check("Başlangıçta fallback aktif",   pool._use_fallback == True)

    pool.add_checkpoint("dummy_1.pt")
    pool.add_checkpoint("dummy_2.pt")
    check("add_checkpoint → size=2",      pool.size == 2)

    # Ring buffer: max_pool_size=5 aşılmamalı
    for i in range(10):
        pool.add_checkpoint(f"dummy_{i}.pt")
    check("Ring buffer max_pool_size=5 korunuyor", pool.size == 5)


# ===========================================================================
print("\n── 2. reset() — boş pool → heuristic fallback ───────────────────")
# ===========================================================================
if not HAS_TORCH:
    skip("Reset (boş pool) testi", "PyTorch yok")
else:
    empty_pool = OpponentPool(
        config        = TEST_CONFIG,
        red_ids       = env.red_ids,
        obs_dim       = OBS_DIM,
        action_dim    = ACTION_DIM,
        device        = torch.device("cpu"),
        fallback      = fallback,
        max_pool_size = 20,
    )
    empty_pool.reset()
    check("Boş pool reset → fallback aktif",  empty_pool._use_fallback == True)
    check("Boş pool reset → actor None",      empty_pool._current_actor is None)


# ===========================================================================
print("\n── 3. reset() — geçerli checkpoint → actor yükleme ─────────────")
# ===========================================================================
if not HAS_TORCH:
    skip("Checkpoint yükleme testi", "PyTorch yok")
else:
    from training.train_mappo import MAPPOActor

    with tempfile.TemporaryDirectory() as tmpdir:
        actor = MAPPOActor(OBS_DIM, ACTION_DIM, hidden=64)
        ckpt_path = Path(tmpdir) / "test_actor.pt"
        torch.save({
            "episode":     10,
            "global_step": 1000,
            "actor":       actor.state_dict(),
            "config":      TEST_CONFIG,
        }, ckpt_path)

        pool_ckpt = OpponentPool(
            config        = TEST_CONFIG,
            red_ids       = env.red_ids,
            obs_dim       = OBS_DIM,
            action_dim    = ACTION_DIM,
            device        = torch.device("cpu"),
            fallback      = fallback,
            max_pool_size = 20,
        )
        pool_ckpt.add_checkpoint(str(ckpt_path))
        pool_ckpt.reset()

        check("Checkpoint yüklendi → fallback False",  pool_ckpt._use_fallback == False)
        check("Checkpoint yüklendi → actor var",       pool_ckpt._current_actor is not None)
        check("Yüklenen actor MAPPOActor örneği",
              isinstance(pool_ckpt._current_actor, MAPPOActor))


# ===========================================================================
print("\n── 4. act() — fallback (boş pool) ──────────────────────────────")
# ===========================================================================
if not HAS_TORCH:
    skip("act() fallback testi", "PyTorch yok")
else:
    env2 = DogfightEnv(TEST_CONFIG)
    obs_dict2   = env2.reset()
    state_dict2 = env2.get_all_states()

    team_map2 = {aid: ("blue" if "blue" in aid else "red") for aid in env2.agent_ids}
    fb2       = MultiHeuristicPolicy(TEST_CONFIG, env2.agent_ids, team_map2)
    pool_fb   = OpponentPool(
        config=TEST_CONFIG, red_ids=env2.red_ids,
        obs_dim=OBS_DIM, action_dim=ACTION_DIM,
        device=torch.device("cpu"), fallback=fb2, max_pool_size=20,
    )
    pool_fb.reset()   # pool boş → fallback aktif

    acts_fb = pool_fb.act(obs_dict2, state_dict2)
    check("Fallback act red_ids kapsıyor",
          all(rid in acts_fb for rid in env2.red_ids))
    check("Fallback act şekil doğru (ACTION_DIM=5)",
          all(acts_fb[rid].shape == (ACTION_DIM,) for rid in env2.red_ids))
    check("Fallback act finite değerler",
          all(np.all(np.isfinite(acts_fb[rid])) for rid in env2.red_ids))


# ===========================================================================
print("\n── 5. act() — neural policy (checkpoint yüklü) ──────────────────")
# ===========================================================================
if not HAS_TORCH:
    skip("act() neural policy testi", "PyTorch yok")
else:
    with tempfile.TemporaryDirectory() as tmpdir:
        actor2     = MAPPOActor(OBS_DIM, ACTION_DIM, hidden=64)
        ckpt_path2 = Path(tmpdir) / "test_actor2.pt"
        torch.save({
            "episode": 20, "global_step": 2000,
            "actor": actor2.state_dict(), "config": TEST_CONFIG,
        }, ckpt_path2)

        env3       = DogfightEnv(TEST_CONFIG)
        obs3       = env3.reset()
        state3     = env3.get_all_states()

        team_map3  = {aid: ("blue" if "blue" in aid else "red") for aid in env3.agent_ids}
        fb3        = MultiHeuristicPolicy(TEST_CONFIG, env3.agent_ids, team_map3)
        pool_nn    = OpponentPool(
            config=TEST_CONFIG, red_ids=env3.red_ids,
            obs_dim=OBS_DIM, action_dim=ACTION_DIM,
            device=torch.device("cpu"), fallback=fb3, max_pool_size=20,
        )
        pool_nn.add_checkpoint(str(ckpt_path2))
        pool_nn.reset()

        acts_nn = pool_nn.act(obs3, state3)
        check("Neural act red_ids kapsıyor",
              all(rid in acts_nn for rid in env3.red_ids))
        check("Neural act şekil doğru (ACTION_DIM=5)",
              all(acts_nn[rid].shape == (ACTION_DIM,) for rid in env3.red_ids))
        check("Neural act finite değerler",
              all(np.all(np.isfinite(acts_nn[rid])) for rid in env3.red_ids))
        # Squash garantileri: da,de,dr ∈ [-1,1]; dt ∈ [0,1]
        for rid in env3.red_ids:
            a = acts_nn[rid]
            check(f"Neural {rid} da ∈ [-1,1]", abs(float(a[0])) <= 1.0 + 1e-5)
            check(f"Neural {rid} dt ∈ [0,1]",
                  float(a[3]) >= -1e-5 and float(a[3]) <= 1.0 + 1e-5)


# ===========================================================================
print("\n── 6. Bozuk checkpoint → heuristic fallback'e geç ───────────────")
# ===========================================================================
if not HAS_TORCH:
    skip("Bozuk checkpoint testi", "PyTorch yok")
else:
    with tempfile.TemporaryDirectory() as tmpdir:
        bad_path = Path(tmpdir) / "bad.pt"
        bad_path.write_text("bu geçersiz bir checkpoint dosyasıdır")

        team_map4 = {aid: ("blue" if "blue" in aid else "red") for aid in env.agent_ids}
        fb4       = MultiHeuristicPolicy(TEST_CONFIG, env.agent_ids, team_map4)
        pool_bad  = OpponentPool(
            config=TEST_CONFIG, red_ids=env.red_ids,
            obs_dim=OBS_DIM, action_dim=ACTION_DIM,
            device=torch.device("cpu"), fallback=fb4, max_pool_size=5,
        )
        pool_bad.add_checkpoint(str(bad_path))
        pool_bad.reset()   # hata bekleniyor, fallback'e geçmeli

        check("Bozuk checkpoint → fallback aktif",  pool_bad._use_fallback == True)
        check("Bozuk checkpoint → actor None",      pool_bad._current_actor is None)


# ===========================================================================
print("\n── 7. Eğitim entegrasyonu — pool snapshot otomatik oluşturma ─────")
# ===========================================================================
if not HAS_TORCH:
    skip("Entegrasyon testi", "PyTorch yok")
else:
    from training.train_mappo import MAPPOTrainer

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = yaml.safe_load(yaml.dump(TEST_CONFIG))
        cfg["logging"]["log_dir"]             = str(Path(tmpdir) / "logs")
        cfg["logging"]["checkpoint_dir"]      = str(Path(tmpdir) / "ckpts")
        cfg["training"]["total_timesteps"]    = 400
        cfg["training"]["n_steps"]            = 16
        cfg["training"]["n_epochs"]           = 2
        cfg["training"]["minibatch_size"]     = 8
        cfg["env"]["max_steps"]               = 20
        cfg["logging"]["log_interval"]        = 5
        cfg["logging"]["checkpoint_interval"] = 100
        cfg["opponent_pool"] = {
            "max_pool_size":        5,
            "start_step":      99999,   # smoke test'te pool kullanılmaz
            "pool_update_interval": 5,  # her 5 episode'da bir snapshot
        }

        try:
            trainer = MAPPOTrainer(cfg, device="cpu")
            trainer.train()

            check("trainer.pool mevcut",
                  hasattr(trainer, "pool"))
            check("Pool OpponentPool örneği",
                  isinstance(trainer.pool, OpponentPool))
            check("Eğitim sonrası pool'da snapshot var",
                  trainer.pool.size > 0,
                  f"pool.size={trainer.pool.size}")

            # Pool snapshot dosyaları gerçekten oluştu mu?
            ckpt_dir = Path(tmpdir) / "ckpts"
            snapshots = list(ckpt_dir.glob("pool_actor_ep*.pt"))
            check("Pool snapshot dosyaları oluştu",
                  len(snapshots) > 0,
                  f"{len(snapshots)} dosya")

        except Exception as e:
            check("Entegrasyon testi tamamlandı", False, str(e))


# ===========================================================================
print("\n" + "=" * 60)
total_n = len(results)
passed  = sum(1 for _, ok in results if ok)
failed  = total_n - passed
print(f"TOPLAM : {total_n}")
print(f"✅ PASS : {passed}")
print(f"❌ FAIL : {failed}")
if not HAS_TORCH:
    print(f"⏭️  SKIP : Tüm testler (PyTorch yok)")
if failed > 0:
    print("\nBaşarısız testler:")
    for name, ok in results:
        if not ok:
            print(f"  ❌ {name}")
    sys.exit(1)
else:
    print("\n🎉 Tüm testler geçti!")
    sys.exit(0)

"""
test_train_mappo.py
===================
MAPPOActor, MAPPOCritic, RolloutBuffer ve MAPPOTrainer testleri.

PyTorch gerektiren testler torch import kontrolü ile koşullu çalışır.
Tüm testler geçmeli — torch yoksa ilgili testler SKIP olur.

Çalıştırma:
    python test_train_mappo.py
"""

import sys
import os
import yaml
import numpy as np
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
    # Skip'ler toplam sayıya dahil edilmez

def almost(a, b, tol=1e-4):
    return abs(float(a) - float(b)) < tol

# ---------------------------------------------------------------------------
# PyTorch kontrolü
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("⚠️  PyTorch bulunamadı — network testleri SKIP edilecek.\n")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

# Hızlı test için küçük config
TEST_CONFIG = yaml.safe_load(yaml.dump(CONFIG))
TEST_CONFIG["env"]["max_steps"]            = 20
TEST_CONFIG["training"]["n_steps"]         = 16
TEST_CONFIG["training"]["n_epochs"]        = 2
TEST_CONFIG["training"]["minibatch_size"]  = 8
TEST_CONFIG["training"]["total_timesteps"] = 200
TEST_CONFIG["training"]["hidden_dim"]      = 64   # küçük network

OBS_DIM    = 49   # 2v2 Faz 0-1
ACTION_DIM = 5
N_AGENTS   = 2    # blue takımı
GLOBAL_DIM = OBS_DIM * N_AGENTS  # 98


# ===========================================================================
print("\n── 1. RolloutBuffer ─────────────────────────────────────────────")
# ===========================================================================
from training.rollout_buffer import RolloutBuffer

N_STEPS = 16
buf = RolloutBuffer(N_STEPS, N_AGENTS, OBS_DIM, ACTION_DIM, GLOBAL_DIM)

# Dummy veri doldur
agent_ids = ["blue_0", "blue_1"]
for t in range(N_STEPS):
    obs_dict = {aid: np.random.randn(OBS_DIM).astype(np.float32)
                for aid in agent_ids}
    act_dict = {aid: np.random.randn(ACTION_DIM).astype(np.float32)
                for aid in agent_ids}
    lp_dict  = {aid: float(np.random.randn()) for aid in agent_ids}
    rew_dict = {aid: float(np.random.randn()) for aid in agent_ids}
    done_dict = {aid: False for aid in agent_ids}
    values   = np.random.randn(N_AGENTS).astype(np.float32)
    gobs     = np.random.randn(GLOBAL_DIM).astype(np.float32)

    buf.add(obs_dict, act_dict, lp_dict, rew_dict, done_dict,
            values, gobs, agent_ids)

check("Buffer ptr = N_STEPS sonrası", buf._ptr == N_STEPS)
check("obs shape  = (16, 2, 49)",  buf.obs.shape    == (N_STEPS, N_AGENTS, OBS_DIM))
check("act shape  = (16, 2, 5)",   buf.actions.shape == (N_STEPS, N_AGENTS, ACTION_DIM))
check("rew shape  = (16, 2)",      buf.rewards.shape == (N_STEPS, N_AGENTS))
check("gobs shape = (16, 98)",     buf.global_obs.shape == (N_STEPS, GLOBAL_DIM))

# GAE hesabı
last_values = np.zeros(N_AGENTS, dtype=np.float32)
adv, ret = buf.compute_gae(last_values, gamma=0.99, gae_lambda=0.95)
check("GAE adv shape = (16, 2)",   adv.shape == (N_STEPS, N_AGENTS))
check("GAE ret shape = (16, 2)",   ret.shape == (N_STEPS, N_AGENTS))
check("GAE adv NaN yok",           not np.any(np.isnan(adv)))
check("GAE ret NaN yok",           not np.any(np.isnan(ret)))
check("GAE ret = adv + values",
      np.allclose(ret, adv + buf.values, atol=1e-5))

# Reset
buf.reset()
check("Buffer reset sonrası ptr=0", buf._ptr == 0)
check("Buffer reset sonrası obs=0", np.allclose(buf.obs, 0.0))


# ===========================================================================
print("\n── 2. MAPPOActor ────────────────────────────────────────────────")
# ===========================================================================
if not HAS_TORCH:
    skip("MAPPOActor testleri", "PyTorch yok")
else:
    from training.train_mappo import MAPPOActor

    actor = MAPPOActor(obs_dim=OBS_DIM, action_dim=ACTION_DIM, hidden=64)

    # Parametre sayısı
    n_params = sum(p.numel() for p in actor.parameters())
    check("Actor parametre sayısı > 0", n_params > 0, f"{n_params:,}")

    # Forward — batch
    obs_batch = torch.randn(4, OBS_DIM)
    mean, log_std, fire_logit = actor.forward(obs_batch)
    check("Actor mean shape = (4, 4)",       tuple(mean.shape)       == (4, ACTION_DIM - 1))
    check("Actor log_std shape = (4,)",      tuple(log_std.shape)    == (ACTION_DIM - 1,))
    check("Actor fire_logit shape = (4, 1)", tuple(fire_logit.shape) == (4, 1))
    check("Actor mean NaN yok",              not torch.any(torch.isnan(mean)))

    # get_dist — ctrl (Normal) + fire (Bernoulli) ayrı head
    ctrl_dist, fire_dist = actor.get_dist(obs_batch)
    check("Actor ctrl_dist type Normal",
          isinstance(ctrl_dist, torch.distributions.Normal))
    check("Actor fire_dist type Bernoulli",
          isinstance(fire_dist, torch.distributions.Bernoulli))

    # act — single obs
    obs_single = torch.randn(1, OBS_DIM)
    raw, lp = actor.act(obs_single, deterministic=False)
    check("Actor act raw shape = (1, 5)", tuple(raw.shape) == (1, ACTION_DIM))
    check("Actor act log_prob scalar",    lp.shape == torch.Size([1]))

    # squash — aralık kontrolü
    raw_t    = torch.randn(8, ACTION_DIM)
    squashed = MAPPOActor.squash(raw_t)
    check("Squash da ∈ [-1,1]",
          bool(squashed[:, 0].abs().max() <= 1.0 + 1e-6))
    check("Squash de ∈ [-1,1]",
          bool(squashed[:, 1].abs().max() <= 1.0 + 1e-6))
    check("Squash dt ∈ [0,1]",
          bool((squashed[:, 3] >= -1e-6).all() and
               (squashed[:, 3] <= 1.0 + 1e-6).all()))
    check("Squash fire passthrough (Bernoulli tarafından üretilir, squash dokunmaz)",
          torch.allclose(squashed[:, 4], raw_t[:, 4]))

    # Deterministic act tekrarlanabilir
    obs_d = torch.randn(1, OBS_DIM)
    raw1, _ = actor.act(obs_d, deterministic=True)
    raw2, _ = actor.act(obs_d, deterministic=True)
    check("Deterministic act tekrarlanabilir",
          torch.allclose(raw1, raw2))


# ===========================================================================
print("\n── 3. MAPPOCritic ───────────────────────────────────────────────")
# ===========================================================================
if not HAS_TORCH:
    skip("MAPPOCritic testleri", "PyTorch yok")
else:
    from training.train_mappo import MAPPOCritic

    critic = MAPPOCritic(global_obs_dim=GLOBAL_DIM, hidden=64)

    n_params_c = sum(p.numel() for p in critic.parameters())
    check("Critic parametre sayısı > 0", n_params_c > 0, f"{n_params_c:,}")

    # Forward — batch
    gobs_batch = torch.randn(4, GLOBAL_DIM)
    val = critic(gobs_batch)
    check("Critic output shape = (4,)", tuple(val.shape) == (4,))
    check("Critic output NaN yok",        not torch.any(torch.isnan(val)))

    # Single
    gobs_single = torch.randn(1, GLOBAL_DIM)
    val_s = critic(gobs_single)
    check("Critic single scalar çıkış", val_s.numel() == 1)


# ===========================================================================
print("\n── 4. MAPPOTrainer init ─────────────────────────────────────────")
# ===========================================================================
if not HAS_TORCH:
    skip("MAPPOTrainer testleri", "PyTorch yok")
else:
    from training.train_mappo import MAPPOTrainer
    import tempfile

    # Log ve checkpoint dizinlerini temp'e yönlendir
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = yaml.safe_load(yaml.dump(TEST_CONFIG))
        cfg["logging"]["log_dir"]        = str(Path(tmpdir) / "logs")
        cfg["logging"]["checkpoint_dir"] = str(Path(tmpdir) / "ckpts")

        trainer = MAPPOTrainer(cfg, device="cpu")

        check("trainer.obs_dim = 49",    trainer.obs_dim    == OBS_DIM)
        check("trainer.action_dim = 5",  trainer.action_dim == ACTION_DIM)
        check("trainer.n_agents = 2",    trainer.n_agents   == N_AGENTS)
        check("trainer.global_obs_dim = 98",
              trainer.global_obs_dim == GLOBAL_DIM)
        check("trainer.train_ids = blue",
              trainer.train_ids == ["blue_0", "blue_1"])

        # Actor/Critic cihaz kontrolü
        actor_dev = next(trainer.actor.parameters()).device
        check("Actor CPU'da",  str(actor_dev) == "cpu")


# ===========================================================================
print("\n── 5. Kısa eğitim döngüsü (smoke test) ─────────────────────────")
# ===========================================================================
if not HAS_TORCH:
    skip("Smoke test", "PyTorch yok")
else:
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = yaml.safe_load(yaml.dump(TEST_CONFIG))
        cfg["logging"]["log_dir"]        = str(Path(tmpdir) / "logs")
        cfg["logging"]["checkpoint_dir"] = str(Path(tmpdir) / "ckpts")
        cfg["logging"]["log_interval"]        = 5
        cfg["logging"]["checkpoint_interval"] = 50
        cfg["training"]["total_timesteps"]    = 400  # çok kısa

        try:
            trainer2 = MAPPOTrainer(cfg, device="cpu")
            trainer2.train()
            check("Kısa eğitim döngüsü tamamlandı", True)
            check("Episode count > 0", trainer2.episode_count > 0,
                  f"ep={trainer2.episode_count}")
            check("Global step > 0",   trainer2.global_step   > 0,
                  f"step={trainer2.global_step}")
            check("Update count > 0",  trainer2._update_count > 0,
                  f"updates={trainer2._update_count}")

            # CSV log oluştu mu?
            csv_path = Path(cfg["logging"]["log_dir"]) / "train_log.csv"
            check("train_log.csv oluştu", csv_path.exists())

        except Exception as e:
            check("Kısa eğitim döngüsü tamamlandı", False, str(e))


# ===========================================================================
print("\n── 6. Checkpoint kaydet / yükle ─────────────────────────────────")
# ===========================================================================
if not HAS_TORCH:
    skip("Checkpoint testleri", "PyTorch yok")
else:
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = yaml.safe_load(yaml.dump(TEST_CONFIG))
        cfg["logging"]["log_dir"]        = str(Path(tmpdir) / "logs")
        cfg["logging"]["checkpoint_dir"] = str(Path(tmpdir) / "ckpts")

        t_save = MAPPOTrainer(cfg, device="cpu")
        t_save.episode_count = 42
        t_save.global_step   = 1000
        ckpt_path = Path(tmpdir) / "test_ckpt.pt"
        t_save._ckpt_dir = Path(tmpdir)
        torch.save({
            "episode":     t_save.episode_count,
            "global_step": t_save.global_step,
            "actor":       t_save.actor.state_dict(),
            "critic":      t_save.critic.state_dict(),
            "opt_actor":   t_save.opt_actor.state_dict(),
            "opt_critic":  t_save.opt_critic.state_dict(),
            "config":      cfg,
        }, ckpt_path)

        # Yükleme
        t_load = MAPPOTrainer(cfg, device="cpu")
        t_load.load_checkpoint(str(ckpt_path))

        check("Checkpoint episode yüklendi",
              t_load.episode_count == 42)
        check("Checkpoint global_step yüklendi",
              t_load.global_step == 1000)

        # Ağırlıklar aynı mı?
        for (n1, p1), (n2, p2) in zip(
            t_save.actor.named_parameters(),
            t_load.actor.named_parameters()
        ):
            if not torch.allclose(p1, p2):
                check(f"Actor ağırlık eşleşme ({n1})", False)
                break
        else:
            check("Actor ağırlıkları eşleşti", True)


# ===========================================================================
print("\n── 7. _build_global_obs ─────────────────────────────────────────")
# ===========================================================================
if not HAS_TORCH:
    skip("_build_global_obs testi", "PyTorch yok")
else:
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = yaml.safe_load(yaml.dump(TEST_CONFIG))
        cfg["logging"]["log_dir"]        = str(Path(tmpdir) / "logs")
        cfg["logging"]["checkpoint_dir"] = str(Path(tmpdir) / "ckpts")

        t3 = MAPPOTrainer(cfg, device="cpu")
        t3.env.reset()

        obs_dict = t3.env._build_obs_dict()
        gobs = t3._build_global_obs(obs_dict)

        check("Global obs shape = (98,)",   gobs.shape == (GLOBAL_DIM,))
        check("Global obs dtype float32",   gobs.dtype == np.float32)
        check("Global obs NaN yok",         not np.any(np.isnan(gobs)))

        # blue_0 ve blue_1 obs'unun concat'ı olmalı
        expected = np.concatenate([obs_dict["blue_0"], obs_dict["blue_1"]])
        check("Global obs = concat(blue_0, blue_1)",
              np.allclose(gobs, expected))


# ===========================================================================
print("\n" + "=" * 60)
total_n = len(results)
passed  = sum(1 for _, ok in results if ok)
failed  = total_n - passed
print(f"TOPLAM : {total_n}")
print(f"✅ PASS : {passed}")
print(f"❌ FAIL : {failed}")
if not HAS_TORCH:
    print(f"⏭️  SKIP : Network testleri (PyTorch yok)")
if failed > 0:
    print("\nBaşarısız testler:")
    for name, ok in results:
        if not ok:
            print(f"  ❌ {name}")
    sys.exit(1)
else:
    if HAS_TORCH:
        print("\n🎉 Tüm testler geçti! eval.py yazımına geçilebilir.")
    else:
        print("\n✅ PyTorch gerektirmeyen testler geçti.")
        print("   PyTorch kurulduktan sonra tam test çalıştırın.")
    sys.exit(0)

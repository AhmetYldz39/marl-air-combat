"""
test_env_step.py
================
DogfightEnv entegrasyon testleri.

Çalıştırma:
    python test_env_step.py

Kategoriler:
    1.  Init — boyutlar, ajan ID'leri
    2.  reset() — obs boyutu, aralık, NaN/Inf
    3.  step() — tek adım dönüş tipi ve boyutları
    4.  Ölü ajan obs sıfır değil (hayatta enemy bilgisi korunur)
    5.  Done — max_steps sonrası beraberlik
    6.  Done — tüm düşmanlar ölünce Blue kazanır
    7.  Episode reward birikimi
    8.  Aggression embedding Faz 2
    9.  GAT mesajı boyut kontrolü (Faz 3)
    10. Deterministik reset seed kontrolü
    11. Tam episode rollout — crash yok, NaN yok
    12. from_yaml yükleyici
"""

import numpy as np
import sys, os, yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.dogfight_env import DogfightEnv, BLUE, RED
from envs.aircraft_model import ACTION_DIM, STATE_ALIVE, STATE_HP
from utils.normalization  import OBS_EGO_DIM, OBS_TEAMMATE_DIM, OBS_ENEMY_DIM, OBS_ROLE_DIM, OBS_GAT_MSG_DIM

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # tests/ → proje kökü
CONFIG_PATH  = PROJECT_ROOT / "configs" / "config.yaml"

PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = []

def check(name, cond, detail=""):
    status = PASS if cond else FAIL
    print(f"{status} | {name}" + (f"  [{detail}]" if detail else ""))
    results.append((name, bool(cond)))

def almost(a, b, tol=1e-5):
    return abs(float(a) - float(b)) < tol

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

# Test için hızlı episode
TEST_CONFIG = yaml.safe_load(yaml.dump(CONFIG))
TEST_CONFIG["env"]["max_steps"] = 50

def make_env(max_steps=50, seed=42):
    cfg = yaml.safe_load(yaml.dump(CONFIG))
    cfg["env"]["max_steps"] = max_steps
    env = DogfightEnv(cfg)
    env.seed(seed)
    return env

def zero_actions(env):
    return {aid: np.zeros(ACTION_DIM, dtype=np.float32) for aid in env.agent_ids}

def random_actions(env, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    actions = {}
    for aid in env.agent_ids:
        a = np.zeros(ACTION_DIM, dtype=np.float32)
        a[0] = float(rng.uniform(-1, 1))   # aileron
        a[1] = float(rng.uniform(-1, 1))   # elevator
        a[2] = float(rng.uniform(-0.3, 0.3))  # rudder
        a[3] = float(rng.uniform(0.5, 1.0))   # throttle
        a[4] = 0.0                             # fire off
        actions[aid] = a
    return actions


# ---------------------------------------------------------------------------
print("\n── 1. Init ──────────────────────────────────────────────────────")
# ---------------------------------------------------------------------------
env = make_env()

check("agent_ids sayısı = 4",    len(env.agent_ids) == 4)
check("blue_ids = ['blue_0','blue_1']", env.blue_ids == ["blue_0", "blue_1"])
check("red_ids  = ['red_0', 'red_1']",  env.red_ids  == ["red_0",  "red_1"])
check("obs_dim = 49 (2v2 Faz 0-1)",     env.obs_dim  == 49, f"{env.obs_dim}")
check("action_dim = 5",                  env.action_dim == 5)
check("n_agents = 4",                    env.n_agents == 4)


# ---------------------------------------------------------------------------
print("\n── 2. reset() ───────────────────────────────────────────────────")
# ---------------------------------------------------------------------------
obs = env.reset()

check("reset() 4 ajan döndürür",  len(obs) == 4)
for aid in env.agent_ids:
    o = obs[aid]
    check(f"{aid} obs boyutu=49",   len(o) == 49, f"{len(o)}")
    check(f"{aid} obs dtype=float32", o.dtype == np.float32)
    check(f"{aid} obs NaN yok",     not np.any(np.isnan(o)))
    check(f"{aid} obs Inf yok",     not np.any(np.isinf(o)))
    check(f"{aid} obs [-1,1]",      bool(np.all(o >= -1.0 - 1e-6) and np.all(o <= 1.0 + 1e-6)))


# ---------------------------------------------------------------------------
print("\n── 3. step() — tek adım dönüş tipleri ──────────────────────────")
# ---------------------------------------------------------------------------
obs2 = env.reset()
actions = zero_actions(env)
obs3, rew, done, info = env.step(actions)

check("obs döndürür (dict)",   isinstance(obs3, dict))
check("rew döndürür (dict)",   isinstance(rew,  dict))
check("done döndürür (dict)",  isinstance(done, dict))
check("info döndürür (dict)",  isinstance(info, dict))
check("'__all__' done key var", "__all__" in done)
check("step 1: __all__=False",  done["__all__"] == False)

for aid in env.agent_ids:
    check(f"{aid} rew float",   isinstance(rew[aid], float))
    check(f"{aid} obs boyutu",  len(obs3[aid]) == 49)
    check(f"{aid} obs NaN yok", not np.any(np.isnan(obs3[aid])))


# ---------------------------------------------------------------------------
print("\n── 4. Hayatta ajan obs sıfır değil ──────────────────────────────")
# ---------------------------------------------------------------------------
env4 = make_env()
obs4 = env4.reset()
for aid in env4.agent_ids:
    check(f"{aid} hayattayken obs sıfır değil",
          not np.allclose(obs4[aid], 0.0))


# ---------------------------------------------------------------------------
print("\n── 5. Done — max_steps sonrası beraberlik ───────────────────────")
# ---------------------------------------------------------------------------
env5 = make_env(max_steps=10, seed=0)
env5.reset()
rng5 = np.random.default_rng(0)
done5 = {"__all__": False}
for _ in range(10):
    _, _, done5, _ = env5.step(random_actions(env5, rng5))

check("10 adım sonrası done=True",  done5["__all__"] == True)
check("10 adım sonrası draw",       done5.get("winner") == "draw")
check("step_count = 10",            env5.get_step_count() == 10)


# ---------------------------------------------------------------------------
print("\n── 6. Done — Blue kazanma (tüm red'ler ölü) ────────────────────")
# ---------------------------------------------------------------------------
env6 = make_env(max_steps=500, seed=1)
env6.reset()

# Red ajanlarını manuel olarak öldür
for rid in env6.red_ids:
    env6._states[rid][STATE_ALIVE] = 0.0
    env6._states[rid][STATE_HP]    = 0.0

# Bir adım at — done tetiklenmeli
_, _, done6, info6 = env6.step(zero_actions(env6))
check("Red ölü → done=True",       done6["__all__"] == True)
check("Red ölü → winner=blue",     done6.get("winner") == BLUE)

# Episode summary kontrolü
for aid in env6.blue_ids:
    check(f"{aid} episode summary var",
          "episode" in info6[aid] and "episode/total_reward" in info6[aid]["episode"])


# ---------------------------------------------------------------------------
print("\n── 7. Episode reward birikimi ───────────────────────────────────")
# ---------------------------------------------------------------------------
env7 = make_env(max_steps=20, seed=2)
env7.reset()
rng7 = np.random.default_rng(2)
total_rews = {aid: 0.0 for aid in env7.agent_ids}
done7 = {"__all__": False}
step7 = 0
while not done7["__all__"]:
    _, rew7, done7, _ = env7.step(random_actions(env7, rng7))
    for aid in env7.agent_ids:
        total_rews[aid] += rew7[aid]
    step7 += 1

for aid in env7.agent_ids:
    check(f"{aid} reward birikimi tutarlı",
          almost(total_rews[aid], env7._episode_rewards[aid], tol=1e-3),
          f"birikim={total_rews[aid]:.3f}, env={env7._episode_rewards[aid]:.3f}")


# ---------------------------------------------------------------------------
print("\n── 8. Aggression embedding — Faz 2 ─────────────────────────────")
# ---------------------------------------------------------------------------
env8 = make_env()
agg_dict = {aid: 0.7 for aid in env8.agent_ids}
obs8 = env8.reset(aggression_dict=agg_dict)

# Faz 2 ile obs boyutu 51 olmalı
for aid in env8.agent_ids:
    check(f"{aid} Faz 2 obs boyutu=51", len(obs8[aid]) == 51, f"{len(obs8[aid])}")
    # Son 2 eleman rol embedding
    role_emb = obs8[aid][-OBS_ROLE_DIM:]
    check(f"{aid} rol embedding[0]=0.7",
          almost(role_emb[0], 0.7, tol=1e-5), f"{role_emb[0]:.4f}")
    check(f"{aid} rol embedding toplamı=1.0",
          almost(float(np.sum(role_emb)), 1.0))


# ---------------------------------------------------------------------------
print("\n── 9. GAT mesajı boyut kontrolü — Faz 3 ────────────────────────")
# ---------------------------------------------------------------------------
env9 = make_env()
agg9 = {aid: 0.5 for aid in env9.agent_ids}
env9.reset(aggression_dict=agg9)
# Her ajan için 1 takım arkadaşından 16 boyutlu mesaj
gat9 = {aid: [np.zeros(OBS_GAT_MSG_DIM, dtype=np.float32)]
        for aid in env9.agent_ids}
_, _, _, _ = env9.step(zero_actions(env9), gat_messages=gat9)
obs9 = env9._build_obs_dict(gat_messages=gat9)

expected_gat = 51 + OBS_GAT_MSG_DIM  # 51 + 16 = 67
for aid in env9.agent_ids:
    check(f"{aid} Faz 3 obs boyutu=67",
          len(obs9[aid]) == expected_gat, f"{len(obs9[aid])}")


# ---------------------------------------------------------------------------
print("\n── 10. Deterministik seed ───────────────────────────────────────")
# ---------------------------------------------------------------------------
env_a = make_env(seed=99)
env_b = make_env(seed=99)
obs_a = env_a.reset()
obs_b = env_b.reset()

all_same = all(np.allclose(obs_a[aid], obs_b[aid]) for aid in env_a.agent_ids)
check("Aynı seed → aynı reset obs", all_same)

env_c = make_env(seed=7)
obs_c = env_c.reset()
all_diff = any(not np.allclose(obs_a[aid], obs_c[aid]) for aid in env_a.agent_ids)
check("Farklı seed → farklı reset obs", all_diff)


# ---------------------------------------------------------------------------
print("\n── 11. Tam episode rollout — crash ve NaN yok ───────────────────")
# ---------------------------------------------------------------------------
env11 = make_env(max_steps=200, seed=42)
env11.reset()
rng11 = np.random.default_rng(42)
done11 = {"__all__": False}
step11 = 0
nan_found = False

while not done11["__all__"]:
    actions11 = random_actions(env11, rng11)
    obs11, rew11, done11, info11 = env11.step(actions11)
    step11 += 1

    # Her adımda NaN/Inf kontrolü
    for aid in env11.agent_ids:
        if np.any(np.isnan(obs11[aid])) or np.any(np.isinf(obs11[aid])):
            nan_found = True
            break
    if nan_found:
        break

check("200 adım rollout tamamlandı",   not nan_found, f"step={step11}")
check("Episode adım sayısı > 0",       step11 > 0)
check("Episode adım sayısı ≤ 200",     step11 <= 200)
check("Rollout NaN/Inf yok",           not nan_found)

# Episode sonunda info summary'de gerekli anahtarlar var
for aid in env11.agent_ids:
    if "episode" in info11.get(aid, {}):
        ep = info11[aid]["episode"]
        for k in ["episode/total_reward", "episode/kills",
                  "episode/steps", "episode/survived"]:
            check(f"Episode summary '{k}' var", k in ep)
        break


# ---------------------------------------------------------------------------
print("\n── 12. from_yaml yükleyici ──────────────────────────────────────")
# ---------------------------------------------------------------------------
try:
    env12 = DogfightEnv.from_yaml(CONFIG_PATH)
    obs12 = env12.reset()
    check("from_yaml yüklendi",       True)
    check("from_yaml obs boyutu=49",  len(obs12["blue_0"]) == 49,
          f"{len(obs12['blue_0'])}")
except Exception as e:
    check("from_yaml yüklendi", False, str(e))


# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
total_n = len(results)
passed  = sum(1 for _, ok in results if ok)
failed  = total_n - passed
print(f"TOPLAM : {total_n}")
print(f"✅ PASS : {passed}")
print(f"❌ FAIL : {failed}")
if failed > 0:
    print("\nBaşarısız testler:")
    for name, ok in results:
        if not ok:
            print(f"  ❌ {name}")
    print("\n⛔ dogfight_env.py düzeltilmeden eğitime geçilmez.")
    sys.exit(1)
else:
    print("\n🎉 Tüm testler geçti! Faz 0 tamamlandı — eğitime geçilebilir.")
    sys.exit(0)

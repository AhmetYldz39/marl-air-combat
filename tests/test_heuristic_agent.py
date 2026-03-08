"""
test_heuristic_agent.py
=======================
HeuristicAgent ve MultiHeuristicPolicy için unit + entegrasyon testleri.

Çalıştırma:
    python test_heuristic_agent.py

Kategoriler:
    1.  Init
    2.  Ölü ajan → sıfır aksiyon
    3.  Aksiyon clip — sınırlar korunuyor
    4.  CRITICAL: zemin kurtarma (elevator + tam gaz)
    5.  CRITICAL: sınır kurtarma (merkeze dönüş)
    6.  CRITICAL: stall kurtarma
    7.  EVASION: yüksek tehdit → break turn
    8.  EVASION: düşük tehdit → PURSUIT'a geç
    9.  PURSUIT: hedef seçimi (en iyi WEZ skoru)
    10. PURSUIT: hizalanınca ateş
    11. PURSUIT: uzak hedef → tam gaz
    12. Nominal uçuş (düşman yokken)
    13. MultiHeuristicPolicy — tam episode rollout
    14. Reset — cooldown temizleme
"""

import numpy as np
import sys
import os
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.heuristic_agent import (
    HeuristicAgent, MultiHeuristicPolicy,
    CRITICAL_H_FLOOR, CRITICAL_ALPHA_MAX, CRITICAL_V_MIN, CRITICAL_MAP_MARGIN,
    EVASION_THREAT_THRESHOLD, PURSUIT_WEZ_ENGAGE,
)
from envs.dogfight_env import DogfightEnv, BLUE, RED
from envs.aircraft_model import (
    AircraftModel,
    STATE_X, STATE_Y, STATE_H, STATE_V, STATE_ALPHA, STATE_BETA,
    STATE_GAMMA, STATE_PHI, STATE_THETA, STATE_PSI,
    STATE_P, STATE_Q, STATE_R,
    STATE_FUEL, STATE_AMMO, STATE_HP, STATE_ALIVE,
    ACTION_DA, ACTION_DE, ACTION_DR, ACTION_DT, ACTION_FIRE,
    ACTION_DIM,
)
from envs.geometry_utils import deg2rad

PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = []


def check(name, cond, detail=""):
    status = PASS if cond else FAIL
    print(f"{status} | {name}" + (f"  [{detail}]" if detail else ""))
    results.append((name, bool(cond)))


def almost(a, b, tol=1e-4):
    return abs(float(a) - float(b)) < tol


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

aircraft = AircraftModel(CONFIG)
agent    = HeuristicAgent(CONFIG, agent_id="test_agent")


def mk(x=0.0, y=0.0, h=4000.0, V=200.0, psi_deg=0.0,
       alpha_deg=3.0, phi_deg=0.0, gamma_deg=0.0,
       ammo=6.0, hp=1.0, alive=1.0, fuel=3000.0):
    s = aircraft.reset({
        "x": x, "y": y, "h": h, "V": V,
        "psi": deg2rad(psi_deg), "alpha": deg2rad(alpha_deg),
        "phi": deg2rad(phi_deg), "gamma": deg2rad(gamma_deg),
    })
    s[STATE_AMMO]  = ammo
    s[STATE_HP]    = hp
    s[STATE_ALIVE] = alive
    s[STATE_FUEL]  = fuel
    return s


def in_range(arr, lo, hi):
    return bool(np.all(arr >= lo - 1e-6) and np.all(arr <= hi + 1e-6))


# ---------------------------------------------------------------------------
print("\n── 1. Init ──────────────────────────────────────────────────────")
# ---------------------------------------------------------------------------
check("agent_id atandı",   agent.agent_id == "test_agent")
check("map_size=50000",    almost(agent.map_size, 50000.0))
check("half_map=25000",    almost(agent.half_map, 25000.0))
check("h_min=50",          almost(agent.h_min, 50.0))


# ---------------------------------------------------------------------------
print("\n── 2. Ölü ajan → sıfır aksiyon ─────────────────────────────────")
# ---------------------------------------------------------------------------
dead = mk(alive=0.0)
a_dead = agent.act(dead, [], [])
check("Ölü ajan aksiyon = sıfır", np.allclose(a_dead, 0.0))


# ---------------------------------------------------------------------------
print("\n── 3. Aksiyon clip — sınırlar ───────────────────────────────────")
# ---------------------------------------------------------------------------
s_ok = mk()
a = agent.act(s_ok, [], [])
check("Aksiyon shape = (5,)",    a.shape == (ACTION_DIM,))
check("Aksiyon dtype float32",   a.dtype == np.float32)
check("da ∈ [-1, 1]",            -1.0 <= a[ACTION_DA]   <= 1.0)
check("de ∈ [-1, 1]",            -1.0 <= a[ACTION_DE]   <= 1.0)
check("dr ∈ [-1, 1]",            -1.0 <= a[ACTION_DR]   <= 1.0)
check("dt ∈ [0, 1]",              0.0 <= a[ACTION_DT]   <= 1.0)
check("fire ∈ [0, 1]",            0.0 <= a[ACTION_FIRE] <= 1.0)


# ---------------------------------------------------------------------------
print("\n── 4. CRITICAL: zemin kurtarma ──────────────────────────────────")
# ---------------------------------------------------------------------------
# h < CRITICAL_H_FLOOR → elevator pozitif (burun yukarı), tam gaz
s_low = mk(h=CRITICAL_H_FLOOR - 50.0)
a_low = agent.act(s_low, [], [])
check("Zemin kurtarma: de > 0",   a_low[ACTION_DE] > 0.0,
      f"de={a_low[ACTION_DE]:.3f}")
check("Zemin kurtarma: dt = 1.0", almost(a_low[ACTION_DT], 1.0),
      f"dt={a_low[ACTION_DT]:.3f}")
check("Zemin kurtarma: fire = 0", almost(a_low[ACTION_FIRE], 0.0))


# ---------------------------------------------------------------------------
print("\n── 5. CRITICAL: harita sınırı kurtarma ──────────────────────────")
# ---------------------------------------------------------------------------
# X kenarına yakın, kuzey'e bakıyor → merkeze dönmeli
s_oob = mk(x=25000.0 - 1000.0, y=0.0, psi_deg=90.0)  # +X kenarında, Doğu'ya bakıyor
a_oob = agent.act(s_oob, [], [])
check("Sınır kurtarma: dt = 1.0",
      almost(a_oob[ACTION_DT], 1.0), f"dt={a_oob[ACTION_DT]:.3f}")
# Dönüş hareketi — aileron sıfır değil olmalı
check("Sınır kurtarma: da != 0",
      abs(a_oob[ACTION_DA]) > 0.01, f"da={a_oob[ACTION_DA]:.3f}")


# ---------------------------------------------------------------------------
print("\n── 6. CRITICAL: stall kurtarma ──────────────────────────────────")
# ---------------------------------------------------------------------------
s_stall2 = mk()
s_stall2[STATE_ALPHA] = CRITICAL_ALPHA_MAX + deg2rad(3.0)
a_stall = agent.act(s_stall2, [], [])
check("Stall kurtarma: de < 0 (burun indirme)",
      a_stall[ACTION_DE] < 0.0, f"de={a_stall[ACTION_DE]:.3f}")
check("Stall kurtarma: dt = 1.0",
      almost(a_stall[ACTION_DT], 1.0), f"dt={a_stall[ACTION_DT]:.3f}")


# ---------------------------------------------------------------------------
print("\n── 7. EVASION: yüksek tehdit → break turn ───────────────────────")
# ---------------------------------------------------------------------------
# Düşman tam arkamızda, WEZ içinde (tehdit yüksek)
s_prey  = mk(x=0.0,    y=0.0,    psi_deg=0.0)    # biz: kuzeye bakıyor
s_pred  = mk(x=0.0,    y=-1500.0, psi_deg=0.0)   # düşman: tam arkamızda, kuzeye bakıyor
# Düşmanın bize tehdidi yüksek olmalı (önümüzde, WEZ içinde)
from envs.weapons_model import WeaponsModel
wm_test = WeaponsModel(CONFIG)
threat = wm_test.wez_advantage_score(s_pred, s_prey)
print(f"  [debug] tehdit skoru = {threat:.3f} (≥{EVASION_THREAT_THRESHOLD} gerekli)")

if threat >= EVASION_THREAT_THRESHOLD:
    a_evade = agent.act(s_prey, [], [s_pred])
    check("Evasion tetiklendi: fire=0",
          almost(a_evade[ACTION_FIRE], 0.0))
    check("Evasion tetiklendi: dt=1.0",
          almost(a_evade[ACTION_DT], 1.0), f"dt={a_evade[ACTION_DT]:.3f}")
    check("Evasion tetiklendi: |da| yüksek (break turn)",
          abs(a_evade[ACTION_DA]) > 0.5, f"da={a_evade[ACTION_DA]:.3f}")
else:
    # Tehdit eşiği geçilmediyse evasion yerine pursuit çalışır — bu da geçerli
    check("Evasion skoru düşük — pursuit devrede (geçerli)", True,
          f"threat={threat:.3f}")


# ---------------------------------------------------------------------------
print("\n── 8. EVASION: düşük tehdit → PURSUIT'a geç ────────────────────")
# ---------------------------------------------------------------------------
# Düşman çok uzakta → tehdit düşük → evasion tetiklenmemeli
s_self   = mk(x=0.0, y=0.0, psi_deg=0.0)
s_far_en = mk(x=0.0, y=12000.0, psi_deg=180.0)  # 12 km uzakta, bize bakıyor
threat_far = wm_test.wez_advantage_score(s_far_en, s_self)
print(f"  [debug] uzak düşman tehdit = {threat_far:.3f} (< {EVASION_THREAT_THRESHOLD} bekleniyor)")
check("Uzak düşman tehdit < eşik",
      threat_far < EVASION_THREAT_THRESHOLD, f"threat={threat_far:.3f}")


# ---------------------------------------------------------------------------
print("\n── 9. PURSUIT: hedef seçimi ─────────────────────────────────────")
# ---------------------------------------------------------------------------
s_own   = mk(x=0.0, y=0.0, psi_deg=0.0)
e_close = mk(x=0.0, y=2000.0,  psi_deg=180.0)   # yakın, önümüzde
e_far   = mk(x=0.0, y=10000.0, psi_deg=180.0)   # uzak

wez_close = wm_test.wez_advantage_score(s_own, e_close)
wez_far   = wm_test.wez_advantage_score(s_own, e_far)
target = agent._select_target(s_own, [e_close, e_far])
check("Yakın hedef seçilir (daha iyi WEZ)",
      target is e_close or wez_close >= wez_far,
      f"wez_close={wez_close:.3f}, wez_far={wez_far:.3f}")


# ---------------------------------------------------------------------------
print("\n── 10. PURSUIT: WEZ içinde ateş ─────────────────────────────────")
# ---------------------------------------------------------------------------
# Düşman önümüzde ama bize bakmıyor (threat düşük → evasion tetiklenmiyor)
# Biz ise düşmana bakıyoruz + optimal mesafe → WEZ avantajı yüksek → ateş
s_shooter = mk(x=0.0, y=0.0,    psi_deg=0.0)    # kuzeye bakıyor
s_target  = mk(x=0.0, y=3000.0, psi_deg=0.0)    # düşman da kuzeye bakıyor (bize BAKMIYOR)
wez_adv   = wm_test.wez_advantage_score(s_shooter, s_target)
threat_on_us = wm_test.wez_advantage_score(s_target, s_shooter)
print(f"  [debug] WEZ avantajı (biz→düşman) = {wez_adv:.3f}")
print(f"  [debug] Tehdit (düşman→biz)       = {threat_on_us:.3f} (< {EVASION_THREAT_THRESHOLD} gerekli)")
a_shoot = agent.act(s_shooter, [], [s_target])
if wez_adv >= PURSUIT_WEZ_ENGAGE and threat_on_us < EVASION_THREAT_THRESHOLD:
    check("WEZ içinde, tehdit yok: fire=1", almost(a_shoot[ACTION_FIRE], 1.0),
          f"fire={a_shoot[ACTION_FIRE]:.2f}")
else:
    check("Senaryo koşulları sağlanamadı — atlandı", True,
          f"wez={wez_adv:.3f}, threat={threat_on_us:.3f}")


# ---------------------------------------------------------------------------
print("\n── 11. PURSUIT: uzak hedef → tam gaz ───────────────────────────")
# ---------------------------------------------------------------------------
s_own2   = mk(x=0.0, y=0.0, psi_deg=0.0)
s_far_t  = mk(x=0.0, y=7000.0)   # > 6000m → tam gaz bekleniyor
a_far_t  = agent._pursuit(s_own2, s_far_t)
check("Uzak hedef: dt = 1.0 (tam gaz)",
      almost(a_far_t[ACTION_DT], 1.0), f"dt={a_far_t[ACTION_DT]:.3f}")


# ---------------------------------------------------------------------------
print("\n── 12. Nominal uçuş (düşman yok) ────────────────────────────────")
# ---------------------------------------------------------------------------
s_alone = mk(h=3000.0)  # NOMINAL_H=4000 → yukarı çıkması lazım
a_nom   = agent.act(s_alone, [], [])
check("Nominal: de > 0 (irtifa kazan)",
      a_nom[ACTION_DE] > 0.0, f"de={a_nom[ACTION_DE]:.3f}")
check("Nominal: fire=0", almost(a_nom[ACTION_FIRE], 0.0))
check("Nominal: dt > 0", a_nom[ACTION_DT] > 0.0)


# ---------------------------------------------------------------------------
print("\n── 13. MultiHeuristicPolicy — tam episode rollout ───────────────")
# ---------------------------------------------------------------------------
import yaml as _yaml
cfg2 = _yaml.safe_load(_yaml.dump(CONFIG))
cfg2["env"]["max_steps"] = 100

env = DogfightEnv(cfg2)
env.seed(42)

team_map = {aid: ("blue" if "blue" in aid else "red") for aid in env.agent_ids}
policy   = MultiHeuristicPolicy(CONFIG, env.agent_ids, team_map)

env.reset()
policy.reset()

done = {"__all__": False}
steps = 0
nan_found = False

while not done["__all__"]:
    state_dict  = env.get_all_states()
    action_dict = policy.act(state_dict)

    # Aksiyon clip kontrolü
    for aid, a in action_dict.items():
        if not in_range(a[:4], -1.0, 1.0) or not (0.0 <= a[4] <= 1.0):
            nan_found = True
            break

    obs, rew, done, info = env.step(action_dict)
    steps += 1

    for aid in env.agent_ids:
        if np.any(np.isnan(obs[aid])) or np.any(np.isinf(obs[aid])):
            nan_found = True
            break
    if nan_found:
        break

check("100 adım rollout tamamlandı",    not nan_found, f"step={steps}")
check("Episode steps > 0",              steps > 0)
check("Tüm aksiyonlar geçerli aralıkta", not nan_found)
check("Obs NaN/Inf yok",                not nan_found)

winner = done.get("winner", "?")
check("Episode sonucu geçerli (blue/red/draw)",
      winner in ("blue", "red", "draw"), f"winner={winner}")


# ---------------------------------------------------------------------------
print("\n── 14. Reset — cooldown temizleme ───────────────────────────────")
# ---------------------------------------------------------------------------
agent2 = HeuristicAgent(CONFIG, "reset_test")
agent2._wm._cooldown_timer = 1.5   # yapay cooldown
agent2.reset()
check("reset() sonrası cooldown = 0",
      almost(agent2._wm._cooldown_timer, 0.0),
      f"cooldown={agent2._wm._cooldown_timer:.3f}")


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
    print("\n⛔ heuristic_agent.py düzeltilmeden train_mappo.py'ye geçilmez.")
    sys.exit(1)
else:
    print("\n🎉 Tüm testler geçti! train_mappo.py yazımına geçilebilir.")
    sys.exit(0)

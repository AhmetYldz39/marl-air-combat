"""
test_reward_model.py
====================
reward_model.py (v2 — sürekli aggression skalası) için unit testler.

Çalıştırma:
    python test_reward_model.py
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.reward_model import (
    RewardModel, aggression_to_embedding,
    AGGRESSION_MIN, AGGRESSION_MAX, AGGRESSION_DEFAULT
)
from envs.weapons_model import WeaponsModel
from envs.aircraft_model import (
    AircraftModel,
    STATE_X, STATE_Y, STATE_H, STATE_V, STATE_PSI, STATE_ALPHA,
    STATE_AMMO, STATE_HP, STATE_ALIVE, STATE_FUEL,
    ACTION_DIM, ACTION_FIRE
)
from envs.geometry_utils import deg2rad

PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = []


def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    msg = f"{status} | {name}"
    if detail:
        msg += f"  [{detail}]"
    print(msg)
    results.append((name, condition))


def almost_equal(a, b, tol=1e-5):
    return abs(a - b) < tol


TEST_CONFIG = {
    "aircraft": {
        "mass": 9100.0, "wingspan": 9.45, "wing_area": 27.87,
        "mean_chord": 3.45, "Ixx": 12875.0, "Iyy": 75674.0,
        "Izz": 85552.0, "Ixz": 1331.0, "max_thrust": 76300.0,
        "SFC": 2.0e-5, "initial_fuel": 3000.0, "initial_ammo": 6,
        "initial_hp": 1.0, "radar_range": 15000.0,
        "V_min": 60.0, "V_max": 600.0,
        "alpha_max": deg2rad(25.0), "alpha_min": deg2rad(-10.0),
        "h_min": 50.0,
    },
    "aero_coeffs": {
        "CL0": 0.0, "CL_alpha": 5.5, "CL_q": 1.8, "CL_de": 0.5,
        "CD0": 0.013, "CD_alpha": 0.5, "CY_beta": -0.35, "CY_da": 0.0,
        "CY_dr": 0.10, "Cl_p": -0.41, "Cl_r": 0.15, "Cl_da": 0.18,
        "Cl_dr": 0.01, "Cm0": 0.04, "Cm_alpha": -0.65, "Cm_q": -12.4,
        "Cm_de": -1.1, "Cn_p": -0.05, "Cn_r": -0.20,
        "Cn_da": -0.02, "Cn_dr": 0.15,
    },
    "control_limits": {
        "aileron_max_deg": 21.5, "elevator_max_deg": 25.0, "rudder_max_deg": 30.0,
    },
    "weapons": {
        "wez_range_max": 8000.0, "wez_range_min": 300.0,
        "wez_angle_max": 30.0, "missile_damage": 0.6,
        "fire_cooldown": 2.0, "min_fire_altitude": 200.0,
    },
    "reward": {
        "w_kill": 10.0, "w_wez": 2.0, "w_tracking": 1.0,
        "w_survival": 0.1, "w_coord": 0.5, "w_resource": 0.1,
        "w_penalty": -5.0, "w_role": 1.0,
    },
    "roles": {
        "w_kill_at_0": 0.5,      "w_kill_at_1": 2.0,
        "w_survival_at_0": 2.0,  "w_survival_at_1": 0.5,
        "w_coord_at_0": 1.5,     "w_coord_at_1": 1.0,
        "role_kill_weight_at_0": 0.0,     "role_kill_weight_at_1": 1.0,
        "role_survival_weight_at_0": 0.5, "role_survival_weight_at_1": 0.0,
        "role_coord_weight_at_0": 0.5,    "role_coord_weight_at_1": 0.0,
    },
    "coord": {"dist_min": 500.0, "dist_max": 8000.0, "dist_opt": 3000.0},
}

aircraft = AircraftModel(TEST_CONFIG)
weapons  = WeaponsModel(TEST_CONFIG)
rm       = RewardModel(TEST_CONFIG)
DT       = 0.05
MAP_SIZE = 50000.0


def make_state(x=0.0, y=0.0, h=3000.0, V=200.0,
               psi_deg=0.0, alpha_deg=3.0,
               ammo=6.0, hp=1.0, alive=1.0, fuel=3000.0):
    s = aircraft.reset({"x": x, "y": y, "h": h, "V": V,
                        "psi": deg2rad(psi_deg), "alpha": deg2rad(alpha_deg)})
    s[STATE_AMMO] = ammo; s[STATE_HP] = hp
    s[STATE_ALIVE] = alive; s[STATE_FUEL] = fuel
    return s


def no_fire():
    return {"fired": False, "hit": False, "damage": 0.0, "kill": False,
            "ammo_remaining": None, "new_target_hp": None,
            "wez_info": {}, "fail_reason": "no_fire_command"}

def kill_fire():
    return {"fired": True, "hit": True, "damage": 0.6, "kill": True,
            "ammo_remaining": 5.0, "new_target_hp": 0.0,
            "wez_info": {}, "fail_reason": None}

def miss_fire():
    return {"fired": True, "hit": False, "damage": 0.0, "kill": False,
            "ammo_remaining": 6.0, "new_target_hp": None,
            "wez_info": {}, "fail_reason": None}


# ---------------------------------------------------------------------------
print("\n── 1. RewardModel Init ──────────────────────────────────────────")
# ---------------------------------------------------------------------------
check("w_kill=10.0",       almost_equal(rm.w_kill,     10.0))
check("w_wez=2.0",         almost_equal(rm.w_wez,       2.0))
check("w_penalty=-5.0",    almost_equal(rm.w_penalty,  -5.0))
check("w_role=1.0",        almost_equal(rm.w_role,      1.0))
check("kill_at_0=0.5",     almost_equal(rm._kill_at_0,  0.5))
check("kill_at_1=2.0",     almost_equal(rm._kill_at_1,  2.0))
check("survival_at_0=2.0", almost_equal(rm._survival_at_0, 2.0))
check("survival_at_1=0.5", almost_equal(rm._survival_at_1, 0.5))


# ---------------------------------------------------------------------------
print("\n── 2. aggression=None — Pasif Mod (Faz 0-1) ─────────────────────")
# ---------------------------------------------------------------------------
weapons.reset()
a = make_state()
_, info = rm.compute(a, [], [], weapons, a.copy(), no_fire(), DT, MAP_SIZE,
                     aggression=None)
check("aggression=None → r_role=0.0",            almost_equal(info["r_role"], 0.0))
check("aggression=None → w_kill_scale=1.0",      almost_equal(info["w_kill_scale"], 1.0))
check("aggression=None → w_survival_scale=1.0",  almost_equal(info["w_survival_scale"], 1.0))
check("aggression=None → w_coord_scale=1.0",     almost_equal(info["w_coord_scale"], 1.0))
check("aggression=None → info['aggression']=None", info["aggression"] is None)


# ---------------------------------------------------------------------------
print("\n── 3. Lerp Skala Doğrulaması ────────────────────────────────────")
# ---------------------------------------------------------------------------
def get_scales(agg):
    weapons.reset()
    _, info = rm.compute(make_state(), [], [], weapons, make_state().copy(),
                         no_fire(), DT, MAP_SIZE, aggression=agg)
    return info["w_kill_scale"], info["w_survival_scale"], info["w_coord_scale"]

ks0, ss0, cs0 = get_scales(0.0)
ks5, ss5, cs5 = get_scales(0.5)
ks1, ss1, cs1 = get_scales(1.0)

check("aggression=0.0 → w_kill_scale=0.5",    almost_equal(ks0, 0.5),  f"{ks0:.4f}")
check("aggression=0.5 → w_kill_scale=1.25",   almost_equal(ks5, 1.25), f"{ks5:.4f}")
check("aggression=1.0 → w_kill_scale=2.0",    almost_equal(ks1, 2.0),  f"{ks1:.4f}")
check("aggression=0.0 → w_survival_scale=2.0", almost_equal(ss0, 2.0), f"{ss0:.4f}")
check("aggression=0.5 → w_survival_scale=1.25",almost_equal(ss5, 1.25),f"{ss5:.4f}")
check("aggression=1.0 → w_survival_scale=0.5", almost_equal(ss1, 0.5), f"{ss1:.4f}")
check("w_kill_scale monoton artan",      ks0 < ks5 < ks1)
check("w_survival_scale monoton azalan", ss0 > ss5 > ss1)
check("w_coord_scale monoton azalan",    cs0 >= cs5 >= cs1)


# ---------------------------------------------------------------------------
print("\n── 4. Kill Katkısı: Agresif > Dengeli > Defansif ────────────────")
# ---------------------------------------------------------------------------
weapons.reset()
a = make_state()
_, i_agg = rm.compute(a, [], [], weapons, a.copy(), kill_fire(), DT, MAP_SIZE, aggression=1.0)
_, i_bal = rm.compute(a, [], [], weapons, a.copy(), kill_fire(), DT, MAP_SIZE, aggression=0.5)
_, i_def = rm.compute(a, [], [], weapons, a.copy(), kill_fire(), DT, MAP_SIZE, aggression=0.0)

check("Agresif kill katkısı = 20.0",
      almost_equal(i_agg["w_kill_contrib"], 20.0), f"{i_agg['w_kill_contrib']:.2f}")
check("Dengeli kill katkısı = 12.5",
      almost_equal(i_bal["w_kill_contrib"], 12.5), f"{i_bal['w_kill_contrib']:.2f}")
check("Defansif kill katkısı = 5.0",
      almost_equal(i_def["w_kill_contrib"], 5.0),  f"{i_def['w_kill_contrib']:.2f}")
check("Agresif > dengeli > defansif kill",
      i_agg["w_kill_contrib"] > i_bal["w_kill_contrib"] > i_def["w_kill_contrib"])


# ---------------------------------------------------------------------------
print("\n── 5. Survival Katkısı: Defansif > Agresif ─────────────────────")
# ---------------------------------------------------------------------------
weapons.reset()
_, i_s_def = rm.compute(a, [], [], weapons, a.copy(), no_fire(), DT, MAP_SIZE, aggression=0.0)
_, i_s_bal = rm.compute(a, [], [], weapons, a.copy(), no_fire(), DT, MAP_SIZE, aggression=0.5)
_, i_s_agg = rm.compute(a, [], [], weapons, a.copy(), no_fire(), DT, MAP_SIZE, aggression=1.0)

check("Defansif > dengeli > agresif survival",
      i_s_def["w_survival_contrib"] > i_s_bal["w_survival_contrib"] > i_s_agg["w_survival_contrib"],
      f"def={i_s_def['w_survival_contrib']:.5f}, agg={i_s_agg['w_survival_contrib']:.5f}")
check("Defansif survival = w_survival×2.0×dt",
      almost_equal(i_s_def["w_survival_contrib"], rm.w_survival * 2.0 * DT))
check("Agresif survival = w_survival×0.5×dt",
      almost_equal(i_s_agg["w_survival_contrib"], rm.w_survival * 0.5 * DT))


# ---------------------------------------------------------------------------
print("\n── 6. WEZ Reward ────────────────────────────────────────────────")
# ---------------------------------------------------------------------------
weapons.reset()
agent_wez = make_state(psi_deg=0.0)
enemy_opt = make_state(x=0.0, y=3200.0)
enemy_far = make_state(x=0.0, y=9000.0)

_, i_opt = rm.compute(agent_wez, [], [enemy_opt], weapons, agent_wez.copy(), no_fire(), DT, MAP_SIZE)
_, i_far = rm.compute(agent_wez, [], [enemy_far], weapons, agent_wez.copy(), no_fire(), DT, MAP_SIZE)

check("Optimal WEZ → r_wez > 0.5", i_opt["r_wez"] > 0.5, f"{i_opt['r_wez']:.3f}")
check("Menzil dışı → r_wez = 0",   almost_equal(i_far["r_wez"], 0.0), f"{i_far['r_wez']:.3f}")
check("Çoklu düşman → max WEZ",
      almost_equal(
          rm.compute(agent_wez, [], [enemy_far, enemy_opt], weapons,
                     agent_wez.copy(), no_fire(), DT, MAP_SIZE)[1]["r_wez"],
          i_opt["r_wez"], tol=1e-4
      ))


# ---------------------------------------------------------------------------
print("\n── 7. Tracking Reward ───────────────────────────────────────────")
# ---------------------------------------------------------------------------
weapons.reset()
agent_tr    = make_state(psi_deg=0.0)
enemy_front = make_state(x=0.0, y=1000.0)
enemy_side  = make_state(x=1000.0, y=0.0)

_, i_front = rm.compute(agent_tr, [], [enemy_front], weapons, agent_tr.copy(), no_fire(), DT, MAP_SIZE)
_, i_side  = rm.compute(agent_tr, [], [enemy_side],  weapons, agent_tr.copy(), no_fire(), DT, MAP_SIZE)

check("Hedef önde → r_tracking ≈ 1.0",
      almost_equal(i_front["r_tracking"], 1.0, tol=1e-3), f"{i_front['r_tracking']:.4f}")
check("Hedef sağda → r_tracking ≈ 0.0",
      almost_equal(i_side["r_tracking"],  0.0, tol=1e-3), f"{i_side['r_tracking']:.4f}")


# ---------------------------------------------------------------------------
print("\n── 8. Koordinasyon Reward ───────────────────────────────────────")
# ---------------------------------------------------------------------------
weapons.reset()
agent_co = make_state()
_, i_opt_co  = rm.compute(agent_co, [make_state(y=3000.0)],  [], weapons, agent_co.copy(), no_fire(), DT, MAP_SIZE)
_, i_near_co = rm.compute(agent_co, [make_state(y=100.0)],   [], weapons, agent_co.copy(), no_fire(), DT, MAP_SIZE)
_, i_far_co  = rm.compute(agent_co, [make_state(y=15000.0)], [], weapons, agent_co.copy(), no_fire(), DT, MAP_SIZE)
_, i_dead_co = rm.compute(agent_co, [make_state(alive=0.0)], [], weapons, agent_co.copy(), no_fire(), DT, MAP_SIZE)

check("Optimal mesafe → r_coord en yüksek",
      i_opt_co["r_coord"] >= i_near_co["r_coord"] and
      i_opt_co["r_coord"] >= i_far_co["r_coord"],
      f"opt={i_opt_co['r_coord']:.3f}, near={i_near_co['r_coord']:.3f}, far={i_far_co['r_coord']:.3f}")
check("r_coord [0,1] aralığında", 0.0 <= i_opt_co["r_coord"] <= 1.0)
check("Ölü takım arkadaşı → r_coord=0", almost_equal(i_dead_co["r_coord"], 0.0))


# ---------------------------------------------------------------------------
print("\n── 9. Resource Reward ───────────────────────────────────────────")
# ---------------------------------------------------------------------------
weapons.reset()
agent_rs = make_state(ammo=6.0)
_, i_miss = rm.compute(agent_rs, [], [], weapons, agent_rs.copy(), miss_fire(), DT, MAP_SIZE)
_, i_norm = rm.compute(agent_rs, [], [], weapons, agent_rs.copy(), no_fire(),   DT, MAP_SIZE)

check("WEZ dışı ateş → r_resource = -1.0",
      almost_equal(i_miss["r_resource"], -1.0, tol=0.1), f"{i_miss['r_resource']:.3f}")
check("Normal uçuş → r_resource ≥ -0.1", i_norm["r_resource"] >= -0.1)


# ---------------------------------------------------------------------------
print("\n── 10. Penalty Reward ───────────────────────────────────────────")
# ---------------------------------------------------------------------------
weapons.reset()
_, i_oob   = rm.compute(make_state(x=26000.0),   [], [], weapons, make_state(x=26000.0).copy(),  no_fire(), DT, MAP_SIZE)
_, i_stall = rm.compute(make_state(alpha_deg=26), [], [], weapons, make_state(alpha_deg=26).copy(), no_fire(), DT, MAP_SIZE)
_, i_low   = rm.compute(make_state(h=120.0),      [], [], weapons, make_state(h=120.0).copy(),    no_fire(), DT, MAP_SIZE)
_, i_ok    = rm.compute(make_state(h=3000.0),     [], [], weapons, make_state(h=3000.0).copy(),   no_fire(), DT, MAP_SIZE)

check("Sınır dışı → r_penalty > 0 & contrib < 0",
      i_oob["r_penalty"] > 0 and i_oob["w_penalty_contrib"] < 0,
      f"penalty={i_oob['r_penalty']:.1f}, contrib={i_oob['w_penalty_contrib']:.1f}")
check("Stall → r_penalty > 0",     i_stall["r_penalty"] > 0.0)
check("Zemin yakını → r_penalty > 0", i_low["r_penalty"] > 0.0)
check("Normal uçuş → r_penalty = 0",
      almost_equal(i_ok["r_penalty"], 0.0), f"{i_ok['r_penalty']:.3f}")


# ---------------------------------------------------------------------------
print("\n── 11. r_role_bonus Monotonluk ──────────────────────────────────")
# ---------------------------------------------------------------------------
weapons.reset()
a = make_state()
role_vals = []
for agg in [0.0, 0.25, 0.5, 0.75, 1.0]:
    _, inf = rm.compute(a, [], [], weapons, a.copy(), kill_fire(), DT, MAP_SIZE, aggression=agg)
    role_vals.append(inf["r_role"])

check("r_role kill varken aggression ile monoton artan",
      all(role_vals[i] <= role_vals[i+1] for i in range(len(role_vals)-1)),
      str([f"{v:.3f}" for v in role_vals]))

weapons.reset()
tm = make_state(y=3000.0)
_, i_rd = rm.compute(a, [tm], [], weapons, a.copy(), no_fire(), DT, MAP_SIZE, aggression=0.0)
_, i_ra = rm.compute(a, [tm], [], weapons, a.copy(), no_fire(), DT, MAP_SIZE, aggression=1.0)
check("Kill yok: defansif r_role ≥ agresif r_role",
      i_rd["r_role"] >= i_ra["r_role"],
      f"def={i_rd['r_role']:.3f}, agg={i_ra['r_role']:.3f}")


# ---------------------------------------------------------------------------
print("\n── 12. aggression_to_embedding ──────────────────────────────────")
# ---------------------------------------------------------------------------
check("embedding(0.0) = [0.0, 1.0]", np.allclose(aggression_to_embedding(0.0), [0.0, 1.0]))
check("embedding(0.5) = [0.5, 0.5]", np.allclose(aggression_to_embedding(0.5), [0.5, 0.5]))
check("embedding(1.0) = [1.0, 0.0]", np.allclose(aggression_to_embedding(1.0), [1.0, 0.0]))
check("embedding shape = (2,)",       aggression_to_embedding(0.3).shape == (2,))
check("embedding sum = 1.0",          almost_equal(float(np.sum(aggression_to_embedding(0.7))), 1.0))
check("RewardModel static method tutarlı",
      np.allclose(RewardModel.aggression_to_embedding(0.7), aggression_to_embedding(0.7)))


# ---------------------------------------------------------------------------
print("\n── 13. Info Sözlüğü Tutarlılığı ─────────────────────────────────")
# ---------------------------------------------------------------------------
weapons.reset()
total, info = rm.compute(make_state(), [], [], weapons, make_state().copy(),
                          no_fire(), DT, MAP_SIZE, aggression=0.5)

contrib_sum = sum(info[k] for k in [
    "w_kill_contrib", "w_wez_contrib", "w_tracking_contrib",
    "w_survival_contrib", "w_coord_contrib", "w_resource_contrib",
    "w_penalty_contrib", "w_role_contrib"
])
check("Total = katkılar toplamı",
      almost_equal(total, contrib_sum, tol=1e-5),
      f"total={total:.6f}, sum={contrib_sum:.6f}")

required_keys = [
    "total", "r_kill", "r_wez", "r_tracking", "r_survival",
    "r_coord", "r_resource", "r_penalty", "r_role",
    "aggression", "w_kill_scale", "w_survival_scale", "w_coord_scale",
    "w_kill_contrib", "w_wez_contrib", "w_tracking_contrib",
    "w_survival_contrib", "w_coord_contrib", "w_resource_contrib",
    "w_penalty_contrib", "w_role_contrib",
]
for k in required_keys:
    check(f"Info key '{k}' mevcut", k in info)


# ---------------------------------------------------------------------------
print("\n── 14. Ölü Ajan → Reward = 0 ────────────────────────────────────")
# ---------------------------------------------------------------------------
weapons.reset()
dead = make_state(alive=0.0)
total_dead, info_dead = rm.compute(dead, [], [], weapons, dead.copy(),
                                    no_fire(), DT, MAP_SIZE, aggression=0.5)
check("Ölü ajan → total=0.0",      almost_equal(total_dead, 0.0))
check("Ölü ajan → r_survival=0.0", almost_equal(info_dead["r_survival"], 0.0))
check("Ölü ajan → r_role=0.0",     almost_equal(info_dead["r_role"], 0.0))


# ---------------------------------------------------------------------------
print("\n── 15. Sınır Dışı aggression → AssertionError ───────────────────")
# ---------------------------------------------------------------------------
weapons.reset()
a = make_state()
for bad, label in [(1.5, "1.5"), (-0.1, "-0.1"), (2.0, "2.0")]:
    try:
        rm.compute(a, [], [], weapons, a.copy(), no_fire(), DT, MAP_SIZE, aggression=bad)
        check(f"aggression={label} → AssertionError", False)
    except AssertionError:
        check(f"aggression={label} → AssertionError ✓", True)


# ---------------------------------------------------------------------------
print("\n── 16. Summarize ────────────────────────────────────────────────")
# ---------------------------------------------------------------------------
weapons.reset()
history = []
for agg in [0.3, 0.5, 0.7]:
    _, inf = rm.compute(make_state(), [], [], weapons, make_state().copy(),
                        no_fire(), DT, MAP_SIZE, aggression=agg)
    history.append(inf)

summary = RewardModel.summarize(history)
check("summarize: sum_total anahtarı var",   "reward/sum_total" in summary)
check("summarize: mean_aggression ≈ 0.5",
      almost_equal(summary.get("reward/mean_aggression", -1), 0.5),
      f"{summary.get('reward/mean_aggression', 'N/A')}")
check("summarize: 3 adım sum_survival ≈ 3×dt",
      almost_equal(summary["reward/sum_r_survival"], 3 * DT, tol=1e-5))

# aggression=None episode → mean_aggression anahtarı olmamalı
weapons.reset()
hist_none = [
    rm.compute(make_state(), [], [], weapons, make_state().copy(),
               no_fire(), DT, MAP_SIZE, aggression=None)[1]
    for _ in range(3)
]
summary_none = RewardModel.summarize(hist_none)
check("aggression=None episode → mean_aggression yok",
      "reward/mean_aggression" not in summary_none)


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
    print("\n⛔ reward_model.py düzeltilmeden bir sonraki adıma geçilmez.")
    sys.exit(1)
else:
    print("\n🎉 Tüm testler geçti! normalization.py yazımına geçilebilir.")
    sys.exit(0)
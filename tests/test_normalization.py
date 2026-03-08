"""
test_normalization.py
=====================
normalization.py için unit testler.

Çalıştırma:
    python test_normalization.py

Test kategorileri:
    1.  Normalizer init
    2.  _norm_pos
    3.  _norm_h
    4.  _norm_V
    5.  _norm_angle
    6.  _norm_pqr
    7.  _norm_fuel / _norm_ammo
    8.  _norm_dist
    9.  ego_obs — boyut ve aralık kontrolü
    10. teammate_obs — boyut, aralık, ölü ajan
    11. enemy_obs — boyut, aralık, ölü düşman
    12. build_obs — 2v2 Faz 0-1 (49)
    13. build_obs — 2v2 Faz 2 rol embedding (51)
    14. build_obs — 2v2 Faz 3 GAT (67)
    15. build_obs — 3v3 Faz 0-1 (70)
    16. build_obs — 3v3 Faz 2 (72)
    17. obs_dim yardımcısı
    18. normalize_action / denormalize_action
    19. Tüm değerler sonlu (NaN/Inf yok)
    20. Simetri: konum işaret değişince obs işaret değişir
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.normalization import (
    Normalizer,
    OBS_EGO_DIM, OBS_TEAMMATE_DIM, OBS_ENEMY_DIM,
    OBS_ROLE_DIM, OBS_GAT_MSG_DIM,
)
from envs.aircraft_model import (
    AircraftModel,
    STATE_P, STATE_Q, STATE_R,
    STATE_FUEL, STATE_AMMO, STATE_HP,
    STATE_ALIVE,
    ACTION_DA, ACTION_DE, ACTION_DR, ACTION_DT, ACTION_FIRE,
)
from envs.geometry_utils import deg2rad

PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = []


def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    print(f"{status} | {name}" + (f"  [{detail}]" if detail else ""))
    results.append((name, condition))


def almost(a, b, tol=1e-5):
    return abs(float(a) - float(b)) < tol


# ---------------------------------------------------------------------------
# Config ve sabitler
# ---------------------------------------------------------------------------
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
    "env": {
        "map_size": 50000.0,
        "h_max":    15000.0,
        "p_max": 3.0, "q_max": 3.0, "r_max": 3.0,
    },
    "reward": {
        "w_kill": 10.0, "w_wez": 2.0, "w_tracking": 1.0,
        "w_survival": 0.1, "w_coord": 0.5, "w_resource": 0.1,
        "w_penalty": -5.0, "w_role": 1.0,
    },
    "roles": {},
    "coord": {"dist_min": 500.0, "dist_max": 8000.0, "dist_opt": 3000.0},
}

aircraft = AircraftModel(TEST_CONFIG)
norm     = Normalizer(TEST_CONFIG)

MAP   = 50000.0
HALF  = MAP / 2.0


def mk(x=0.0, y=0.0, h=3000.0, V=200.0, psi_deg=0.0, alpha_deg=3.0,
       p=0.0, q=0.0, r=0.0, fuel=3000.0, ammo=6.0, hp=1.0, alive=1.0):
    s = aircraft.reset({"x": x, "y": y, "h": h, "V": V,
                        "psi": deg2rad(psi_deg), "alpha": deg2rad(alpha_deg)})
    s[STATE_P] = p;  s[STATE_Q] = q; s[STATE_R] = r
    s[STATE_FUEL] = fuel; s[STATE_AMMO] = ammo
    s[STATE_HP] = hp; s[STATE_ALIVE] = alive
    return s


def in_range(arr, lo=-1.0, hi=1.0):
    return bool(np.all(arr >= lo - 1e-6) and np.all(arr <= hi + 1e-6))


# ---------------------------------------------------------------------------
print("\n── 1. Normalizer Init ───────────────────────────────────────────")
# ---------------------------------------------------------------------------
check("map_size=50000", almost(norm.map_size, 50000.0))
check("half_map=25000", almost(norm.half_map, 25000.0))
check("h_max=15000",    almost(norm.h_max,    15000.0))
check("V_max=600",      almost(norm.V_max,      600.0))
check("init_fuel=3000", almost(norm.init_fuel, 3000.0))
check("init_ammo=6",    almost(norm.init_ammo,    6.0))


# ---------------------------------------------------------------------------
print("\n── 2. _norm_pos ─────────────────────────────────────────────────")
# ---------------------------------------------------------------------------
p0  = norm._norm_pos(0.0,   0.0)
p1  = norm._norm_pos(25000, 25000)
pm  = norm._norm_pos(-25000, -25000)
pov = norm._norm_pos(30000, 30000)   # sınır dışı → clip

check("Merkez (0,0) → [0,0]",      np.allclose(p0, [0.0, 0.0]))
check("(+half, +half) → [1,1]",    np.allclose(p1, [1.0, 1.0], atol=1e-4))
check("(-half, -half) → [-1,-1]",  np.allclose(pm, [-1.0, -1.0], atol=1e-4))
check("Sınır dışı clip → max=1.0", in_range(pov, -1.0, 1.0))


# ---------------------------------------------------------------------------
print("\n── 3. _norm_h ───────────────────────────────────────────────────")
# ---------------------------------------------------------------------------
check("h=h_min → ≈ 0.0",    almost(norm._norm_h(norm.h_min),   0.0, tol=1e-3))
check("h=h_max → 1.0",      almost(norm._norm_h(norm.h_max),   1.0, tol=1e-4))
check("h=orta  → ~0.5",     0.45 < norm._norm_h((norm.h_min + norm.h_max) / 2) < 0.55)
check("h < h_min → clip 0", almost(norm._norm_h(0.0), 0.0))
check("h > h_max → clip 1", almost(norm._norm_h(20000.0), 1.0))


# ---------------------------------------------------------------------------
print("\n── 4. _norm_V ───────────────────────────────────────────────────")
# ---------------------------------------------------------------------------
check("V=V_min → ≈ 0.0", almost(norm._norm_V(norm.V_min), 0.0, tol=1e-4))
check("V=V_max → 1.0",   almost(norm._norm_V(norm.V_max), 1.0, tol=1e-4))
check("V=330   → ~0.5",  0.45 < norm._norm_V(330.0) < 0.55)
check("V < V_min → 0",   almost(norm._norm_V(0.0), 0.0))
check("V > V_max → 1",   almost(norm._norm_V(800.0), 1.0))


# ---------------------------------------------------------------------------
print("\n── 5. _norm_angle ───────────────────────────────────────────────")
# ---------------------------------------------------------------------------
check("angle=0     → 0.0",  almost(norm._norm_angle(0.0), 0.0))
# wrap_to_pi(π) = ±π (float sınırında), normalize [-1, 1] aralığında olmalı
check("angle=π     → aralıkta [-1,1]",
      -1.0 <= norm._norm_angle(np.pi) <= 1.0)
check("angle=-π    → -1.0 veya 1.0 (float sınırı)",
      almost(abs(norm._norm_angle(-np.pi)), 1.0, tol=1e-4))
check("angle=π/2   → 0.5",  almost(norm._norm_angle(np.pi/2), 0.5, tol=1e-4))
check("angle=3π (wrap) → aralıkta",
      -1.0 <= norm._norm_angle(3 * np.pi) <= 1.0)


# ---------------------------------------------------------------------------
print("\n── 6. _norm_pqr ─────────────────────────────────────────────────")
# ---------------------------------------------------------------------------
check("p=0      → 0.0",  almost(norm._norm_pqr(0.0,    3.0), 0.0))
check("p=p_max  → 1.0",  almost(norm._norm_pqr(3.0,    3.0), 1.0))
check("p=-p_max → -1.0", almost(norm._norm_pqr(-3.0,   3.0), -1.0))
check("p=1.5    → 0.5",  almost(norm._norm_pqr(1.5,    3.0), 0.5))
check("p=10 → clip 1.0", almost(norm._norm_pqr(10.0,   3.0), 1.0))


# ---------------------------------------------------------------------------
print("\n── 7. _norm_fuel / _norm_ammo ───────────────────────────────────")
# ---------------------------------------------------------------------------
check("fuel=full  → 1.0", almost(norm._norm_fuel(3000.0), 1.0))
check("fuel=0     → 0.0", almost(norm._norm_fuel(0.0),    0.0))
check("fuel=1500  → 0.5", almost(norm._norm_fuel(1500.0), 0.5, tol=1e-3))
check("ammo=full  → 1.0", almost(norm._norm_ammo(6.0),    1.0, tol=1e-3))
check("ammo=0     → 0.0", almost(norm._norm_ammo(0.0),    0.0))
check("ammo=3     → 0.5", almost(norm._norm_ammo(3.0),    0.5, tol=1e-3))


# ---------------------------------------------------------------------------
print("\n── 8. _norm_dist ────────────────────────────────────────────────")
# ---------------------------------------------------------------------------
check("dist=0        → 0.0", almost(norm._norm_dist(0.0),     0.0))
check("dist=map_size → 1.0", almost(norm._norm_dist(MAP),     1.0, tol=1e-4))
check("dist=half_map → 0.5", almost(norm._norm_dist(HALF),    0.5, tol=1e-4))
check("dist > map → clip 1", almost(norm._norm_dist(60000.0), 1.0))


# ---------------------------------------------------------------------------
print("\n── 9. ego_obs — boyut ve aralık ─────────────────────────────────")
# ---------------------------------------------------------------------------
s = mk()
ego = norm.ego_obs(s)

check(f"ego_obs boyutu = {OBS_EGO_DIM}", len(ego) == OBS_EGO_DIM, f"{len(ego)}")
check("ego_obs dtype float32",  ego.dtype == np.float32)
check("ego_obs [-1, 1] aralığı", in_range(ego, -1.0, 1.0))
check("ego_obs NaN yok",  not np.any(np.isnan(ego)))
check("ego_obs Inf yok",  not np.any(np.isinf(ego)))

# Değer spot kontrolleri
check("ego h_norm pozitif (h=3000)", ego[2] > 0.0, f"h_norm={ego[2]:.3f}")
check("ego V_norm pozitif (V=200)",  ego[3] > 0.0, f"V_norm={ego[3]:.3f}")
check("ego hp=1.0",                  almost(ego[15], 1.0))

# Kenar durumlar
s_max = mk(x=25000, y=25000, h=15000, V=600, fuel=3000, ammo=6, hp=1.0)
ego_max = norm.ego_obs(s_max)
check("Maks değerler → [-1,1]",     in_range(ego_max, -1.0, 1.0))

s_min = mk(x=-25000, y=-25000, h=50, V=60, fuel=0, ammo=0, hp=0.0)
ego_min = norm.ego_obs(s_min)
check("Min değerler → [-1,1]",      in_range(ego_min, -1.0, 1.0))


# ---------------------------------------------------------------------------
print("\n── 10. teammate_obs ─────────────────────────────────────────────")
# ---------------------------------------------------------------------------
agent = mk(x=0.0, y=0.0, h=3000.0)
tm    = mk(x=1000.0, y=2000.0, h=4000.0, hp=0.8)

tm_obs = norm.teammate_obs(agent, tm)
check(f"teammate_obs boyutu = {OBS_TEAMMATE_DIM}", len(tm_obs) == OBS_TEAMMATE_DIM, f"{len(tm_obs)}")
check("teammate_obs dtype float32", tm_obs.dtype == np.float32)
check("teammate_obs [-1,1]",        in_range(tm_obs, -1.0, 1.0))
check("teammate_obs NaN/Inf yok",   not (np.any(np.isnan(tm_obs)) or np.any(np.isinf(tm_obs))))
check("hp_teammate = 0.8",          almost(tm_obs[7], 0.8, tol=1e-4))
check("alive = 1.0",                almost(tm_obs[8], 1.0))

# Ölü takım arkadaşı → sıfır vektör
tm_dead = mk(alive=0.0)
tm_dead_obs = norm.teammate_obs(agent, tm_dead)
check("Ölü takım → sıfır vektör",   np.allclose(tm_dead_obs, 0.0))

# Aynı konumda → mesafe=0, NaN yok
tm_same = mk()
tm_same_obs = norm.teammate_obs(agent, tm_same)
check("Aynı konum → NaN yok", not np.any(np.isnan(tm_same_obs)))


# ---------------------------------------------------------------------------
print("\n── 11. enemy_obs ────────────────────────────────────────────────")
# ---------------------------------------------------------------------------
agent_e = mk(x=0.0, y=0.0, psi_deg=0.0)
enemy   = mk(x=0.0, y=3000.0, psi_deg=180.0, hp=0.6)

en_obs = norm.enemy_obs(agent_e, enemy)
check(f"enemy_obs boyutu = {OBS_ENEMY_DIM}", len(en_obs) == OBS_ENEMY_DIM, f"{len(en_obs)}")
check("enemy_obs dtype float32", en_obs.dtype == np.float32)
check("enemy_obs [-1,1]",        in_range(en_obs, -1.0, 1.0))
check("enemy_obs NaN/Inf yok",   not (np.any(np.isnan(en_obs)) or np.any(np.isinf(en_obs))))
check("hp_enemy = 0.6",          almost(en_obs[10], 0.6, tol=1e-4))
check("alive = 1.0",             almost(en_obs[11], 1.0))

# Mesafe [0,1]
check("dist_norm [0,1]", 0.0 <= en_obs[8] <= 1.0, f"dist_norm={en_obs[8]:.3f}")
# Tehdit skoru [0,1]
check("threat [0,1]",    0.0 <= en_obs[9] <= 1.0, f"threat={en_obs[9]:.3f}")

# Ölü düşman → sıfır vektör
en_dead = mk(alive=0.0)
en_dead_obs = norm.enemy_obs(agent_e, en_dead)
check("Ölü düşman → sıfır vektör", np.allclose(en_dead_obs, 0.0))

# Aynı konum → NaN yok
en_same = mk()
check("Aynı konum düşman → NaN yok",
      not np.any(np.isnan(norm.enemy_obs(agent_e, en_same))))


# ---------------------------------------------------------------------------
print("\n── 12. build_obs — 2v2 Faz 0-1 (49) ────────────────────────────")
# ---------------------------------------------------------------------------
a  = mk()
t1 = mk(x=1000.0, y=500.0)
e1 = mk(x=-500.0, y=3000.0)
e2 = mk(x=2000.0, y=1500.0)

obs_2v2 = norm.build_obs(a, [t1], [e1, e2], aggression=None)
expected_2v2 = OBS_EGO_DIM + OBS_TEAMMATE_DIM * 1 + OBS_ENEMY_DIM * 2  # 49

check(f"2v2 Faz 0-1 boyut = {expected_2v2}",
      len(obs_2v2) == expected_2v2, f"{len(obs_2v2)}")
check("2v2 obs [-1,1]",    in_range(obs_2v2, -1.0, 1.0))
check("2v2 obs NaN/Inf yok",
      not (np.any(np.isnan(obs_2v2)) or np.any(np.isinf(obs_2v2))))
check("2v2 obs dtype float32", obs_2v2.dtype == np.float32)


# ---------------------------------------------------------------------------
print("\n── 13. build_obs — 2v2 Faz 2 rol embedding (51) ────────────────")
# ---------------------------------------------------------------------------
obs_2v2_role = norm.build_obs(a, [t1], [e1, e2], aggression=0.7)
expected_2v2_role = expected_2v2 + OBS_ROLE_DIM  # 51

check(f"2v2 Faz 2 boyut = {expected_2v2_role}",
      len(obs_2v2_role) == expected_2v2_role, f"{len(obs_2v2_role)}")
check("2v2 Faz 2 obs [-1,1]", in_range(obs_2v2_role, -1.0, 1.0))

# Rol embedding doğru değerlerde mi?
role_emb = obs_2v2_role[-OBS_ROLE_DIM:]
check("rol embedding[0] = 0.7", almost(role_emb[0], 0.7, tol=1e-5))
check("rol embedding[1] = 0.3", almost(role_emb[1], 0.3, tol=1e-5))
check("rol embedding toplamı = 1.0", almost(float(np.sum(role_emb)), 1.0))

# Faz 0-1 ile başlangıç kısmı aynı
check("Faz 2 önceki kısım değişmedi",
      np.allclose(obs_2v2_role[:expected_2v2], obs_2v2))


# ---------------------------------------------------------------------------
print("\n── 14. build_obs — 2v2 Faz 3 GAT mesajı (67) ───────────────────")
# ---------------------------------------------------------------------------
np.random.seed(42)
gat_msg = [np.random.randn(OBS_GAT_MSG_DIM).astype(np.float32)]
obs_2v2_gat = norm.build_obs(a, [t1], [e1, e2], aggression=0.7, gat_messages=gat_msg)
expected_2v2_gat = expected_2v2_role + OBS_GAT_MSG_DIM * 1  # 67

check(f"2v2 Faz 3 boyut = {expected_2v2_gat}",
      len(obs_2v2_gat) == expected_2v2_gat, f"{len(obs_2v2_gat)}")
# GAT mesajları normalize edilmez (ham iletişim vektörü) — rol+önceki kısım değişmemeli
n_role = expected_2v2_role  # 51
check("Faz 3 rol+önceki kısım değişmedi (GAT hariç)",
      np.allclose(obs_2v2_gat[:n_role], obs_2v2_role))


# ---------------------------------------------------------------------------
print("\n── 15. build_obs — 3v3 Faz 0-1 (70) ────────────────────────────")
# ---------------------------------------------------------------------------
t2 = mk(x=-1000.0, y=1500.0)
e3 = mk(x=3000.0,  y=-500.0)

obs_3v3 = norm.build_obs(a, [t1, t2], [e1, e2, e3], aggression=None)
expected_3v3 = OBS_EGO_DIM + OBS_TEAMMATE_DIM * 2 + OBS_ENEMY_DIM * 3  # 70

check(f"3v3 Faz 0-1 boyut = {expected_3v3}",
      len(obs_3v3) == expected_3v3, f"{len(obs_3v3)}")
check("3v3 obs [-1,1]",     in_range(obs_3v3, -1.0, 1.0))
check("3v3 obs NaN/Inf yok",
      not (np.any(np.isnan(obs_3v3)) or np.any(np.isinf(obs_3v3))))


# ---------------------------------------------------------------------------
print("\n── 16. build_obs — 3v3 Faz 2 (72) ──────────────────────────────")
# ---------------------------------------------------------------------------
obs_3v3_role = norm.build_obs(a, [t1, t2], [e1, e2, e3], aggression=0.0)
expected_3v3_role = expected_3v3 + OBS_ROLE_DIM  # 72

check(f"3v3 Faz 2 boyut = {expected_3v3_role}",
      len(obs_3v3_role) == expected_3v3_role, f"{len(obs_3v3_role)}")
# aggression=0.0 → embedding=[0.0, 1.0]
role_emb_3v3 = obs_3v3_role[-OBS_ROLE_DIM:]
check("aggression=0.0 embedding = [0,1]",
      np.allclose(role_emb_3v3, [0.0, 1.0], atol=1e-5))


# ---------------------------------------------------------------------------
print("\n── 17. obs_dim yardımcısı ────────────────────────────────────────")
# ---------------------------------------------------------------------------
check("obs_dim 2v2 Faz 0-1 = 49",
      norm.obs_dim(n_teammates=1, n_enemies=2) == 49)
check("obs_dim 2v2 Faz 2   = 51",
      norm.obs_dim(n_teammates=1, n_enemies=2, with_role=True) == 51)
check("obs_dim 2v2 Faz 3   = 67",
      norm.obs_dim(n_teammates=1, n_enemies=2, with_role=True, with_gat=True) == 67)
check("obs_dim 3v3 Faz 0-1 = 70",
      norm.obs_dim(n_teammates=2, n_enemies=3) == 70)
check("obs_dim 3v3 Faz 2   = 72",
      norm.obs_dim(n_teammates=2, n_enemies=3, with_role=True) == 72)
check("obs_dim 3v3 Faz 3   = 104",
      norm.obs_dim(n_teammates=2, n_enemies=3, with_role=True, with_gat=True) == 104)


# ---------------------------------------------------------------------------
print("\n── 18. normalize_action / denormalize_action ────────────────────")
# ---------------------------------------------------------------------------
# Geçerli aksiyon
valid_action = np.array([0.5, -0.3, 0.1, 0.8, 0.6], dtype=np.float32)
norm_a = Normalizer.normalize_action(valid_action)
check("Geçerli aksiyon değişmeden geçer",  np.allclose(norm_a, valid_action))

# Sınır dışı → clip
oob_action = np.array([1.5, -2.0, 0.1, -0.5, 1.2], dtype=np.float32)
clipped = Normalizer.normalize_action(oob_action)
check("Aileron clip [-1,1]",  -1.0 <= clipped[ACTION_DA]   <= 1.0)
check("Elevator clip [-1,1]", -1.0 <= clipped[ACTION_DE]   <= 1.0)
check("Rudder clip [-1,1]",   -1.0 <= clipped[ACTION_DR]   <= 1.0)
check("Throttle clip [0,1]",   0.0 <= clipped[ACTION_DT]   <= 1.0)
check("Fire clip [0,1]",       0.0 <= clipped[ACTION_FIRE] <= 1.0)
check("Throttle negatif → 0.0", almost(clipped[ACTION_DT], 0.0))

# denormalize = normalize (şu an identity)
denorm_a = Normalizer.denormalize_action(valid_action)
check("denormalize_action = normalize_action (identity)",
      np.allclose(denorm_a, norm_a))


# ---------------------------------------------------------------------------
print("\n── 19. NaN / Inf güvenlik taraması ──────────────────────────────")
# ---------------------------------------------------------------------------
# Çeşitli uç konumlar
test_states = [
    mk(x=24999, y=24999, h=14999, V=599),
    mk(x=-24999, y=-24999, h=51, V=61),
    mk(x=0, y=0, h=7500, V=330, p=2.9, q=2.9, r=2.9),
    mk(fuel=1.0, ammo=1.0, hp=0.01),
    mk(hp=0.0, alive=1.0),
]
for i, st in enumerate(test_states):
    ego_t = norm.ego_obs(st)
    check(f"Uç durum {i+1} ego_obs NaN/Inf yok",
          not (np.any(np.isnan(ego_t)) or np.any(np.isinf(ego_t))))


# ---------------------------------------------------------------------------
print("\n── 20. Simetri: konum işaret değişince obs işaret değişir ───────")
# ---------------------------------------------------------------------------
a_pos = mk(x=5000.0,  y=0.0)
a_neg = mk(x=-5000.0, y=0.0)
ego_pos = norm.ego_obs(a_pos)
ego_neg = norm.ego_obs(a_neg)

# x normalize değeri işaret değişmeli
check("x_norm simetri: +x → pozitif, -x → negatif",
      ego_pos[0] > 0 and ego_neg[0] < 0,
      f"+x={ego_pos[0]:.3f}, -x={ego_neg[0]:.3f}")
check("x_norm simetri: büyüklük eşit",
      almost(abs(ego_pos[0]), abs(ego_neg[0]), tol=1e-4))


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
    print("\n⛔ normalization.py düzeltilmeden bir sonraki adıma geçilmez.")
    sys.exit(1)
else:
    print("\n🎉 Tüm testler geçti! dogfight_env.py yazımına geçilebilir.")
    sys.exit(0)
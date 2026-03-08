"""
test_aircraft_model.py
======================
aircraft_model.py için fizik doğrulama testleri.

Çalıştırma:
    python test_aircraft_model.py

Test kategorileri:
    1. ISA atmosfer modeli
    2. Kontrol yüzeyi clip
    3. Trim durumu (düz ve seviyeli uçuş)
    4. Enerji korunumu (yaklaşık)
    5. Singularity koruması (gamma, theta ±90°)
    6. Stall tespiti
    7. Momentum tepkileri (aileron, elevator, rudder)
    8. Yakıt tüketimi
    9. Ixz etkisi (roll-yaw coupling)
    10. ISA yoğunluk sıralaması
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.aircraft_model import (
    AircraftModel,
    STATE_X, STATE_Y, STATE_H, STATE_V, STATE_ALPHA, STATE_BETA,
    STATE_GAMMA, STATE_PHI, STATE_THETA, STATE_PSI,
    STATE_P, STATE_Q, STATE_R, STATE_FUEL, STATE_AMMO,
    STATE_HP, STATE_RADAR, STATE_ALIVE, STATE_DIM,
    ACTION_DA, ACTION_DE, ACTION_DR, ACTION_DT, ACTION_FIRE, ACTION_DIM
)
from envs.geometry_utils import deg2rad, rad2deg

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


def almost_equal(a, b, tol=1e-4):
    return abs(a - b) < tol


# ---------------------------------------------------------------------------
# Test config — config.yaml yapısını taklit eder
# ---------------------------------------------------------------------------
TEST_CONFIG = {
    "aircraft": {
        "mass":         9100.0,
        "wingspan":     9.45,
        "wing_area":    27.87,
        "mean_chord":   3.45,
        "Ixx":          12875.0,
        "Iyy":          75674.0,
        "Izz":          85552.0,
        "Ixz":          1331.0,
        "max_thrust":   76300.0,
        "SFC":          2.0e-5,
        "initial_fuel": 3000.0,
        "initial_ammo": 6,
        "initial_hp":   1.0,
        "radar_range":  15000.0,
        "V_min":        60.0,
        "V_max":        600.0,
        "alpha_max":    deg2rad(25.0),
        "alpha_min":    deg2rad(-10.0),
        "h_min":        50.0,
    },
    "aero_coeffs": {
        "CL0":      0.0,
        "CL_alpha": 5.5,
        "CL_q":     1.8,
        "CL_de":    0.5,
        "CD0":      0.013,
        "CD_alpha": 0.5,
        "CY_beta":  -0.35,
        "CY_da":    0.0,
        "CY_dr":    0.10,
        "Cl_p":     -0.41,
        "Cl_r":     0.15,
        "Cl_da":    0.18,
        "Cl_dr":    0.01,
        "Cm0":      0.04,
        "Cm_alpha": -0.65,
        "Cm_q":     -12.4,
        "Cm_de":    -1.1,
        "Cn_p":     -0.05,
        "Cn_r":     -0.20,
        "Cn_da":    -0.02,
        "Cn_dr":    0.15,
    },
    "control_limits": {
        "aileron_max_deg":  21.5,
        "elevator_max_deg": 25.0,
        "rudder_max_deg":   30.0,
    }
}

aircraft = AircraftModel(TEST_CONFIG)
DT = 0.05   # adım süresi (config.yaml ile uyumlu)


def make_state(h=3000.0, V=200.0, alpha_deg=3.0, gamma_deg=0.0,
               phi_deg=0.0, theta_deg=3.0, psi_deg=0.0,
               p=0.0, q=0.0, r=0.0, fuel=3000.0):
    """Test için hızlı state oluşturucu."""
    return aircraft.reset({
        "x": 0.0, "y": 0.0, "h": h,
        "V": V,
        "alpha": deg2rad(alpha_deg),
        "beta":  0.0,
        "gamma": deg2rad(gamma_deg),
        "phi":   deg2rad(phi_deg),
        "theta": deg2rad(theta_deg),
        "psi":   deg2rad(psi_deg),
        "p": p, "q": q, "r": r,
    })


def neutral_action(throttle=0.5):
    """Nötr aksiyon: sıfır kontrol yüzeyi."""
    a = np.zeros(ACTION_DIM)
    a[ACTION_DT] = throttle
    return a


# ---------------------------------------------------------------------------
print("\n── 1. Reset ve State Boyutu ──────────────────────────────────────")
# ---------------------------------------------------------------------------

state0 = make_state()
check("State boyutu STATE_DIM=18", len(state0) == STATE_DIM,
      f"len={len(state0)}")
check("Başlangıç alive=1", state0[STATE_ALIVE] == 1.0)
check("Başlangıç fuel=initial_fuel",
      almost_equal(state0[STATE_FUEL], 3000.0))
check("Başlangıç hp=1.0", almost_equal(state0[STATE_HP], 1.0))
check("Başlangıç ammo=6", almost_equal(state0[STATE_AMMO], 6.0))


# ---------------------------------------------------------------------------
print("\n── 2. ISA Atmosfer Modeli ────────────────────────────────────────")
# ---------------------------------------------------------------------------

rho0, T0 = aircraft._isa_atmosphere(0.0)
check("ISA deniz seviyesi rho=1.225", almost_equal(rho0, 1.225, tol=0.005),
      f"rho={rho0:.4f}")
check("ISA deniz seviyesi T=288.15K", almost_equal(T0, 288.15, tol=0.1),
      f"T={T0:.2f}K")

rho5k, T5k = aircraft._isa_atmosphere(5000.0)
check("ISA 5000m rho < deniz seviyesi", rho5k < rho0,
      f"rho5k={rho5k:.4f}")
check("ISA 5000m rho > 0", rho5k > 0.0)

rho11k, T11k = aircraft._isa_atmosphere(11000.0)
check("ISA tropopoz T≈216.65K", almost_equal(T11k, 216.65, tol=0.5),
      f"T11k={T11k:.2f}K")

rho15k, _ = aircraft._isa_atmosphere(15000.0)
check("ISA 15000m rho < 11000m rho", rho15k < rho11k,
      f"rho15k={rho15k:.4f}")

# Yoğunluk her zaman pozitif ve azalan
rhos = [aircraft._isa_atmosphere(h)[0] for h in [0, 2000, 5000, 8000, 11000, 15000]]
check("ISA rho monoton azalıyor", all(rhos[i] > rhos[i+1] for i in range(len(rhos)-1)))

# Negatif irtifa → 0 olarak alınmalı
rho_neg, _ = aircraft._isa_atmosphere(-100.0)
check("ISA negatif irtifa → deniz seviyesi", almost_equal(rho_neg, rho0, tol=0.001))


# ---------------------------------------------------------------------------
print("\n── 3. Kontrol Yüzeyi Limitleri ──────────────────────────────────")
# ---------------------------------------------------------------------------

# Aşırı aksiyon clip edilmeli
action_over = np.array([2.0, -3.0, 1.5, 2.0, 0.5])
clipped = aircraft._clip_action(action_over)
check("Aileron clip +1", almost_equal(clipped[ACTION_DA], 1.0))
check("Elevator clip -1", almost_equal(clipped[ACTION_DE], -1.0))
check("Rudder clip +1", almost_equal(clipped[ACTION_DR], 1.0))
check("Throttle clip +1", almost_equal(clipped[ACTION_DT], 1.0))

action_under = np.array([-2.0, 0.5, -2.0, -1.0, -0.5])
clipped2 = aircraft._clip_action(action_under)
check("Aileron clip -1", almost_equal(clipped2[ACTION_DA], -1.0))
check("Throttle clip 0", almost_equal(clipped2[ACTION_DT], 0.0))


# ---------------------------------------------------------------------------
print("\n── 4. Step Fonksiyonu — Temel ───────────────────────────────────")
# ---------------------------------------------------------------------------

state0 = make_state()
action0 = neutral_action(throttle=0.6)
state1 = aircraft.step(state0, action0, DT)

check("Step sonrası STATE_DIM korunuyor", len(state1) == STATE_DIM)
check("Step sonrası alive=1", state1[STATE_ALIVE] == 1.0)

# Ölü uçak step ile değişmemeli
state_dead = make_state()
state_dead[STATE_ALIVE] = 0.0
state_dead2 = aircraft.step(state_dead, action0, DT)
check("Ölü uçak step ile değişmez",
      np.allclose(state_dead, state_dead2))


# ---------------------------------------------------------------------------
print("\n── 5. Yakıt Tüketimi ────────────────────────────────────────────")
# ---------------------------------------------------------------------------

state_fuel = make_state(fuel=3000.0)
action_full = neutral_action(throttle=1.0)
action_idle = neutral_action(throttle=0.0)

state_after_full = aircraft.step(state_fuel.copy(), action_full, DT)
state_after_idle = aircraft.step(state_fuel.copy(), action_idle, DT)

check("Tam gaz yakıt azalıyor",
      state_after_full[STATE_FUEL] < state_fuel[STATE_FUEL])
check("Rölanti yakıt azalmıyor (thrust=0)",
      almost_equal(state_after_idle[STATE_FUEL], state_fuel[STATE_FUEL], tol=0.01))
check("Tam gaz > rölanti yakıt tüketimi",
      state_after_full[STATE_FUEL] < state_after_idle[STATE_FUEL])

# Yakıt negatife düşmemeli
state_empty = make_state(fuel=0.0)
state_empty[STATE_FUEL] = 0.0
state_after_empty = aircraft.step(state_empty, action_full, DT)
check("Yakıt negatife düşmez", state_after_empty[STATE_FUEL] >= 0.0)


# ---------------------------------------------------------------------------
print("\n── 6. Momentum Tepkileri ────────────────────────────────────────")
# ---------------------------------------------------------------------------

state_base = make_state(p=0.0, q=0.0, r=0.0)

# Pozitif aileron → sağ roll → p artmalı
action_aileron_pos = neutral_action(0.5)
action_aileron_pos[ACTION_DA] = 1.0
s_after_aileron = aircraft.step(state_base.copy(), action_aileron_pos, DT)
check("Pozitif aileron → p artıyor",
      s_after_aileron[STATE_P] > state_base[STATE_P],
      f"p_before={state_base[STATE_P]:.4f}, p_after={s_after_aileron[STATE_P]:.4f}")

# Negatif elevator (burnu aşağı) → q azalmalı
action_elev_neg = neutral_action(0.5)
action_elev_neg[ACTION_DE] = -1.0
s_after_elev = aircraft.step(state_base.copy(), action_elev_neg, DT)
check("Negatif elevator → q değişiyor (burnu aşağı)",
      s_after_elev[STATE_Q] != state_base[STATE_Q],
      f"q_diff={s_after_elev[STATE_Q] - state_base[STATE_Q]:.4f}")

# Pozitif rudder → sağ yaw → r artmalı
action_rudder_pos = neutral_action(0.5)
action_rudder_pos[ACTION_DR] = 1.0
s_after_rudder = aircraft.step(state_base.copy(), action_rudder_pos, DT)
check("Pozitif rudder → r değişiyor",
      s_after_rudder[STATE_R] != state_base[STATE_R],
      f"r_diff={s_after_rudder[STATE_R] - state_base[STATE_R]:.4f}")

# Antimetrik test: ters aileron ters etki
action_aileron_neg = neutral_action(0.5)
action_aileron_neg[ACTION_DA] = -1.0
s_after_aileron_neg = aircraft.step(state_base.copy(), action_aileron_neg, DT)
check("Ters aileron → ters p",
      s_after_aileron_neg[STATE_P] < state_base[STATE_P])


# ---------------------------------------------------------------------------
print("\n── 7. Singularity Koruması ──────────────────────────────────────")
# ---------------------------------------------------------------------------

# Gamma ≈ ±85° — singularity bölgesi
state_steep = make_state(gamma_deg=84.9, theta_deg=84.9)
action_s = neutral_action(0.8)
try:
    s_after_steep = aircraft.step(state_steep, action_s, DT)
    # NaN veya inf olmamalı
    check("Steep climb NaN/Inf yok",
          not np.any(np.isnan(s_after_steep)) and
          not np.any(np.isinf(s_after_steep)),
          f"gamma={rad2deg(s_after_steep[STATE_GAMMA]):.1f}°")
    check("Steep climb gamma sınırda kalıyor",
          abs(s_after_steep[STATE_GAMMA]) <= deg2rad(86.0))
except Exception as e:
    check("Steep climb crash yok", False, str(e))

# Theta ≈ ±85°
state_theta_lim = make_state(theta_deg=84.9)
try:
    s_theta = aircraft.step(state_theta_lim, action_s, DT)
    check("Theta limit NaN/Inf yok",
          not np.any(np.isnan(s_theta)) and
          not np.any(np.isinf(s_theta)))
except Exception as e:
    check("Theta limit crash yok", False, str(e))

# V çok küçük (stall yakını)
state_slow = make_state(V=65.0)
try:
    s_slow = aircraft.step(state_slow, neutral_action(0.3), DT)
    check("Düşük hız NaN/Inf yok",
          not np.any(np.isnan(s_slow)) and
          not np.any(np.isinf(s_slow)))
except Exception as e:
    check("Düşük hız crash yok", False, str(e))


# ---------------------------------------------------------------------------
print("\n── 8. İrtifa Limiti (Zemin) ─────────────────────────────────────")
# ---------------------------------------------------------------------------

# Düşük irtifada gamma < 0 olmamalı (zemin çarpışmasını önler)
state_ground = make_state(h=52.0, gamma_deg=-10.0)
action_dive = neutral_action(0.5)
action_dive[ACTION_DE] = -0.5

# Birkaç step simüle et
s = state_ground.copy()
for _ in range(20):
    s = aircraft.step(s, action_dive, DT)

check("Zemin sınırında irtifa h_min'de kalıyor",
      s[STATE_H] >= aircraft.h_min - 1.0,
      f"h={s[STATE_H]:.1f}m, h_min={aircraft.h_min}m")


# ---------------------------------------------------------------------------
print("\n── 9. Stall Tespiti ─────────────────────────────────────────────")
# ---------------------------------------------------------------------------

# Normal uçuşta stall olmamalı
state_normal = make_state(V=200.0, alpha_deg=5.0)
check("Normal uçuşta stall yok",
      not aircraft.is_stalled(state_normal))

# Yüksek alpha → stall
state_stall_alpha = make_state(alpha_deg=26.0)
check("Yüksek alpha → stall",
      aircraft.is_stalled(state_stall_alpha))

# Düşük hız → stall
state_stall_v = make_state(V=50.0)
check("Düşük hız → stall",
      aircraft.is_stalled(state_stall_v))


# ---------------------------------------------------------------------------
print("\n── 10. Harita Sınırı Tespiti ────────────────────────────────────")
# ---------------------------------------------------------------------------

MAP_SIZE = 50000.0

state_center = make_state()
state_center[STATE_X] = 0.0
state_center[STATE_Y] = 0.0
check("Merkez → harita içinde",
      not aircraft.is_out_of_bounds(state_center, MAP_SIZE))

state_edge = make_state()
state_edge[STATE_X] = 26000.0  # > 25000
check("Sınır dışı X → out_of_bounds",
      aircraft.is_out_of_bounds(state_edge, MAP_SIZE))

state_low = make_state(h=10.0)
check("Düşük irtifa → out_of_bounds",
      aircraft.is_out_of_bounds(state_low, MAP_SIZE))


# ---------------------------------------------------------------------------
print("\n── 11. Enerji Korunumu (Yaklaşık) ───────────────────────────────")
# ---------------------------------------------------------------------------

# Tam gaz, düz uçuş, 10 saniye
# Toplam mekanik enerji ≈ artan kinetik + potansiyel enerji
# (thrust > drag olduğu için enerji artmalı)
state_energy = make_state(h=3000.0, V=200.0, gamma_deg=0.0)
action_full_throttle = neutral_action(throttle=1.0)

m = aircraft.mass
g = 9.80665

E0 = 0.5 * m * state_energy[STATE_V]**2 + m * g * state_energy[STATE_H]

s = state_energy.copy()
for _ in range(int(10.0 / DT)):  # 10 saniye
    s = aircraft.step(s, action_full_throttle, DT)

E1 = 0.5 * m * s[STATE_V]**2 + m * g * s[STATE_H]

check("Tam gaz → enerji artıyor (thrust > drag)",
      E1 > E0,
      f"E0={E0/1e6:.3f}MJ, E1={E1/1e6:.3f}MJ")

# Rölanti, düz uçuş, 10 saniye
# Enerji azalmalı (drag > thrust=0)
state_energy2 = make_state(h=3000.0, V=300.0, gamma_deg=0.0)
action_idle2 = neutral_action(throttle=0.0)
E0b = 0.5 * m * state_energy2[STATE_V]**2 + m * g * state_energy2[STATE_H]

s2 = state_energy2.copy()
for _ in range(int(10.0 / DT)):
    s2 = aircraft.step(s2, action_idle2, DT)

E1b = 0.5 * m * s2[STATE_V]**2 + m * g * s2[STATE_H]

check("Rölanti → enerji azalıyor (drag > thrust)",
      E1b < E0b,
      f"E0={E0b/1e6:.3f}MJ, E1={E1b/1e6:.3f}MJ")


# ---------------------------------------------------------------------------
print("\n── 12. Ixz Roll-Yaw Coupling ────────────────────────────────────")
# ---------------------------------------------------------------------------

# Ixz=0 vs Ixz=1331 karşılaştırması
# Saf roll manevrında yaw coupling görülmeli

config_no_ixz = {k: v for k, v in TEST_CONFIG.items()}
config_no_ixz["aircraft"] = {**TEST_CONFIG["aircraft"], "Ixz": 0.0}
aircraft_no_ixz = AircraftModel(config_no_ixz)

state_roll = make_state(p=0.0, q=0.0, r=0.0)
action_roll = neutral_action(0.5)
action_roll[ACTION_DA] = 1.0  # tam aileron

# 1 saniye roll
s_ixz    = state_roll.copy()
s_no_ixz = state_roll.copy()

for _ in range(int(1.0 / DT)):
    s_ixz    = aircraft.step(s_ixz, action_roll, DT)
    s_no_ixz = aircraft_no_ixz.step(s_no_ixz, action_roll, DT)

# Ixz varsa r (yaw rate) farklı olmalı
r_diff = abs(s_ixz[STATE_R] - s_no_ixz[STATE_R])
check("Ixz roll-yaw coupling etkisi gözlemleniyor",
      r_diff > 1e-6,
      f"r_ixz={s_ixz[STATE_R]:.4f}, r_no_ixz={s_no_ixz[STATE_R]:.4f}, diff={r_diff:.6f}")


# ---------------------------------------------------------------------------
print("\n── 13. ISA Thrust İrtifa Etkisi ─────────────────────────────────")
# ---------------------------------------------------------------------------

# Yüksek irtifada thrust daha düşük olmalı
thrust_sl  = aircraft._compute_thrust(1.0, 200.0, 0.0)
thrust_5k  = aircraft._compute_thrust(1.0, 200.0, 5000.0)
thrust_10k = aircraft._compute_thrust(1.0, 200.0, 10000.0)

check("Thrust irtifayla azalıyor (SL > 5km)",
      thrust_sl > thrust_5k,
      f"SL={thrust_sl:.0f}N, 5km={thrust_5k:.0f}N")
check("Thrust irtifayla azalıyor (5km > 10km)",
      thrust_5k > thrust_10k,
      f"5km={thrust_5k:.0f}N, 10km={thrust_10k:.0f}N")
check("Thrust her zaman pozitif",
      thrust_10k > 0.0)


# ---------------------------------------------------------------------------
print("\n── 14. NaN / Inf Kapsamlı Tarama ────────────────────────────────")
# ---------------------------------------------------------------------------

# Farklı koşullarda 100 random step — hiç NaN/Inf olmamalı
rng = np.random.default_rng(42)
nan_found = False

for trial in range(20):
    s = aircraft.reset({
        "x": rng.uniform(-5000, 5000),
        "y": rng.uniform(-5000, 5000),
        "h": rng.uniform(500, 10000),
        "V": rng.uniform(80, 400),
        "alpha": deg2rad(rng.uniform(-5, 20)),
        "beta":  deg2rad(rng.uniform(-5, 5)),
        "gamma": deg2rad(rng.uniform(-30, 30)),
        "phi":   deg2rad(rng.uniform(-60, 60)),
        "theta": deg2rad(rng.uniform(-30, 30)),
        "psi":   deg2rad(rng.uniform(-180, 180)),
        "p": rng.uniform(-0.5, 0.5),
        "q": rng.uniform(-0.3, 0.3),
        "r": rng.uniform(-0.3, 0.3),
    })
    for step_i in range(50):
        a = rng.uniform(-1, 1, ACTION_DIM)
        a[ACTION_DT] = abs(a[ACTION_DT])  # throttle ≥ 0
        s = aircraft.step(s, a, DT)
        if np.any(np.isnan(s)) or np.any(np.isinf(s)):
            nan_found = True
            print(f"  ⚠️  Trial {trial}, Step {step_i}: NaN/Inf bulundu!")
            print(f"  State: {s}")
            break

check("20 random trial × 50 step → NaN/Inf yok", not nan_found)


# ---------------------------------------------------------------------------
# Sonuç Özeti
# ---------------------------------------------------------------------------
print("\n" + "="*60)
total  = len(results)
passed = sum(1 for _, ok in results if ok)
failed = total - passed

print(f"TOPLAM : {total}")
print(f"✅ PASS : {passed}")
print(f"❌ FAIL : {failed}")

if failed > 0:
    print("\nBaşarısız testler:")
    for name, ok in results:
        if not ok:
            print(f"  ❌ {name}")
    print("\n⛔ aircraft_model.py düzeltilmeden bir sonraki adıma geçilmez.")
    sys.exit(1)
else:
    print("\n🎉 Tüm testler geçti! weapons_model.py yazımına geçilebilir.")
    sys.exit(0)

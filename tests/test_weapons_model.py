"""
test_weapons_model.py
=====================
weapons_model.py için unit testler.

Çalıştırma:
    python test_weapons_model.py
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.weapons_model import WeaponsModel
from envs.aircraft_model import (
    AircraftModel,
    STATE_X, STATE_Y, STATE_H, STATE_PSI,
    STATE_AMMO, STATE_HP, STATE_ALIVE, STATE_FUEL,
    ACTION_DIM, ACTION_FIRE
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


def almost_equal(a, b, tol=1e-6):
    return abs(a - b) < tol


# ---------------------------------------------------------------------------
# Test config
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
        "wez_range_max":      8000.0,
        "wez_range_min":      300.0,
        "wez_angle_max":      30.0,      # derece — WeaponsModel içinde rad'a çevrilir
        "missile_damage":     0.6,
        "fire_cooldown":      2.0,
        "min_fire_altitude":  200.0,
    }
}

aircraft = AircraftModel(TEST_CONFIG)
DT = 0.05


def make_state(x=0.0, y=0.0, h=3000.0, psi_deg=0.0,
               ammo=6.0, hp=1.0, alive=1.0):
    """Test için minimal state oluşturucu."""
    s = aircraft.reset({"x": x, "y": y, "h": h, "psi": deg2rad(psi_deg)})
    s[STATE_AMMO]  = ammo
    s[STATE_HP]    = hp
    s[STATE_ALIVE] = alive
    return s


def fire_action(fire=1.0):
    a = np.zeros(ACTION_DIM)
    a[ACTION_FIRE] = fire
    return a


def no_fire_action():
    return fire_action(fire=0.0)


# ---------------------------------------------------------------------------
print("\n── 1. WeaponsModel Init ve Reset ────────────────────────────────")
# ---------------------------------------------------------------------------

wm = WeaponsModel(TEST_CONFIG)
check("Init: wez_range_max=8000", almost_equal(wm.wez_range_max, 8000.0))
check("Init: wez_angle_max=30° rad",
      almost_equal(wm.wez_angle_max, deg2rad(30.0), tol=1e-5))
check("Init: missile_damage=0.6", almost_equal(wm.missile_damage, 0.6))
check("Init: fire_cooldown=2.0", almost_equal(wm.fire_cooldown, 2.0))

wm.reset()
check("Reset: cooldown=0", almost_equal(wm._cooldown_timer, 0.0))
check("Reset: can_fire=True", wm.can_fire)


# ---------------------------------------------------------------------------
print("\n── 2. WEZ Geometri — Temel Durumlar ─────────────────────────────")
# ---------------------------------------------------------------------------

wm.reset()

# Hedef tam önümüzde, optimal mesafede → WEZ içinde
shooter = make_state(x=0.0, y=0.0, h=3000.0, psi_deg=0.0)    # Kuzey'e bakıyor
target  = make_state(x=0.0, y=3200.0, h=3000.0, psi_deg=180.0)  # Kuzey'de, bize dönük

wez = wm.compute_wez(shooter, target)
check("Optimal pozisyon → WEZ içinde", wez["in_wez"],
      f"dist={wez['distance']:.0f}m, ata={rad2deg(wez['ata']):.1f}°")
check("Optimal WEZ advantage > 0.5", wez["wez_advantage"] > 0.5,
      f"adv={wez['wez_advantage']:.3f}")
check("ATA optimal ≈ 0", almost_equal(wez["ata"], 0.0, tol=0.01),
      f"ata={rad2deg(wez['ata']):.2f}°")

# Hedef tam arkamızda → WEZ dışı
target_behind = make_state(x=0.0, y=-3200.0, h=3000.0)
wez_behind = wm.compute_wez(shooter, target_behind)
check("Hedef arkada → WEZ dışı", not wez_behind["in_wez"],
      f"ata={rad2deg(wez_behind['ata']):.1f}°")

# Hedef çok uzakta → WEZ dışı
target_far = make_state(x=0.0, y=9000.0, h=3000.0)
wez_far = wm.compute_wez(shooter, target_far)
check("Hedef çok uzakta → WEZ dışı", not wez_far["in_wez"],
      f"dist={wez_far['distance']:.0f}m")
check("Çok uzakta range_factor=0", almost_equal(wez_far["range_factor"], 0.0))

# Hedef açı limitinde (30° sınırı)
# Kuzey'e bakan uçak, hedef tam Doğu'da → ATA=90° → dışarıda
target_east = make_state(x=3200.0, y=0.0, h=3000.0)
wez_east = wm.compute_wez(shooter, target_east)
check("Hedef 90° açıda → WEZ dışı (açı)", not wez_east["in_wez"],
      f"ata={rad2deg(wez_east['ata']):.1f}°")
check("90° açıda angle_factor=0", almost_equal(wez_east["angle_factor"], 0.0))


# ---------------------------------------------------------------------------
print("\n── 3. WEZ Mesafe Sınırları ──────────────────────────────────────")
# ---------------------------------------------------------------------------

wm.reset()
shooter_ref = make_state(x=0.0, y=0.0, h=3000.0, psi_deg=0.0)

# Minimum menzil altında → dışarıda
target_tooclose = make_state(x=0.0, y=200.0, h=3000.0)  # 200m < 300m min
wez_close = wm.compute_wez(shooter_ref, target_tooclose)
check("Min menzil altı → WEZ dışı",
      not wez_close["in_wez"],
      f"dist={wez_close['distance']:.0f}m < min={wm.wez_range_min:.0f}m")

# Maksimum menzil üstünde → dışarıda
target_toofar = make_state(x=0.0, y=8100.0, h=3000.0)
wez_toofar = wm.compute_wez(shooter_ref, target_toofar)
check("Maks menzil üstü → WEZ dışı",
      not wez_toofar["in_wez"],
      f"dist={wez_toofar['distance']:.0f}m > max={wm.wez_range_max:.0f}m")

# Geçerli menzil içinde → içeride
target_valid = make_state(x=0.0, y=3000.0, h=3000.0)
wez_valid = wm.compute_wez(shooter_ref, target_valid)
check("Geçerli menzil → WEZ içinde",
      wez_valid["in_wez"],
      f"dist={wez_valid['distance']:.0f}m")


# ---------------------------------------------------------------------------
print("\n── 4. Ateş İşleme — Başarılı İsabet ────────────────────────────")
# ---------------------------------------------------------------------------

wm.reset()

shooter = make_state(x=0.0, y=0.0, h=3000.0, psi_deg=0.0, ammo=6.0, hp=1.0)
target  = make_state(x=0.0, y=3200.0, h=3000.0, psi_deg=180.0, hp=1.0)
action  = fire_action(fire=1.0)

result = wm.process_fire(shooter, target, action, DT)

check("WEZ içinde ateş → fired=True", result["fired"])
check("WEZ içinde ateş → hit=True", result["hit"])
check("İsabet sonrası hasar=0.6",
      almost_equal(result["damage"], 0.6))
check("Hedef HP 1.0 - 0.6 = 0.4",
      almost_equal(result["new_target_hp"], 0.4))
check("Mühimmat 1 azaldı",
      almost_equal(result["ammo_remaining"], 5.0))
check("Kill=False (hp=0.4 > 0)",
      not result["kill"])
check("fail_reason=None", result["fail_reason"] is None)


# ---------------------------------------------------------------------------
print("\n── 5. Ateş İşleme — Kill ────────────────────────────────────────")
# ---------------------------------------------------------------------------

wm.reset()
shooter_k = make_state(x=0.0, y=0.0, h=3000.0, psi_deg=0.0, ammo=6.0)
target_k  = make_state(x=0.0, y=3200.0, h=3000.0, psi_deg=180.0, hp=0.5)

result_k = wm.process_fire(shooter_k, target_k, fire_action(), DT)
check("HP=0.5, hasar=0.6 → kill=True", result_k["kill"])
check("Kill sonrası new_hp=0.0",
      almost_equal(result_k["new_target_hp"], 0.0))


# ---------------------------------------------------------------------------
print("\n── 6. Ateş İşleme — Başarısız Durumlar ─────────────────────────")
# ---------------------------------------------------------------------------

# a) Fire komutu yok
wm.reset()
shooter_f = make_state(ammo=6.0)
target_f  = make_state(x=0.0, y=3200.0)
res_nofire = wm.process_fire(shooter_f, target_f, no_fire_action(), DT)
check("Fire=0 → fired=False", not res_nofire["fired"])
check("Fire=0 → fail_reason='no_fire_command'",
      res_nofire["fail_reason"] == "no_fire_command")

# b) Mühimmat yok
wm.reset()
shooter_noammo = make_state(ammo=0.0)
res_noammo = wm.process_fire(shooter_noammo, target_f, fire_action(), DT)
check("Mühimmat yok → fail_reason='no_ammo'",
      res_noammo["fail_reason"] == "no_ammo")

# c) Cooldown
wm.reset()
wm._cooldown_timer = 1.5  # cooldown aktif
shooter_cd = make_state(ammo=6.0)
res_cd = wm.process_fire(shooter_cd, target_f, fire_action(), DT)
check("Cooldown aktif → fail_reason='cooldown'",
      "cooldown" in res_cd["fail_reason"])

# d) Nişancı ölü
wm.reset()
shooter_dead = make_state(ammo=6.0, alive=0.0)
res_dead = wm.process_fire(shooter_dead, target_f, fire_action(), DT)
check("Nişancı ölü → fail_reason='shooter_dead'",
      res_dead["fail_reason"] == "shooter_dead")

# e) Hedef ölü
wm.reset()
shooter_alive = make_state(ammo=6.0)
target_dead   = make_state(x=0.0, y=3200.0, alive=0.0)
res_tdead = wm.process_fire(shooter_alive, target_dead, fire_action(), DT)
check("Hedef ölü → fail_reason='target_dead'",
      res_tdead["fail_reason"] == "target_dead")

# f) WEZ dışında ateş → mühimmat harcanmaz
wm.reset()
shooter_outside = make_state(x=0.0, y=0.0, ammo=6.0)
target_outside  = make_state(x=0.0, y=-3200.0)  # arkada
res_outside = wm.process_fire(shooter_outside, target_outside, fire_action(), DT)
check("WEZ dışı → fired=True, hit=False",
      res_outside["fired"] and not res_outside["hit"])
check("WEZ dışı → mühimmat harcanmaz",
      almost_equal(res_outside["ammo_remaining"], 6.0))

# g) Minimum irtifa altında
wm.reset()
shooter_low = make_state(h=100.0, ammo=6.0)  # 100m < 200m min
target_low  = make_state(x=0.0, y=3200.0, h=100.0)
res_low = wm.process_fire(shooter_low, target_low, fire_action(), DT)
check("Min irtifa altı → fail_reason='below_min_altitude'",
      res_low["fail_reason"] == "below_min_altitude")


# ---------------------------------------------------------------------------
print("\n── 7. Cooldown Mekanizması ──────────────────────────────────────")
# ---------------------------------------------------------------------------

wm.reset()
shooter_c = make_state(x=0.0, y=0.0, h=3000.0, psi_deg=0.0, ammo=6.0)
target_c  = make_state(x=0.0, y=3200.0, h=3000.0, psi_deg=180.0)

# İlk ateş — başarılı, cooldown başlamalı
res1 = wm.process_fire(shooter_c, target_c, fire_action(), DT)
check("İlk ateş başarılı", res1["hit"])
check("İlk ateş sonrası cooldown başladı",
      wm._cooldown_timer > 0.0,
      f"timer={wm._cooldown_timer:.2f}s")

# Hemen tekrar ateş → cooldown
res2 = wm.process_fire(shooter_c, target_c, fire_action(), DT)
check("Cooldown süresinde 2. ateş engellendi",
      "cooldown" in res2["fail_reason"])

# Cooldown bitince tekrar ateş edebilmeli
n_ticks = int(wm.fire_cooldown / DT) + 5
for _ in range(n_ticks):
    wm.tick(DT)

check("Cooldown sonrası can_fire=True", wm.can_fire)

shooter_c2 = make_state(x=0.0, y=0.0, h=3000.0, psi_deg=0.0, ammo=5.0)
res3 = wm.process_fire(shooter_c2, target_c, fire_action(), DT)
check("Cooldown sonrası ateş başarılı", res3["hit"])


# ---------------------------------------------------------------------------
print("\n── 8. antenna_train_cos ─────────────────────────────────────────")
# ---------------------------------------------------------------------------

wm.reset()

# Hedef tam önde → cos(0) = 1.0
shooter_atc = make_state(x=0.0, y=0.0, psi_deg=0.0)
target_front = make_state(x=0.0, y=1000.0)
cos_front = wm.antenna_train_cos(shooter_atc, target_front)
check("ATA cos: hedef önde = 1.0",
      almost_equal(cos_front, 1.0, tol=1e-4),
      f"cos={cos_front:.4f}")

# Hedef 90° sağda → cos(π/2) ≈ 0
target_right_atc = make_state(x=1000.0, y=0.0)
cos_right = wm.antenna_train_cos(shooter_atc, target_right_atc)
check("ATA cos: hedef sağda ≈ 0.0",
      almost_equal(cos_right, 0.0, tol=1e-4),
      f"cos={cos_right:.4f}")

# Ölü nişancı → 0
shooter_dead_atc = make_state(alive=0.0)
cos_dead = wm.antenna_train_cos(shooter_dead_atc, target_front)
check("Ölü nişancı cos = 0.0",
      almost_equal(cos_dead, 0.0))


# ---------------------------------------------------------------------------
print("\n── 9. WEZ Advantage Skoru Sürekliliği ───────────────────────────")
# ---------------------------------------------------------------------------

wm.reset()
shooter_adv = make_state(x=0.0, y=0.0, h=3000.0, psi_deg=0.0)

# Mesafe arttıkça avantaj azalmalı (optimal mesafeden uzaklaşınca)
distances = [500, 1000, 3200, 5000, 7000, 8500]
advantages = []
for d in distances:
    t = make_state(x=0.0, y=float(d), h=3000.0)
    adv = wm.wez_advantage_score(shooter_adv, t)
    advantages.append(adv)

# 8500m menzil dışında → 0
check("Menzil dışı advantage=0",
      almost_equal(advantages[-1], 0.0),
      f"adv_8500m={advantages[-1]:.3f}")

# 500m > wez_range_min(300m) → WEZ içinde, Gaussian pozitif sinyal (normal davranış)
# Gerçek min menzil altı: 200m < 300m
target_belowmin = make_state(x=0.0, y=200.0, h=3000.0)
adv_belowmin = wm.wez_advantage_score(shooter_adv, target_belowmin)
check("Min menzil altı (200m) advantage=0",
      almost_equal(adv_belowmin, 0.0),
      f"adv_200m={adv_belowmin:.3f}")

# 3200m optimal civarı → en yüksek
check("Optimal mesafe en yüksek avantaj",
      advantages[2] == max(advantages),
      f"advs={[f'{a:.3f}' for a in advantages]}")


# ---------------------------------------------------------------------------
print("\n── 10. Ölü Uçak WEZ Sonuçları ───────────────────────────────────")
# ---------------------------------------------------------------------------

wm.reset()
dead_shooter = make_state(alive=0.0)
live_target  = make_state(x=0.0, y=3200.0)

wez_dead = wm.compute_wez(dead_shooter, live_target)
check("Ölü nişancı WEZ: in_wez=False", not wez_dead["in_wez"])
check("Ölü nişancı WEZ: advantage=0", almost_equal(wez_dead["wez_advantage"], 0.0))
check("Ölü nişancı WEZ: distance=inf",
      wez_dead["distance"] == float("inf"))


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
    print("\n⛔ weapons_model.py düzeltilmeden bir sonraki adıma geçilmez.")
    sys.exit(1)
else:
    print("\n🎉 Tüm testler geçti! reward_model.py yazımına geçilebilir.")
    sys.exit(0)

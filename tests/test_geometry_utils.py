"""
test_geometry_utils.py
======================
geometry_utils.py için unit testler.

Çalıştırma:
    python test_geometry_utils.py

Her test PASS/FAIL çıktısı verir.
Tüm testler geçmeden aircraft_model.py yazımına geçilmez.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.geometry_utils import (
    relative_position, distance_3d, distance_horizontal,
    bearing_angle, elevation_angle, aspect_angle,
    antenna_train_angle, flight_path_to_heading,
    wrap_to_pi, wrap_to_2pi, deg2rad, rad2deg,
    ned_to_enu, enu_to_unity,
    threat_score, rotation_matrix_body_to_wind,
    rotation_matrix_wind_to_body, euler_to_rotation_matrix
)

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


def vec_almost_equal(a, b, tol=1e-6):
    return np.allclose(a, b, atol=tol)


# ---------------------------------------------------------------------------
print("\n── 1. relative_position ──────────────────────────────────────────")
# ---------------------------------------------------------------------------

p1 = np.array([0.0, 0.0, 0.0])
p2 = np.array([3.0, 4.0, 0.0])
result = relative_position(p1, p2)
check("relative_position temel", vec_almost_equal(result, [3, 4, 0]))

# Ters yön
result2 = relative_position(p2, p1)
check("relative_position ters", vec_almost_equal(result2, [-3, -4, 0]))

# Aynı nokta
result3 = relative_position(p1, p1)
check("relative_position aynı nokta", vec_almost_equal(result3, [0, 0, 0]))


# ---------------------------------------------------------------------------
print("\n── 2. distance_3d ────────────────────────────────────────────────")
# ---------------------------------------------------------------------------

p1 = np.array([0.0, 0.0, 0.0])
p2 = np.array([3.0, 4.0, 0.0])
check("distance_3d 2D pitagor (3-4-5)", almost_equal(distance_3d(p1, p2), 5.0))

p3 = np.array([0.0, 0.0, 10.0])
check("distance_3d dikey", almost_equal(distance_3d(p1, p3), 10.0))

p4 = np.array([1.0, 1.0, 1.0])
check("distance_3d 3D sqrt(3)", almost_equal(distance_3d(p1, p4), np.sqrt(3.0)))

check("distance_3d simetrik", almost_equal(distance_3d(p1, p2), distance_3d(p2, p1)))
check("distance_3d sıfır", almost_equal(distance_3d(p1, p1), 0.0))


# ---------------------------------------------------------------------------
print("\n── 3. bearing_angle ──────────────────────────────────────────────")
# ---------------------------------------------------------------------------

origin = np.array([0.0, 0.0, 0.0])

# Kuzey'de hedef: bearing = 0
north = np.array([0.0, 1000.0, 0.0])
check("bearing Kuzey = 0", almost_equal(bearing_angle(origin, north), 0.0))

# Doğu'da hedef: bearing = π/2
east = np.array([1000.0, 0.0, 0.0])
check("bearing Doğu = π/2", almost_equal(bearing_angle(origin, east), np.pi / 2))

# Güney'de hedef: bearing = π
south = np.array([0.0, -1000.0, 0.0])
check("bearing Güney = π", almost_equal(bearing_angle(origin, south), np.pi))

# Batı'da hedef: bearing = 3π/2
west = np.array([-1000.0, 0.0, 0.0])
check("bearing Batı = 3π/2", almost_equal(bearing_angle(origin, west), 3 * np.pi / 2))

# İrtifa farkı bearing'i etkilememeli (yatay açı)
north_high = np.array([0.0, 1000.0, 5000.0])
check("bearing irtifadan bağımsız", almost_equal(bearing_angle(origin, north_high), 0.0))


# ---------------------------------------------------------------------------
print("\n── 4. elevation_angle ────────────────────────────────────────────")
# ---------------------------------------------------------------------------

origin = np.array([0.0, 0.0, 0.0])

# Tam yukarıda: elevation = π/2
above = np.array([0.0, 0.0, 1000.0])
check("elevation tam yukarı = π/2", almost_equal(elevation_angle(origin, above), np.pi / 2, tol=1e-4))

# Aynı irtifada: elevation = 0
side = np.array([1000.0, 0.0, 0.0])
check("elevation yatay = 0", almost_equal(elevation_angle(origin, side), 0.0))

# 45° yukarı
diag = np.array([1000.0, 0.0, 1000.0])
check("elevation 45°", almost_equal(elevation_angle(origin, diag), np.pi / 4, tol=1e-4))

# Aşağıda: negatif
below = np.array([0.0, 0.0, -1000.0])
check("elevation aşağı = -π/2", almost_equal(elevation_angle(origin, below), -np.pi / 2, tol=1e-4))


# ---------------------------------------------------------------------------
print("\n── 5. aspect_angle ───────────────────────────────────────────────")
# ---------------------------------------------------------------------------

# Hedef Kuzey'e bakıyor (psi=0), nişancı hedefin önünde (tam Kuzey'de)
# → Hedefin tam önü = AA = 0° (baş kafası)
pos_target  = np.array([0.0, 0.0, 0.0])
pos_shooter = np.array([0.0, 1000.0, 0.0])   # Kuzey'de
psi_target  = 0.0                              # Hedefe Kuzey'e bakıyor
aa = aspect_angle(pos_target, pos_shooter, psi_target)
check("aspect angle: nişancı önde = 0°", almost_equal(aa, 0.0, tol=1e-4),
      f"aa={rad2deg(aa):.1f}°")

# Nişancı hedefin arkasında (Güney'de)
# → AA = 180° (kuyruk)
pos_shooter2 = np.array([0.0, -1000.0, 0.0])
aa2 = aspect_angle(pos_target, pos_shooter2, psi_target)
check("aspect angle: nişancı arkada = 180°", almost_equal(aa2, np.pi, tol=1e-4),
      f"aa={rad2deg(aa2):.1f}°")

# Nişancı hedefin sağında (Doğu'da)
# → AA = 90°
pos_shooter3 = np.array([1000.0, 0.0, 0.0])
aa3 = aspect_angle(pos_target, pos_shooter3, psi_target)
check("aspect angle: nişancı sağda = 90°", almost_equal(aa3, np.pi / 2, tol=1e-4),
      f"aa={rad2deg(aa3):.1f}°")

# AA [0, π] aralığında olmalı (simetrik)
check("aspect angle [0, π] aralığı", 0.0 <= aa3 <= np.pi)


# ---------------------------------------------------------------------------
print("\n── 6. antenna_train_angle ────────────────────────────────────────")
# ---------------------------------------------------------------------------

origin = np.array([0.0, 0.0, 0.0])
psi_self = 0.0  # Kuzey'e bakıyoruz

# Hedef tam önümüzde (Kuzey) → ATA = 0
target_front = np.array([0.0, 1000.0, 0.0])
ata1 = antenna_train_angle(origin, target_front, psi_self)
check("ATA hedef önde = 0", almost_equal(ata1, 0.0, tol=1e-4),
      f"ata={rad2deg(ata1):.1f}°")

# Hedef sağımızda (Doğu) → ATA = +π/2
target_right = np.array([1000.0, 0.0, 0.0])
ata2 = antenna_train_angle(origin, target_right, psi_self)
check("ATA hedef sağda = +90°", almost_equal(ata2, np.pi / 2, tol=1e-4),
      f"ata={rad2deg(ata2):.1f}°")

# Hedef solumuzda (Batı) → ATA = -π/2
target_left = np.array([-1000.0, 0.0, 0.0])
ata3 = antenna_train_angle(origin, target_left, psi_self)
check("ATA hedef solda = -90°", almost_equal(ata3, -np.pi / 2, tol=1e-4),
      f"ata={rad2deg(ata3):.1f}°")

# ATA (-π, π] aralığında olmalı
check("ATA (-π, π] aralığı", -np.pi < ata3 <= np.pi)

# cos(ATA) reward testi: önde maksimum
check("cos(ATA) önde = 1.0", almost_equal(np.cos(ata1), 1.0, tol=1e-4))
check("cos(ATA) sağda = 0.0", almost_equal(np.cos(ata2), 0.0, tol=1e-4))


# ---------------------------------------------------------------------------
print("\n── 7. wrap_to_pi ve wrap_to_2pi ──────────────────────────────────")
# ---------------------------------------------------------------------------

check("wrap_to_pi: 3π → -π+ε", almost_equal(wrap_to_pi(3 * np.pi), -np.pi, tol=1e-9) or
      almost_equal(abs(wrap_to_pi(3 * np.pi)), np.pi, tol=1e-9))
check("wrap_to_pi: -π → -π", almost_equal(wrap_to_pi(-np.pi), -np.pi, tol=1e-9) or
      almost_equal(abs(wrap_to_pi(-np.pi)), np.pi, tol=1e-9))
check("wrap_to_pi: 0 → 0", almost_equal(wrap_to_pi(0.0), 0.0))
check("wrap_to_pi: 2π → 0", almost_equal(wrap_to_pi(2 * np.pi), 0.0, tol=1e-9))

check("wrap_to_2pi: -π → π", almost_equal(wrap_to_2pi(-np.pi), np.pi, tol=1e-9))
check("wrap_to_2pi: 0 → 0", almost_equal(wrap_to_2pi(0.0), 0.0))
check("wrap_to_2pi: 3π → π", almost_equal(wrap_to_2pi(3 * np.pi), np.pi, tol=1e-9))


# ---------------------------------------------------------------------------
print("\n── 8. ned_to_enu ve enu_to_unity ─────────────────────────────────")
# ---------------------------------------------------------------------------

# NED → ENU: x_ned=1(K), y_ned=2(D), z_ned=3(Aşağı) → x_enu=2(D), y_enu=1(K), z_enu=-3(Y)
x_e, y_e, z_e = ned_to_enu(1.0, 2.0, 3.0)
check("ned_to_enu x (Doğu)", almost_equal(x_e, 2.0))
check("ned_to_enu y (Kuzey)", almost_equal(y_e, 1.0))
check("ned_to_enu z (Yukarı)", almost_equal(z_e, -3.0))

# ENU → Unity: x_enu=1(D), y_enu=2(K), z_enu=3(Y) → x_u=1(Sağ), y_u=3(Yukarı), z_u=2(İleri)
x_u, y_u, z_u = enu_to_unity(1.0, 2.0, 3.0)
check("enu_to_unity x (Sağ=Doğu)", almost_equal(x_u, 1.0))
check("enu_to_unity y (Yukarı)", almost_equal(y_u, 3.0))
check("enu_to_unity z (İleri=Kuzey)", almost_equal(z_u, 2.0))


# ---------------------------------------------------------------------------
print("\n── 9. threat_score ───────────────────────────────────────────────")
# ---------------------------------------------------------------------------

range_max = 8000.0
angle_max = deg2rad(30.0)  # ±30° yarı açı

# Maksimum tehdit: yakın, düşman bize dönük, biz düşmanın önündeyiz
score_max = threat_score(
    distance=100.0,         # çok yakın
    ata=deg2rad(5.0),       # düşman neredeyse bize dönük
    aspect=deg2rad(5.0),    # biz düşmanın önündeyiz
    range_max=range_max,
    angle_max_rad=angle_max
)
check("threat_score yüksek tehdit > 0.8", score_max > 0.8,
      f"score={score_max:.3f}")

# Minimum tehdit: uzak, düşman bize sırtı dönük
score_min = threat_score(
    distance=9000.0,        # menzil dışı
    ata=deg2rad(170.0),     # düşman bize sırtı dönük
    aspect=deg2rad(170.0),  # biz düşmanın arkasındayız
    range_max=range_max,
    angle_max_rad=angle_max
)
check("threat_score düşük tehdit = 0", almost_equal(score_min, 0.0, tol=1e-4),
      f"score={score_min:.3f}")

# Skor [0,1] aralığında olmalı
check("threat_score [0,1] aralığı max", 0.0 <= score_max <= 1.0)
check("threat_score [0,1] aralığı min", 0.0 <= score_min <= 1.0)


# ---------------------------------------------------------------------------
print("\n── 10. rotation_matrix_body_to_wind ──────────────────────────────")
# ---------------------------------------------------------------------------

# α=0, β=0 → birim matris olmalı
R = rotation_matrix_body_to_wind(0.0, 0.0)
check("body_to_wind α=0,β=0 → birim matris", vec_almost_equal(R, np.eye(3)))

# R * R^T = I (ortogonal matris)
alpha_test = deg2rad(5.0)
beta_test  = deg2rad(2.0)
R2 = rotation_matrix_body_to_wind(alpha_test, beta_test)
RRT = R2 @ R2.T
check("body_to_wind ortogonal (R·Rᵀ = I)", vec_almost_equal(RRT, np.eye(3), tol=1e-10))

# Wind→Body = Body→Wind'in tersi olmalı
R_wb = rotation_matrix_wind_to_body(alpha_test, beta_test)
check("wind_to_body = body_to_wind^T", vec_almost_equal(R_wb, R2.T))


# ---------------------------------------------------------------------------
print("\n── 11. euler_to_rotation_matrix ──────────────────────────────────")
# ---------------------------------------------------------------------------

# φ=0, θ=0, ψ=0 → birim matris
R_euler = euler_to_rotation_matrix(0.0, 0.0, 0.0)
check("euler_to_dcm sıfır açılar → birim matris", vec_almost_equal(R_euler, np.eye(3)))

# Ortogonal olmalı
phi_t   = deg2rad(10.0)
theta_t = deg2rad(5.0)
psi_t   = deg2rad(45.0)
R_e = euler_to_rotation_matrix(phi_t, theta_t, psi_t)
check("euler_to_dcm ortogonal", vec_almost_equal(R_e @ R_e.T, np.eye(3), tol=1e-10))

# det = +1 olmalı (sağ-el sistemi)
det = np.linalg.det(R_e)
check("euler_to_dcm det = +1", almost_equal(det, 1.0, tol=1e-10),
      f"det={det:.10f}")


# ---------------------------------------------------------------------------
print("\n── 12. flight_path_to_heading ────────────────────────────────────")
# ---------------------------------------------------------------------------

# γ=0 (yatay), ψ=0 (Kuzey) → [0, 1, 0]
v = flight_path_to_heading(0.0, 0.0)
check("flight_path yatay kuzey = [0,1,0]", vec_almost_equal(v, [0, 1, 0]))

# γ=π/2 (tam yukarı) → [0, 0, 1]
v2 = flight_path_to_heading(np.pi / 2, 0.0)
check("flight_path tam yukarı = [0,0,1]", vec_almost_equal(v2, [0, 0, 1], tol=1e-6))

# γ=0, ψ=π/2 (Doğu) → [1, 0, 0]
v3 = flight_path_to_heading(0.0, np.pi / 2)
check("flight_path yatay doğu = [1,0,0]", vec_almost_equal(v3, [1, 0, 0], tol=1e-6))

# Birim vektör olmalı
check("flight_path birim vektör", almost_equal(np.linalg.norm(v3), 1.0, tol=1e-9))


# ---------------------------------------------------------------------------
# Sonuç Özeti
# ---------------------------------------------------------------------------
print("\n" + "="*60)
total   = len(results)
passed  = sum(1 for _, ok in results if ok)
failed  = total - passed

print(f"TOPLAM : {total}")
print(f"✅ PASS : {passed}")
print(f"❌ FAIL : {failed}")

if failed > 0:
    print("\nBaşarısız testler:")
    for name, ok in results:
        if not ok:
            print(f"  ❌ {name}")
    print("\n⛔ geometry_utils.py düzeltilmeden bir sonraki adıma geçilmez.")
    sys.exit(1)
else:
    print("\n🎉 Tüm testler geçti! aircraft_model.py yazımına geçilebilir.")
    sys.exit(0)

"""
test_trim_solver.py
===================
TrimSolver testleri — scipy olmadan da çalışır.
"""

import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = []

def check(name, cond, detail=""):
    status = PASS if cond else FAIL
    print(f"{status} | {name}" + (f"  [{detail}]" if detail else ""))
    results.append((name, bool(cond)))

def almost(a, b, tol=1e-3):
    return abs(float(a) - float(b)) < tol

# ---------------------------------------------------------------------------
# Dummy aircraft (aircraft_model gerektirmeden test)
# ---------------------------------------------------------------------------
import types
def make_dummy_aircraft():
    a = types.SimpleNamespace()
    # F-16 katsayıları (config.yaml'dan)
    a.mass       = 9100.0
    a.wing_area  = 27.87
    a.mean_chord = 3.45
    a.wingspan   = 9.45
    a.max_thrust = 76300.0
    a.CL0        = 0.05
    a.CL_alpha   = 5.5
    a.CL_de      = 0.5
    a.CD0        = 0.016
    a.CD_alpha   = 0.3
    a.Cm0        = 0.03
    a.Cm_alpha   = -0.8
    a.Cm_de      = -1.5
    a.de_max     = np.deg2rad(25.0)
    a.da_max     = np.deg2rad(21.5)
    a.dr_max     = np.deg2rad(30.0)
    a.V_min      = 60.0
    a.V_max      = 600.0
    return a

aircraft = make_dummy_aircraft()

from envs.trim_solver import TrimSolver, TrimResult, _SCIPY_AVAILABLE
print(f"scipy available: {_SCIPY_AVAILABLE}")

solver = TrimSolver(aircraft)

# ===========================================================================
print("\n── 1. TrimSolver init ───────────────────────────────────────────")
# ===========================================================================
check("mass yüklendi",       almost(solver.mass, 9100.0))
check("CL_alpha yüklendi",   almost(solver.CL_alpha, 5.5))
check("de_max rad cinsinden", solver.de_max > 0.1)

# ===========================================================================
print("\n── 2. ISA Atmosfer ──────────────────────────────────────────────")
# ===========================================================================
rho0, T0 = solver._isa_atmosphere(0.0)
check("Deniz seviyesi rho ≈ 1.225",  almost(rho0, 1.225, tol=0.01))
check("Deniz seviyesi T ≈ 288.15",   almost(T0, 288.15, tol=0.1))

rho_4k, T_4k = solver._isa_atmosphere(4000.0)
check("4000m rho < deniz seviyesi",  rho_4k < rho0)
check("4000m T < deniz seviyesi",    T_4k < T0)
check("rho pozitif",                 rho_4k > 0.0)

# ===========================================================================
print("\n── 3. Aerodinamik (wing-level) ──────────────────────────────────")
# ===========================================================================
rho, _ = solver._isa_atmosphere(4000.0)
q_bar  = 0.5 * rho * 200.0 ** 2
alpha  = np.deg2rad(5.0)
de_rad = np.deg2rad(-2.0)

aero = solver._aero_wing_level(alpha, de_rad, q_bar)
check("L > 0",   aero["L"] > 0.0)
check("D > 0",   aero["D"] > 0.0)
check("CL > 0",  aero["CL"] > 0.0)
check("CD > 0",  aero["CD"] > 0.0)

# ===========================================================================
print("\n── 4. Denge denklemleri residual ────────────────────────────────")
# ===========================================================================
# Rastgele x ile residual büyük olmalı
x_bad = np.array([0.0, 0.0, 0.0])
res_bad = solver._residuals(x_bad, V=200.0, h=4000.0)
check("Kötü x ile residual büyük",  np.linalg.norm(res_bad) > 100.0)

# ===========================================================================
print("\n── 5. Trim çözümü — temel koşullar ─────────────────────────────")
# ===========================================================================
test_cases = [
    (150.0, 3000.0, "Düşük hız, alçak irtifa"),
    (200.0, 4000.0, "Nominal koşul"),
    (250.0, 6000.0, "Yüksek hız, yüksek irtifa"),
    (280.0, 8000.0, "Maksimum spawn koşulu"),
]

for V, h, label in test_cases:
    res = solver.solve(V, h)
    check(f"{label} — çözüm başarılı",
          res.success,
          f"residual={res.residual:.2f}")
    check(f"{label} — alpha fiziksel aralık [0°,15°]",
          0.0 <= np.rad2deg(res.alpha) <= 15.0,
          f"alpha={np.rad2deg(res.alpha):.2f}°")
    check(f"{label} — dt ∈ [0,1]",
          0.0 <= res.dt <= 1.0,
          f"dt={res.dt:.3f}")
    check(f"{label} — de ∈ [-1,1]",
          -1.0 <= res.de <= 1.0,
          f"de={res.de:.3f}")
    # Trim doğrulaması: residual küçük olmalı
    check(f"{label} — residual < 50 N/Nm",
          res.residual < 50.0,
          f"residual={res.residual:.4f}")

# ===========================================================================
print("\n── 6. Trim doğrulaması — denge denklemleri ─────────────────────")
# ===========================================================================
res = solver.solve(V=200.0, h=4000.0)
if res.success:
    de_rad = res.de * solver.de_max
    rho, _ = solver._isa_atmosphere(res.h)
    q_bar  = 0.5 * rho * res.V ** 2
    aero   = solver._aero_wing_level(res.alpha, de_rad, q_bar)
    thrust = solver._thrust(res.dt, res.V, res.h)
    weight = solver.mass * 9.80665

    f1 = thrust * np.cos(res.alpha) - aero["D"]
    f2 = aero["L"] + thrust * np.sin(res.alpha) - weight
    f3 = aero["M"]

    check("V_dot ≈ 0  (thrust = drag)",      abs(f1) < 100.0,  f"|f1|={abs(f1):.2f} N")
    check("γ_dot ≈ 0  (lift = weight)",      abs(f2) < 100.0,  f"|f2|={abs(f2):.2f} N")
    check("q_dot ≈ 0  (pitch moment = 0)",   abs(f3) < 500.0,  f"|f3|={abs(f3):.2f} Nm")

    print(f"  [trim] V={res.V:.0f} m/s, h={res.h:.0f} m")
    print(f"         alpha={np.rad2deg(res.alpha):.2f}°, "
          f"de={res.de:.3f}, dt={res.dt:.3f}")
    print(f"         L={aero['L']:.0f} N, W={weight:.0f} N, T={thrust:.0f} N")

# ===========================================================================
print("\n── 7. Lookup tablosu ────────────────────────────────────────────")
# ===========================================================================
table = solver.build_lookup_table(
    V_range=(150.0, 280.0), h_range=(3000.0, 8000.0),
    n_V=5, n_h=4
)
check("Tablo V_arr uzunluğu = 5",  len(table["V_arr"]) == 5)
check("Tablo h_arr uzunluğu = 4",  len(table["h_arr"]) == 4)
check("alpha tablosu shape (5,4)", table["alpha"].shape == (5, 4))
check("Tablo başarı oranı > 0.8",
      table["ok"].mean() > 0.8,
      f"ok_rate={table['ok'].mean():.2f}")

# Lookup interpolasyon
res_lk = solver.lookup(V=200.0, h=5000.0, table=table)
check("Lookup çözümü döndü",       isinstance(res_lk, TrimResult))
check("Lookup alpha fiziksel",
      0.0 <= np.rad2deg(res_lk.alpha) <= 15.0,
      f"alpha={np.rad2deg(res_lk.alpha):.2f}°")

# Tablo dışı → solve() çağrılır
res_oor = solver.lookup(V=100.0, h=1000.0, table=table)
check("Tablo dışı → doğrudan solve", isinstance(res_oor, TrimResult))

# ===========================================================================
print("\n── 8. Spawn uyumluluğu ──────────────────────────────────────────")
# ===========================================================================
# dogfight_env spawn aralığı: V=[150,280], h=[3000,8000]
import yaml
config_path = PROJECT_ROOT / "configs" / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

V_min = config["env"]["spawn_V_min"]
V_max = config["env"]["spawn_V_max"]
h_min = config["env"]["spawn_h_min"]
h_max = config["env"]["spawn_h_max"]

rng = np.random.default_rng(42)
n_spawn = 20
ok_count = 0
for _ in range(n_spawn):
    V = float(rng.uniform(V_min, V_max))
    h = float(rng.uniform(h_min, h_max))
    r = solver.solve(V, h)
    if r.success:
        ok_count += 1

check(f"Spawn aralığında trim başarı > %90",
      ok_count / n_spawn >= 0.9,
      f"{ok_count}/{n_spawn}")

# ===========================================================================
print("\n" + "=" * 60)
total_n  = len(results)
passed_n = sum(1 for _, ok in results if ok)
failed_n = total_n - passed_n
print(f"TOPLAM : {total_n}")
print(f"✅ PASS : {passed_n}")
print(f"❌ FAIL : {failed_n}")
if failed_n > 0:
    print("\nBaşarısız testler:")
    for name, ok in results:
        if not ok:
            print(f"  ❌ {name}")
    sys.exit(1)
else:
    print("\n🎉 Tüm testler geçti! dogfight_env.py entegrasyonuna geçilebilir.")
    sys.exit(0)

"""
trim_solver.py
==============
Wing-level trim hesaplama — F-16 spawn başlangıç koşulu için.

Trim Tanımı:
    Belirli bir (V, h) koşulunda uçağın dengede olduğu
    (alpha, de, dt) üçlüsü.

Wing-level trim varsayımları:
    phi   = 0  (kanatlar düz)
    beta  = 0  (yan kayma yok)
    p=q=r = 0  (angular hız yok)
    gamma = 0  (düz uçuş, h_dot = 0)

Denge denklemleri (3 denklem, 3 bilinmeyen):
    [1] V_dot   = 0  →  T·cos(α) - D = m·g·sin(γ) = 0
    [2] γ_dot   = 0  →  L·cos(φ) + T·sin(α) = m·g·cos(γ) = m·g
    [3] M_pitch = 0  →  Cm(α, de) = 0

Çözüm: scipy.optimize.fsolve ile Newton iterasyonu
Süre: ~1 ms per trim (eğitim hızını etkilemez)

Kullanım:
    from trim_solver import TrimSolver
    solver = TrimSolver(aircraft_model)
    result = solver.solve(V=200.0, h=4000.0)
    # result.alpha, result.de, result.dt, result.success

Bu dosya değişirse etkilenen dosyalar:
    - dogfight_env.py  (spawn'da trim kullanılıyor)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

# Sabitler
G_ACCEL = 9.80665   # m/s²
RHO0_ISA = 1.225    # kg/m³

# scipy opsiyonel — yoksa fallback kullan
try:
    from scipy.optimize import fsolve
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


@dataclass
class TrimResult:
    """Trim çözüm sonucu."""
    alpha:    float          # hücum açısı (rad)
    de:       float          # elevator normalize [-1, 1]
    dt:       float          # throttle [0, 1]
    V:        float          # airspeed (m/s)
    h:        float          # irtifa (m)
    success:  bool           # çözüm başarılı mı
    residual: float          # denge denklemi artığı (küçük = iyi)
    iterations: int          # iterasyon sayısı


class TrimSolver:
    """
    Wing-level trim çözücü.

    aircraft parametreleri aircraft_model.AircraftModel'den alınır.
    Doğrudan model nesnesine bağlı değil — katsayıları kopyalar.
    Bu sayede trim_solver bağımsız test edilebilir.
    """

    def __init__(self, aircraft):
        """
        Parameters
        ----------
        aircraft : AircraftModel instance veya aynı attribute'lara sahip nesne
        """
        # Kütle ve geometri
        self.mass        = float(aircraft.mass)
        self.wing_area   = float(aircraft.wing_area)
        self.mean_chord  = float(aircraft.mean_chord)
        self.wingspan    = float(aircraft.wingspan)
        self.max_thrust  = float(aircraft.max_thrust)

        # Aerodinamik katsayılar (trim için gereken subset)
        self.CL0      = float(aircraft.CL0)
        self.CL_alpha = float(aircraft.CL_alpha)
        self.CL_de    = float(aircraft.CL_de)
        self.CD0      = float(aircraft.CD0)
        self.CD_alpha = float(aircraft.CD_alpha)
        self.Cm0      = float(aircraft.Cm0)
        self.Cm_alpha = float(aircraft.Cm_alpha)
        self.Cm_de    = float(aircraft.Cm_de)

        # Kontrol limitleri
        self.de_max   = float(aircraft.de_max)    # rad
        self.da_max   = float(aircraft.da_max)    # rad (bilgi amaçlı)
        self.dr_max   = float(aircraft.dr_max)    # rad

        # Hız sınırları
        self.V_min    = float(aircraft.V_min)
        self.V_max    = float(aircraft.V_max)

    # -----------------------------------------------------------------------
    # ISA Atmosfer (aircraft_model ile aynı)
    # -----------------------------------------------------------------------

    @staticmethod
    def _isa_atmosphere(h: float) -> tuple:
        """Basitleştirilmiş ISA: troposfer (h < 11000 m)."""
        T0, L, R, g = 288.15, 0.0065, 287.05, G_ACCEL
        h = max(0.0, min(h, 11000.0))
        T   = T0 - L * h
        rho = RHO0_ISA * (T / T0) ** (g / (L * R) - 1.0)
        return float(rho), float(T)

    # -----------------------------------------------------------------------
    # Aerodinamik (wing-level, p=q=r=0, beta=0)
    # -----------------------------------------------------------------------

    def _aero_wing_level(self, alpha: float, de_rad: float,
                          q_bar: float) -> dict:
        """
        Wing-level koşulda aerodinamik kuvvet ve moment.
        p=q=r=0, beta=0 → boyutsuz dönme terimleri sıfır.
        """
        CL = self.CL0 + self.CL_alpha * alpha + self.CL_de * de_rad
        CD = self.CD0 + self.CD_alpha * alpha ** 2
        Cm = self.Cm0 + self.Cm_alpha * alpha + self.Cm_de * de_rad

        L = q_bar * self.wing_area * CL
        D = q_bar * self.wing_area * CD
        M = q_bar * self.wing_area * self.mean_chord * Cm

        return {"L": L, "D": D, "M": M, "CL": CL, "CD": CD, "Cm": Cm}

    def _thrust(self, dt: float, V: float, h: float) -> float:
        """aircraft_model ile aynı thrust modeli."""
        rho, _ = self._isa_atmosphere(h)
        rho_ratio = rho / RHO0_ISA
        return float(self.max_thrust * dt * (rho_ratio ** 0.7))

    # -----------------------------------------------------------------------
    # Denge Denklemleri
    # -----------------------------------------------------------------------

    def _residuals(self, x: np.ndarray, V: float, h: float) -> np.ndarray:
        """
        Trim denge denklemleri.

        x = [alpha (rad), de_norm [-1,1], dt [0,1]]

        Denklemler:
            f1 = T·cos(α) - D - m·g·sin(γ)   = 0   (V_dot = 0, γ=0 → sin=0)
            f2 = L + T·sin(α) - m·g·cos(γ)   = 0   (γ_dot = 0, γ=0 → cos=1)
            f3 = M_pitch                       = 0   (q_dot = 0)
        """
        alpha  = float(x[0])
        de_norm = float(x[1])
        dt     = float(np.clip(x[2], 0.0, 1.0))

        de_rad = de_norm * self.de_max

        rho, _  = self._isa_atmosphere(h)
        q_bar   = 0.5 * rho * V ** 2
        aero    = self._aero_wing_level(alpha, de_rad, q_bar)
        thrust  = self._thrust(dt, V, h)
        weight  = self.mass * G_ACCEL

        # γ = 0 → sin(γ)=0, cos(γ)=1
        f1 = thrust * np.cos(alpha) - aero["D"]            # V_dot = 0
        f2 = aero["L"] + thrust * np.sin(alpha) - weight   # γ_dot = 0
        f3 = aero["M"]                                      # q_dot = 0

        return np.array([f1, f2, f3], dtype=np.float64)

    # -----------------------------------------------------------------------
    # Ana Çözücü
    # -----------------------------------------------------------------------

    def solve(self, V: float, h: float,
              alpha_guess: float = None,
              tol: float = 1e-6,
              max_iter: int = 100) -> TrimResult:
        """
        (V, h) için wing-level trim hesapla.

        Parameters
        ----------
        V           : airspeed (m/s)
        h           : irtifa (m)
        alpha_guess : başlangıç hücum açısı tahmini (rad). None → otomatik
        tol         : çözüm toleransı
        max_iter    : maksimum iterasyon

        Returns
        -------
        TrimResult
        """
        V = float(np.clip(V, self.V_min, self.V_max))
        h = float(np.clip(h, 50.0, 15000.0))

        # Başlangıç tahmini
        if alpha_guess is None:
            # CL gereksinimi: L = W → CL_req = W / (q_bar * S)
            rho, _ = self._isa_atmosphere(h)
            q_bar  = 0.5 * rho * V ** 2
            weight = self.mass * G_ACCEL
            CL_req = weight / max(q_bar * self.wing_area, 1.0)
            # CL = CL0 + CL_alpha * alpha → alpha_guess
            alpha0 = (CL_req - self.CL0) / max(self.CL_alpha, 1e-6)
            alpha0 = float(np.clip(alpha0, np.deg2rad(0.0), np.deg2rad(15.0)))
        else:
            alpha0 = float(alpha_guess)

        # de başlangıcı: Cm = 0 → de = -(Cm0 + Cm_alpha*alpha) / Cm_de
        de0_rad = -(self.Cm0 + self.Cm_alpha * alpha0) / max(abs(self.Cm_de), 1e-6)
        de0_norm = float(np.clip(de0_rad / self.de_max, -1.0, 1.0))

        # dt başlangıcı: thrust = drag (yaklaşık)
        rho, _ = self._isa_atmosphere(h)
        q_bar  = 0.5 * rho * V ** 2
        CD_est = self.CD0 + self.CD_alpha * alpha0 ** 2
        D_est  = q_bar * self.wing_area * CD_est
        T_max_eff = self.max_thrust * (rho / RHO0_ISA) ** 0.7
        dt0   = float(np.clip(D_est / max(T_max_eff, 1.0), 0.05, 0.95))

        x0 = np.array([alpha0, de0_norm, dt0], dtype=np.float64)

        if _SCIPY_AVAILABLE:
            result = self._solve_scipy(x0, V, h, tol, max_iter)
        else:
            result = self._solve_newton(x0, V, h, tol, max_iter)

        return result

    def _solve_scipy(self, x0: np.ndarray, V: float, h: float,
                     tol: float, max_iter: int) -> TrimResult:
        """scipy.optimize.fsolve ile çözüm."""
        info = {}
        try:
            sol, info_dict, ier, msg = fsolve(
                self._residuals, x0,
                args=(V, h),
                full_output=True,
                xtol=tol,
                maxfev=max_iter * 10,
            )
            residual = float(np.linalg.norm(self._residuals(sol, V, h)))
            success  = (ier == 1) and (residual < 1.0)
            nfev     = info_dict.get("nfev", 0)
        except Exception:
            sol      = x0
            residual = float(np.linalg.norm(self._residuals(x0, V, h)))
            success  = False
            nfev     = 0

        return self._build_result(sol, V, h, success, residual, nfev)

    def _solve_newton(self, x0: np.ndarray, V: float, h: float,
                      tol: float, max_iter: int) -> TrimResult:
        """
        scipy yoksa basit Newton-Raphson — sonuç aynı kalite.
        Jacobian sonlu fark ile yaklaşık hesaplanır.
        """
        x   = x0.copy()
        eps = 1e-7
        n   = len(x)

        for iteration in range(max_iter):
            f  = self._residuals(x, V, h)
            if np.linalg.norm(f) < tol:
                break

            # Jacobian (sonlu fark)
            J = np.zeros((n, n), dtype=np.float64)
            for j in range(n):
                xp    = x.copy(); xp[j] += eps
                J[:, j] = (self._residuals(xp, V, h) - f) / eps

            # Newton adımı: Δx = -J⁻¹ f
            try:
                dx = np.linalg.solve(J, -f)
            except np.linalg.LinAlgError:
                break

            # Adım kısıtlaması (divergence koruması)
            step_norm = np.linalg.norm(dx)
            if step_norm > 0.5:
                dx *= 0.5 / step_norm

            x = x + dx

        residual = float(np.linalg.norm(self._residuals(x, V, h)))
        success  = residual < 1.0

        return self._build_result(x, V, h, success, residual, iteration + 1)

    def _build_result(self, x: np.ndarray, V: float, h: float,
                      success: bool, residual: float,
                      iterations: int) -> TrimResult:
        """Çözüm vektöründen TrimResult oluştur."""
        alpha   = float(x[0])
        de_norm = float(np.clip(x[1], -1.0, 1.0))
        dt      = float(np.clip(x[2],  0.0,  1.0))

        # Fiziksel sınır kontrolü
        alpha = float(np.clip(alpha, np.deg2rad(-5.0), np.deg2rad(20.0)))

        # Sınır dışına çıktıysa başarısız say
        if abs(x[1]) > 1.05 or x[2] < -0.05 or x[2] > 1.05:
            success = False

        return TrimResult(
            alpha     = alpha,
            de        = de_norm,
            dt        = dt,
            V         = V,
            h         = h,
            success   = success,
            residual  = residual,
            iterations= iterations,
        )

    # -----------------------------------------------------------------------
    # Tablo Önbelleği (eğitim hızlandırma)
    # -----------------------------------------------------------------------

    def build_lookup_table(self,
                            V_range: tuple = (150.0, 280.0),
                            h_range: tuple = (3000.0, 8000.0),
                            n_V: int = 10,
                            n_h: int = 8) -> dict:
        """
        Sık kullanılan (V, h) değerleri için trim tablosu oluştur.
        Eğitim başında bir kez çağrılır, spawn'da interpolasyon kullanılır.

        Returns
        -------
        table : dict — (V_arr, h_arr, alpha_table, de_table, dt_table)
        """
        V_arr = np.linspace(V_range[0], V_range[1], n_V)
        h_arr = np.linspace(h_range[0], h_range[1], n_h)

        alpha_table = np.zeros((n_V, n_h))
        de_table    = np.zeros((n_V, n_h))
        dt_table    = np.zeros((n_V, n_h))
        ok_table    = np.zeros((n_V, n_h), dtype=bool)

        for i, V in enumerate(V_arr):
            for j, h in enumerate(h_arr):
                res = self.solve(V, h)
                alpha_table[i, j] = res.alpha
                de_table[i, j]    = res.de
                dt_table[i, j]    = res.dt
                ok_table[i, j]    = res.success

        return {
            "V_arr": V_arr, "h_arr": h_arr,
            "alpha": alpha_table,
            "de":    de_table,
            "dt":    dt_table,
            "ok":    ok_table,
        }

    def lookup(self, V: float, h: float, table: dict) -> TrimResult:
        """
        Önceden hesaplanmış tablodan bilinear interpolasyon.
        Tablo sınırları dışındaysa doğrudan solve() çağrılır.
        """
        V_arr = table["V_arr"]
        h_arr = table["h_arr"]

        if V < V_arr[0] or V > V_arr[-1] or h < h_arr[0] or h > h_arr[-1]:
            return self.solve(V, h)

        # Bilinear interpolasyon
        i = int(np.searchsorted(V_arr, V) - 1)
        j = int(np.searchsorted(h_arr, h) - 1)
        i = int(np.clip(i, 0, len(V_arr) - 2))
        j = int(np.clip(j, 0, len(h_arr) - 2))

        tV = (V - V_arr[i]) / max(V_arr[i+1] - V_arr[i], 1e-9)
        th = (h - h_arr[j]) / max(h_arr[j+1] - h_arr[j], 1e-9)

        def interp2(tbl):
            return float(
                (1-tV)*(1-th)*tbl[i, j]   + tV*(1-th)*tbl[i+1, j] +
                (1-tV)*th    *tbl[i, j+1] + tV*th    *tbl[i+1, j+1]
            )

        alpha = interp2(table["alpha"])
        de    = interp2(table["de"])
        dt    = interp2(table["dt"])

        return TrimResult(
            alpha=alpha, de=de, dt=dt,
            V=V, h=h, success=True, residual=0.0, iterations=0
        )

"""
aircraft_model.py
=================
6-DoF uçak dinamik modeli — JSBSim F-16 katsayıları, RK4 entegrasyon.

Koordinat sistemi: ENU (East-North-Up)
    x = Doğu  (m)
    y = Kuzey (m)
    z = Yukarı / İrtifa (m)

State vektörü indeksleri (STATE_* sabitleri):
    0  : x        — Doğu konumu (m)
    1  : y        — Kuzey konumu (m)
    2  : h        — İrtifa (m)
    3  : V        — Airspeed (m/s)
    4  : alpha    — Hücum açısı (rad)
    5  : beta     — Kayma açısı (rad)
    6  : gamma    — Uçuş yolu açısı (rad)
    7  : phi      — Roll açısı (rad)
    8  : theta    — Pitch açısı (rad)
    9  : psi      — Yaw/Heading açısı (rad)
    10 : p        — Roll rate (rad/s)
    11 : q        — Pitch rate (rad/s)
    12 : r        — Yaw rate (rad/s)
    13 : fuel     — Kalan yakıt (kg)
    14 : ammo     — Kalan mühimmat (adet)
    15 : hp       — Hasar durumu (0–1)
    16 : radar_range — Aktif sensör menzili (m)
    17 : alive    — 1=hayatta, 0=imha

Action vektörü indeksleri (ACTION_* sabitleri):
    0 : delta_a  — Aileron  (normalize, -1 to 1)
    1 : delta_e  — Elevator (normalize, -1 to 1)
    2 : delta_r  — Rudder   (normalize, -1 to 1)
    3 : delta_t  — Throttle (normalize,  0 to 1)
    4 : fire     — Ateş komutu (0 to 1, 0.5 eşiği)

Bağımlılıklar:
    - numpy
    - geometry_utils.py  (rotation_matrix_body_to_wind, euler_to_rotation_matrix)

Bu dosya değişirse etkilenen dosyalar:
    - weapons_model.py   (state indeksleri)
    - reward_model.py    (state indeksleri)
    - dogfight_env.py    (step döngüsü)
    - normalization.py   (state boyutları)
    - test_aircraft_model.py
"""

import numpy as np
from envs.geometry_utils import (
    rotation_matrix_body_to_wind,
    euler_to_rotation_matrix,
    deg2rad
)

# ---------------------------------------------------------------------------
# State indeks sabitleri — tüm dosyalar bu sabitleri import eder
# ---------------------------------------------------------------------------
STATE_X      = 0
STATE_Y      = 1
STATE_H      = 2
STATE_V      = 3
STATE_ALPHA  = 4
STATE_BETA   = 5
STATE_GAMMA  = 6
STATE_PHI    = 7
STATE_THETA  = 8
STATE_PSI    = 9
STATE_P      = 10
STATE_Q      = 11
STATE_R      = 12
STATE_FUEL   = 13
STATE_AMMO   = 14
STATE_HP     = 15
STATE_RADAR  = 16
STATE_ALIVE  = 17

STATE_DIM = 18

# Action indeks sabitleri
ACTION_DA   = 0   # aileron
ACTION_DE   = 1   # elevator
ACTION_DR   = 2   # rudder
ACTION_DT   = 3   # throttle
ACTION_FIRE = 4

ACTION_DIM = 5

# ---------------------------------------------------------------------------
# Fiziksel sabitler
# ---------------------------------------------------------------------------
G_ACCEL   = 9.80665    # m/s² — standart yerçekimi ivmesi
R_AIR     = 287.05     # J/(kg·K) — hava gaz sabiti
GAMMA_AIR = 1.4        # havanın özgül ısı oranı

# ISA atmosfer sabitleri
T0_ISA    = 288.15     # K — deniz seviyesi sıcaklığı
RHO0_ISA  = 1.225      # kg/m³ — deniz seviyesi yoğunluğu
L_LAPSE   = 0.0065     # K/m — troposfer ısıl düşüş oranı
H_TROPO   = 11000.0    # m — tropopoz yüksekliği
T_TROPO   = 216.65     # K — tropopoz sıcaklığı
RHO_TROPO = 0.3639     # kg/m³ — tropopoz yoğunluğu


class AircraftModel:
    """
    6-DoF uçak dinamik modeli.

    Kullanım:
        cfg = yaml.safe_load(open("configs/config.yaml"))
        aircraft = AircraftModel(cfg)
        state = aircraft.reset(init_conditions)
        for _ in range(steps):
            state = aircraft.step(state, action, dt)
    """

    def __init__(self, config: dict):
        """
        Parameters
        ----------
        config : dict
            config.yaml'dan yüklenen tam config sözlüğü.
            'aircraft' ve 'aero_coeffs' anahtarları kullanılır.
        """
        ac  = config["aircraft"]
        aero = config["aero_coeffs"]
        ctrl = config.get("control_limits", {})

        # ── Kütle ve geometri ──────────────────────────────────────────
        self.mass       = float(ac["mass"])          # kg
        self.wingspan   = float(ac["wingspan"])      # m  (b)
        self.wing_area  = float(ac["wing_area"])     # m² (S)
        self.mean_chord = float(ac["mean_chord"])    # m  (c̄)

        # ── Atalet momentleri ──────────────────────────────────────────
        self.Ixx = float(ac["Ixx"])   # kg·m²
        self.Iyy = float(ac["Iyy"])
        self.Izz = float(ac["Izz"])
        self.Ixz = float(ac["Ixz"])   # çapraz terim — ihmal edilmez

        # Ixz'li inertia matrisinin tersi (3x3 simetrik matris)
        # [Ixx   0  -Ixz]
        # [  0  Iyy    0]
        # [-Ixz  0   Izz]
        self._inertia_matrix = np.array([
            [ self.Ixx,  0.0,      -self.Ixz],
            [ 0.0,       self.Iyy,  0.0     ],
            [-self.Ixz,  0.0,       self.Izz]
        ])
        self._inertia_inv = np.linalg.inv(self._inertia_matrix)

        # ── İtki ve yakıt ──────────────────────────────────────────────
        self.max_thrust     = float(ac["max_thrust"])      # N
        self.SFC            = float(ac["SFC"])             # kg/N/s
        self.initial_fuel   = float(ac["initial_fuel"])    # kg
        self.initial_ammo   = int(ac["initial_ammo"])
        self.initial_hp     = float(ac.get("initial_hp", 1.0))
        self.radar_range_default = float(ac.get("radar_range", 15000.0))  # m

        # ── Uçuş zarfı ────────────────────────────────────────────────
        self.V_min      = float(ac.get("V_min", 60.0))     # m/s
        self.V_max      = float(ac.get("V_max", 600.0))    # m/s
        self.alpha_max  = float(ac.get("alpha_max", deg2rad(25.0)))  # rad
        self.alpha_min  = float(ac.get("alpha_min", deg2rad(-10.0)))
        self.h_min      = float(ac.get("h_min", 50.0))     # m — zemin limiti

        # ── Kontrol yüzeyi limitleri (F-16 gerçek değerleri) ──────────
        # Normalize [-1,1] → gerçek açı (rad) dönüşümü için
        self.da_max = deg2rad(float(ctrl.get("aileron_max_deg",  21.5)))
        self.de_max = deg2rad(float(ctrl.get("elevator_max_deg", 25.0)))
        self.dr_max = deg2rad(float(ctrl.get("rudder_max_deg",   30.0)))

        # ── Aerodinamik katsayılar (JSBSim F-16 referanslı) ───────────
        # Kaldırma (Lift)
        self.CL0      = float(aero["CL0"])
        self.CL_alpha = float(aero["CL_alpha"])
        self.CL_q     = float(aero["CL_q"])
        self.CL_de    = float(aero["CL_de"])

        # Sürükleme (Drag) — parabolic polar
        self.CD0      = float(aero["CD0"])
        self.CD_alpha = float(aero["CD_alpha"])

        # Yan kuvvet (Side force)
        self.CY_beta  = float(aero["CY_beta"])
        self.CY_da    = float(aero.get("CY_da", 0.0))
        self.CY_dr    = float(aero["CY_dr"])

        # Roll moment
        self.Cl_p     = float(aero["Cl_p"])
        self.Cl_r     = float(aero["Cl_r"])
        self.Cl_da    = float(aero["Cl_da"])
        self.Cl_dr    = float(aero.get("Cl_dr", 0.01))

        # Pitch moment
        self.Cm0      = float(aero["Cm0"])
        self.Cm_alpha = float(aero["Cm_alpha"])
        self.Cm_q     = float(aero["Cm_q"])
        self.Cm_de    = float(aero["Cm_de"])

        # Yaw moment
        self.Cn_p     = float(aero["Cn_p"])
        self.Cn_r     = float(aero["Cn_r"])
        self.Cn_da    = float(aero["Cn_da"])
        self.Cn_dr    = float(aero["Cn_dr"])

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def reset(self, init_conditions: dict) -> np.ndarray:
        """
        Başlangıç durumunu oluşturur.

        Parameters
        ----------
        init_conditions : dict
            Anahtarlar: x, y, h, V, alpha, beta, gamma,
                        phi, theta, psi, p, q, r
            Verilmeyenler için varsayılan değerler kullanılır.

        Returns
        -------
        np.ndarray, shape (STATE_DIM,) — başlangıç state vektörü
        """
        s = np.zeros(STATE_DIM)

        s[STATE_X]     = float(init_conditions.get("x",     0.0))
        s[STATE_Y]     = float(init_conditions.get("y",     0.0))
        s[STATE_H]     = float(init_conditions.get("h",     3000.0))
        s[STATE_V]     = float(init_conditions.get("V",     200.0))
        s[STATE_ALPHA] = float(init_conditions.get("alpha", deg2rad(3.0)))
        s[STATE_BETA]  = float(init_conditions.get("beta",  0.0))
        s[STATE_GAMMA] = float(init_conditions.get("gamma", 0.0))
        s[STATE_PHI]   = float(init_conditions.get("phi",   0.0))
        s[STATE_THETA] = float(init_conditions.get("theta", deg2rad(3.0)))
        s[STATE_PSI]   = float(init_conditions.get("psi",   0.0))
        s[STATE_P]     = float(init_conditions.get("p",     0.0))
        s[STATE_Q]     = float(init_conditions.get("q",     0.0))
        s[STATE_R]     = float(init_conditions.get("r",     0.0))

        s[STATE_FUEL]  = self.initial_fuel
        s[STATE_AMMO]  = float(self.initial_ammo)
        s[STATE_HP]    = self.initial_hp
        s[STATE_RADAR] = self.radar_range_default
        s[STATE_ALIVE] = 1.0

        return s

    def step(self, state: np.ndarray,
             action: np.ndarray,
             dt: float) -> np.ndarray:
        """
        RK4 entegrasyonu ile bir adım ilerler.

        Parameters
        ----------
        state  : np.ndarray, shape (STATE_DIM,)
        action : np.ndarray, shape (ACTION_DIM,)
                 Normalize değerler: delta_a/e/r ∈ [-1,1], delta_t ∈ [0,1]
        dt     : float — adım süresi (saniye)

        Returns
        -------
        np.ndarray, shape (STATE_DIM,) — yeni state
        """
        if state[STATE_ALIVE] < 0.5:
            return state.copy()

        # Aksiyonu clip et — normalize sınırlar
        action = self._clip_action(action)

        # RK4 — sadece dinamik değişkenler entegre edilir (0:13)
        # Resource ve durum değişkenleri (fuel, ammo, hp, radar, alive)
        # ayrıca güncellenir
        k1 = self._derivatives(state, action)
        k2 = self._derivatives(state + 0.5 * dt * k1, action)
        k3 = self._derivatives(state + 0.5 * dt * k2, action)
        k4 = self._derivatives(state + dt * k3, action)

        new_state = state.copy()
        new_state[:STATE_FUEL] += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)[:STATE_FUEL]

        # Yakıt tüketimi
        thrust = self._compute_thrust(action[ACTION_DT], state[STATE_V],
                                      state[STATE_H])
        fuel_burn = self.SFC * thrust * dt
        new_state[STATE_FUEL] = max(0.0, state[STATE_FUEL] - fuel_burn)

        # Resource değişkenleri değişmez (ammo, hp, radar dışarıdan yönetilir)
        new_state[STATE_AMMO]  = state[STATE_AMMO]
        new_state[STATE_HP]    = state[STATE_HP]
        new_state[STATE_RADAR] = state[STATE_RADAR]
        new_state[STATE_ALIVE] = state[STATE_ALIVE]

        # Fizik limitlerini uygula
        new_state = self._apply_limits(new_state)

        return new_state

    def is_stalled(self, state: np.ndarray) -> bool:
        """Uçak stall durumunda mı?"""
        return (abs(state[STATE_ALPHA]) > self.alpha_max or
                state[STATE_V] < self.V_min)

    def is_out_of_bounds(self, state: np.ndarray, map_size: float) -> bool:
        """Uçak harita dışına çıktı mı?"""
        return (abs(state[STATE_X]) > map_size / 2 or
                abs(state[STATE_Y]) > map_size / 2 or
                state[STATE_H] < self.h_min)

    # -----------------------------------------------------------------------
    # Atmosfer Modeli
    # -----------------------------------------------------------------------

    def _isa_atmosphere(self, h: float):
        """
        ISA (International Standard Atmosphere) modeli.
        Troposfer (0-11000m) ve alt stratosfer (11000-20000m).

        Parameters
        ----------
        h : float — irtifa (m), negatif → 0 olarak alınır

        Returns
        -------
        rho : float — hava yoğunluğu (kg/m³)
        T   : float — sıcaklık (K)
        """
        h = max(0.0, h)

        if h <= H_TROPO:
            # Troposfer: doğrusal sıcaklık düşüşü
            T   = T0_ISA - L_LAPSE * h
            rho = RHO0_ISA * (T / T0_ISA) ** (G_ACCEL / (L_LAPSE * R_AIR) - 1.0)
        else:
            # Alt stratosfer: izothermal
            T   = T_TROPO
            rho = RHO_TROPO * np.exp(-G_ACCEL / (R_AIR * T_TROPO) * (h - H_TROPO))

        return float(rho), float(T)

    # -----------------------------------------------------------------------
    # Aerodinamik Katsayılar
    # -----------------------------------------------------------------------

    def _aero_coefficients(self, state: np.ndarray,
                           ctrl_rad: np.ndarray) -> dict:
        """
        Normalize aksiyondan gerçek açılara dönüştürülmüş kontrol yüzeyi
        ve mevcut state kullanarak aerodinamik katsayıları hesaplar.

        Parameters
        ----------
        state    : np.ndarray — mevcut state
        ctrl_rad : np.ndarray — [da, de, dr] gerçek açılar (rad)

        Returns
        -------
        dict — CL, CD, CY, Cl, Cm, Cn
        """
        alpha = state[STATE_ALPHA]
        beta  = state[STATE_BETA]
        p     = state[STATE_P]
        q     = state[STATE_Q]
        r     = state[STATE_R]
        V     = max(state[STATE_V], 1.0)   # sıfıra bölme koruması

        da, de, dr = ctrl_rad[0], ctrl_rad[1], ctrl_rad[2]

        # Boyutsuz dönme hızları (non-dimensional rates)
        pb_2V = p * self.wingspan   / (2.0 * V)
        qc_2V = q * self.mean_chord / (2.0 * V)
        rb_2V = r * self.wingspan   / (2.0 * V)

        CL = (self.CL0
              + self.CL_alpha * alpha
              + self.CL_q     * qc_2V
              + self.CL_de    * de)

        CD = self.CD0 + self.CD_alpha * alpha ** 2

        CY = (self.CY_beta * beta
              + self.CY_da  * da
              + self.CY_dr  * dr)

        Cl = (self.Cl_p  * pb_2V
              + self.Cl_r * rb_2V
              + self.Cl_da * da
              + self.Cl_dr * dr)

        Cm = (self.Cm0
              + self.Cm_alpha * alpha
              + self.Cm_q     * qc_2V
              + self.Cm_de    * de)

        Cn = (self.Cn_p  * pb_2V
              + self.Cn_r * rb_2V
              + self.Cn_da * da
              + self.Cn_dr * dr)

        return {"CL": CL, "CD": CD, "CY": CY,
                "Cl": Cl, "Cm": Cm, "Cn": Cn}

    # -----------------------------------------------------------------------
    # İtki Modeli
    # -----------------------------------------------------------------------

    def _compute_thrust(self, delta_t: float, V: float, h: float) -> float:
        """
        Throttle, hız ve irtifaya göre thrust hesaplar.
        Basitleştirilmiş model: max_thrust * delta_t * (rho/rho0)^0.7

        Parameters
        ----------
        delta_t : float — throttle [0, 1]
        V       : float — airspeed (m/s)
        h       : float — irtifa (m)

        Returns
        -------
        float — thrust (N)
        """
        rho, _ = self._isa_atmosphere(h)
        rho_ratio = rho / RHO0_ISA
        # İrtifayla thrust düşüşü — basitleştirilmiş
        thrust = self.max_thrust * delta_t * (rho_ratio ** 0.7)
        return float(thrust)

    # -----------------------------------------------------------------------
    # Türevler (Kalp)
    # -----------------------------------------------------------------------

    def _derivatives(self, state: np.ndarray,
                     action: np.ndarray) -> np.ndarray:
        """
        Tüm state değişkenlerinin türevlerini hesaplar.
        RK4'te f(state, action) olarak çağrılır.

        Parameters
        ----------
        state  : np.ndarray, shape (STATE_DIM,)
        action : np.ndarray, shape (ACTION_DIM,)

        Returns
        -------
        np.ndarray, shape (STATE_DIM,) — türevler (d/dt)
        Resource değişkenleri (fuel dahil) bu fonksiyonda sıfır döner,
        step() içinde ayrıca hesaplanır.
        """
        # State'i çöz
        h     = state[STATE_H]
        V     = max(state[STATE_V], 1.0)
        alpha = state[STATE_ALPHA]
        beta  = state[STATE_BETA]
        gamma = state[STATE_GAMMA]
        phi   = state[STATE_PHI]
        theta = state[STATE_THETA]
        psi   = state[STATE_PSI]
        p     = state[STATE_P]
        q     = state[STATE_Q]
        r     = state[STATE_R]

        # Kontrol yüzeyi açıları (normalize → gerçek rad)
        da = action[ACTION_DA] * self.da_max
        de = action[ACTION_DE] * self.de_max
        dr = action[ACTION_DR] * self.dr_max
        dt = np.clip(action[ACTION_DT], 0.0, 1.0)
        ctrl_rad = np.array([da, de, dr])

        # Atmosfer
        rho, _ = self._isa_atmosphere(h)
        q_bar  = 0.5 * rho * V ** 2   # dinamik basınç (Pa)

        # Aerodinamik katsayılar
        coeffs = self._aero_coefficients(state, ctrl_rad)

        # Kuvvetler (wind ekseninde)
        L = q_bar * self.wing_area * coeffs["CL"]   # kaldırma
        D = q_bar * self.wing_area * coeffs["CD"]   # sürükleme
        Y = q_bar * self.wing_area * coeffs["CY"]   # yan kuvvet

        # Momentler (body ekseninde)
        La = q_bar * self.wing_area * self.wingspan   * coeffs["Cl"]
        Ma = q_bar * self.wing_area * self.mean_chord * coeffs["Cm"]
        Na = q_bar * self.wing_area * self.wingspan   * coeffs["Cn"]

        # İtki
        thrust = self._compute_thrust(dt, V, h)

        # ── Translasyonel Dinamik (Wind ekseni) ──────────────────────
        # Bank açısı μ: phi ≈ roll açısı wind ekseninde
        mu = phi

        cos_g = np.cos(gamma)
        sin_g = np.sin(gamma)
        # Singularity koruması: cos(gamma) = 0 durumunda
        cos_g_safe = np.sign(cos_g) * max(abs(cos_g), 1e-4)

        V_dot = ((thrust * np.cos(alpha) - D) / self.mass
                 - G_ACCEL * sin_g)

        gamma_dot = ((L * np.cos(mu) + thrust * np.sin(alpha)
                      - self.mass * G_ACCEL * cos_g)
                     / (self.mass * V))

        psi_dot = (L * np.sin(mu)) / (self.mass * V * cos_g_safe)

        # ── Translasyonel Kinematik (ENU pozisyon) ───────────────────
        x_dot = V * cos_g * np.sin(psi)   # Doğu
        y_dot = V * cos_g * np.cos(psi)   # Kuzey
        h_dot = V * sin_g                 # Yukarı (irtifa artışı)

        # ── Rotasyonel Kinematik (Euler açı türevleri) ───────────────
        cos_phi   = np.cos(phi)
        sin_phi   = np.sin(phi)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        # Singularity: theta = ±90° → cos(theta) = 0
        cos_theta_safe = np.sign(cos_theta) * max(abs(cos_theta), 1e-4)

        phi_dot   = p + (q * sin_phi + r * cos_phi) * np.tan(theta)
        theta_dot = q * cos_phi - r * sin_phi
        psi_body_dot = (q * sin_phi + r * cos_phi) / cos_theta_safe

        # ── Rotasyonel Dinamik (Euler momentleri, Ixz dahil) ─────────
        # Jiroskopik terimler
        gyro_roll  = (self.Iyy - self.Izz) * q * r - self.Ixz * (p * q)
        gyro_pitch = (self.Izz - self.Ixx) * p * r + self.Ixz * (p ** 2 - r ** 2)
        gyro_yaw   = (self.Ixx - self.Iyy) * p * q + self.Ixz * (q * r)

        # Moment denklemi: I · [ṗ, q̇, ṙ]ᵀ = [La+gyro, Ma+gyro, Na+gyro]ᵀ
        rhs = np.array([
            La + gyro_roll,
            Ma + gyro_pitch,
            Na + gyro_yaw
        ])
        pqr_dot = self._inertia_inv @ rhs
        p_dot, q_dot, r_dot = pqr_dot

        # ── α ve β türevleri ─────────────────────────────────────────
        # Body ekseni hız bileşenleri
        u = V * np.cos(alpha) * np.cos(beta)
        v = V * np.sin(beta)
        w = V * np.sin(alpha) * np.cos(beta)

        # Kuvvetlerin body eksenine dönüşümü
        # F_body = R_wb @ [−D, Y, −L]_wind
        # Wind ekseninde: x_w=−drag, y_w=side, z_w=−lift
        f_wind = np.array([-D, Y, -L])
        R_wb = rotation_matrix_body_to_wind(alpha, beta).T  # wind→body
        f_body = R_wb @ f_wind + np.array([
            thrust * np.cos(alpha) * np.cos(beta),  # thrust body-x
            0.0,                                      # thrust body-y
            0.0                                       # thrust body-z
        ])
        Fx, Fy, Fz = f_body

        # Body ekseni hız türevleri
        u_dot = Fx / self.mass - G_ACCEL * sin_theta     + r * v - q * w
        v_dot = Fy / self.mass + G_ACCEL * cos_theta * sin_phi  - r * u + p * w
        w_dot = Fz / self.mass + G_ACCEL * cos_theta * cos_phi  + q * u - p * v

        # u,v,w → V,α,β türevleri
        V_sq   = u**2 + v**2 + w**2
        V_safe = max(np.sqrt(V_sq), 1.0)
        u_safe = max(abs(u), 1e-4) * np.sign(u) if abs(u) > 1e-4 else 1e-4

        # Zincir kuralı ile
        V_dot_body = (u * u_dot + v * v_dot + w * w_dot) / V_safe
        alpha_dot  = (u_safe * w_dot - w * u_dot) / (u_safe**2 + w**2)
        beta_dot   = (v_dot * V_safe - v * V_dot_body) / (V_sq - v**2 + 1e-8) ** 0.5

        # ── Türev vektörünü oluştur ───────────────────────────────────
        dstate = np.zeros(STATE_DIM)

        dstate[STATE_X]     = x_dot
        dstate[STATE_Y]     = y_dot
        dstate[STATE_H]     = h_dot
        dstate[STATE_V]     = V_dot_body     # body ekseninden hesaplanan
        dstate[STATE_ALPHA] = alpha_dot
        dstate[STATE_BETA]  = beta_dot
        dstate[STATE_GAMMA] = gamma_dot      # wind ekseninden hesaplanan
        dstate[STATE_PHI]   = phi_dot
        dstate[STATE_THETA] = theta_dot
        dstate[STATE_PSI]   = psi_body_dot   # body ekseninden heading
        dstate[STATE_P]     = p_dot
        dstate[STATE_Q]     = q_dot
        dstate[STATE_R]     = r_dot
        # 13-17 (fuel, ammo, hp, radar, alive) → sıfır (step'te ayrıca)

        return dstate

    # -----------------------------------------------------------------------
    # Limit ve Clip Fonksiyonları
    # -----------------------------------------------------------------------

    def _clip_action(self, action: np.ndarray) -> np.ndarray:
        """Aksiyonu fiziksel sınırlara kırpar."""
        a = action.copy()
        a[ACTION_DA] = np.clip(a[ACTION_DA], -1.0,  1.0)
        a[ACTION_DE] = np.clip(a[ACTION_DE], -1.0,  1.0)
        a[ACTION_DR] = np.clip(a[ACTION_DR], -1.0,  1.0)
        a[ACTION_DT] = np.clip(a[ACTION_DT],  0.0,  1.0)
        a[ACTION_FIRE] = np.clip(a[ACTION_FIRE], 0.0, 1.0)
        return a

    def _apply_limits(self, state: np.ndarray) -> np.ndarray:
        """
        Fizik entegrasyonu sonrası sınır ve kararlılık kontrolü.
        Açıları normalize eder, hız ve irtifa limitlerini uygular.
        """
        s = state.copy()

        # Hız alt sınırı (stall koruması — hard clip, ceza reward_model'de)
        s[STATE_V] = np.clip(s[STATE_V], self.V_min * 0.5, self.V_max * 1.2)

        # Hücum açısı — aerodinamik geçerlilik
        s[STATE_ALPHA] = np.clip(s[STATE_ALPHA], deg2rad(-20.0), deg2rad(30.0))

        # Kayma açısı — yan kayma limiti
        s[STATE_BETA] = np.clip(s[STATE_BETA], deg2rad(-30.0), deg2rad(30.0))

        # Uçuş yolu açısı — singularity koruması
        s[STATE_GAMMA] = np.clip(s[STATE_GAMMA], deg2rad(-85.0), deg2rad(85.0))

        # Pitch açısı — singularity koruması
        s[STATE_THETA] = np.clip(s[STATE_THETA], deg2rad(-85.0), deg2rad(85.0))

        # Açıları (-π, π] aralığına normalize et
        s[STATE_PHI]   = float((s[STATE_PHI]   + np.pi) % (2*np.pi) - np.pi)
        s[STATE_PSI]   = float((s[STATE_PSI]   + np.pi) % (2*np.pi) - np.pi)
        s[STATE_GAMMA] = float((s[STATE_GAMMA] + np.pi) % (2*np.pi) - np.pi)
        # Yeniden clip (normalize sonrası)
        s[STATE_GAMMA] = np.clip(s[STATE_GAMMA], deg2rad(-85.0), deg2rad(85.0))

        # İrtifa — zemin limiti
        if s[STATE_H] < self.h_min:
            s[STATE_H]     = self.h_min
            s[STATE_GAMMA] = max(0.0, s[STATE_GAMMA])  # yer temasında yukarı

        # Angular hız sınırları — overflow koruması
        # MAPPO eğitimi başında rastgele aksiyonlar p/q/r'yi patlatabilir
        s[STATE_P] = np.clip(s[STATE_P], -6.0, 6.0)   # ~2π rad/s max
        s[STATE_Q] = np.clip(s[STATE_Q], -1.0, 1.0)
        s[STATE_R] = np.clip(s[STATE_R], -1.0, 1.0)

        # NaN/Inf temizliği — son savunma hattı
        for i in range(len(s)):
            if not np.isfinite(s[i]):
                s[i] = 0.0

        # Yakıt tükenmişse thrust yok, alive devam eder (pilotaj sona erer)
        s[STATE_FUEL] = max(0.0, s[STATE_FUEL])

        # HP sınırı
        s[STATE_HP] = np.clip(s[STATE_HP], 0.0, 1.0)

        return s
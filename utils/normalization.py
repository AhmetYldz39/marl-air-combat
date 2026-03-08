"""
normalization.py
================
Observation ve action normalize/denormalize fonksiyonları.

Tüm limit ve ölçek değerleri config.yaml'dan okunur.
Normalize edilmiş değerler genellikle [-1, 1] veya [0, 1] aralığındadır.

Normalize stratejileri:
    Konum (x, y)    : harita boyutuna göre [-1, 1]
    İrtifa (h)      : [h_min, h_max] → [0, 1]
    Hız (V)         : [V_min, V_max] → [0, 1]
    Açılar (rad)    : [-π, π] → [-1, 1]  veya  [0, π] → [0, 1]
    Angular hızlar  : [-p_max, p_max] → [-1, 1]
    Yakıt           : [0, initial_fuel] → [0, 1]
    Mühimmat        : [0, initial_ammo] → [0, 1]
    HP              : zaten [0, 1]
    Radar menzili   : [0, radar_range_max] → [0, 1]
    Alive           : zaten {0, 1}

Observation vektörü yapısı (Faz 0-1, iletişimsiz, rol yok):
    Ego         : 16 boyut — kendi normalize state
    Takım ark.  : 9 boyut × n_teammates (her biri için)
    Düşman      : 12 boyut × n_enemies  (her biri için)
    ──────────────────────────────────────────
    2v2 toplam  : 16 + 9 + 12×2 = 49 boyut

    Faz 2 eklemeleri:
    + Rol embedding  : +2 boyut → 51
    + GAT mesajı     : +16 boyut × n_teammates → 67 (2v2)

Bağımlılıklar:
    - numpy
    - aircraft_model.py (STATE_* sabitleri, STATE_DIM)
    - geometry_utils.py (bearing_angle, elevation_angle, distance_3d,
                         antenna_train_angle, aspect_angle, threat_score,
                         wrap_to_pi)

Bu dosya değişirse etkilenen dosyalar:
    - dogfight_env.py  (observation oluşturma)
    - models/mappo_actor.py (giriş boyutu)
"""

import numpy as np
from envs.aircraft_model import (
    STATE_X, STATE_Y, STATE_H, STATE_V,
    STATE_ALPHA, STATE_BETA, STATE_GAMMA,
    STATE_PHI, STATE_THETA, STATE_PSI,
    STATE_P, STATE_Q, STATE_R,
    STATE_FUEL, STATE_AMMO, STATE_HP,
    STATE_RADAR, STATE_ALIVE,
    STATE_DIM,
)
from envs.geometry_utils import (
    bearing_angle, elevation_angle, distance_3d,
    antenna_train_angle, aspect_angle, threat_score,
    wrap_to_pi, deg2rad
)

# Observation bileşen boyutları — dogfight_env.py bu sabitleri kullanır
OBS_EGO_DIM        = 16
OBS_TEAMMATE_DIM   = 9
OBS_ENEMY_DIM      = 12
OBS_ROLE_DIM       = 2    # Faz 2: aggression embedding
OBS_GAT_MSG_DIM    = 16   # Faz 2: GAT mesajı (teammate başına)

EPS = 1e-9


class Normalizer:
    """
    State → normalize observation dönüşümü.

    Her ajan için ayrı Normalizer örneği oluşturulabilir,
    ama paylaşılan parametreler config'den geldiği için
    tüm ajanlar aynı örneği paylaşabilir.

    Kullanım:
        norm = Normalizer(config)

        # Ego observation
        ego_obs = norm.ego_obs(agent_state)

        # Takım arkadaşı observation
        tm_obs = norm.teammate_obs(agent_state, teammate_state)

        # Düşman observation
        en_obs = norm.enemy_obs(agent_state, enemy_state)

        # Tam observation vektörü (2v2, iletişimsiz, rol yok)
        obs = norm.build_obs(
            agent_state, teammate_states, enemy_states,
            aggression=None
        )
    """

    def __init__(self, config: dict):
        ac    = config["aircraft"]
        env   = config["env"]
        wpn   = config["weapons"]

        # ── Pozisyon ──────────────────────────────────────────────────
        self.map_size      = float(env["map_size"])          # m
        self.half_map      = self.map_size / 2.0

        # ── İrtifa ────────────────────────────────────────────────────
        self.h_min         = float(ac.get("h_min",  50.0))   # m
        self.h_max         = float(env.get("h_max", 15000.0))# m

        # ── Hız ───────────────────────────────────────────────────────
        self.V_min         = float(ac.get("V_min",  60.0))   # m/s
        self.V_max         = float(ac.get("V_max", 600.0))   # m/s

        # ── Angular hız sınırları ─────────────────────────────────────
        self.p_max         = float(env.get("p_max", 3.0))    # rad/s
        self.q_max         = float(env.get("q_max", 3.0))
        self.r_max         = float(env.get("r_max", 3.0))

        # ── Resource ──────────────────────────────────────────────────
        self.init_fuel     = float(ac["initial_fuel"])        # kg
        self.init_ammo     = float(ac["initial_ammo"])        # adet
        self.radar_max     = float(ac.get("radar_range", 15000.0))  # m

        # ── WEZ / Tehdit ──────────────────────────────────────────────
        self.wez_range_max = float(wpn["wez_range_max"])      # m
        self.wez_angle_max = deg2rad(float(wpn["wez_angle_max"]))  # rad

    # -----------------------------------------------------------------------
    # Yardımcı Normalize Fonksiyonları
    # -----------------------------------------------------------------------

    def _norm_pos(self, x: float, y: float) -> np.ndarray:
        """Pozisyon → [-1, 1] × [-1, 1]"""
        return np.array([
            np.clip(x / (self.half_map + EPS), -1.0, 1.0),
            np.clip(y / (self.half_map + EPS), -1.0, 1.0),
        ])

    def _norm_h(self, h: float) -> float:
        """İrtifa → [0, 1]"""
        return float(np.clip(
            (h - self.h_min) / (self.h_max - self.h_min + EPS),
            0.0, 1.0
        ))

    def _norm_V(self, V: float) -> float:
        """Airspeed → [0, 1]"""
        return float(np.clip(
            (V - self.V_min) / (self.V_max - self.V_min + EPS),
            0.0, 1.0
        ))

    def _norm_angle(self, angle_rad: float) -> float:
        """Açı (rad, [-π, π]) → [-1, 1]"""
        return float(np.clip(wrap_to_pi(angle_rad) / np.pi, -1.0, 1.0))

    def _norm_pqr(self, rate: float, rate_max: float) -> float:
        """Angular hız → [-1, 1]"""
        return float(np.clip(rate / (rate_max + EPS), -1.0, 1.0))

    def _norm_fuel(self, fuel: float) -> float:
        """Yakıt → [0, 1]"""
        return float(np.clip(fuel / (self.init_fuel + EPS), 0.0, 1.0))

    def _norm_ammo(self, ammo: float) -> float:
        """Mühimmat → [0, 1]"""
        return float(np.clip(ammo / (self.init_ammo + EPS), 0.0, 1.0))

    def _norm_dist(self, dist: float) -> float:
        """Mesafe (0–map_size) → [0, 1]"""
        return float(np.clip(dist / (self.map_size + EPS), 0.0, 1.0))

    def _norm_rel_pos(self, dx: float, dy: float, dh: float) -> np.ndarray:
        """Göreceli pozisyon → [-1, 1]³"""
        return np.array([
            np.clip(dx / (self.map_size + EPS), -1.0, 1.0),
            np.clip(dy / (self.map_size + EPS), -1.0, 1.0),
            np.clip(dh / (self.h_max   + EPS), -1.0, 1.0),
        ])

    # -----------------------------------------------------------------------
    # Ego Observation (16 boyut)
    # -----------------------------------------------------------------------

    def ego_obs(self, state: np.ndarray) -> np.ndarray:
        """
        Kendi state'inden ego observation vektörü oluşturur.

        Boyutlar (16):
            0-1  : x_norm, y_norm        — normalize konum [-1,1]
            2    : h_norm                — normalize irtifa [0,1]
            3    : V_norm                — normalize hız [0,1]
            4    : alpha_norm            — hücum açısı [-1,1]
            5    : beta_norm             — kayma açısı [-1,1]
            6    : gamma_norm            — uçuş yolu açısı [-1,1]
            7    : phi_norm              — roll [-1,1]
            8    : theta_norm            — pitch [-1,1]
            9    : psi_norm              — heading [-1,1]
            10   : p_norm                — roll rate [-1,1]
            11   : q_norm                — pitch rate [-1,1]
            12   : r_norm                — yaw rate [-1,1]
            13   : fuel_norm             — yakıt [0,1]
            14   : ammo_norm             — mühimmat [0,1]
            15   : hp                   — HP [0,1] (zaten normalize)
        """
        pos_norm = self._norm_pos(state[STATE_X], state[STATE_Y])

        obs = np.array([
            pos_norm[0],
            pos_norm[1],
            self._norm_h(state[STATE_H]),
            self._norm_V(state[STATE_V]),
            self._norm_angle(state[STATE_ALPHA]),
            self._norm_angle(state[STATE_BETA]),
            self._norm_angle(state[STATE_GAMMA]),
            self._norm_angle(state[STATE_PHI]),
            self._norm_angle(state[STATE_THETA]),
            self._norm_angle(state[STATE_PSI]),
            self._norm_pqr(state[STATE_P], self.p_max),
            self._norm_pqr(state[STATE_Q], self.q_max),
            self._norm_pqr(state[STATE_R], self.r_max),
            self._norm_fuel(state[STATE_FUEL]),
            self._norm_ammo(state[STATE_AMMO]),
            float(np.clip(state[STATE_HP], 0.0, 1.0)),
        ], dtype=np.float32)

        assert len(obs) == OBS_EGO_DIM, f"Ego obs boyutu hatalı: {len(obs)}"
        return obs

    # -----------------------------------------------------------------------
    # Teammate Observation (9 boyut)
    # -----------------------------------------------------------------------

    def teammate_obs(self, agent_state: np.ndarray,
                     teammate_state: np.ndarray) -> np.ndarray:
        """
        Takım arkadaşına ait relative observation vektörü.

        Boyutlar (9):
            0-2 : rel_x, rel_y, rel_h   — göreceli konum (normalize) [-1,1]
            3   : rel_V_norm             — göreceli hız farkı [-1,1]
            4   : bearing_norm           — ajan→takım bearing [-1,1]
            5   : elevation_norm         — ajan→takım elevation [-1,1]
            6   : dist_norm              — mesafe [0,1]
            7   : hp_teammate            — HP [0,1]
            8   : alive                  — hayatta mı {0,1}
        """
        if teammate_state[STATE_ALIVE] < 0.5:
            return np.zeros(OBS_TEAMMATE_DIM, dtype=np.float32)

        agent_pos    = agent_state[[STATE_X, STATE_Y, STATE_H]]
        teammate_pos = teammate_state[[STATE_X, STATE_Y, STATE_H]]

        dx = teammate_state[STATE_X] - agent_state[STATE_X]
        dy = teammate_state[STATE_Y] - agent_state[STATE_Y]
        dh = teammate_state[STATE_H] - agent_state[STATE_H]

        rel_pos  = self._norm_rel_pos(dx, dy, dh)
        dist     = distance_3d(agent_pos, teammate_pos)
        bearing  = bearing_angle(agent_pos, teammate_pos)
        elev     = elevation_angle(agent_pos, teammate_pos)
        rel_V    = teammate_state[STATE_V] - agent_state[STATE_V]

        obs = np.array([
            rel_pos[0],
            rel_pos[1],
            rel_pos[2],
            float(np.clip(rel_V / (self.V_max + EPS), -1.0, 1.0)),
            self._norm_angle(bearing),
            self._norm_angle(elev),
            self._norm_dist(dist),
            float(np.clip(teammate_state[STATE_HP], 0.0, 1.0)),
            float(teammate_state[STATE_ALIVE]),
        ], dtype=np.float32)

        assert len(obs) == OBS_TEAMMATE_DIM, f"Teammate obs boyutu hatalı: {len(obs)}"
        return obs

    # -----------------------------------------------------------------------
    # Enemy Observation (12 boyut)
    # -----------------------------------------------------------------------

    def enemy_obs(self, agent_state: np.ndarray,
                  enemy_state: np.ndarray) -> np.ndarray:
        """
        Düşmana ait relative observation vektörü.

        Boyutlar (12):
            0-2 : rel_x, rel_y, rel_h   — göreceli konum (normalize) [-1,1]
            3   : rel_V_norm             — göreceli hız farkı [-1,1]
            4   : bearing_norm           — ajan→düşman bearing [-1,1]
            5   : elevation_norm         — ajan→düşman elevation [-1,1]
            6   : ata_norm               — Antenna Train Angle [-1,1]
            7   : aspect_norm            — Aspect Angle [0,1]
            8   : dist_norm              — mesafe [0,1]
            9   : threat                 — tehdit skoru [0,1]
            10  : hp_enemy               — düşman HP tahmini [0,1]
            11  : alive                  — hayatta mı {0,1}
        """
        if enemy_state[STATE_ALIVE] < 0.5:
            return np.zeros(OBS_ENEMY_DIM, dtype=np.float32)

        agent_pos = agent_state[[STATE_X, STATE_Y, STATE_H]]
        enemy_pos = enemy_state[[STATE_X, STATE_Y, STATE_H]]

        dx = enemy_state[STATE_X] - agent_state[STATE_X]
        dy = enemy_state[STATE_Y] - agent_state[STATE_Y]
        dh = enemy_state[STATE_H] - agent_state[STATE_H]

        rel_pos    = self._norm_rel_pos(dx, dy, dh)
        dist       = distance_3d(agent_pos, enemy_pos)
        bearing    = bearing_angle(agent_pos, enemy_pos)
        elev       = elevation_angle(agent_pos, enemy_pos)
        ata        = antenna_train_angle(agent_pos, enemy_pos, float(agent_state[STATE_PSI]))
        aa         = aspect_angle(enemy_pos, agent_pos, float(enemy_state[STATE_PSI]))
        rel_V      = enemy_state[STATE_V] - agent_state[STATE_V]
        ts         = threat_score(dist, ata, aa, self.wez_range_max, self.wez_angle_max)

        obs = np.array([
            rel_pos[0],
            rel_pos[1],
            rel_pos[2],
            float(np.clip(rel_V / (self.V_max + EPS), -1.0, 1.0)),
            self._norm_angle(bearing),
            self._norm_angle(elev),
            self._norm_angle(ata),                      # ATA [-1,1]
            float(np.clip(aa / np.pi, 0.0, 1.0)),       # Aspect [0,1]
            self._norm_dist(dist),
            float(np.clip(ts, 0.0, 1.0)),
            float(np.clip(enemy_state[STATE_HP], 0.0, 1.0)),
            float(enemy_state[STATE_ALIVE]),
        ], dtype=np.float32)

        assert len(obs) == OBS_ENEMY_DIM, f"Enemy obs boyutu hatalı: {len(obs)}"
        return obs

    # -----------------------------------------------------------------------
    # Tam Observation Vektörü
    # -----------------------------------------------------------------------

    def build_obs(self,
                  agent_state:     np.ndarray,
                  teammate_states: list,
                  enemy_states:    list,
                  aggression:      float = None,
                  gat_messages:    list  = None) -> np.ndarray:
        """
        Tüm bileşenlerden tam observation vektörü oluşturur.

        Parameters
        ----------
        agent_state      : np.ndarray — ajanın state'i
        teammate_states  : list[np.ndarray] — takım arkadaşları
        enemy_states     : list[np.ndarray] — düşmanlar
        aggression       : float | None
                           None → rol embedding eklenmez (Faz 0-1)
                           float → [aggression, 1-aggression] eklenir (Faz 2)
        gat_messages     : list[np.ndarray] | None
                           None → GAT mesajı eklenmez (Faz 0-2)
                           list → her takım arkadaşından 16 boyutlu mesaj (Faz 3)

        Returns
        -------
        np.ndarray — normalize edilmiş tam observation vektörü

        Boyut örnekleri:
            2v2, Faz 0-1 (rol yok, GAT yok)  : 16 + 9 + 12×2       = 49
            2v2, Faz 2   (rol var, GAT yok)   : 16 + 9 + 12×2 + 2   = 51
            2v2, Faz 3   (rol var, GAT var)   : 16 + 9 + 12×2 + 2+16= 67
            3v3, Faz 0-1                       : 16 + 9×2 + 12×3     = 70
            3v3, Faz 2                         : 70 + 2               = 72
            3v3, Faz 3                         : 72 + 16×2            = 104
        """
        parts = []

        # 1. Ego (16)
        parts.append(self.ego_obs(agent_state))

        # 2. Takım arkadaşları (9 × n_teammates)
        for tm_state in teammate_states:
            parts.append(self.teammate_obs(agent_state, tm_state))

        # 3. Düşmanlar (12 × n_enemies)
        for en_state in enemy_states:
            parts.append(self.enemy_obs(agent_state, en_state))

        # 4. Rol embedding (2) — Faz 2+
        if aggression is not None:
            a = float(np.clip(aggression, 0.0, 1.0))
            parts.append(np.array([a, 1.0 - a], dtype=np.float32))

        # 5. GAT mesajları (16 × n_teammates) — Faz 3+
        if gat_messages is not None:
            for msg in gat_messages:
                parts.append(np.asarray(msg, dtype=np.float32))

        return np.concatenate(parts, axis=0)

    def obs_dim(self,
                n_teammates:  int,
                n_enemies:    int,
                with_role:    bool = False,
                with_gat:     bool = False) -> int:
        """
        Verilen konfigürasyon için beklenen observation boyutunu döndürür.
        dogfight_env.py ve model init'te kullanılır.
        """
        dim = OBS_EGO_DIM
        dim += OBS_TEAMMATE_DIM * n_teammates
        dim += OBS_ENEMY_DIM    * n_enemies
        if with_role:
            dim += OBS_ROLE_DIM
        if with_gat:
            dim += OBS_GAT_MSG_DIM * n_teammates
        return dim

    # -----------------------------------------------------------------------
    # Action Normalize / Denormalize
    # -----------------------------------------------------------------------

    @staticmethod
    def normalize_action(action: np.ndarray) -> np.ndarray:
        """
        Ham aksiyon → normalize.
        Aksiyonlar zaten [-1,1] / [0,1] aralığında geldiği için
        şu an sadece clip işlemi yapılır.
        aircraft_model._clip_action() ile tutarlı.
        """
        from envs.aircraft_model import ACTION_DA, ACTION_DE, ACTION_DR, ACTION_DT, ACTION_FIRE
        a = action.copy().astype(np.float32)
        a[ACTION_DA]   = np.clip(a[ACTION_DA],   -1.0, 1.0)
        a[ACTION_DE]   = np.clip(a[ACTION_DE],   -1.0, 1.0)
        a[ACTION_DR]   = np.clip(a[ACTION_DR],   -1.0, 1.0)
        a[ACTION_DT]   = np.clip(a[ACTION_DT],    0.0, 1.0)
        a[ACTION_FIRE] = np.clip(a[ACTION_FIRE],  0.0, 1.0)
        return a

    @staticmethod
    def denormalize_action(action: np.ndarray) -> np.ndarray:
        """
        Normalize aksiyon → ham (şu an identity, ileride ölçekleme eklenebilir).
        """
        return Normalizer.normalize_action(action)

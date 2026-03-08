"""
heuristic_agent.py
==================
Kural tabanlı baseline ajan — MAPPO'nun üzerine geçmesi gereken referans.

Davranış mimarisi (öncelik sırası):
    1. CRITICAL  — Zemin / stall / sınır kurtarma (hayatta kalma zorunlu)
    2. EVASION   — Düşman WEZ'indeyse kaç
    3. PURSUIT   — En tehlikeli düşmanı takip et, WEZ'e gir, ateş et

Her davranış bir "delta aksiyon" üretir.
Öncelik sırası: CRITICAL > EVASION > PURSUIT.
Aynı adımda sadece bir davranış aktif olur.

Aksiyon uzayı (aircraft_model.py ACTION_* ile uyumlu):
    [da, de, dr, dt, fire]
    da  ∈ [-1, 1]  aileron
    de  ∈ [-1, 1]  elevator
    dr  ∈ [-1, 1]  rudder
    dt  ∈ [ 0, 1]  throttle
    fire∈ [ 0, 1]  ateş komutu (>0.5 ateş)

Bağımlılıklar:
    - numpy
    - aircraft_model.py  (STATE_* sabitleri)
    - geometry_utils.py  (bearing_angle, elevation_angle, distance_3d,
                          antenna_train_angle, wrap_to_pi, deg2rad)
    - weapons_model.py   (wez_advantage_score)

Bu dosya değişirse etkilenen dosyalar:
    - training/train_mappo.py   (opponent olarak kullanılır)
    - evaluation/eval.py        (baseline karşılaştırma)
    - test_heuristic_agent.py
"""

import numpy as np
from envs.aircraft_model import (
    STATE_X, STATE_Y, STATE_H, STATE_V,
    STATE_ALPHA, STATE_BETA, STATE_GAMMA,
    STATE_PHI, STATE_THETA, STATE_PSI,
    STATE_P, STATE_Q, STATE_R,
    STATE_FUEL, STATE_AMMO, STATE_HP, STATE_ALIVE,
    ACTION_DA, ACTION_DE, ACTION_DR, ACTION_DT, ACTION_FIRE,
    ACTION_DIM,
)
from envs.geometry_utils import (
    bearing_angle, elevation_angle, distance_3d,
    antenna_train_angle, aspect_angle, wrap_to_pi, deg2rad, rad2deg
)
from envs.weapons_model import WeaponsModel


# ---------------------------------------------------------------------------
# Davranış sabitleri
# ---------------------------------------------------------------------------

# CRITICAL eşikleri
CRITICAL_H_FLOOR    = 300.0   # m   — bu altında zemin kurtarma devreye girer
CRITICAL_ALPHA_MAX  = deg2rad(22.0)  # rad — stall öncesi kurtarma
CRITICAL_V_MIN      = 80.0    # m/s — çok düşük hız kurtarma
CRITICAL_MAP_MARGIN = 3000.0  # m   — harita kenarına bu kadar kaldığında dön

# EVASION eşikleri
EVASION_THREAT_THRESHOLD = 0.55  # threat_score bu üstündeyse kaç
EVASION_BREAK_ROLL       = 1.0   # tam aileron (break turn)
EVASION_UNLOAD_DE        = -0.3  # hafif burun indirme (hız kazanma)

# PURSUIT sabitleri
PURSUIT_BANK_GAIN     = 2.0    # bearing hatası → aileron gain
PURSUIT_PITCH_GAIN    = 1.5    # elevation hatası → elevator gain
PURSUIT_HEADING_TOL   = deg2rad(5.0)   # bu kadar hizalıysa düz uç
PURSUIT_WEZ_ENGAGE    = 0.6    # wez_advantage bu üstündeyse ateş
PURSUIT_THROTTLE_FULL = 1.0
PURSUIT_THROTTLE_CRSE = 0.7

# Nominal uçuş
NOMINAL_THROTTLE = 0.65
NOMINAL_H        = 4000.0     # m — hedef irtifa (düşman yokken)


class HeuristicAgent:
    """
    Kural tabanlı tek ajan.

    DogfightEnv'deki her ajan için ayrı örnek oluşturulur.

    Kullanım:
        agent = HeuristicAgent(config, agent_id="red_0")
        action = agent.act(own_state, teammate_states, enemy_states)
    """

    def __init__(self, config: dict, agent_id: str = "heuristic"):
        self.agent_id      = agent_id
        self.map_size      = float(config["env"]["map_size"])
        self.half_map      = self.map_size / 2.0
        self._wm           = WeaponsModel(config)

        # Aircraft limits (kurtarma kontrolü için)
        ac = config["aircraft"]
        self.h_min   = float(ac.get("h_min",  50.0))
        self.V_min   = float(ac.get("V_min",  60.0))
        self.V_max   = float(ac.get("V_max", 600.0))

    def reset(self):
        """Episode başında çağrılır. WeaponsModel cooldown sıfırla."""
        self._wm.reset()

    # -----------------------------------------------------------------------
    # Ana Karar Metodu
    # -----------------------------------------------------------------------

    def act(self,
            own_state:       np.ndarray,
            teammate_states: list,
            enemy_states:    list) -> np.ndarray:
        """
        Mevcut state'e göre aksiyon üretir.

        Parameters
        ----------
        own_state        : np.ndarray shape (STATE_DIM,)
        teammate_states  : list[np.ndarray] — takım arkadaşları (kullanılmıyor, genişleme için)
        enemy_states     : list[np.ndarray] — düşmanlar

        Returns
        -------
        action : np.ndarray shape (ACTION_DIM,) — normalize, clip edilmiş
        """
        # Ölüyse sıfır aksiyon
        if own_state[STATE_ALIVE] < 0.5:
            return np.zeros(ACTION_DIM, dtype=np.float32)

        # Hayatta düşmanlar
        alive_enemies = [e for e in enemy_states if e[STATE_ALIVE] > 0.5]

        # ── Öncelik 1: CRITICAL ───────────────────────────────────────
        action = self._critical_recovery(own_state)
        if action is not None:
            return self._clip(action)

        # ── Öncelik 2: EVASION ───────────────────────────────────────
        if alive_enemies:
            action = self._evasion(own_state, alive_enemies)
            if action is not None:
                return self._clip(action)

        # ── Öncelik 3: PURSUIT ───────────────────────────────────────
        if alive_enemies:
            target = self._select_target(own_state, alive_enemies)
            return self._clip(self._pursuit(own_state, target))

        # Düşman kalmadıysa nominal uçuş
        return self._clip(self._nominal_flight(own_state))

    # -----------------------------------------------------------------------
    # Davranış 1: CRITICAL RECOVERY
    # -----------------------------------------------------------------------

    def _critical_recovery(self, s: np.ndarray) -> np.ndarray | None:
        """
        Hayati tehlike durumlarında override aksiyon.
        None döndürürse tehlike yok, bir sonraki davranışa geç.
        """
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        triggered = False

        # Zemin yaklaşımı — burun kaldır, tam gaz
        if s[STATE_H] < CRITICAL_H_FLOOR:
            pull_strength = np.clip(
                (CRITICAL_H_FLOOR - s[STATE_H]) / CRITICAL_H_FLOOR, 0.3, 1.0
            )
            action[ACTION_DE] = float(pull_strength)   # burun yukarı
            action[ACTION_DT] = PURSUIT_THROTTLE_FULL
            action[ACTION_DA] = float(-s[STATE_PHI] * 0.5)  # kanatları düzelt
            triggered = True

        # Stall — yüklemi azalt, hız kazan
        elif abs(s[STATE_ALPHA]) > CRITICAL_ALPHA_MAX:
            action[ACTION_DE] = -0.5    # burun indirme (alpha azalt)
            action[ACTION_DT] = PURSUIT_THROTTLE_FULL
            triggered = True

        # Çok düşük hız
        elif s[STATE_V] < CRITICAL_V_MIN:
            action[ACTION_DE] = -0.3
            action[ACTION_DT] = PURSUIT_THROTTLE_FULL
            triggered = True

        # Harita sınırı — döndürme manevrası
        elif self._near_boundary(s):
            return self._boundary_turn(s)

        return action if triggered else None

    def _near_boundary(self, s: np.ndarray) -> bool:
        return (abs(s[STATE_X]) > self.half_map - CRITICAL_MAP_MARGIN or
                abs(s[STATE_Y]) > self.half_map - CRITICAL_MAP_MARGIN)

    def _boundary_turn(self, s: np.ndarray) -> np.ndarray:
        """Harita merkezine doğru sert dönüş."""
        action = np.zeros(ACTION_DIM, dtype=np.float32)

        # Merkeze bearing hesapla
        own_pos     = np.array([s[STATE_X], s[STATE_Y], s[STATE_H]])
        center_pos  = np.array([0.0, 0.0, s[STATE_H]])
        target_bear = bearing_angle(own_pos, center_pos)
        bear_err    = wrap_to_pi(target_bear - s[STATE_PSI])

        action[ACTION_DA] = float(np.clip(bear_err * 1.5, -1.0, 1.0))
        action[ACTION_DE] = 0.2   # hafif çekiş
        action[ACTION_DT] = PURSUIT_THROTTLE_FULL
        return action

    # -----------------------------------------------------------------------
    # Davranış 2: EVASION
    # -----------------------------------------------------------------------

    def _evasion(self, s: np.ndarray,
                 enemies: list) -> np.ndarray | None:
        """
        En tehlikeli düşmanın tehdit skoru eşiği aşıyorsa kaç.
        Break turn: en tehlikeli düşmandan zıt yöne sert dönüş.
        """
        # En yüksek tehdit skoru
        max_threat = 0.0
        worst_enemy = None
        for e in enemies:
            threat = self._compute_threat(e, s)  # düşmanın bize tehdidi
            if threat > max_threat:
                max_threat  = threat
                worst_enemy = e

        if max_threat < EVASION_THREAT_THRESHOLD:
            return None  # tehdit yok, EVASION devreye girme

        # Break turn — düşmandan en uzak yöne sert aileron
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        own_pos   = np.array([s[STATE_X],   s[STATE_Y],   s[STATE_H]])
        enemy_pos = np.array([worst_enemy[STATE_X], worst_enemy[STATE_Y],
                               worst_enemy[STATE_H]])

        bear_to_enemy = bearing_angle(own_pos, enemy_pos)
        bear_err      = wrap_to_pi(bear_to_enemy - s[STATE_PSI])

        # Düşman solumuzda → sağa break (aileron +1), sağımızda → sola break
        break_dir = -np.sign(bear_err) if abs(bear_err) > 0.1 else 1.0
        action[ACTION_DA] = float(break_dir * EVASION_BREAK_ROLL)
        action[ACTION_DE] = EVASION_UNLOAD_DE
        action[ACTION_DT] = PURSUIT_THROTTLE_FULL
        action[ACTION_FIRE] = 0.0

        return action

    def _compute_threat(self, enemy_state: np.ndarray,
                         own_state: np.ndarray) -> float:
        """
        Düşmanın bize olan tehdit skoru.
        weapons_model.wez_advantage_score'u ters perspektiften çağırır:
        düşman shooter, biz target.
        """
        return float(self._wm.wez_advantage_score(enemy_state, own_state))

    # -----------------------------------------------------------------------
    # Davranış 3: PURSUIT
    # -----------------------------------------------------------------------

    def _select_target(self, s: np.ndarray,
                        enemies: list) -> np.ndarray:
        """
        En yüksek WEZ avantajı sağlayan düşmanı seç.
        WEZ avantajı eşitse en yakın düşman seçilir.
        """
        best_score = -np.inf
        best_enemy = enemies[0]
        own_pos = np.array([s[STATE_X], s[STATE_Y], s[STATE_H]])

        for e in enemies:
            wez_adv = float(self._wm.wez_advantage_score(s, e))
            dist    = distance_3d(own_pos,
                                   np.array([e[STATE_X], e[STATE_Y], e[STATE_H]]))
            # Bileşik skor: WEZ avantajı ağırlıklı, mesafe ikincil
            score = wez_adv * 2.0 - dist / 50000.0
            if score > best_score:
                best_score = score
                best_enemy = e

        return best_enemy

    def _pursuit(self, s: np.ndarray,
                  target: np.ndarray) -> np.ndarray:
        """
        Hedefe proportional navigation benzeri kovalama.

        1. Bearing hatası → aileron (bank-to-turn)
        2. Elevation hatası → elevator
        3. WEZ içindeyse → ateş
        4. Throttle: uzaksa tam gaz, yakınsa kıs
        """
        action = np.zeros(ACTION_DIM, dtype=np.float32)

        own_pos    = np.array([s[STATE_X],      s[STATE_Y],      s[STATE_H]])
        target_pos = np.array([target[STATE_X], target[STATE_Y], target[STATE_H]])

        # Yön hataları
        target_bear = bearing_angle(own_pos, target_pos)
        target_elev = elevation_angle(own_pos, target_pos)
        bear_err    = wrap_to_pi(target_bear - s[STATE_PSI])
        elev_err    = target_elev - s[STATE_GAMMA]

        # Aileron: bearing hatasını düzelt (bank-to-turn)
        da = float(np.clip(bear_err * PURSUIT_BANK_GAIN, -1.0, 1.0))

        # Elevator: elevation hatasını düzelt
        # Hizalanmışsa pitch stabilize et (theta → 0'a çek)
        if abs(bear_err) < PURSUIT_HEADING_TOL:
            de = float(np.clip(elev_err * PURSUIT_PITCH_GAIN, -1.0, 1.0))
        else:
            # Dönerken hafif çekiş (koordineli dönüş)
            de = float(np.clip(abs(da) * 0.3, 0.0, 0.5))

        # Throttle: mesafeye göre
        dist = distance_3d(own_pos, target_pos)
        if dist > 6000.0:
            dt = PURSUIT_THROTTLE_FULL
        elif dist > 2000.0:
            dt = PURSUIT_THROTTLE_CRSE
        else:
            dt = NOMINAL_THROTTLE  # çok yakınsa kıs (overshootu önle)

        # Ateş kararı: WEZ avantajı eşiği aştıysa
        wez_adv = float(self._wm.wez_advantage_score(s, target))
        fire    = 1.0 if wez_adv >= PURSUIT_WEZ_ENGAGE else 0.0

        action[ACTION_DA]   = da
        action[ACTION_DE]   = de
        action[ACTION_DT]   = dt
        action[ACTION_FIRE] = fire
        return action

    # -----------------------------------------------------------------------
    # Nominal Uçuş (düşman yokken)
    # -----------------------------------------------------------------------

    def _nominal_flight(self, s: np.ndarray) -> np.ndarray:
        """
        Hedef irtifaya çık, kanatları düzelt, sabit hızda uç.
        """
        action = np.zeros(ACTION_DIM, dtype=np.float32)

        h_err = NOMINAL_H - s[STATE_H]
        de    = float(np.clip(h_err / 2000.0, -0.3, 0.3))
        da    = float(np.clip(-s[STATE_PHI] * 0.5, -1.0, 1.0))  # roll sıfırla

        action[ACTION_DA] = da
        action[ACTION_DE] = de
        action[ACTION_DT] = NOMINAL_THROTTLE
        return action

    # -----------------------------------------------------------------------
    # Yardımcı
    # -----------------------------------------------------------------------

    @staticmethod
    def _clip(action: np.ndarray) -> np.ndarray:
        """Aksiyon sınırlarına clip uygula."""
        a = action.copy().astype(np.float32)
        a[ACTION_DA]   = np.clip(a[ACTION_DA],   -1.0, 1.0)
        a[ACTION_DE]   = np.clip(a[ACTION_DE],   -1.0, 1.0)
        a[ACTION_DR]   = np.clip(a[ACTION_DR],   -1.0, 1.0)
        a[ACTION_DT]   = np.clip(a[ACTION_DT],    0.0, 1.0)
        a[ACTION_FIRE] = np.clip(a[ACTION_FIRE],  0.0, 1.0)
        return a


# ---------------------------------------------------------------------------
# Çok Ajanlı Sarmalayıcı
# ---------------------------------------------------------------------------

class MultiHeuristicPolicy:
    """
    DogfightEnv ile doğrudan çalışan çok-ajanlı sarmalayıcı.
    Tüm ajanlar için aynı kural setini uygular.

    Kullanım:
        policy = MultiHeuristicPolicy(config, agent_ids)
        action_dict = policy.act(obs_dict=None, state_dict=state_dict)
    """

    def __init__(self, config: dict, agent_ids: list,
                 team_map: dict):
        """
        Parameters
        ----------
        config     : dict
        agent_ids  : list[str]   — tüm ajan ID'leri
        team_map   : dict[str, str] — {agent_id: "blue"/"red"}
        """
        self._agents = {
            aid: HeuristicAgent(config, agent_id=aid)
            for aid in agent_ids
        }
        self._team_map = team_map
        self._agent_ids = agent_ids

        # Takım → üye ID'leri
        self._team_members: dict[str, list] = {}
        for aid, team in team_map.items():
            self._team_members.setdefault(team, []).append(aid)

    def reset(self):
        for agent in self._agents.values():
            agent.reset()

    def act(self, state_dict: dict) -> dict:
        """
        Parameters
        ----------
        state_dict : dict[str, np.ndarray] — env.get_all_states() çıktısı

        Returns
        -------
        action_dict : dict[str, np.ndarray]
        """
        action_dict = {}
        for aid, agent in self._agents.items():
            own_state = state_dict[aid]
            team      = self._team_map[aid]

            teammates = [state_dict[tid]
                         for tid in self._team_members[team]
                         if tid != aid]
            enemies   = [state_dict[eid]
                         for eid in self._agent_ids
                         if self._team_map[eid] != team]

            action_dict[aid] = agent.act(own_state, teammates, enemies)

        return action_dict

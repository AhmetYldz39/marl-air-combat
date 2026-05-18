"""
dogfight_env.py
===============
MARL hava muharebe simülasyon ortamı.

Gym-benzeri arayüz, çok-ajanlı (CTDE paradigması).
Faz 0-1: 2v2, iletişimsiz, rol yok.
Faz 2+ : aggression embedding, GAT mesajları (dışarıdan enjekte edilir).

API:
    env = DogfightEnv(config)
    obs_dict = env.reset()
    obs_dict, rew_dict, done_dict, info_dict = env.step(action_dict)

Ajan ID'leri:
    "blue_0", "blue_1"   — Blue takımı
    "red_0",  "red_1"    — Red takımı
    (3v3'te "blue_2", "red_2" eklenir)

Bağımlılıklar:
    - aircraft_model.py
    - weapons_model.py
    - reward_model.py
    - normalization.py
    - geometry_utils.py

Bu dosya değişirse etkilenen dosyalar:
    - training/train_mappo.py
    - evaluation/eval.py
    - test_env_step.py
"""

import numpy as np
import yaml
from collections import deque
from copy import deepcopy

from envs.aircraft_model import (
    AircraftModel,
    STATE_X, STATE_Y, STATE_H, STATE_V,
    STATE_ALPHA, STATE_PHI, STATE_PSI, STATE_FUEL,
    STATE_AMMO, STATE_HP, STATE_ALIVE,
    STATE_DIM, ACTION_DIM,
    ACTION_DA, ACTION_DE, ACTION_DR, ACTION_DT,
)
from envs.weapons_model  import WeaponsModel
from envs.reward_model   import RewardModel
from utils.normalization  import Normalizer
from envs.geometry_utils import deg2rad, bearing_angle, wrap_to_pi, distance_3d
from envs.trim_solver import TrimSolver

# ---------------------------------------------------------------------------
# CRITICAL recovery sabitleri (Blue MAPPO agent OOB koruması)
# ---------------------------------------------------------------------------
_CRIT_H_FLOOR    = 300.0    # m   — zemin kurtarma eşiği
_CRIT_MAP_MARGIN = 5000.0   # m   — harita kenarı döndürme eşiği (3000→5000)


# ---------------------------------------------------------------------------
# Takım sabitleri
# ---------------------------------------------------------------------------
BLUE = "blue"
RED  = "red"
TEAMS = [BLUE, RED]


class DogfightEnv:
    """
    Çok-ajanlı hava muharebe ortamı.

    Her takımda n_agents_per_team adet ajan vardır.
    Tüm ajanlar aynı AircraftModel ve WeaponsModel parametrelerini
    paylaşır (homojen takım).

    Attributes
    ----------
    agent_ids     : list[str]   — tüm ajan ID'leri
    obs_dim       : int         — tek ajan observation boyutu
    action_dim    : int         — tek ajan aksiyon boyutu (= ACTION_DIM = 5)
    n_agents      : int         — toplam ajan sayısı
    """

    def __init__(self, config: dict):
        self.config = config
        env_cfg     = config["env"]

        # ── Parametreler ──────────────────────────────────────────────
        self.dt                 = float(env_cfg["dt"])
        self.max_steps          = int(env_cfg["max_steps"])
        self.map_size           = float(env_cfg["map_size"])
        self._max_n_per_team    = int(env_cfg["n_agents_per_team"])  # config'deki max değer (2)

        # Spawn aralıkları
        self.spawn_x_range      = float(env_cfg.get("spawn_x_range",   10000.0))
        self.spawn_y_range      = float(env_cfg.get("spawn_y_range",   10000.0))
        self.spawn_h_min        = float(env_cfg.get("spawn_h_min",      3000.0))
        self.spawn_h_max        = float(env_cfg.get("spawn_h_max",      8000.0))
        self.spawn_V_min        = float(env_cfg.get("spawn_V_min",       150.0))
        self.spawn_V_max        = float(env_cfg.get("spawn_V_max",       280.0))
        self.spawn_team_offset  = float(env_cfg.get("spawn_team_offset", 8000.0))

        # Curriculum-v2 spawn parametreleri
        cur = config.get("curriculum_v2", {})
        self._phase1_dist_min    = float(cur.get("phase1_spawn_dist_min",  500.0))
        self._phase1_dist_max    = float(cur.get("phase1_spawn_dist_max", 1500.0))
        self._dynamic_spawn_dist = float(cur.get("phase15_dist_start",   2000.0))
        self._normal_spawn_dist     = float(cur.get("phase3_spawn_dist",     6000.0))
        self._normal_spawn_dist_max = float(cur.get("phase3_spawn_dist_max", 12000.0))

        # Team kill bonus (config'den)
        self._team_kill_bonus = float(config.get("reward", {}).get("team_kill_bonus", 0.0))

        # Curriculum fazı: 1=WEZ-close, 2=dinamik-dist, 3=1v1-normal, 4=2v2-normal
        self._curriculum_phase = 1

        # ── Alt modüller ──────────────────────────────────────────────
        self._aircraft   = AircraftModel(config)
        self._normalizer = Normalizer(config)

        # Dead-agent için sıfır state (obs padding'de kullanılır)
        self._dummy_state = np.zeros(STATE_DIM, dtype=np.float32)

        # ── Ajan ID'leri (başlangıç: Faz 1 = 1v1) ────────────────────
        self.n_per_team = 1
        self._rebuild_agent_ids()

        # ── Boyutlar ──────────────────────────────────────────────────
        # obs_dim her zaman max topoloji (2v2) boyutunda — ağ sabit kalır
        n_tm_max = self._max_n_per_team - 1   # 1
        n_en_max = self._max_n_per_team        # 2
        self.obs_dim    = self._normalizer.obs_dim(n_tm_max, n_en_max)  # 50
        self.action_dim = ACTION_DIM

        # ── Episode state ─────────────────────────────────────────────
        self._states: dict[str, np.ndarray] = {}
        self._step_count  = 0
        self._done        = False
        self._episode_rewards: dict[str, float] = {}
        self._info_history:    dict[str, list]  = {}

        # ── Düşman obs geçmişi (OpponentModel için) ────────────────────
        # Her Blue ajan için son _opp_hist_steps adım düşman obs (24D/adım) tutulur.
        # Trainer bu buffer'ı okur, OpponentModel çalıştırır, opp_intent'i obs'a ekler.
        opp_cfg = config.get("opponent_model", {})
        self._opp_hist_steps  = int(opp_cfg.get("history_window", 20))
        self._opp_enemy_dim   = 12   # OBS_ENEMY_DIM
        self._opp_step_dim    = self._max_n_per_team * self._opp_enemy_dim  # 24
        self._enemy_history: dict[str, deque] = {}  # doldurulur reset()'te

        # RNG (seed dışarıdan set edilebilir)
        self.rng = np.random.default_rng(seed=None)

        # Trim çözücü + lookup tablosu (spawn başlangıç koşulu)
        self._trim_solver = TrimSolver(self._aircraft)
        self._trim_table  = self._trim_solver.build_lookup_table(
            V_range=(self.spawn_V_min, self.spawn_V_max),
            h_range=(self.spawn_h_min, self.spawn_h_max),
            n_V=10, n_h=8,
        )

    # -----------------------------------------------------------------------
    # Curriculum
    # -----------------------------------------------------------------------

    def _rebuild_agent_ids(self):
        """n_per_team'e göre ajan listelerini ve alt modülleri yeniden oluşturur."""
        self.blue_ids  = [f"blue_{i}" for i in range(self.n_per_team)]
        self.red_ids   = [f"red_{i}"  for i in range(self.n_per_team)]
        self.agent_ids = self.blue_ids + self.red_ids
        self.n_agents  = len(self.agent_ids)
        self._team_of  = {aid: BLUE for aid in self.blue_ids}
        self._team_of.update({aid: RED for aid in self.red_ids})
        self._weapons  = {aid: WeaponsModel(self.config) for aid in self.agent_ids}
        self._rewards  = {aid: RewardModel(self.config)  for aid in self.agent_ids}

    def set_curriculum_phase(self, phase: int):
        """
        Curriculum fazını günceller.

        Dahili faz numaraları:
          1 = Faz-1   : 1v1 WEZ-close
          2 = Faz-1.5 : 1v1 dinamik mesafe
          3 = Faz-2   : 1v1 normal
          4 = Faz-3   : 2v2 normal
        """
        self._curriculum_phase = phase
        new_n = 1 if phase < 4 else self._max_n_per_team
        if new_n != self.n_per_team:
            self.n_per_team = new_n
            self._rebuild_agent_ids()

    def set_dynamic_spawn_dist(self, dist: float):
        """Faz 1.5 için spawn mesafesini günceller."""
        self._dynamic_spawn_dist = float(dist)

    # -----------------------------------------------------------------------
    # Seed
    # -----------------------------------------------------------------------

    def seed(self, seed: int):
        self.rng = np.random.default_rng(seed)

    # -----------------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------------

    def reset(self, aggression_dict: dict = None) -> dict:
        """
        Episode'u sıfırlar ve ilk observation'ları döndürür.

        Parameters
        ----------
        aggression_dict : dict[str, float] | None
            Her ajan için aggression değeri.
            None → tüm ajanlar için aggression=None (Faz 0-1 pasif mod).

        Returns
        -------
        obs_dict : dict[str, np.ndarray]
            Her ajan için normalize observation vektörü.
        """
        self._step_count = 0
        self._done       = False

        # Aggression
        if aggression_dict is None:
            aggression_dict = {aid: None for aid in self.agent_ids}
        self._aggression = aggression_dict

        # Silah cooldown sıfırla
        for wm in self._weapons.values():
            wm.reset()

        # Episode istatistikleri
        self._episode_rewards  = {aid: 0.0 for aid in self.agent_ids}
        self._info_history     = {aid: []  for aid in self.agent_ids}

        # Spawn
        self._states = self._spawn_agents()

        # Önceki aksiyon takibi (smoothness reward için)
        self._prev_actions = {aid: None for aid in self.agent_ids}

        # Kapanma hızı ödülü için önceki mesafe
        self._prev_distances: dict[str, float] = {aid: None for aid in self.agent_ids}

        # WEZ streak sayacı (N ardışık adım WEZ içindeyse bonus)
        self._wez_streaks: dict[str, int] = {aid: 0 for aid in self.agent_ids}

        # Düşman geçmişi sıfırla — her Blue ajan için _opp_hist_steps × sıfır vektör
        zero_step = np.zeros(self._opp_step_dim, dtype=np.float32)
        self._enemy_history = {
            aid: deque(
                [zero_step.copy() for _ in range(self._opp_hist_steps)],
                maxlen=self._opp_hist_steps,
            )
            for aid in self.blue_ids
        }

        return self._build_obs_dict()

    # -----------------------------------------------------------------------
    # Step
    # -----------------------------------------------------------------------

    def step(self, action_dict: dict,
             gat_messages: dict = None,
             role_support_probs: dict = None,
             role_vecs: dict = None) -> tuple:
        """
        Bir adım ilerler.

        Parameters
        ----------
        action_dict : dict[str, np.ndarray]
            Her ajan için ACTION_DIM boyutlu aksiyon vektörü.
            Ölü ajanlar için sıfır vektör gönderilebilir (yoksayılır).
        gat_messages : dict[str, list[np.ndarray]] | None
            Faz 3+ : her ajan için takım arkadaşlarından gelen GAT mesajları.
            None → Faz 0-2 (mesajsız).

        Returns
        -------
        obs_dict  : dict[str, np.ndarray]
        rew_dict  : dict[str, float]
        done_dict : dict[str, bool]  — "__all__" anahtarı da içerir
        info_dict : dict[str, dict]
        """
        assert not self._done, "Episode bitti. reset() çağırın."

        prev_states = {aid: s.copy() for aid, s in self._states.items()}
        fire_results = {}

        # ── 1. Fizik adımı + ateş işlemi ─────────────────────────────
        for aid in self.agent_ids:
            state = self._states[aid]
            if state[STATE_ALIVE] < 0.5:
                fire_results[aid] = None
                continue

            raw_act = action_dict.get(aid, np.zeros(ACTION_DIM, dtype=np.float32))
            # Blue MAPPO agent için OOB / zemin kurtarma override
            if aid in self.blue_ids:
                recovery = self._critical_recovery_blue(state)
                if recovery is not None:
                    raw_act = recovery
            action = self._normalizer.normalize_action(raw_act)

            # Fizik
            self._states[aid] = self._aircraft.step(state, action, self.dt)

            # Ateş
            enemies = self._get_enemies(aid)
            # Tek hedef: en yakın hayatta düşman
            target_state = self._closest_alive(self._states[aid], enemies)
            if target_state is not None:
                fire_result = self._weapons[aid].process_fire(
                    self._states[aid], target_state, action, self.dt
                )
                # Hasar uygula
                if fire_result["hit"]:
                    target_id = self._find_state_owner(target_state, enemies)
                    if target_id is not None:
                        self._states[target_id][STATE_HP] = float(
                            fire_result["new_target_hp"]
                        )
                        if fire_result["kill"]:
                            self._states[target_id][STATE_ALIVE] = 0.0
                # Ammo güncelle (fired=True durumunda)
                if fire_result["fired"]:
                    self._states[aid][STATE_AMMO] = float(
                        fire_result["ammo_remaining"]
                    )
            else:
                fire_result = self._weapons[aid]._no_fire_result(
                    self._weapons[aid]._empty_wez(), "no_target"
                )
            fire_results[aid] = fire_result

        # Cooldown tick (ateş komutundan bağımsız olarak her adımda)
        for aid in self.agent_ids:
            self._weapons[aid].tick(self.dt)

        self._step_count += 1

        # Düşman geçmişini güncelle (tüm Blue ajanlar için)
        self._update_enemy_history()

        # ── 2. Reward hesabı ──────────────────────────────────────────
        rew_dict  = {}
        info_dict = {}

        for aid in self.agent_ids:
            state      = self._states[aid]
            teammates  = self._get_teammates(aid)
            enemies    = self._get_enemies(aid)
            agg        = self._aggression.get(aid)

            # Aksiyon deltası: smoothness reward için (episode başında None)
            raw_action  = action_dict.get(aid, np.zeros(self.action_dim, dtype=np.float32))
            prev_act    = self._prev_actions.get(aid)
            action_delta = (raw_action - prev_act) if prev_act is not None else None
            self._prev_actions[aid] = raw_action.copy()

            # Kapanma hızı için önceki mesafe
            prev_dist = self._prev_distances.get(aid)

            role_sp = (role_support_probs.get(aid, 1.0)
                       if role_support_probs and aid in self.blue_ids
                       else 1.0)
            role_v = (role_vecs.get(aid, None)
                      if role_vecs and aid in self.blue_ids
                      else None)
            total_rew, rew_info = self._rewards[aid].compute(
                agent_state        = state,
                teammate_states    = teammates,
                enemy_states       = enemies,
                weapons_model      = self._weapons[aid],
                prev_state         = prev_states[aid],
                fire_result        = fire_results[aid],
                dt                 = self.dt,
                map_size           = self.map_size,
                aggression         = agg,
                prev_action        = action_delta,
                prev_distance      = prev_dist,
                current_action     = raw_action,
                role_support_prob  = role_sp,
                role_vec           = role_v,
            )

            # Mevcut adımda en yakın düşman mesafesini kaydet
            self._prev_distances[aid] = self._nearest_enemy_dist(aid)

            rew_dict[aid]  = float(total_rew)
            info_dict[aid] = rew_info
            self._episode_rewards[aid] += float(total_rew)
            self._info_history[aid].append(rew_info)

            # WEZ streak bonusu: N=5 ardışık adım WEZ içindeyse +0.3
            if aid in self.blue_ids:
                fr = fire_results.get(aid)
                in_wez = (fr is not None and
                          fr.get("wez_info", {}).get("in_wez", False))
                if in_wez:
                    self._wez_streaks[aid] += 1
                    if self._wez_streaks[aid] % 5 == 0:
                        rew_dict[aid] += 0.3
                        self._episode_rewards[aid] += 0.3
                else:
                    self._wez_streaks[aid] = 0

        # ── 3. Done kontrolü ─────────────────────────────────────────
        done_dict = self._check_done()
        if done_dict["__all__"]:
            self._done = True
            # Team kill bonus: tüm Red'ler ölünce her Blue ajana bonus
            if done_dict.get("winner") == BLUE and self._team_kill_bonus > 0.0:
                for aid in self.blue_ids:
                    if self._states[aid][STATE_ALIVE] > 0.5:
                        rew_dict[aid] = rew_dict.get(aid, 0.0) + self._team_kill_bonus
                        self._episode_rewards[aid] += self._team_kill_bonus
            # Episode özeti her ajana ekle
            for aid in self.agent_ids:
                info_dict[aid]["episode"] = self._build_episode_summary(aid)

        # ── 4. Observation ───────────────────────────────────────────
        obs_dict = self._build_obs_dict(gat_messages=gat_messages)

        return obs_dict, rew_dict, done_dict, info_dict

    # -----------------------------------------------------------------------
    # Spawn
    # -----------------------------------------------------------------------

    def _update_enemy_history(self):
        """
        Her Blue ajan için güncel düşman obs (24D) geçmişe ekler.
        Her adımda _step_count artışından sonra çağrılır.
        """
        for aid in self.blue_ids:
            state = self._states.get(aid)
            if state is None:
                continue
            parts = []
            for eid in self.red_ids:
                es = self._states.get(eid, self._dummy_state)
                parts.append(self._normalizer.enemy_obs(state, es))
            # Kalan slotları sıfırla doldur (1v1 fazlarda 2. düşman yok)
            while len(parts) < self._max_n_per_team:
                parts.append(np.zeros(self._opp_enemy_dim, dtype=np.float32))
            step_obs = np.concatenate(parts[:self._max_n_per_team], axis=0)  # 24D
            if aid in self._enemy_history:
                self._enemy_history[aid].append(step_obs)

    def get_enemy_history_flat(self, aid: str) -> np.ndarray:
        """
        Belirtilen Blue ajan için düşman obs geçmişini düzleştirilmiş olarak döndürür.
        Döndürür: (HISTORY_STEPS * N_ENEMIES * ENEMY_OBS_DIM,) = (480,) float32
        Sıfır vektörle başlatılmış: ilk episode'da geçmiş dolana kadar sıfır.
        """
        hist = self._enemy_history.get(aid)
        if hist is None:
            return np.zeros(self._opp_hist_steps * self._opp_step_dim, dtype=np.float32)
        return np.concatenate(list(hist), axis=0)

    def _nearest_enemy_dist(self, aid: str) -> float:
        """aid'nin en yakın hayatta düşmanına 3D mesafe (m). Düşman yoksa inf."""
        state = self._states.get(aid)
        if state is None or state[STATE_ALIVE] < 0.5:
            return np.inf
        pos = state[[STATE_X, STATE_Y, STATE_H]]
        enemy_ids = self.red_ids if aid in self.blue_ids else self.blue_ids
        best = np.inf
        for eid in enemy_ids:
            es = self._states.get(eid)
            if es is None or es[STATE_ALIVE] < 0.5:
                continue
            d = distance_3d(pos, es[[STATE_X, STATE_Y, STATE_H]])
            if d < best:
                best = d
        return best

    def _spawn_agents(self) -> dict:
        """Curriculum fazına göre doğru spawn metodunu çağırır."""
        if self._curriculum_phase == 1:
            return self._spawn_agents_wez_close()
        if self._curriculum_phase == 2:
            return self._spawn_agents_dynamic_dist()
        return self._spawn_agents_normal()

    def _spawn_agents_wez_close(self) -> dict:
        """
        Faz 1: 1v1, WEZ içi yakın spawn.

        Blue rastgele konumda spawn edilir.
        Red, Blue'nun ±25° önünde _phase1_dist_min–_phase1_dist_max mesafede
        Blue'ya doğru bakacak şekilde spawn edilir.
        Her iki ajan da birbirinin WEZ'inde başlar.
        """
        states = {}

        # Blue spawn
        x_b   = float(self.rng.uniform(-self.spawn_x_range * 0.4,
                                        self.spawn_x_range * 0.4))
        y_b   = float(self.rng.uniform(-self.spawn_y_range * 0.4,
                                        self.spawn_y_range * 0.4))
        h_b   = float(self.rng.uniform(self.spawn_h_min, self.spawn_h_max))
        V_b   = float(self.rng.uniform(self.spawn_V_min, self.spawn_V_max))
        psi_init = float(self.rng.uniform(-np.pi, np.pi))

        # Red: Blue'nun önünde WEZ konisi içinde
        dist     = float(self.rng.uniform(self._phase1_dist_min,
                                           self._phase1_dist_max))
        ang_off  = float(self.rng.uniform(-deg2rad(25.0), deg2rad(25.0)))
        ang_to_r = psi_init + ang_off

        # ENU: x=Doğu, y=Kuzey  →  sin(psi)=Doğu bileşeni, cos(psi)=Kuzey bileşeni
        x_r  = x_b + dist * np.sin(ang_to_r)
        y_r  = y_b + dist * np.cos(ang_to_r)
        h_r  = float(np.clip(h_b + self.rng.uniform(-300.0, 300.0),
                              self.spawn_h_min, self.spawn_h_max))
        V_r  = float(self.rng.uniform(self.spawn_V_min, self.spawn_V_max))

        # Heading hizalama: her ajan karşısındakine ±45° içinde yönelsin
        ang_to_red  = float(np.arctan2(y_r - y_b, x_r - x_b))
        ang_to_blue = float(np.arctan2(y_b - y_r, x_b - x_r))
        psi_b = ang_to_red  + float(self.rng.uniform(-np.pi / 4, np.pi / 4))
        psi_r = ang_to_blue + float(self.rng.uniform(-np.pi / 4, np.pi / 4))

        for aid, x, y, h, V, psi in [
            (self.blue_ids[0], x_b, y_b, h_b, V_b, psi_b),
            (self.red_ids[0],  x_r, y_r, h_r, V_r, psi_r),
        ]:
            trim = self._trim_solver.lookup(V, h, self._trim_table)
            if not trim.success:
                trim = self._trim_solver.solve(V, h)
            init = {"x": x, "y": y, "h": h, "V": V,
                    "psi": psi, "alpha": trim.alpha}
            s = self._aircraft.reset(init)
            trim_action = np.zeros(5, dtype=np.float32)
            trim_action[1] = float(trim.de)
            trim_action[3] = float(trim.dt)
            s = self._aircraft.step(s, trim_action, self.dt)
            states[aid] = s

        return states

    def _spawn_agents_dynamic_dist(self) -> dict:
        """
        Faz 1.5: 1v1, dinamik spawn mesafesi.

        _dynamic_spawn_dist CurriculumManager tarafından dışarıdan güncellenir.
        Yapı WEZ-close ile aynı: Blue rastgele, Red Blue'nun önünde dist mesafede.
        """
        states = {}

        x_b   = float(self.rng.uniform(-self.spawn_x_range * 0.4,
                                        self.spawn_x_range * 0.4))
        y_b   = float(self.rng.uniform(-self.spawn_y_range * 0.4,
                                        self.spawn_y_range * 0.4))
        h_b   = float(self.rng.uniform(self.spawn_h_min, self.spawn_h_max))
        V_b   = float(self.rng.uniform(self.spawn_V_min, self.spawn_V_max))
        psi_init = float(self.rng.uniform(-np.pi, np.pi))

        # Mesafe: _dynamic_spawn_dist ± %20 rastgele
        dist    = float(self._dynamic_spawn_dist *
                        self.rng.uniform(0.8, 1.2))
        ang_off = float(self.rng.uniform(-deg2rad(30.0), deg2rad(30.0)))
        ang_to_r = psi_init + ang_off

        x_r  = x_b + dist * np.sin(ang_to_r)
        y_r  = y_b + dist * np.cos(ang_to_r)
        h_r  = float(np.clip(h_b + self.rng.uniform(-500.0, 500.0),
                              self.spawn_h_min, self.spawn_h_max))
        V_r  = float(self.rng.uniform(self.spawn_V_min, self.spawn_V_max))

        # Heading hizalama: her ajan karşısındakine ±45° içinde yönelsin
        ang_to_red  = float(np.arctan2(y_r - y_b, x_r - x_b))
        ang_to_blue = float(np.arctan2(y_b - y_r, x_b - x_r))
        psi_b = ang_to_red  + float(self.rng.uniform(-np.pi / 4, np.pi / 4))
        psi_r = ang_to_blue + float(self.rng.uniform(-np.pi / 4, np.pi / 4))

        for aid, x, y, h, V, psi in [
            (self.blue_ids[0], x_b, y_b, h_b, V_b, psi_b),
            (self.red_ids[0],  x_r, y_r, h_r, V_r, psi_r),
        ]:
            trim = self._trim_solver.lookup(V, h, self._trim_table)
            if not trim.success:
                trim = self._trim_solver.solve(V, h)
            init = {"x": x, "y": y, "h": h, "V": V,
                    "psi": psi, "alpha": trim.alpha}
            s = self._aircraft.reset(init)
            trim_action = np.zeros(5, dtype=np.float32)
            trim_action[1] = float(trim.de)
            trim_action[3] = float(trim.dt)
            s = self._aircraft.step(s, trim_action, self.dt)
            states[aid] = s

        return states

    def _spawn_agents_normal(self) -> dict:
        """
        Faz 2/3: Normal spawn — mesafe sınırlı, heading hizalamalı.

        Blue rastgele konumda spawn edilir.
        Red, Blue'dan _normal_spawn_dist ± %20 mesafede rastgele yönde spawn edilir.
        Her iki ajan heading'i karşılıklı olarak ±45° içinde hizalanır.

        Faz 4 (2v2): Her takımdan 2 ajan, Blue etrafında ±500m x-offset ile spawn edilir.
        """
        states = {}

        # 1v1: Blue_0 ve Red_0
        x_b = float(self.rng.uniform(-self.spawn_x_range * 0.4,
                                      self.spawn_x_range * 0.4))
        y_b = float(self.rng.uniform(-self.spawn_y_range * 0.4,
                                      self.spawn_y_range * 0.4))
        h_b = float(self.rng.uniform(self.spawn_h_min, self.spawn_h_max))
        V_b = float(self.rng.uniform(self.spawn_V_min, self.spawn_V_max))

        # Red'in konumunu Blue'dan 6000-12000m arası rastgele mesafede belirle
        dist    = float(self.rng.uniform(self._normal_spawn_dist,
                                         self._normal_spawn_dist_max))
        # Rastgele yön açısı (tüm 360°)
        ang_br  = float(self.rng.uniform(-np.pi, np.pi))
        x_r     = x_b + dist * np.cos(ang_br)   # math convention: cos=East
        y_r     = y_b + dist * np.sin(ang_br)    # math convention: sin=North

        # Harita sınırı kontrolü — Red dışarı çıkmasın
        half = self.map_size * 0.4
        x_r  = float(np.clip(x_r, -half, half))
        y_r  = float(np.clip(y_r, -half, half))

        h_r = float(np.clip(h_b + self.rng.uniform(-500.0, 500.0),
                             self.spawn_h_min, self.spawn_h_max))
        V_r = float(self.rng.uniform(self.spawn_V_min, self.spawn_V_max))

        # Heading hizalama: her ajan karşısındakine ±45° içinde yönelsin
        # math convention: psi = atan2(dy, dx); 0=East, CCW
        ang_to_red  = float(np.arctan2(y_r - y_b, x_r - x_b))
        ang_to_blue = float(np.arctan2(y_b - y_r, x_b - x_r))
        psi_b = ang_to_red  + float(self.rng.uniform(-np.pi / 4, np.pi / 4))
        psi_r = ang_to_blue + float(self.rng.uniform(-np.pi / 4, np.pi / 4))

        spawn_list = [(self.blue_ids[0], x_b, y_b, h_b, V_b, psi_b),
                      (self.red_ids[0],  x_r, y_r, h_r, V_r, psi_r)]

        # 2v2 (Faz 4): wingman'lar ±1000m x-offset ile aynı mesafede
        if self.n_per_team == 2:
            x_b2 = x_b + float(self.rng.uniform(-1000.0, 1000.0))
            y_b2 = y_b + float(self.rng.uniform(-500.0,   500.0))
            h_b2 = float(np.clip(h_b + self.rng.uniform(-300.0, 300.0),
                                  self.spawn_h_min, self.spawn_h_max))
            V_b2 = float(self.rng.uniform(self.spawn_V_min, self.spawn_V_max))
            ang_b2 = float(np.arctan2(y_r - y_b2, x_r - x_b2))
            psi_b2 = ang_b2 + float(self.rng.uniform(-np.pi / 4, np.pi / 4))

            x_r2 = x_r + float(self.rng.uniform(-1000.0, 1000.0))
            y_r2 = y_r + float(self.rng.uniform(-500.0,   500.0))
            h_r2 = float(np.clip(h_r + self.rng.uniform(-300.0, 300.0),
                                  self.spawn_h_min, self.spawn_h_max))
            V_r2 = float(self.rng.uniform(self.spawn_V_min, self.spawn_V_max))
            ang_r2 = float(np.arctan2(y_b - y_r2, x_b - x_r2))
            psi_r2 = ang_r2 + float(self.rng.uniform(-np.pi / 4, np.pi / 4))

            spawn_list += [
                (self.blue_ids[1], x_b2, y_b2, h_b2, V_b2, psi_b2),
                (self.red_ids[1],  x_r2, y_r2, h_r2, V_r2, psi_r2),
            ]

        for aid, x, y, h, V, psi in spawn_list:
            trim = self._trim_solver.lookup(V, h, self._trim_table)
            if not trim.success:
                trim = self._trim_solver.solve(V, h)
            init = {"x": x, "y": y, "h": h, "V": V,
                    "psi": psi, "alpha": trim.alpha}
            s = self._aircraft.reset(init)
            trim_action = np.zeros(5, dtype=np.float32)
            trim_action[1] = float(trim.de)
            trim_action[3] = float(trim.dt)
            s = self._aircraft.step(s, trim_action, self.dt)
            states[aid] = s

        return states

    # -----------------------------------------------------------------------
    # Observation Oluşturma
    # -----------------------------------------------------------------------

    def _build_obs_dict(self, gat_messages: dict = None) -> dict:
        """
        Her ajan için normalize observation vektörü döndürür.

        obs_dim her zaman max topoloji (2v2) boyutundadır.
        1v1 fazlarda eksik teammate/enemy slotları dummy (ölü) state ile doldurulur
        → normalizer zeros döndürür → ağ boyutu sabit kalır (50D).
        """
        obs_dict = {}
        max_n_tm = self._max_n_per_team - 1  # 1
        max_n_en = self._max_n_per_team       # 2

        for aid in self.agent_ids:
            state      = self._states[aid]
            teammates  = self._get_teammates(aid)
            enemies    = self._get_enemies(aid)
            agg        = self._aggression.get(aid)
            gat_msgs   = gat_messages.get(aid) if gat_messages else None

            # Eksik slotları dummy dead-state ile doldur
            while len(teammates) < max_n_tm:
                teammates.append(self._dummy_state)
            while len(enemies) < max_n_en:
                enemies.append(self._dummy_state)

            # Cooldown normalize: kalan süre / max cooldown → [0,1]
            wep            = self._weapons[aid]
            cooldown_norm  = float(np.clip(
                wep._cooldown_timer / max(wep.fire_cooldown, 1e-9), 0.0, 1.0
            ))

            obs_dict[aid] = self._normalizer.build_obs(
                agent_state     = state,
                teammate_states = teammates,
                enemy_states    = enemies,
                aggression      = agg,
                gat_messages    = gat_msgs,
                cooldown_norm   = cooldown_norm,
            )
        return obs_dict

    # -----------------------------------------------------------------------
    # CRITICAL Recovery — Blue MAPPO Agent OOB Koruması
    # -----------------------------------------------------------------------

    def _critical_recovery_blue(self, s: np.ndarray):
        """
        Blue MAPPO agent için zemin ve harita sınırı kurtarma override.

        Heuristic agent'taki CRITICAL mantığını MAPPO'ya uygular.
        None döndürürse tehlike yok — ajan kendi aksiyonunu kullanır.
        Aksiyon uzayı heuristic ile aynı: [da, de, dr, dt, fire] ∈ [-1,1]/[0,1].
        """
        action = np.zeros(ACTION_DIM, dtype=np.float32)

        # Zemin yaklaşımı — burun kaldır, tam gaz
        if s[STATE_H] < _CRIT_H_FLOOR:
            pull = float(np.clip(
                (_CRIT_H_FLOOR - s[STATE_H]) / _CRIT_H_FLOOR, 0.3, 1.0
            ))
            action[ACTION_DE] = pull          # burun yukarı
            action[ACTION_DT] = 1.0           # tam gaz
            action[ACTION_DA] = float(-s[STATE_PHI] * 0.5)  # kanat düzelt
            return action

        # Harita sınırı — merkeze döndürme manevrası
        half = self.map_size / 2.0
        if (abs(s[STATE_X]) > half - _CRIT_MAP_MARGIN or
                abs(s[STATE_Y]) > half - _CRIT_MAP_MARGIN):
            own_pos    = np.array([s[STATE_X], s[STATE_Y], s[STATE_H]])
            center_pos = np.array([0.0, 0.0, s[STATE_H]])
            bear_to_center = bearing_angle(own_pos, center_pos)
            bear_err       = wrap_to_pi(bear_to_center - s[STATE_PSI])
            action[ACTION_DA] = float(np.clip(bear_err * 1.5, -1.0, 1.0))
            action[ACTION_DR] = float(np.clip(bear_err * 0.5, -1.0, 1.0))  # koordineli dönüş
            action[ACTION_DE] = 0.2    # hafif çekiş
            action[ACTION_DT] = 1.0   # tam gaz — throttle_reward'dan bağımsız override
            return action

        return None

    # -----------------------------------------------------------------------
    # Done Kontrolü
    # -----------------------------------------------------------------------

    def _check_done(self) -> dict:
        """
        Episode bitiş koşullarını kontrol eder.

        Koşullar:
            1. Blue takımındaki tüm ajanlar ölü → Red kazandı
            2. Red  takımındaki tüm ajanlar ölü → Blue kazandı
            3. max_steps aşıldı → Beraberlik
        """
        blue_alive = any(
            self._states[aid][STATE_ALIVE] > 0.5 for aid in self.blue_ids
        )
        red_alive  = any(
            self._states[aid][STATE_ALIVE] > 0.5 for aid in self.red_ids
        )

        time_up = self._step_count >= self.max_steps

        episode_over = (not blue_alive) or (not red_alive) or time_up

        done_dict = {aid: episode_over for aid in self.agent_ids}
        done_dict["__all__"] = episode_over

        # Sonuç bilgisi (info'ya episode summary'de eklenir)
        if not blue_alive and red_alive:
            done_dict["winner"] = RED
        elif blue_alive and not red_alive:
            done_dict["winner"] = BLUE
        else:
            done_dict["winner"] = "draw"

        return done_dict

    # -----------------------------------------------------------------------
    # Episode Özeti
    # -----------------------------------------------------------------------

    def _build_episode_summary(self, agent_id: str) -> dict:
        """
        Episode sonunda ajan başına özet istatistik döndürür.
        logger.py → WandB/TensorBoard'a gönderilir.
        """
        reward_summary = RewardModel.summarize(self._info_history[agent_id])

        # Kill sayısı
        kills = sum(
            1 for info in self._info_history[agent_id]
            if info.get("r_kill", 0.0) > 0.5
        )

        return {
            **reward_summary,
            "episode/total_reward":  self._episode_rewards[agent_id],
            "episode/kills":         kills,
            "episode/steps":         self._step_count,
            "episode/survived":      float(self._states[agent_id][STATE_ALIVE]),
            "episode/fuel_remaining":float(self._states[agent_id][STATE_FUEL]),
            "episode/ammo_remaining":float(self._states[agent_id][STATE_AMMO]),
            "episode/hp_final":      float(self._states[agent_id][STATE_HP]),
        }

    # -----------------------------------------------------------------------
    # Yardımcı Metodlar
    # -----------------------------------------------------------------------

    def _get_teammates(self, agent_id: str) -> list:
        """Ajan ile aynı takımdaki diğer ajanların state listesi."""
        team = self._team_of[agent_id]
        ids  = self.blue_ids if team == BLUE else self.red_ids
        return [self._states[aid] for aid in ids if aid != agent_id]

    def _get_enemies(self, agent_id: str) -> list:
        """Ajan ile farklı takımdaki ajanların state listesi."""
        team = self._team_of[agent_id]
        ids  = self.red_ids if team == BLUE else self.blue_ids
        return [self._states[aid] for aid in ids]

    def _get_enemy_ids(self, agent_id: str) -> list:
        """Düşman ajan ID'leri."""
        team = self._team_of[agent_id]
        return self.red_ids if team == BLUE else self.blue_ids

    def _closest_alive(self, agent_state: np.ndarray,
                        enemy_states: list) -> np.ndarray | None:
        """Hayatta düşmanlar arasında en yakın olanın state'ini döndürür."""
        best_dist = np.inf
        best_state = None
        ax, ay, ah = (agent_state[STATE_X],
                      agent_state[STATE_Y],
                      agent_state[STATE_H])
        for es in enemy_states:
            if es[STATE_ALIVE] < 0.5:
                continue
            dist = float(np.sqrt(
                (es[STATE_X] - ax) ** 2 +
                (es[STATE_Y] - ay) ** 2 +
                (es[STATE_H] - ah) ** 2
            ))
            if dist < best_dist:
                best_dist  = dist
                best_state = es
        return best_state

    def _find_state_owner(self, target_state: np.ndarray,
                           candidates: list) -> str | None:
        """
        Verilen state numpy dizisine pointer eşitliği ile sahip ajan ID'sini bulur.
        _get_enemies'den dönen liste self._states referanslarını içerir.
        """
        for aid, s in self._states.items():
            if s is target_state:
                return aid
        return None

    # -----------------------------------------------------------------------
    # Gözlem ve State Erişimi (eval / replay için)
    # -----------------------------------------------------------------------

    def get_state(self, agent_id: str) -> np.ndarray:
        """Ham (normalize edilmemiş) state vektörü."""
        return self._states[agent_id].copy()

    def get_all_states(self) -> dict:
        """Tüm ajanların ham state sözlüğü."""
        return {aid: s.copy() for aid, s in self._states.items()}

    def get_step_count(self) -> int:
        return self._step_count

    def is_done(self) -> bool:
        return self._done

    @property
    def blue_alive(self) -> list:
        return [aid for aid in self.blue_ids
                if self._states[aid][STATE_ALIVE] > 0.5]

    @property
    def red_alive(self) -> list:
        return [aid for aid in self.red_ids
                if self._states[aid][STATE_ALIVE] > 0.5]

    # -----------------------------------------------------------------------
    # Config yükleyici (kolaylık)
    # -----------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> "DogfightEnv":
        """YAML dosyasından ortam oluşturur."""
        with open(path) as f:
            config = yaml.safe_load(f)
        return cls(config)

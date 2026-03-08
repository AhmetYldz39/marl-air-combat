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
from copy import deepcopy

from envs.aircraft_model import (
    AircraftModel,
    STATE_X, STATE_Y, STATE_H, STATE_V,
    STATE_ALPHA, STATE_PSI, STATE_FUEL,
    STATE_AMMO, STATE_HP, STATE_ALIVE,
    STATE_DIM, ACTION_DIM,
)
from envs.weapons_model  import WeaponsModel
from envs.reward_model   import RewardModel
from utils.normalization  import Normalizer
from envs.geometry_utils import deg2rad


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
        self.n_per_team         = int(env_cfg["n_agents_per_team"])

        # Spawn aralıkları
        self.spawn_x_range      = float(env_cfg.get("spawn_x_range",   10000.0))
        self.spawn_y_range      = float(env_cfg.get("spawn_y_range",   10000.0))
        self.spawn_h_min        = float(env_cfg.get("spawn_h_min",      3000.0))
        self.spawn_h_max        = float(env_cfg.get("spawn_h_max",      8000.0))
        self.spawn_V_min        = float(env_cfg.get("spawn_V_min",       150.0))
        self.spawn_V_max        = float(env_cfg.get("spawn_V_max",       280.0))
        self.spawn_team_offset  = float(env_cfg.get("spawn_team_offset", 8000.0))

        # ── Ajan ID'leri ──────────────────────────────────────────────
        self.blue_ids = [f"blue_{i}" for i in range(self.n_per_team)]
        self.red_ids  = [f"red_{i}"  for i in range(self.n_per_team)]
        self.agent_ids = self.blue_ids + self.red_ids
        self.n_agents  = len(self.agent_ids)

        # Takım üyeliği: ajan_id → takım adı
        self._team_of = {aid: BLUE for aid in self.blue_ids}
        self._team_of.update({aid: RED for aid in self.red_ids})

        # ── Alt modüller ──────────────────────────────────────────────
        self._aircraft  = AircraftModel(config)
        self._normalizer = Normalizer(config)

        # Her ajan için ayrı WeaponsModel ve RewardModel
        self._weapons = {aid: WeaponsModel(config) for aid in self.agent_ids}
        self._rewards = {aid: RewardModel(config)  for aid in self.agent_ids}

        # ── Boyutlar ──────────────────────────────────────────────────
        n_tm = self.n_per_team - 1   # takım arkadaşı sayısı
        n_en = self.n_per_team       # düşman sayısı
        self.obs_dim    = self._normalizer.obs_dim(n_tm, n_en)
        self.action_dim = ACTION_DIM

        # ── Episode state ─────────────────────────────────────────────
        self._states: dict[str, np.ndarray] = {}
        self._step_count  = 0
        self._done        = False
        self._episode_rewards: dict[str, float] = {}
        self._info_history:    dict[str, list]  = {}

        # RNG (seed dışarıdan set edilebilir)
        self.rng = np.random.default_rng(seed=None)

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

        return self._build_obs_dict()

    # -----------------------------------------------------------------------
    # Step
    # -----------------------------------------------------------------------

    def step(self, action_dict: dict,
             gat_messages: dict = None) -> tuple:
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

            action = self._normalizer.normalize_action(
                action_dict.get(aid, np.zeros(ACTION_DIM, dtype=np.float32))
            )

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
            else:
                fire_result = self._weapons[aid]._no_fire_result(
                    self._weapons[aid]._empty_wez(), "no_target"
                )
            fire_results[aid] = fire_result

        # Cooldown tick (ateş komutundan bağımsız olarak her adımda)
        for aid in self.agent_ids:
            self._weapons[aid].tick(self.dt)

        self._step_count += 1

        # ── 2. Reward hesabı ──────────────────────────────────────────
        rew_dict  = {}
        info_dict = {}

        for aid in self.agent_ids:
            state      = self._states[aid]
            teammates  = self._get_teammates(aid)
            enemies    = self._get_enemies(aid)
            agg        = self._aggression.get(aid)

            total_rew, rew_info = self._rewards[aid].compute(
                agent_state     = state,
                teammate_states = teammates,
                enemy_states    = enemies,
                weapons_model   = self._weapons[aid],
                prev_state      = prev_states[aid],
                fire_result     = fire_results[aid],
                dt              = self.dt,
                map_size        = self.map_size,
                aggression      = agg,
            )

            rew_dict[aid]  = float(total_rew)
            info_dict[aid] = rew_info
            self._episode_rewards[aid] += float(total_rew)
            self._info_history[aid].append(rew_info)

        # ── 3. Done kontrolü ─────────────────────────────────────────
        done_dict = self._check_done()
        if done_dict["__all__"]:
            self._done = True
            # Episode özeti her ajana ekle
            for aid in self.agent_ids:
                info_dict[aid]["episode"] = self._build_episode_summary(aid)

        # ── 4. Observation ───────────────────────────────────────────
        obs_dict = self._build_obs_dict(gat_messages=gat_messages)

        return obs_dict, rew_dict, done_dict, info_dict

    # -----------------------------------------------------------------------
    # Spawn
    # -----------------------------------------------------------------------

    def _spawn_agents(self) -> dict:
        """
        İki takımı karşılıklı rastgele spawn eder.

        Blue takımı: +y offset (Kuzey)
        Red  takımı: -y offset (Güney)
        Her takım içinde x'te rastgele dağılım.
        """
        states = {}
        half_offset = self.spawn_team_offset / 2.0

        for team, ids, y_sign in [(BLUE, self.blue_ids, +1.0),
                                   (RED,  self.red_ids,  -1.0)]:
            for i, aid in enumerate(ids):
                x = float(self.rng.uniform(-self.spawn_x_range,
                                            self.spawn_x_range))
                y = float(y_sign * half_offset +
                           self.rng.uniform(-self.spawn_y_range * 0.3,
                                             self.spawn_y_range * 0.3))
                h = float(self.rng.uniform(self.spawn_h_min, self.spawn_h_max))
                V = float(self.rng.uniform(self.spawn_V_min, self.spawn_V_max))

                # Heading: Blue → Güney'e (Red'e doğru), Red → Kuzey'e
                psi = np.pi if team == BLUE else 0.0
                psi += float(self.rng.uniform(-0.3, 0.3))  # küçük randomizasyon

                init = {
                    "x": x, "y": y, "h": h, "V": V,
                    "psi": psi, "alpha": deg2rad(3.0),
                }
                s = self._aircraft.reset(init)
                states[aid] = s

        return states

    # -----------------------------------------------------------------------
    # Observation Oluşturma
    # -----------------------------------------------------------------------

    def _build_obs_dict(self, gat_messages: dict = None) -> dict:
        """Her ajan için normalize observation vektörü döndürür."""
        obs_dict = {}
        for aid in self.agent_ids:
            state      = self._states[aid]
            teammates  = self._get_teammates(aid)
            enemies    = self._get_enemies(aid)
            agg        = self._aggression.get(aid)
            gat_msgs   = gat_messages.get(aid) if gat_messages else None

            obs_dict[aid] = self._normalizer.build_obs(
                agent_state     = state,
                teammate_states = teammates,
                enemy_states    = enemies,
                aggression      = agg,
                gat_messages    = gat_msgs,
            )
        return obs_dict

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

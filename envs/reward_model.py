"""
reward_model.py
===============
Ağırlıklı reward hesaplama modeli.

Reward bileşenleri:
    r_kill          : Düşman imhası (+10 anlık)
    r_wez_advantage : WEZ avantajı (sürekli, Gaussian tabanlı)
    r_tracking      : Radar izleme (cos(ATA))
    r_survival      : Hayatta kalma (her adım küçük pozitif)
    r_team_coord    : Takım koordinasyonu (mesafe Gaussian)
    r_resource      : Kaynak verimliliği (yakıt/ammo israfı cezası)
    r_penalty       : Tehlikeli davranış (sınır dışı, stall, çarpışma)
    r_role_bonus    : Aggression skalasına uyumlu davranış bonusu

Rol sistemi — Sürekli Aggression Skalası:
    aggression ∈ [0.0, 1.0]
        0.0 = tam defansif  (hayatta kal, takımı koru)
        0.5 = dengeli       (varsayılan, Faz 2 başlangıcı)
        1.0 = tam agresif   (kill odaklı, risk al)

    Ağırlık ölçekleri lineer interpolasyonla (lerp) hesaplanır:
        w_kill_scale     = lerp(0.5 → 2.0, aggression)
        w_survival_scale = lerp(2.0 → 0.5, aggression)
        w_coord_scale    = lerp(1.5 → 1.0, aggression)

    aggression | w_kill | w_survival | w_coord
    -----------|--------|-----------|--------
         0.0   |   0.5  |    2.0    |   1.5
         0.25  |  0.875 |   1.625   |  1.375
         0.5   |  1.25  |   1.25    |  1.25
         0.75  |  1.625 |   0.875   |  1.125
         1.0   |   2.0  |    0.5    |   1.0

    Faz 0-1: aggression=None → tüm scale=1.0 (pasif mod)
    Faz 2+ : aggression=0.5 ile başla, curriculum/runtime'da değiştir

    Observation embedding (Faz 2):
        role_embedding = [aggression, 1.0 - aggression]  ← 2 boyut

Tüm ağırlıklar ve lerp uç noktaları config.yaml'dan okunur.

Bağımlılıklar:
    - numpy
    - weapons_model.py  (wez_advantage_score, antenna_train_cos)
    - aircraft_model.py (STATE_* sabitleri)
    - geometry_utils.py (distance_3d)

Bu dosya değişirse etkilenen dosyalar:
    - dogfight_env.py        (her step'te reward hesabı)
    - test_reward_model.py
    - normalization.py       (role_embedding boyutu = 2)
"""

import numpy as np
from envs.aircraft_model import (
    STATE_X, STATE_Y, STATE_H,
    STATE_V, STATE_PSI, STATE_ALIVE,
    STATE_AMMO, STATE_HP, STATE_FUEL,
    STATE_ALPHA,
)
from envs.geometry_utils import distance_3d, deg2rad
from envs.weapons_model import WeaponsModel

# Aggression sınırları
AGGRESSION_MIN     = 0.0
AGGRESSION_MAX     = 1.0
AGGRESSION_DEFAULT = 0.5   # Faz 2 başlangıcı


def _lerp(a: float, b: float, t: float) -> float:
    """Lineer interpolasyon. t=[0,1], t=0→a, t=1→b. Otomatik clip."""
    return float(a + (b - a) * float(np.clip(t, 0.0, 1.0)))


def aggression_to_embedding(aggression: float) -> np.ndarray:
    """
    Aggression skalarını 2 boyutlu observation embedding'e çevirir.
    Faz 2'de obs vektörüne eklenir.

        0.0 → [0.0, 1.0]  tam defansif
        0.5 → [0.5, 0.5]  dengeli
        1.0 → [1.0, 0.0]  tam agresif
    """
    a = float(np.clip(aggression, 0.0, 1.0))
    return np.array([a, 1.0 - a], dtype=np.float32)


class RewardModel:
    """
    Tek bir ajan için her adımda reward hesaplar.
    Her ajan için ayrı örnek oluşturulur.
    dogfight_env.step() içinde çağrılır.
    """

    def __init__(self, config: dict):
        r     = config["reward"]
        roles = config.get("roles", {})
        ac    = config["aircraft"]
        coord = config.get("coord", {})

        # ── Temel ağırlıklar ──────────────────────────────────────────
        self.w_kill     = float(r["w_kill"])
        self.w_wez      = float(r["w_wez"])
        self.w_tracking = float(r["w_tracking"])
        self.w_survival = float(r["w_survival"])
        self.w_coord    = float(r["w_coord"])
        self.w_resource = float(r["w_resource"])
        self.w_penalty  = float(r["w_penalty"])
        self.w_role     = float(r.get("w_role", 1.0))

        # ── Aggression lerp uç noktaları ──────────────────────────────
        # Ağırlık ölçekleri: aggression=0 (defansif) → aggression=1 (agresif)
        self._kill_at_0     = float(roles.get("w_kill_at_0",     0.5))
        self._kill_at_1     = float(roles.get("w_kill_at_1",     2.0))
        self._survival_at_0 = float(roles.get("w_survival_at_0", 2.0))
        self._survival_at_1 = float(roles.get("w_survival_at_1", 0.5))
        self._coord_at_0    = float(roles.get("w_coord_at_0",    1.5))
        self._coord_at_1    = float(roles.get("w_coord_at_1",    1.0))

        # r_role_bonus bileşen ağırlıkları lerp uç noktaları
        self._role_kill_at_0     = float(roles.get("role_kill_weight_at_0",     0.0))
        self._role_kill_at_1     = float(roles.get("role_kill_weight_at_1",     1.0))
        self._role_survival_at_0 = float(roles.get("role_survival_weight_at_0", 0.5))
        self._role_survival_at_1 = float(roles.get("role_survival_weight_at_1", 0.0))
        self._role_coord_at_0    = float(roles.get("role_coord_weight_at_0",    0.5))
        self._role_coord_at_1    = float(roles.get("role_coord_weight_at_1",    0.0))

        # ── Uçak referans parametreleri ───────────────────────────────
        self._alpha_max = float(ac.get("alpha_max", deg2rad(25.0)))
        self._V_min     = float(ac.get("V_min", 60.0))
        self._h_min     = float(ac.get("h_min", 50.0))
        self._init_fuel = float(ac["initial_fuel"])

        # ── Koordinasyon mesafe parametreleri ─────────────────────────
        self._coord_dist_min = float(coord.get("dist_min", 500.0))
        self._coord_dist_max = float(coord.get("dist_max", 8000.0))
        self._coord_dist_opt = float(coord.get("dist_opt", 3000.0))

    # -----------------------------------------------------------------------
    # Aggression → Ölçekler
    # -----------------------------------------------------------------------

    def _compute_scales(self, aggression) -> dict:
        """
        aggression=None  → pasif (scale=1.0)
        aggression=float → lerp ile ölçekleme
        """
        if aggression is None:
            return {"w_kill_scale": 1.0, "w_survival_scale": 1.0, "w_coord_scale": 1.0}
        a = float(np.clip(aggression, 0.0, 1.0))
        return {
            "w_kill_scale":     _lerp(self._kill_at_0,     self._kill_at_1,     a),
            "w_survival_scale": _lerp(self._survival_at_0, self._survival_at_1, a),
            "w_coord_scale":    _lerp(self._coord_at_0,    self._coord_at_1,    a),
        }

    # -----------------------------------------------------------------------
    # Ana Hesaplama
    # -----------------------------------------------------------------------

    def compute(self,
                agent_state:     np.ndarray,
                teammate_states: list,
                enemy_states:    list,
                weapons_model:   WeaponsModel,
                prev_state:      np.ndarray,
                fire_result:     dict,
                dt:              float,
                map_size:        float,
                aggression:      float = None) -> tuple:
        """
        Bir adım için toplam reward hesaplar.

        Parameters
        ----------
        aggression : float | None
            None    → pasif (Faz 0-1), scale=1.0
            0.0–1.0 → sürekli skala (Faz 2+)
                      0.0=defansif, 0.5=dengeli, 1.0=agresif
        """
        if aggression is not None:
            assert AGGRESSION_MIN <= aggression <= AGGRESSION_MAX, (
                f"Geçersiz aggression={aggression}. [{AGGRESSION_MIN},{AGGRESSION_MAX}]"
            )

        if agent_state[STATE_ALIVE] < 0.5:
            return 0.0, self._zero_info(aggression)

        scales = self._compute_scales(aggression)

        # Bileşenler
        r_kill     = self._kill_reward(fire_result)
        r_wez      = self._wez_reward(agent_state, enemy_states, weapons_model)
        r_tracking = self._tracking_reward(agent_state, enemy_states, weapons_model)
        r_survival = self._survival_reward(dt)
        r_coord    = self._coord_reward(agent_state, teammate_states)
        r_resource = self._resource_reward(agent_state, prev_state, fire_result)
        r_penalty  = self._penalty_reward(agent_state, map_size)
        r_role     = self._role_bonus(r_kill, r_survival, r_coord, aggression)

        # Efektif ağırlıklar
        w_kill_eff     = self.w_kill     * scales["w_kill_scale"]
        w_survival_eff = self.w_survival * scales["w_survival_scale"]
        w_coord_eff    = self.w_coord    * scales["w_coord_scale"]

        total = (
            w_kill_eff      * r_kill     +
            self.w_wez      * r_wez      +
            self.w_tracking * r_tracking +
            w_survival_eff  * r_survival +
            w_coord_eff     * r_coord    +
            self.w_resource * r_resource +
            self.w_penalty  * r_penalty  +
            self.w_role     * r_role
        )

        info = {
            "total": float(total),
            "r_kill": float(r_kill), "r_wez": float(r_wez),
            "r_tracking": float(r_tracking), "r_survival": float(r_survival),
            "r_coord": float(r_coord), "r_resource": float(r_resource),
            "r_penalty": float(r_penalty), "r_role": float(r_role),
            "aggression":       aggression,
            "w_kill_scale":     float(scales["w_kill_scale"]),
            "w_survival_scale": float(scales["w_survival_scale"]),
            "w_coord_scale":    float(scales["w_coord_scale"]),
            "w_kill_contrib":     float(w_kill_eff      * r_kill),
            "w_wez_contrib":      float(self.w_wez      * r_wez),
            "w_tracking_contrib": float(self.w_tracking * r_tracking),
            "w_survival_contrib": float(w_survival_eff  * r_survival),
            "w_coord_contrib":    float(w_coord_eff     * r_coord),
            "w_resource_contrib": float(self.w_resource * r_resource),
            "w_penalty_contrib":  float(self.w_penalty  * r_penalty),
            "w_role_contrib":     float(self.w_role     * r_role),
        }

        return float(total), info

    # -----------------------------------------------------------------------
    # Reward Bileşenleri
    # -----------------------------------------------------------------------

    def _kill_reward(self, fire_result):
        if fire_result is None:
            return 0.0
        return 1.0 if fire_result.get("kill", False) else 0.0

    def _wez_reward(self, agent_state, enemy_states, weapons_model):
        if not enemy_states:
            return 0.0
        best = 0.0
        for es in enemy_states:
            if es[STATE_ALIVE] < 0.5:
                continue
            best = max(best, weapons_model.wez_advantage_score(agent_state, es))
        return float(best)

    def _tracking_reward(self, agent_state, enemy_states, weapons_model):
        if not enemy_states:
            return 0.0
        best = -1.0
        for es in enemy_states:
            if es[STATE_ALIVE] < 0.5:
                continue
            best = max(best, weapons_model.antenna_train_cos(agent_state, es))
        return float(best) if best > -1.0 else 0.0

    def _survival_reward(self, dt):
        return float(dt)

    def _coord_reward(self, agent_state, teammate_states):
        if not teammate_states:
            return 0.0
        agent_pos = agent_state[[STATE_X, STATE_Y, STATE_H]]
        total, n = 0.0, 0
        for tm in teammate_states:
            if tm[STATE_ALIVE] < 0.5:
                continue
            n += 1
            dist  = distance_3d(agent_pos, tm[[STATE_X, STATE_Y, STATE_H]])
            sigma = (self._coord_dist_max - self._coord_dist_min) * 0.3
            total += float(np.exp(
                -((dist - self._coord_dist_opt) ** 2) / (2 * sigma ** 2)
            ))
        return float(total / n) if n > 0 else 0.0

    def _resource_reward(self, agent_state, prev_state, fire_result):
        penalty = 0.0
        if fire_result is not None:
            if fire_result.get("fired", False) and not fire_result.get("hit", False):
                penalty -= 1.0
        fuel_burn = float(prev_state[STATE_FUEL]) - float(agent_state[STATE_FUEL])
        if fuel_burn > 0:
            excess = max(0.0, fuel_burn - self._init_fuel * 0.005)
            penalty -= excess / (self._init_fuel + 1e-9)
        return float(penalty)

    def _penalty_reward(self, agent_state, map_size):
        penalty = 0.0
        half = map_size / 2.0
        if abs(agent_state[STATE_X]) > half or abs(agent_state[STATE_Y]) > half:
            penalty += 1.0
        if agent_state[STATE_H] < self._h_min + 100.0:
            penalty += 1.0
        elif agent_state[STATE_H] < self._h_min + 300.0:
            penalty += 0.5
        if abs(agent_state[STATE_ALPHA]) > self._alpha_max:
            penalty += 1.0
        if agent_state[STATE_V] < self._V_min * 1.2:
            penalty += 0.3
        return float(penalty)

    def _role_bonus(self, r_kill, r_survival, r_coord, aggression):
        """
        Aggression skalasına uyumlu karma bonus.

        aggression=None → 0.0 (pasif)
        aggression=1.0  → tamamen kill'den
        aggression=0.0  → tamamen survival+coord'dan
        aggression=0.5  → karma
        """
        if aggression is None:
            return 0.0
        a            = float(np.clip(aggression, 0.0, 1.0))
        kill_w       = _lerp(self._role_kill_at_0,     self._role_kill_at_1,     a)
        survival_w   = _lerp(self._role_survival_at_0, self._role_survival_at_1, a)
        coord_w      = _lerp(self._role_coord_at_0,    self._role_coord_at_1,    a)
        survival_norm = float(np.clip(r_survival * 10.0, 0.0, 1.0))
        bonus = kill_w * r_kill + survival_w * survival_norm + coord_w * r_coord
        return float(np.clip(bonus, 0.0, 1.0))

    # -----------------------------------------------------------------------
    # Yardımcılar
    # -----------------------------------------------------------------------

    def _zero_info(self, aggression) -> dict:
        scales = self._compute_scales(aggression)
        keys = ["total", "r_kill", "r_wez", "r_tracking", "r_survival",
                "r_coord", "r_resource", "r_penalty", "r_role",
                "w_kill_contrib", "w_wez_contrib", "w_tracking_contrib",
                "w_survival_contrib", "w_coord_contrib", "w_resource_contrib",
                "w_penalty_contrib", "w_role_contrib"]
        info = {k: 0.0 for k in keys}
        info.update({"aggression": aggression, **scales})
        return info

    @staticmethod
    def aggression_to_embedding(aggression: float) -> np.ndarray:
        """Aggression → [aggression, 1-aggression] embedding (Faz 2)."""
        return aggression_to_embedding(aggression)

    @staticmethod
    def summarize(info_history: list) -> dict:
        """Episode özet istatistikleri → WandB/TensorBoard."""
        if not info_history:
            return {}
        keys = ["total", "r_kill", "r_wez", "r_tracking", "r_survival",
                "r_coord", "r_resource", "r_penalty", "r_role"]
        summary = {}
        for k in keys:
            vals = [info[k] for info in info_history if k in info]
            summary[f"reward/sum_{k}"]  = float(np.sum(vals))
            summary[f"reward/mean_{k}"] = float(np.mean(vals))
        agg_vals = [info["aggression"] for info in info_history
                    if info.get("aggression") is not None]
        if agg_vals:
            summary["reward/mean_aggression"] = float(np.mean(agg_vals))
        return summary
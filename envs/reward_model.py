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
        self.w_pursuit         = float(r.get("w_pursuit", 1.0))
        self.pursuit_norm_dist = float(r.get("pursuit_norm_dist", 20000.0))
        self.w_smooth_ctrl     = float(r.get("w_smooth_ctrl",
                                             r.get("w_smooth", 0.05)))
        self.w_smooth_throttle = float(r.get("w_smooth_throttle", 0.003))
        self.w_throttle_ctx    = float(r.get("w_throttle_ctx", 0.0))
        self.ammo_miss_penalty = float(r.get("ammo_miss_penalty", 2.0))
        self.w_support            = float(r.get("w_support", 1.5))
        self.w_role_match         = float(r.get("w_role_match", 1.0))
        self.w_sniper_position    = float(r.get("w_sniper_position",    1.5))
        self.w_defensive_survival = float(r.get("w_defensive_survival", 5.0))
        self.w_sniper_patience    = float(r.get("w_sniper_patience",    1.0))
        self.w_evasion            = float(r.get("w_evasion",            2.0))

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
                agent_state:        np.ndarray,
                teammate_states:    list,
                enemy_states:       list,
                weapons_model:      WeaponsModel,
                prev_state:         np.ndarray,
                fire_result:        dict,
                dt:                 float,
                map_size:           float,
                aggression:         float = None,
                prev_action:        np.ndarray = None,
                prev_distance:      float = None,
                current_action:     np.ndarray = None,
                role_support_prob:  float = 1.0,
                role_vec:           np.ndarray = None) -> tuple:
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
        r_kill       = self._kill_reward(fire_result)
        r_wez        = self._wez_reward(agent_state, enemy_states, weapons_model)
        r_tracking   = self._tracking_reward(agent_state, enemy_states, weapons_model)
        r_survival   = self._survival_reward(dt)
        r_coord      = self._coord_reward(agent_state, teammate_states)
        r_resource   = self._resource_reward(agent_state, prev_state, fire_result)
        r_penalty    = self._penalty_reward(agent_state, map_size)
        r_role       = self._role_bonus(r_kill, r_survival, r_coord, aggression)
        r_fire_ready    = self._fire_ready_reward(fire_result, weapons_model, enemy_states)
        r_pursuit       = self._pursuit_reward(agent_state, enemy_states)
        r_smooth_ctrl   = self._smooth_ctrl_reward(prev_action)
        r_smooth_thr    = self._smooth_throttle_reward(prev_action)
        r_closing_raw   = self._closing_reward(agent_state, enemy_states, prev_distance)
        r_closing       = self._closing_reward_role(r_closing_raw, role_vec)
        r_throttle_ctx  = self._throttle_reward(agent_state, enemy_states,
                                                 current_action, weapons_model)
        r_support       = self._support_reward(agent_state, teammate_states,
                                               enemy_states, weapons_model,
                                               role_support_prob)
        r_support_raw   = self._support_reward(agent_state, teammate_states,
                                               enemy_states, weapons_model,
                                               role_support_prob=1.0)
        r_role_match    = self._role_match_reward(role_vec, fire_result, agent_state,
                                                  teammate_states, enemy_states,
                                                  weapons_model, prev_distance)
        r_sniper_pos      = self._sniper_position_reward(agent_state, enemy_states)
        r_sniper_patience = self._sniper_patience_reward(fire_result, agent_state,
                                                         enemy_states, weapons_model)
        r_evasion         = self._evasion_reward(agent_state, enemy_states, prev_distance)

        # Role-conditional weights (one-hot soft weights from Gumbel-Softmax)
        _sniper_w = float(role_vec[0]) if role_vec is not None else 0.0
        _def_w    = float(role_vec[2]) if role_vec is not None else 0.0

        # Efektif ağırlıklar
        w_kill_eff     = self.w_kill     * scales["w_kill_scale"]
        w_survival_eff = self.w_survival * scales["w_survival_scale"]
        w_coord_eff    = self.w_coord    * scales["w_coord_scale"]

        total = (
            w_kill_eff              * r_kill          +
            self.w_wez              * r_wez            +
            self.w_tracking         * r_tracking       +
            w_survival_eff          * r_survival       +
            w_coord_eff             * r_coord          +
            self.w_resource         * r_resource       +
            self.w_penalty          * r_penalty        +
            self.w_role             * r_role           +
            r_fire_ready                               +  # sabit ağırlık: 0.5 iç içe
            self.w_pursuit          * r_pursuit        +
            self.w_smooth_ctrl      * r_smooth_ctrl    +
            self.w_smooth_throttle  * r_smooth_thr     +
            self.w_throttle_ctx     * r_throttle_ctx   +
            r_closing                                  +  # PURSUIT-only (*0.0001)
            self.w_support          * r_support        +
            self.w_role_match       * r_role_match     +
            self.w_sniper_position    * _sniper_w * r_sniper_pos      +  # SNIPER: pozisyon
            self.w_sniper_patience    * _sniper_w * r_sniper_patience  +  # SNIPER: sabırlı bekleme
            self.w_defensive_survival * _def_w   * r_survival          +  # DEFENSIVE: 2x survival
            self.w_evasion            * _def_w   * r_evasion              # DEFENSIVE: uzaklaşma
        )

        info = {
            "total": float(total),
            "r_kill": float(r_kill), "r_wez": float(r_wez),
            "r_tracking": float(r_tracking), "r_survival": float(r_survival),
            "r_coord": float(r_coord), "r_resource": float(r_resource),
            "r_penalty": float(r_penalty), "r_role": float(r_role),
            "r_fire_ready": float(r_fire_ready), "r_pursuit": float(r_pursuit),
            "r_smooth_ctrl": float(r_smooth_ctrl),
            "r_smooth_throttle": float(r_smooth_thr),
            "r_throttle_ctx": float(r_throttle_ctx),
            "r_closing": float(r_closing),
            "r_closing_raw": float(r_closing_raw),
            "r_sniper_pos": float(r_sniper_pos),
            "r_sniper_patience": float(r_sniper_patience),
            "r_support": float(r_support),
            "r_support_raw": float(r_support_raw),
            "r_evasion": float(r_evasion),
            "r_role_match": float(r_role_match),
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
            "w_pursuit_contrib":  float(self.w_pursuit  * r_pursuit),
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

    def _fire_ready_reward(self, fire_result, weapons_model, enemy_states) -> float:
        """
        WEZ içinde cooldown=0 iken fire_cmd=True ise +0.5 bonus.
        WEZ içinde cooldown>0 iken fire_cmd=True ise -0.1 ceza
        (cooldown farkında olmadan ateşlemeyi caydır).

        fire_result None ise (ajan ölü) → 0.0
        """
        if fire_result is None:
            return 0.0

        in_wez   = fire_result.get("wez_info", {}).get("in_wez", False)
        fired    = fire_result.get("fired", False)
        reason   = fire_result.get("fail_reason", "")
        fire_cmd = fired or (reason != "no_fire_command" and reason != "")

        if not in_wez:
            return 0.0

        # WEZ içinde, cooldown=0, ateş edildi → bonus
        if fired:
            return 0.5

        # WEZ içinde, cooldown nedeniyle ateş edilemedi → küçük ceza
        if "cooldown" in reason:
            return -0.1

        return 0.0

    def _resource_reward(self, agent_state, prev_state, fire_result):
        penalty = 0.0
        if fire_result is not None:
            fired   = fire_result.get("fired", False)
            in_wez  = fire_result.get("wez_info", {}).get("in_wez", False)
            if fired and not in_wez:
                penalty -= self.ammo_miss_penalty   # WEZ dışı ateş
            elif fired and not fire_result.get("hit", False):
                penalty -= 1.0                      # WEZ içi ıskalama
        fuel_burn = float(prev_state[STATE_FUEL]) - float(agent_state[STATE_FUEL])
        if fuel_burn > 0:
            excess = max(0.0, fuel_burn - self._init_fuel * 0.005)
            penalty -= excess / (self._init_fuel + 1e-9)
        return float(penalty)

    def _smooth_ctrl_reward(self, prev_action: np.ndarray) -> float:
        """Aileron/elevator/rudder ([:3]) jitter cezası. w_smooth_ctrl ile çarpılır."""
        if prev_action is None:
            return 0.0
        return -float(np.sum(prev_action[:3] ** 2))

    def _smooth_throttle_reward(self, prev_action: np.ndarray) -> float:
        """Throttle ([3]) değişim cezası. w_smooth_throttle ile çarpılır (gevşek)."""
        if prev_action is None:
            return 0.0
        return -float(prev_action[3] ** 2)

    def _throttle_reward(self, agent_state: np.ndarray, enemy_states: list,
                         current_action: np.ndarray, weapons_model: WeaponsModel) -> float:
        """
        ATA + mesafe bazlı throttle yönlendirmesi.

        Düşman arkada (ATA cos < -0.3):
            yakın (<3 km) → kes (engage fırsatı, overshoot)
            uzak (>8 km)  → bas (yeniden konumlan)
            arası         → doğrusal hedef
        Düşman önde (ATA cos > 0.5):
            bas (kovala)

        w_throttle_ctx ile çarpılır. Dönen değer [0, 0.15].
        """
        if current_action is None or not enemy_states:
            return 0.0
        agent_pos = agent_state[[STATE_X, STATE_Y, STATE_H]]
        best_cos, best_dist = -2.0, np.inf
        for es in enemy_states:
            if es[STATE_ALIVE] < 0.5:
                continue
            cos_val = weapons_model.antenna_train_cos(agent_state, es)
            d       = distance_3d(agent_pos, es[[STATE_X, STATE_Y, STATE_H]])
            if d < best_dist:
                best_dist = d
                best_cos  = cos_val
        if best_cos < -2.0 or np.isinf(best_dist):
            return 0.0
        throttle = float(np.clip(current_action[3], 0.0, 1.0))
        D_NEAR, D_FAR = 3000.0, 8000.0
        if best_cos < -0.3:             # Düşman arkada
            if best_dist < D_NEAR:
                return (1.0 - throttle) * 0.12   # kes
            elif best_dist > D_FAR:
                return throttle * 0.08            # bas
            else:
                t      = (best_dist - D_NEAR) / (D_FAR - D_NEAR)
                target = t                         # 0=kes, 1=bas
                return (1.0 - abs(throttle - target)) * 0.10
        elif best_cos > 0.5:            # Düşman önde
            return throttle * 0.15
        return 0.0

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

    def _pursuit_reward(self, agent_state, enemy_states) -> float:
        """
        Düşmana yaklaşma teşviki.
        En yakın canlı düşman mesafesine göre [0, 0.3] aralığında reward.
        dist=0m  → 0.3 (tam yakın)
        dist=20000m → 0.0 (harita sınırında)

        2v2'de bir düşman ölünce kalan düşmana r_pursuit 2x çarpanla uygulanır.
        Bu, ikinci kill'i de tamamlamayı teşvik eder.
        """
        if not enemy_states:
            return 0.0
        agent_pos = agent_state[[STATE_X, STATE_Y, STATE_H]]
        best_dist = np.inf
        alive_count = 0
        for es in enemy_states:
            if es[STATE_ALIVE] < 0.5:
                continue
            alive_count += 1
            d = distance_3d(agent_pos, es[[STATE_X, STATE_Y, STATE_H]])
            if d < best_dist:
                best_dist = d
        if np.isinf(best_dist):
            return 0.0
        base = float(max(0.0, 1.0 - best_dist / self.pursuit_norm_dist) * 0.3)
        # Bir düşman ölünce (yalnızca 1 kaldı) 3x çarpan — w_pursuit efektif 3.0
        # İkinci kill'i tamamlamayı güçlü biçimde teşvik eder
        multiplier = 3.0 if alive_count == 1 else 1.0
        return base * multiplier

    def _closing_reward(self, agent_state, enemy_states, prev_distance) -> float:
        """
        Düşmana kapanma hızı ödülü (koşulsuz ham değer).
        r_closing = max(0, d_prev - d_now) * 0.0001
        compute() içinde _closing_reward_role() ile rol filtresi uygulanır.
        """
        if prev_distance is None or not enemy_states:
            return 0.0
        if agent_state[STATE_ALIVE] < 0.5:
            return 0.0
        agent_pos = agent_state[[STATE_X, STATE_Y, STATE_H]]
        best_dist = np.inf
        for es in enemy_states:
            if es[STATE_ALIVE] < 0.5:
                continue
            d = distance_3d(agent_pos, es[[STATE_X, STATE_Y, STATE_H]])
            if d < best_dist:
                best_dist = d
        if np.isinf(best_dist):
            return 0.0
        return float(max(0.0, prev_distance - best_dist) * 0.0001)

    def _closing_reward_role(self, r_closing_raw: float,
                             role_vec: np.ndarray = None) -> float:
        """
        Rol bazlı kapanma filtresi.
        PURSUIT (index 1): tam aktif  — role_vec[1] ile ölçeklenir
        Diğer roller      : 0.0       (role_vec[0,2,3] ≈ 0 one-hot durumunda)
        role_vec=None     : koşulsuz (geriye uyumluluk, Faz-1)
        """
        if role_vec is None:
            return r_closing_raw
        return r_closing_raw * float(role_vec[1])

    def _sniper_position_reward(self, agent_state, enemy_states) -> float:
        """
        SNIPER rolü için WEZ optimal mesafe pozisyon ödülü.
        Gaussian: tepe ~3000m, σ=2000m, max=0.3
        compute() içinde role_vec[0] ile ağırlıklandırılır.
        """
        if not enemy_states or agent_state[STATE_ALIVE] < 0.5:
            return 0.0
        agent_pos = agent_state[[STATE_X, STATE_Y, STATE_H]]
        best_dist = np.inf
        for es in enemy_states:
            if es[STATE_ALIVE] < 0.5:
                continue
            d = distance_3d(agent_pos, es[[STATE_X, STATE_Y, STATE_H]])
            if d < best_dist:
                best_dist = d
        if np.isinf(best_dist):
            return 0.0
        opt_dist = 3000.0
        sigma    = 2000.0
        return float(np.exp(-((best_dist - opt_dist) ** 2) / (2 * sigma ** 2)) * 0.3)

    def _sniper_patience_reward(self, fire_result, agent_state,
                                enemy_states, weapons_model) -> float:
        """
        SNIPER sabırlı pozisyon ödülü.
        WEZ içindeyken ateş etmeden bekliyorsa +0.5 — optimal atış anını bekler.
        compute() içinde role_vec[0] ile ağırlıklandırılır.
        """
        if fire_result is None or agent_state[STATE_ALIVE] < 0.5:
            return 0.0
        if not enemy_states:
            return 0.0
        in_wez = fire_result.get("wez_info", {}).get("in_wez", False)
        if not in_wez:
            return 0.0
        fired = fire_result.get("fired", False)
        if fired:
            return 0.0  # zaten ateş etti — ayrıca fire_ready ödülü alır
        return 0.5

    def _evasion_reward(self, agent_state, enemy_states, prev_distance) -> float:
        """
        DEFENSIVE rolü için uzaklaşma ödülü — r_closing_raw'ın tersi.
        Düşmana olan mesafe arttıkça pozitif; compute() içinde role_vec[2] ile ölçeklenir.
        """
        if not enemy_states or agent_state[STATE_ALIVE] < 0.5 or prev_distance is None:
            return 0.0
        agent_pos = agent_state[[STATE_X, STATE_Y, STATE_H]]
        best_dist = np.inf
        for es in enemy_states:
            if es[STATE_ALIVE] < 0.5:
                continue
            d = distance_3d(agent_pos, es[[STATE_X, STATE_Y, STATE_H]])
            if d < best_dist:
                best_dist = d
        if np.isinf(best_dist):
            return 0.0
        return float(max(0.0, best_dist - prev_distance) * 0.0001)

    def _support_reward(self, agent_state: np.ndarray,
                        teammate_states: list,
                        enemy_states:    list,
                        weapons_model:   WeaponsModel,
                        role_support_prob: float = 1.0) -> float:
        """
        Takım arkadaşı angajmandayken ikinci düşmana yönel.

        Koşullar:
          1. Takım arkadaşı hayatta ve en az 1 düşmanı WEZ içinde (score > 0.3)
          2. Ajan 2. düşmana (takım arkadaşının hedef almadığı) yöneliyor

        r_raw = (tracking_term + dist_term) * 0.5   ∈ [0, ~0.65]
        Döndürür: w_support bu metodun dışında uygulanır.
        role_support_prob: RoleSelector[support] ağırlığı — soft eğitimde [0,1],
                           inference'da {0,1}. Faz-3 öncesi her zaman 1.0 kalır.
        """
        if not teammate_states or not enemy_states:
            return 0.0
        if agent_state[STATE_ALIVE] < 0.5:
            return 0.0

        alive_teammates = [tm for tm in teammate_states if tm[STATE_ALIVE] > 0.5]
        if not alive_teammates:
            return 0.0

        alive_enemies = [es for es in enemy_states if es[STATE_ALIVE] > 0.5]
        if not alive_enemies:
            return 0.0

        # Takım arkadaşı WEZ içinde mi? Hedef aldığı düşman indeksini bul.
        teammate_wez   = False
        tm_target_idx  = 0
        for tm in alive_teammates:
            best_score = 0.0
            for idx, es in enumerate(alive_enemies):
                score = weapons_model.wez_advantage_score(tm, es)
                if score > best_score:
                    best_score    = score
                    tm_target_idx = idx
            if best_score > 0.3:
                teammate_wez = True
                break

        if not teammate_wez:
            return 0.0

        # Ajanın hedeflemesi gereken düşman: takım arkadaşının hedefinden farklı
        if len(alive_enemies) > 1:
            second_idx   = 1 - tm_target_idx if tm_target_idx == 0 else 0
            second_enemy = alive_enemies[second_idx]
        else:
            second_enemy = alive_enemies[0]  # tek düşman varsa aynı hedef — destek = baskı

        agent_pos  = agent_state[[STATE_X, STATE_Y, STATE_H]]
        second_pos = second_enemy[[STATE_X, STATE_Y, STATE_H]]
        d          = distance_3d(agent_pos, second_pos)

        tracking_term = float(max(0.0, weapons_model.antenna_train_cos(agent_state, second_enemy)))
        dist_term     = float(max(0.0, 1.0 - d / self.pursuit_norm_dist)) * 0.3

        r_raw = (tracking_term + dist_term) * 0.5
        return float(r_raw * float(np.clip(role_support_prob, 0.0, 1.0)))

    def _role_match_reward(self, role_vec, fire_result, agent_state,
                           teammate_states, enemy_states, weapons_model,
                           prev_distance) -> float:
        """
        RoleSelector çıktısına uyumlu davranış bonusu.

        SNIPER    (0) + kill          → +3.0
        PURSUIT   (1) + dist_closing  → +0.5
        SUPPORT   (3) + tm_in_wez     → +0.5
        DEFENSIVE (2) + hp < 0.5      → +0.5
        """
        if role_vec is None or agent_state[STATE_ALIVE] < 0.5:
            return 0.0
        r = 0.0
        # SNIPER: kill yaptıysa güçlü bonus
        if fire_result is not None and fire_result.get("kill", False):
            r += 3.0 * float(role_vec[0])
        # PURSUIT: mesafe kapandıysa
        if prev_distance is not None:
            agent_pos = agent_state[[STATE_X, STATE_Y, STATE_H]]
            best_dist = np.inf
            for es in enemy_states:
                if es[STATE_ALIVE] < 0.5:
                    continue
                d = distance_3d(agent_pos, es[[STATE_X, STATE_Y, STATE_H]])
                if d < best_dist:
                    best_dist = d
            if not np.isinf(best_dist) and prev_distance > best_dist:
                r += 0.5 * float(role_vec[1])
        # SUPPORT: takım arkadaşı WEZ içindeyse
        for tm in teammate_states:
            if tm[STATE_ALIVE] < 0.5:
                continue
            for es in enemy_states:
                if es[STATE_ALIVE] < 0.5:
                    continue
                if weapons_model.wez_advantage_score(tm, es) > 0.3:
                    r += 0.5 * float(role_vec[3])
                    break
            break
        # DEFENSIVE: düşük HP
        if float(agent_state[STATE_HP]) < 0.5:
            r += 0.5 * float(role_vec[2])
        return float(r)

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
                "r_coord", "r_resource", "r_penalty", "r_role", "r_fire_ready",
                "r_pursuit", "r_smooth_ctrl", "r_smooth_throttle",
                "r_throttle_ctx", "r_closing", "r_closing_raw",
                "r_sniper_pos", "r_sniper_patience",
                "r_support", "r_support_raw", "r_evasion", "r_role_match",
                "w_kill_contrib", "w_wez_contrib", "w_tracking_contrib",
                "w_survival_contrib", "w_coord_contrib", "w_resource_contrib",
                "w_penalty_contrib", "w_role_contrib", "w_pursuit_contrib"]
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
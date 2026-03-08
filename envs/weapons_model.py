"""
weapons_model.py
================
Silah sistemi modeli — WEZ hesaplama ve hasar mekaniği.

WEZ (Weapon Engagement Zone):
    Bir uçağın düşmana ateş edebileceği konik bölge.
    Koni ekseni: ateş eden uçağın heading yönü
    Koni yarı açısı: wez_angle_max (±30°, literatür standardı)
    Koni menzili: wez_range_min – wez_range_max

Hasar mekaniği: Deterministik
    WEZ içindeyse + fire >= 0.5 + ammo > 0 + cooldown bitti
    → hedef hp -= missile_damage
    → ammo -= 1
    → cooldown başlar

Koordinat sistemi: ENU (geometry_utils ile uyumlu)

Bağımlılıklar:
    - numpy
    - geometry_utils.py  (antenna_train_angle, aspect_angle, distance_3d)
    - aircraft_model.py  (STATE_* sabitleri)

Bu dosya değişirse etkilenen dosyalar:
    - reward_model.py    (wez_advantage, kill reward)
    - dogfight_env.py    (step döngüsünde fire işleme)
    - test_weapons_model.py
"""

import numpy as np
from envs.geometry_utils import (
    antenna_train_angle,
    aspect_angle,
    distance_3d,
    deg2rad,
    threat_score as _threat_score
)
from envs.aircraft_model import (
    STATE_X, STATE_Y, STATE_H,
    STATE_V, STATE_PSI, STATE_ALIVE,
    STATE_AMMO, STATE_HP, STATE_FUEL,
    ACTION_FIRE
)


class WeaponsModel:
    """
    Silah sistemi — WEZ kontrolü ve hasar uygulaması.

    Her ajan için ayrı bir WeaponsModel örneği oluşturulur.
    dogfight_env.py step döngüsünde çağrılır.

    Kullanım:
        weapons = WeaponsModel(config)
        weapons.reset()
        fire_result = weapons.process_fire(shooter_state, target_state, action)
        wez_info    = weapons.compute_wez(shooter_state, target_state)
    """

    def __init__(self, config: dict):
        """
        Parameters
        ----------
        config : dict — config.yaml'dan tam config sözlüğü.
                 'weapons' anahtarı kullanılır.
        """
        w = config["weapons"]

        # WEZ geometri parametreleri
        self.wez_range_max  = float(w["wez_range_max"])   # m
        self.wez_range_min  = float(w.get("wez_range_min", 300.0))  # m
        self.wez_angle_max  = deg2rad(float(w["wez_angle_max"]))    # rad (±30°)

        # Hasar parametreleri
        self.missile_damage   = float(w["missile_damage"])    # hp düşüşü
        self.fire_cooldown    = float(w.get("fire_cooldown", 2.0))   # saniye
        self.min_fire_altitude = float(w.get("min_fire_altitude", 200.0))  # m

        # Dahili durum
        self._cooldown_timer = 0.0   # kalan bekleme süresi (saniye)

    def reset(self):
        """Episode başında silah durumunu sıfırla."""
        self._cooldown_timer = 0.0

    def tick(self, dt: float):
        """
        Her step'te cooldown sayacını güncelle.
        dogfight_env.step() içinde process_fire'dan ÖNCE çağrılır.

        Parameters
        ----------
        dt : float — adım süresi (saniye)
        """
        self._cooldown_timer = max(0.0, self._cooldown_timer - dt)

    @property
    def can_fire(self) -> bool:
        """Cooldown bitti mi?"""
        return self._cooldown_timer <= 0.0

    # -----------------------------------------------------------------------
    # WEZ Hesaplama
    # -----------------------------------------------------------------------

    def compute_wez(self, shooter_state: np.ndarray,
                    target_state: np.ndarray) -> dict:
        """
        Nişancının hedefe yönelik WEZ bilgilerini hesaplar.

        Parameters
        ----------
        shooter_state : np.ndarray, shape (STATE_DIM,) — nişancı state
        target_state  : np.ndarray, shape (STATE_DIM,) — hedef state

        Returns
        -------
        dict :
            in_wez          : bool   — hedef WEZ içinde mi?
            distance        : float  — 3D mesafe (m)
            ata             : float  — Antenna Train Angle (rad)
            aspect          : float  — Aspect Angle (rad)
            range_factor    : float  — mesafe skoru [0,1]
            angle_factor    : float  — açı skoru [0,1]
            wez_advantage   : float  — toplam WEZ avantajı [0,1]
            threat_score    : float  — hedefin bize tehdidi [0,1]
        """
        # Hayatta değilse WEZ yok
        if (shooter_state[STATE_ALIVE] < 0.5 or
                target_state[STATE_ALIVE] < 0.5):
            return self._empty_wez()

        shooter_pos = shooter_state[[STATE_X, STATE_Y, STATE_H]]
        target_pos  = target_state[[STATE_X, STATE_Y, STATE_H]]
        psi_shooter = float(shooter_state[STATE_PSI])
        psi_target  = float(target_state[STATE_PSI])

        # Geometrik hesaplar
        dist = distance_3d(shooter_pos, target_pos)
        ata  = antenna_train_angle(shooter_pos, target_pos, psi_shooter)
        aa   = aspect_angle(target_pos, shooter_pos, psi_target)

        # Mesafe faktörü: WEZ menzili içindeyse [0,1]
        # Önce hard sınır kontrolü — min veya max dışındaysa kesin 0
        if dist < self.wez_range_min or dist > self.wez_range_max:
            range_factor = 0.0
        else:
            # Geçerli menzil içinde: Gaussian ile sürekli sinyal
            # Optimal menzil: wez_range_max'ın %40'ı
            r_opt = self.wez_range_max * 0.4
            range_factor = float(np.exp(-((dist - r_opt) ** 2) /
                                        (2 * (self.wez_range_max * 0.3) ** 2)))
            range_factor = max(0.0, min(1.0, range_factor))

        # Açı faktörü: ±wez_angle_max içindeyse [0,1]
        angle_factor = max(0.0, 1.0 - abs(ata) / self.wez_angle_max)
        angle_factor = float(angle_factor if abs(ata) <= self.wez_angle_max else 0.0)

        # WEZ içinde mi? (hard boolean)
        in_wez = (
            self.wez_range_min <= dist <= self.wez_range_max and
            abs(ata) <= self.wez_angle_max
        )

        # WEZ avantaj skoru (reward_model için sürekli sinyal)
        wez_advantage = float(range_factor * angle_factor)

        # Karşı tarafın bize tehdidi
        ts = _threat_score(
            distance=dist,
            ata=float(target_state[STATE_PSI] - shooter_state[STATE_PSI]),
            aspect=aa,
            range_max=self.wez_range_max,
            angle_max_rad=self.wez_angle_max
        )

        return {
            "in_wez":        in_wez,
            "distance":      dist,
            "ata":           ata,
            "aspect":        aa,
            "range_factor":  range_factor,
            "angle_factor":  angle_factor,
            "wez_advantage": wez_advantage,
            "threat_score":  ts,
        }

    # -----------------------------------------------------------------------
    # Ateş İşleme
    # -----------------------------------------------------------------------

    def process_fire(self, shooter_state: np.ndarray,
                     target_state: np.ndarray,
                     action: np.ndarray,
                     dt: float) -> dict:
        """
        Ateş komutunu işler, koşullar sağlanıyorsa hasar uygular.

        Hasar koşulları (deterministik, hepsi sağlanmalı):
            1. fire aksiyon >= 0.5
            2. Nişancı hayatta
            3. Hedef hayatta
            4. Mühimmat > 0
            5. Cooldown bitti
            6. Nişancı minimum irtifanın üzerinde
            7. Hedef WEZ içinde

        Parameters
        ----------
        shooter_state : np.ndarray — nişancı state (değiştirilmez)
        target_state  : np.ndarray — hedef state  (değiştirilmez)
        action        : np.ndarray — nişancının aksiyonu
        dt            : float      — adım süresi (saniye)

        Returns
        -------
        dict :
            fired           : bool  — ateş edildi mi?
            hit             : bool  — isabet oldu mu?
            damage          : float — uygulanan hasar (0 veya missile_damage)
            kill            : bool  — hedef hp <= 0'a düştü mü?
            ammo_remaining  : float — kalan mühimmat
            new_target_hp   : float — hedefin yeni HP'si
            wez_info        : dict  — compute_wez() çıktısı
            fail_reason     : str   — ateş edilmediyse neden
        """
        wez_info = self.compute_wez(shooter_state, target_state)

        # Ateş komutu var mı?
        fire_cmd = float(action[ACTION_FIRE]) >= 0.5

        if not fire_cmd:
            return self._no_fire_result(wez_info, "no_fire_command")

        if shooter_state[STATE_ALIVE] < 0.5:
            return self._no_fire_result(wez_info, "shooter_dead")

        if target_state[STATE_ALIVE] < 0.5:
            return self._no_fire_result(wez_info, "target_dead")

        if shooter_state[STATE_AMMO] <= 0:
            return self._no_fire_result(wez_info, "no_ammo")

        if not self.can_fire:
            return self._no_fire_result(wez_info, "cooldown",
                                         f"{self._cooldown_timer:.1f}s kaldı")

        if shooter_state[STATE_H] < self.min_fire_altitude:
            return self._no_fire_result(wez_info, "below_min_altitude")

        # Ateş edildi — WEZ kontrolü
        fired = True
        hit   = wez_info["in_wez"]

        if hit:
            damage        = self.missile_damage
            new_target_hp = max(0.0, float(target_state[STATE_HP]) - damage)
            kill          = new_target_hp <= 0.0
            self._cooldown_timer = self.fire_cooldown
        else:
            # WEZ dışında ateş → mühimmat harcanmaz, cooldown başlamaz
            damage        = 0.0
            new_target_hp = float(target_state[STATE_HP])
            kill          = False

        return {
            "fired":          fired,
            "hit":            hit,
            "damage":         damage,
            "kill":           kill,
            "ammo_remaining": float(shooter_state[STATE_AMMO]) - (1.0 if hit else 0.0),
            "new_target_hp":  new_target_hp,
            "wez_info":       wez_info,
            "fail_reason":    None,
        }

    # -----------------------------------------------------------------------
    # WEZ Avantaj Skoru (reward_model için)
    # -----------------------------------------------------------------------

    def wez_advantage_score(self, shooter_state: np.ndarray,
                             target_state: np.ndarray) -> float:
        """
        Sürekli WEZ avantaj skoru [0, 1].
        reward_model.py'de r_wez_advantage hesabında kullanılır.

        Parameters
        ----------
        shooter_state : np.ndarray
        target_state  : np.ndarray

        Returns
        -------
        float — WEZ avantajı [0, 1]
        """
        wez = self.compute_wez(shooter_state, target_state)
        return wez["wez_advantage"]

    def antenna_train_cos(self, shooter_state: np.ndarray,
                          target_state: np.ndarray) -> float:
        """
        cos(ATA) — radar izleme reward'u için.
        1.0 = hedef tam önümüzde, 0.0 = dik açıda, -1.0 = arkamızda.

        reward_model.py'de r_tracking hesabında kullanılır.
        """
        if (shooter_state[STATE_ALIVE] < 0.5 or
                target_state[STATE_ALIVE] < 0.5):
            return 0.0

        shooter_pos = shooter_state[[STATE_X, STATE_Y, STATE_H]]
        target_pos  = target_state[[STATE_X, STATE_Y, STATE_H]]
        psi_shooter = float(shooter_state[STATE_PSI])

        ata = antenna_train_angle(shooter_pos, target_pos, psi_shooter)
        return float(np.cos(ata))

    # -----------------------------------------------------------------------
    # Yardımcı
    # -----------------------------------------------------------------------

    def _empty_wez(self) -> dict:
        """Hayatta olmayan uçaklar için boş WEZ sonucu."""
        return {
            "in_wez":        False,
            "distance":      float("inf"),
            "ata":           0.0,
            "aspect":        0.0,
            "range_factor":  0.0,
            "angle_factor":  0.0,
            "wez_advantage": 0.0,
            "threat_score":  0.0,
        }

    def _no_fire_result(self, wez_info: dict,
                         reason: str,
                         detail: str = "") -> dict:
        """Ateş edilmeyen durumlar için standart sonuç."""
        return {
            "fired":          False,
            "hit":            False,
            "damage":         0.0,
            "kill":           False,
            "ammo_remaining": None,   # değişmedi
            "new_target_hp":  None,   # değişmedi
            "wez_info":       wez_info,
            "fail_reason":    reason + (f" ({detail})" if detail else ""),
        }

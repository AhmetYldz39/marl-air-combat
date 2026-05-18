"""
geometry_utils.py
=================
3D geometri yardımcı fonksiyonları.

Koordinat sistemi: ENU (East-North-Up)
    x = Doğu  (m)
    y = Kuzey (m)
    z = Yukarı / İrtifa (m)

Açı convention:
    psi (heading/yaw) : Kuzey'den saat yönünde, radyan
                        0 = Kuzey, π/2 = Doğu, π = Güney
    gamma (flight path angle) : Yataydan yukarı pozitif, radyan
    phi (roll)         : Sağa yatış pozitif, radyan

Bağımlılıklar: Sadece numpy — config veya diğer modüllere bağımlılık YOK.
Bu dosya değişirse etkilenen dosyalar:
    - aircraft_model.py  (bearing, elevation kullanır)
    - weapons_model.py   (aspect_angle, antenna_train_angle, wez_check kullanır)
    - dogfight_env.py    (relative_position, distance_3d kullanır)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------
TWO_PI = 2.0 * np.pi
HALF_PI = 0.5 * np.pi
EPS = 1e-9  # sıfıra bölme koruması


# ---------------------------------------------------------------------------
# 1. Temel Vektör İşlemleri
# ---------------------------------------------------------------------------

def relative_position(pos_self: np.ndarray, pos_other: np.ndarray) -> np.ndarray:
    """
    İki nokta arasındaki göreceli konum vektörünü döndürür (ENU).

    Parameters
    ----------
    pos_self  : np.ndarray, shape (3,) — [x, y, z] kendi konumu
    pos_other : np.ndarray, shape (3,) — [x, y, z] hedef konumu

    Returns
    -------
    np.ndarray, shape (3,) — [dx, dy, dz] = pos_other - pos_self
    """
    return pos_other - pos_self


def distance_3d(pos_self: np.ndarray, pos_other: np.ndarray) -> float:
    """
    İki nokta arasındaki 3D Öklid mesafesini döndürür (metre).

    Parameters
    ----------
    pos_self  : np.ndarray, shape (3,)
    pos_other : np.ndarray, shape (3,)

    Returns
    -------
    float — mesafe (m), her zaman >= 0
    """
    delta = relative_position(pos_self, pos_other)
    return float(np.linalg.norm(delta))


def distance_horizontal(pos_self: np.ndarray, pos_other: np.ndarray) -> float:
    """
    Yatay düzlemdeki (x-y) mesafe (metre).

    Parameters
    ----------
    pos_self  : np.ndarray, shape (3,)
    pos_other : np.ndarray, shape (3,)

    Returns
    -------
    float — yatay mesafe (m)
    """
    delta = relative_position(pos_self, pos_other)
    return float(np.linalg.norm(delta[:2]))


# ---------------------------------------------------------------------------
# 2. Açı Hesaplamaları
# ---------------------------------------------------------------------------

def bearing_angle(pos_self: np.ndarray, pos_other: np.ndarray) -> float:
    """
    Kendi konumundan hedefe olan yatay yön açısı (bearing).
    Kuzey'den saat yönünde, radyan. [0, 2π)

    Parameters
    ----------
    pos_self  : np.ndarray, shape (3,)
    pos_other : np.ndarray, shape (3,)

    Returns
    -------
    float — bearing açısı (rad), [0, 2π)

    Notes
    -----
    ENU'da: atan2(dx, dy) — x=Doğu, y=Kuzey
    Kuzey = 0, Doğu = π/2, Güney = π, Batı = 3π/2
    """
    delta = relative_position(pos_self, pos_other)
    angle = np.arctan2(delta[0], delta[1])  # atan2(Doğu, Kuzey)
    return float(angle % TWO_PI)


def elevation_angle(pos_self: np.ndarray, pos_other: np.ndarray) -> float:
    """
    Kendi konumundan hedefe olan dikey açı (elevation).
    Yataydan yukarı pozitif, radyan. (-π/2, π/2)

    Parameters
    ----------
    pos_self  : np.ndarray, shape (3,)
    pos_other : np.ndarray, shape (3,)

    Returns
    -------
    float — elevation açısı (rad)
    """
    delta = relative_position(pos_self, pos_other)
    dz = delta[2]
    dh = np.linalg.norm(delta[:2])
    return float(np.arctan2(dz, dh + EPS))


def aspect_angle(pos_target: np.ndarray,
                 pos_shooter: np.ndarray,
                 psi_target: float) -> float:
    """
    Aspect angle: Hedefin kendi başlık açısından bakışa göre
    nişancının hangi yönde olduğunu gösterir.

    Tanım: Hedefin burnundan itibaren nişancıya olan açı.
    0° = hedefin tam önünde (baş kafası)
    180° = hedefin tam arkasında (kuyruğu)
    Literatür standardı: [0°, 180°], simetrik (sol/sağ ayrımı yok)

    Parameters
    ----------
    pos_target  : np.ndarray, shape (3,) — hedefin konumu
    pos_shooter : np.ndarray, shape (3,) — nişancının konumu
    psi_target  : float — hedefin heading açısı (rad, ENU convention)

    Returns
    -------
    float — aspect angle (rad), [0, π]

    Notes
    -----
    Hava muharebesi literatüründe "AA" olarak da geçer.
    AA = 0° → hedef bize dönük (tehlikeli)
    AA = 180° → hedefin arkasındayız (avantajlı pozisyon)
    """
    # Hedeften nişancıya göreceli vektör
    delta = relative_position(pos_target, pos_shooter)

    # Hedefin heading yönü vektörü (ENU: kuzey=0, doğu=π/2)
    heading_vec = np.array([np.sin(psi_target), np.cos(psi_target), 0.0])

    # delta vektörünün yatay bileşeni
    delta_h = np.array([delta[0], delta[1], 0.0])
    delta_h_norm = np.linalg.norm(delta_h)

    if delta_h_norm < EPS:
        return 0.0  # aynı yatay konumda — tanımsız, 0 döndür

    delta_h_unit = delta_h / delta_h_norm

    # Dot product → açı
    dot = np.clip(np.dot(heading_vec, delta_h_unit), -1.0, 1.0)
    return float(np.arccos(dot))  # [0, π]


def antenna_train_angle(pos_self: np.ndarray,
                        pos_target: np.ndarray,
                        psi_self: float) -> float:
    """
    Antenna Train Angle (ATA): Kendi başlık açısından hedefe olan açı.
    Radar'ın hedefi izlemek için döndürülmesi gereken açı.

    Tanım: Kendi burnumuzdan itibaren hedefe olan yatay açı.
    0° = hedef tam önümüzde
    ±180° = hedef tam arkamızda
    İşaret: sol negatif, sağ pozitif (pilot convention)

    Parameters
    ----------
    pos_self   : np.ndarray, shape (3,) — kendi konumuz
    pos_target : np.ndarray, shape (3,) — hedef konumu
    psi_self   : float — kendi heading açımız (rad, ENU convention)

    Returns
    -------
    float — ATA (rad), (-π, π]

    Notes
    -----
    Reward fonksiyonunda: cos(ATA) → 1 ise hedef tam önümüzde (iyi)
    WEZ kontrolünde: |ATA| < wez_angle_max ise ateş koridorundayız
    """
    delta = relative_position(pos_self, pos_target)
    delta_norm = np.linalg.norm(delta)

    if delta_norm < EPS:
        return 0.0

    delta_unit = delta / delta_norm
    # Heading birim vektörü (yatay, ENU: kuzey=0, doğu=π/2)
    heading = np.array([np.sin(psi_self), np.cos(psi_self), 0.0])

    # 3D off-boresight açısı (unsigned)
    dot = float(np.clip(np.dot(heading, delta_unit), -1.0, 1.0))
    ata = float(np.arccos(dot))

    # İşaret: heading × delta_unit z-bileşeni pozitifse hedef sol tarafta
    cross_z = heading[0] * delta_unit[1] - heading[1] * delta_unit[0]
    if cross_z > 0.0:
        ata = -ata

    return float(wrap_to_pi(ata))


def flight_path_to_heading(gamma: float, psi: float) -> np.ndarray:
    """
    Uçuş yolu açısı ve heading'den 3D hız yön vektörünü hesaplar (ENU).

    Parameters
    ----------
    gamma : float — uçuş yolu açısı (rad), yukarı pozitif
    psi   : float — heading açısı (rad), kuzey=0, doğu=π/2

    Returns
    -------
    np.ndarray, shape (3,) — birim hız yön vektörü [vx, vy, vz]
    """
    cos_g = np.cos(gamma)
    return np.array([
        cos_g * np.sin(psi),   # Doğu bileşeni
        cos_g * np.cos(psi),   # Kuzey bileşeni
        np.sin(gamma)          # Yukarı bileşeni
    ])


# ---------------------------------------------------------------------------
# 3. Açı Normalizasyon Yardımcıları
# ---------------------------------------------------------------------------

def wrap_to_pi(angle: float) -> float:
    """
    Açıyı (-π, π] aralığına normalize eder.

    Parameters
    ----------
    angle : float — herhangi bir açı (rad)

    Returns
    -------
    float — normalize edilmiş açı (rad), (-π, π]
    """
    return float((angle + np.pi) % TWO_PI - np.pi)


def wrap_to_2pi(angle: float) -> float:
    """
    Açıyı [0, 2π) aralığına normalize eder.

    Parameters
    ----------
    angle : float — herhangi bir açı (rad)

    Returns
    -------
    float — normalize edilmiş açı (rad), [0, 2π)
    """
    return float(angle % TWO_PI)


def deg2rad(degrees: float) -> float:
    """Derece → radyan dönüşümü."""
    return float(np.radians(degrees))


def rad2deg(radians: float) -> float:
    """Radyan → derece dönüşümü."""
    return float(np.degrees(radians))


# ---------------------------------------------------------------------------
# 4. Koordinat Dönüşümleri
# ---------------------------------------------------------------------------

def ned_to_enu(x_ned: float, y_ned: float, z_ned: float):
    """
    NED (North-East-Down) → ENU (East-North-Up) dönüşümü.
    JSBSim katsayı referansları için kullanılır.

    Parameters
    ----------
    x_ned : float — Kuzey bileşeni (NED)
    y_ned : float — Doğu bileşeni (NED)
    z_ned : float — Aşağı bileşeni (NED)

    Returns
    -------
    tuple — (x_enu, y_enu, z_enu)
        x_enu = Doğu
        y_enu = Kuzey
        z_enu = Yukarı
    """
    return float(y_ned), float(x_ned), float(-z_ned)


def enu_to_unity(x_enu: float, y_enu: float, z_enu: float):
    """
    ENU → Unity sol-el koordinat sistemi dönüşümü.
    Unity bridge (Faz 5) için hazır tutulur.

    Parameters
    ----------
    x_enu : float — Doğu (ENU)
    y_enu : float — Kuzey (ENU)
    z_enu : float — Yukarı (ENU)

    Returns
    -------
    tuple — (x_unity, y_unity, z_unity)
        x_unity = Sağ  (← ENU Doğu)
        y_unity = Yukarı (← ENU Yukarı)
        z_unity = İleri (← ENU Kuzey)
    """
    return float(x_enu), float(z_enu), float(y_enu)


# ---------------------------------------------------------------------------
# 5. Tehdit Skoru
# ---------------------------------------------------------------------------

def threat_score(distance: float,
                 ata: float,
                 aspect: float,
                 range_max: float,
                 angle_max_rad: float) -> float:
    """
    Bir hedefin bize yönelik tehdit skorunu hesaplar. [0, 1]

    Yüksek tehdit:
        - Düşman yakın
        - Düşmanın ATA'sı küçük (bize dönük)
        - Aspect angle'ımız küçük (düşmanın önündeyiz)

    Parameters
    ----------
    distance      : float — 3D mesafe (m)
    ata           : float — düşmanın bize olan ATA (rad)
    aspect        : float — bizim aspect angle'ımız (rad), [0, π]
    range_max     : float — maksimum tehdit menzili (m), config'den
    angle_max_rad : float — maksimum tehdit açısı (rad), config'den

    Returns
    -------
    float — tehdit skoru [0, 1], 1 = maksimum tehdit

    Notes
    -----
    Formül:
        range_factor  = max(0, 1 - distance/range_max)
        angle_factor  = max(0, 1 - |ata|/angle_max)
        aspect_factor = max(0, 1 - aspect/π)   (önde = tehlikeli)
        score = range_factor * angle_factor * aspect_factor
    """
    range_factor = max(0.0, 1.0 - distance / (range_max + EPS))
    angle_factor = max(0.0, 1.0 - abs(ata) / (angle_max_rad + EPS))
    aspect_factor = max(0.0, 1.0 - aspect / (np.pi + EPS))
    return float(range_factor * angle_factor * aspect_factor)


# ---------------------------------------------------------------------------
# 6. Rotasyon Matrisleri
# ---------------------------------------------------------------------------

def rotation_matrix_body_to_wind(alpha: float, beta: float) -> np.ndarray:
    """
    Body ekseninden wind eksenine dönüşüm matrisi.
    α (hücum açısı) ve β (kayma açısı) kullanır.

    aircraft_model.py'de u,v,w → V,α,β dönüşümünde kullanılır.

    Parameters
    ----------
    alpha : float — hücum açısı (rad)
    beta  : float — kayma açısı (rad)

    Returns
    -------
    np.ndarray, shape (3, 3) — dönüşüm matrisi
    """
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta),  np.sin(beta)

    return np.array([
        [ ca*cb,  sb,  sa*cb],
        [-ca*sb,  cb, -sa*sb],
        [-sa,     0.0,  ca  ]
    ])


def rotation_matrix_wind_to_body(alpha: float, beta: float) -> np.ndarray:
    """
    Wind ekseninden body eksenine dönüşüm matrisi (yukarıdakinin transpozu).

    Parameters
    ----------
    alpha : float — hücum açısı (rad)
    beta  : float — kayma açısı (rad)

    Returns
    -------
    np.ndarray, shape (3, 3)
    """
    return rotation_matrix_body_to_wind(alpha, beta).T


def euler_to_rotation_matrix(phi: float, theta: float, psi: float) -> np.ndarray:
    """
    Euler açılarından (ZYX convention) body→inertial dönüşüm matrisi.
    aircraft_model.py'de kullanılır.

    Parameters
    ----------
    phi   : float — roll  (rad)
    theta : float — pitch (rad)
    psi   : float — yaw   (rad)

    Returns
    -------
    np.ndarray, shape (3, 3) — DCM (Direction Cosine Matrix)
    """
    cp, sp = np.cos(phi),   np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cy, sy = np.cos(psi),   np.sin(psi)

    return np.array([
        [ct*cy,             ct*sy,            -st   ],
        [sp*st*cy - cp*sy,  sp*st*sy + cp*cy,  sp*ct],
        [cp*st*cy + sp*sy,  cp*st*sy - sp*cy,  cp*ct]
    ])
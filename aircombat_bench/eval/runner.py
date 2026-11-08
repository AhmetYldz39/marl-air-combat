"""
aircombat_bench/eval/runner.py
==============================
Unified evaluation runner — schema v1.

Handles MAPPO, MAPPO+GAT, MAPPO+GAT+OM+Role, QMIX, FACMAC and FACMAC+OM+R
from a single command-line entry point, replacing the former eval.py /
eval_facmac.py / eval_qmix.py split.

Output schema v1 (JSON)
-----------------------
{
  "schema_version": "1.0",
  "package_version": "<pkg>",
  "env_version": "<env>",
  "git_commit": "<short-sha>",
  "timestamp_utc": "<ISO-8601>",
  "scenario": "2v2_normal",
  "opponent": "heuristic-v1",
  "checkpoint": {"path": "…", "sha256": "…"},
  "algorithm": "<detected>",
  "n_episodes": 500,
  "seed": 42,
  "deterministic": true,
  "metrics": {
      "win_rate": 0.842, "win_ci95": [0.807, 0.871],
      "loss_rate": 0.006, "draw_rate": 0.152,
      "kill_per_ep": 1.78, "kill_std": …,
      "mean_ep_len": …, "survival_rate": …,
      "oob_rate": …, "one_kill_stall_rate": …
  },
  "episodes": [
      {"idx": 0, "ep_seed": 42, "outcome": "win", "kills": 2, "steps": 271, …}
  ]
}

CI method: Wilson score (Wilson, 1927), z=1.96 (95%).

Note on seeding (T2.3): The current implementation seeds the env once with
``seed`` before the episode loop, matching the original evaluation scripts that
produced the published baseline numbers.  T2.3 will switch to per-episode
seeding (``seed + episode_index``) so that individual episodes can be replayed
in isolation; that change will require a full re-baseline.

Usage
-----
python -m aircombat_bench.eval.runner --checkpoint checkpoints/facmac_omr_ep27000.pt
aircombat-eval --checkpoint checkpoints/facmac_omr_ep27000.pt --episodes 500 --seed 42
"""

from __future__ import annotations

import os

# ── Tek-thread pin (T2.5 Step 4b) ────────────────────────────────────────────
# CPU'da bit-tekrarlanabilirlik yalnızca sabitlenmiş bir konfigürasyonda
# garanti edilir: thread sayısı BLAS indirgeme sırasını değiştirir, bu da
# kayan nokta sonucunu son bitlerde oynatır ve uzun episode'larda yörüngeyi
# ayırır.  Bu yüzden thread sayısı runner tarafından sabitlenir; kullanıcının
# shell ortamı sonucu sessizce değiştiremez.
#
# OMP/MKL değişkenleri torch import edilmeden ÖNCE ayarlanmalı — bu modül
# torch'u aşağıda (lazy) import ettiği için burada ayarlamak etkili olur.
# Çağıran torch'u bizden önce import etmişse env değişkeni geç kalır; bu
# durumda aşağıdaki torch.set_num_threads(1) yine de devreye girer.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

CANONICAL_NUM_THREADS = 1

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

# ── Module-level constants ───────────────────────────────────────────────────
ROOT            = Path(__file__).resolve().parent.parent.parent
SCHEMA_VERSION  = "1.0"
DEFAULT_SCENARIO = "2v2_normal"
_BASE_OBS_DIM   = 50          # tüm checkpoint türleri için base obs boyutu
_WEZ_RANGE_MIN  = 300.0
_WEZ_RANGE_MAX  = 8_000.0
_WEZ_ANGLE_MAX  = np.radians(30.0)

# ── Package / env versioning ─────────────────────────────────────────────────
from aircombat_bench import __version__ as _PKG_VERSION
from aircombat_bench.envs.dogfight_env import DogfightEnv, BLUE, RED, ENV_VERSION
from aircombat_bench.agents.heuristic_agent import (
    MultiHeuristicPolicy, HEURISTIC_VERSION,
)
from aircombat_bench.envs.aircraft_model import (
    STATE_ALIVE, STATE_X, STATE_Y, STATE_H, STATE_PSI, STATE_AMMO,
)
from aircombat_bench.envs.geometry_utils import (
    distance_3d, bearing_angle, antenna_train_angle, wrap_to_pi,
)


# ===========================================================================
# Runtime fingerprint (T2.5 Step 4b)
# ===========================================================================

def pin_threads(n_threads: int = CANONICAL_NUM_THREADS) -> None:
    """torch thread sayısını sabitle.

    Kanonik eval tek thread'de koşar.  Bu fonksiyon run_eval tarafından
    çağrılır — kullanıcının shell'i değil.
    """
    import torch
    torch.set_num_threads(n_threads)
    try:
        torch.set_num_interop_threads(n_threads)
    except RuntimeError:
        # Paralel iş zaten başlamışsa torch bunu reddeder; intra-op pin
        # (yukarıdaki set_num_threads) belirleyici olan, o geçerli kaldı.
        pass


def _blas_backend() -> str:
    """Aktif BLAS/hızlandırma backend'ini isimlendir."""
    import torch
    parts = []
    try:
        if torch.backends.mkl.is_available():
            parts.append("mkl")
    except Exception:
        pass
    try:
        if torch.backends.mkldnn.is_available():
            parts.append("mkldnn")
    except Exception:
        pass
    if not parts:
        parts.append("unknown")
    return "+".join(parts)


def _cpu_model() -> str:
    """İnsan okunur CPU tanımı (platform'a göre en iyi çaba)."""
    val = platform.processor() or ""
    if platform.system() == "Windows":
        val = os.environ.get("PROCESSOR_IDENTIFIER", val)
    elif platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        val = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass
    return val.strip() or "unknown"


def runtime_block(device) -> Dict[str, Any]:
    """Schema v1 `runtime` bloğu — sonucu yeniden üretmek için gereken
    çalışma zamanı parmak izi.

    Sonuçlar **eşleşen** bir runtime konfigürasyonunda bit-tekrarlanabilirdir.
    Farklı konfigürasyonlar arasında sonuçlar birbirine bildirilen güven
    aralığı içinde uyar — birebir değil.
    """
    import torch
    dev_type = getattr(device, "type", str(device))
    block: Dict[str, Any] = {
        "device":         dev_type,
        "torch_version":  torch.__version__,
        "blas_backend":   _blas_backend(),
        "num_threads":    torch.get_num_threads(),
        "cpu_model":      _cpu_model(),
        "os":             platform.platform(),
        "python_version": platform.python_version(),
    }
    if dev_type == "cuda":
        try:
            block["gpu_model"] = torch.cuda.get_device_name(
                device.index if getattr(device, "index", None) is not None else 0
            )
        except Exception:
            block["gpu_model"] = "unknown"
        block["cuda_version"] = torch.version.cuda or "unknown"
    return block


# ===========================================================================
# Utility functions
# ===========================================================================

def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score 95% güven aralığı → (lo, hi).

    Yöntem: Wilson (1927), z=1.96 (iki-kuyruklu %95).
    """
    if n == 0:
        return 0.0, 0.0
    p   = k / n
    den = 1.0 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    rng = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(max(0.0, ctr - rng)), float(min(1.0, ctr + rng))


def _sha256(path: str) -> str:
    """Dosyanın SHA-256 özetini hesapla."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    """Mevcut HEAD'in kısa SHA'sını döndür; hata halinde 'unknown'."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT),
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return out.strip().decode()
    except Exception:
        return "unknown"


def _git_dirty_files() -> Optional[List[str]]:
    """Commit'lenmemiş değişiklik taşıyan izlenen dosyaların listesi (T2.6.6).

    None → git durumu belirlenemedi.  Takip edilmeyen dosyalar (`??`) sayılmaz;
    çıktıyı etkilemezler.
    """
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(ROOT),
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except Exception:
        return None
    files = []
    for line in out.decode("utf-8", "replace").splitlines():
        if line.strip() and not line.startswith("??"):
            files.append(line[3:].strip())
    return files


def _git_dirty() -> bool:
    """Çalışma ağacında commit'lenmemiş değişiklik var mı? (T2.6.6)

    Kirli bir ağaçta üretilen çıktı, kaydedilen commit'ten yeniden
    üretilemez — bu yüzden kanonik dosyalarda `git_dirty: true` reddedilir.
    Belirlenemezse güvenli taraf seçilir: True.

    NOT: Bu bayrak deponun HERHANGİ bir yerindeki kirliliği işaretler —
    dokümantasyon değişikliği de dâhil.  Kirliliğin değerlendirme yolunu
    etkileyip etkilemediğini ayırt etmek için çıktıdaki `git_dirty_files`
    listesine bakılmalıdır.
    """
    files = _git_dirty_files()
    return True if files is None else bool(files)


def _repo_relative_path(path: str) -> str:
    """Yolu depo köküne göreceli, POSIX ayraçlı biçime çevir (T2.6.3).

    Schema v1 çıktısında mutlak yol ya da kullanıcı ev dizini bulunmamalıdır:
    makineye özgüdür ve dosyayı taşınamaz kılar.  Depo dışındaki yollar
    olduğu gibi bırakılır (yalnızca ayraç normalize edilir) — sessizce yanlış
    bir göreceli yol uydurmaktansa doğrulayıcının reddetmesi yeğdir.
    """
    p = Path(path)
    try:
        rel = p.resolve().relative_to(ROOT.resolve())
        return rel.as_posix()
    except (ValueError, OSError):
        return p.as_posix()


#: schema v1 yollarında bulunmaması gereken kalıplar (T2.6.3).
_HOME_MARKERS = ("/Users/", "\\Users\\", "/home/", "\\home\\",
                 "C:/", "C:\\", "c:/", "c:\\", "%USERPROFILE%", "~/")


def path_is_portable(path: str) -> bool:
    """Yol taşınabilir mi (mutlak değil, ev dizini içermiyor)?"""
    if not path:
        return False
    if Path(path).is_absolute() or path.startswith(("/", "\\")):
        return False
    if len(path) > 1 and path[1] == ":":          # Windows sürücü harfi
        return False
    return not any(m in path for m in _HOME_MARKERS)


def _resolve_device(device: str) -> "torch.device":
    """'auto' → CUDA veya CPU; aksi hâlde birebir torch.device."""
    import torch
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


# ===========================================================================
# Rol müdahale modları (T2.9 §2) — çıkarım anında, eğitim yok
# ===========================================================================

#: Rol vektörü boyut sırası.
ROLE_NAMES = ("sniper", "pursuit", "defensive", "support")
ROLE_SNIPER, ROLE_PURSUIT, ROLE_DEFENSIVE, ROLE_SUPPORT = 0, 1, 2, 3

ROLE_MODE_LEARNED         = "learned"          # C0 — atayıcı çıktısı
ROLE_MODE_ZEROED          = "zeroed"           # C1 — sıfır vektör
ROLE_MODE_FIXED_DISTINCT  = "fixed_distinct"   # C2 — ajana bağlı sabit
ROLE_MODE_FIXED_IDENTICAL = "fixed_identical"  # C3 — ikisine aynı sabit
ROLE_MODE_SHUFFLED        = "shuffled"         # C4 — ajanlar arası takas
ROLE_MODE_UNIFORM         = "uniform"          # C5 — uniform, bilgi taşımayan

ROLE_MODES = (ROLE_MODE_LEARNED, ROLE_MODE_ZEROED, ROLE_MODE_FIXED_DISTINCT,
              ROLE_MODE_FIXED_IDENTICAL, ROLE_MODE_SHUFFLED,
              ROLE_MODE_UNIFORM)

#: C2 — ÖNCEDEN KAYITLI seçim.  06'daki en sık ortak atama:
#: agent0=support, agent1=pursuit (825/2000 = %41.25).  Sonuca bakılarak
#: seçilmemiştir; tek bir ön-kayıtlı çift, çıktısı ne olursa olsun raporlanır.
ROLE_FIXED_DISTINCT = (ROLE_SUPPORT, ROLE_PURSUIT)

#: C3 — ÖNCEDEN KAYITLI seçim.  İki ajana da pursuit: skorla en güçlü
#: ilişkili boyut, yani ayırt edici olmayan bir sinyalin yardımcı olabileceği
#: en elverişli tek seçim.  Null çıkarsa sonuç bu yüzden daha güçlüdür.
ROLE_FIXED_IDENTICAL = ROLE_PURSUIT


def _onehot(idx: int, dim: int = 4) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[idx] = 1.0
    return v


def apply_role_mode(roles: List[np.ndarray], mode: str,
                    scale: float = 1.0) -> List[np.ndarray]:
    """Atayıcının çıktısına çıkarım-anı müdahalesi uygula.

    `roles` atayıcının ürettiği vektörlerdir ve bu fonksiyon ÇAĞRILMADAN ÖNCE
    kaydedilmiş olmalıdır — böylece "atayıcı ne üretti" ile "aktör ne gördü"
    ayrı ölçülebilir.

    scale : sonucu çarpan katsayı (T2.10 C6).  Eğitim dağılımı L2 normu 1.0
            olan one-hot vektörlerden oluşur; scale < 1 büyüklüğü düşürerek
            biçimi korur ve böylece büyüklük ile bilgiyi ayırır.
    """
    if mode == ROLE_MODE_LEARNED:
        out = roles
    elif mode == ROLE_MODE_ZEROED:
        out = [np.zeros_like(r) for r in roles]
    elif mode == ROLE_MODE_FIXED_DISTINCT:
        out = [_onehot(ROLE_FIXED_DISTINCT[i % len(ROLE_FIXED_DISTINCT)]).astype(r.dtype)
               for i, r in enumerate(roles)]
    elif mode == ROLE_MODE_FIXED_IDENTICAL:
        out = [_onehot(ROLE_FIXED_IDENTICAL).astype(r.dtype) for r in roles]
    elif mode == ROLE_MODE_UNIFORM:
        # C5 — her bileşen eşit ve sıfırdan farklı, atayıcı çıktısıyla aynı
        # şekilde normalize (toplam 1).  C1 kadar az rol bilgisi taşır:
        # rol tercihi yok, ajan ayrımı yok — ama sıfır değil.
        # NOT: ‖uniform‖₂ = 0.5 = ‖0.5·one-hot‖₂, yani C6@ε=0.5 ile
        # L2-eşleşmiştir; o çift büyüklüğü sabit tutup biçimi izole eder.
        n = len(ROLE_NAMES)
        out = [np.full(n, 1.0 / n, dtype=r.dtype) for r in roles]
    elif mode == ROLE_MODE_SHUFFLED:
        # İki ajan → permütasyon = takas.  Rol vektörlerinin marjinal
        # dağılımı korunur, ajanlara atanması bozulur.
        if len(roles) == 2:
            out = [roles[1].copy(), roles[0].copy()]
        else:
            out = list(reversed([r.copy() for r in roles]))
    else:
        raise ValueError(f"Bilinmeyen role_mode: {mode!r}. Geçerli: {ROLE_MODES}")

    if scale != 1.0:
        out = [(r * scale).astype(r.dtype) for r in out]
    return out


def _rule_based_fire(
    blue_ids: List[str],
    red_ids: List[str],
    obs_dict: dict,
    states: dict,
    obs_dim: int = _BASE_OBS_DIM,
) -> Dict[str, float]:
    """WEZ koşulu sağlanıyorsa fire=1.0, aksi hâlde 0.0.

    FACMAC actor'ı 4D ctrl çıkarır; 5. boyut bu fonksiyon tarafından eklenir.
    """
    fire: Dict[str, float] = {}
    for aid in blue_ids:
        obs        = obs_dict.get(aid, np.zeros(obs_dim, dtype=np.float32))
        fire[aid]  = 0.0
        if obs[16] > 1e-4:          # cooldown_norm > 0 → henüz ateşleyemez
            continue
        b = states.get(aid)
        if b is None or b[STATE_ALIVE] < 0.5 or b[STATE_AMMO] < 0.5:
            continue
        b_pos = b[[STATE_X, STATE_Y, STATE_H]]
        for rid in red_ids:
            r = states.get(rid)
            if r is None or r[STATE_ALIVE] < 0.5:
                continue
            r_pos = r[[STATE_X, STATE_Y, STATE_H]]
            dist  = distance_3d(b_pos, r_pos)
            if dist < _WEZ_RANGE_MIN or dist > _WEZ_RANGE_MAX:
                continue
            if abs(antenna_train_angle(b_pos, r_pos, b[STATE_PSI])) <= _WEZ_ANGLE_MAX:
                fire[aid] = 1.0
                break
    return fire


# ===========================================================================
# Algorithm detection
# ===========================================================================

def detect_algorithm(ckpt: dict) -> str:
    """Checkpoint anahtarlarından algoritma etiketini otomatik tespit et.

    Döndürülen etiketler: ``mappo``, ``mappo_gat``, ``mappo_gat_om``,
    ``qmix``, ``facmac``, ``facmac_omr``.

    Ayırt edici anahtarlar:
        QMIX:          ``agent_net`` (üst düzey)
        FACMACActorOM: actor'da ``fc1_base.weight``
        FACMACActor:   actor'da ``ctrl_head.weight`` (mean_head/fire_head yok)
        GATMAPPOActor+OM: actor'da ``fc1_intent.weight``
        GATMAPPOActor:    actor'da ``fc1_old.weight``
        MAPPOActor:    actor'da ``mean_head.weight`` + ``fire_head.weight``
    """
    if "agent_net" in ckpt:                           # QMIX
        return "qmix"
    actor_sd = ckpt.get("actor", {})
    if "fc1_base.weight" in actor_sd:                 # FACMACActorOM (FACMAC+OM+R)
        return "facmac_omr"
    if "ctrl_head.weight" in actor_sd:                # FACMACActor (pure FACMAC)
        return "facmac"
    if "fc1_intent.weight" in actor_sd:               # GATMAPPOActor + OM + Role
        return "mappo_gat_om"
    if "fc1_old.weight" in actor_sd:                  # GATMAPPOActor (GAT-only)
        return "mappo_gat"
    # Fallback: MAPPOActor (mean_head + fire_head)
    return "mappo"


# ===========================================================================
# Policy wrappers (one per algorithm family)
# ===========================================================================

class _BasePolicy(ABC):
    """Tüm politika sarmalayıcıları için temel arayüz."""

    algorithm: str
    obs_dim: int

    # Yapısal ablasyon (T2.5 Step 4c / eval 08–09): True ise rol vektörü
    # aktöre sıfır olarak verilir.  Yalnızca OM+Role modellerinde anlamlı;
    # diğer politikalarda etkisizdir.
    #
    # T2.9: ablate_role artık role_mode'un kısayolu — geriye dönük uyumluluk
    # için korunuyor (ablate_role=True  ⇔  role_mode="zeroed").
    ablate_role: bool = False
    role_mode: str = ROLE_MODE_LEARNED
    role_scale: float = 1.0

    def reset(self) -> None:
        """Episode başında çağrılır (OM geçmişi vb. sıfırlanır)."""

    @abstractmethod
    def get_blue_actions(self, obs_dict: dict, deterministic: bool) -> dict:
        """obs_dict → {agent_id: 5D action array}."""


# ---------------------------------------------------------------------------

class _LegacySequentialRoleAssigner:
    """
    Eski sequential cent_role formatı için uyumluluk katmanı (MAPPO ep44000).

    mlp0: [intent(6) + team_state(6)] = 12D → role0(4)
    mlp1: [intent(6) + team_state(6) + role0(4)] = 16D → role1(4)
    """

    def __init__(self) -> None:
        import torch.nn as nn
        self.mlp0 = nn.Sequential(nn.Linear(12, 64), nn.ReLU(), nn.Linear(64, 4))
        self.mlp1 = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 4))

    def to(self, device):
        self.mlp0 = self.mlp0.to(device)
        self.mlp1 = self.mlp1.to(device)
        return self

    def eval(self):
        self.mlp0.eval()
        self.mlp1.eval()
        return self

    def load_state_dict(self, sd, strict=True):
        import torch.nn as nn
        self.mlp0.load_state_dict(
            {k[len("mlp0."):]: v for k, v in sd.items() if k.startswith("mlp0.")},
            strict=strict,
        )
        self.mlp1.load_state_dict(
            {k[len("mlp1."):]: v for k, v in sd.items() if k.startswith("mlp1.")},
            strict=strict,
        )

    @property
    def parameters(self):
        import itertools
        return itertools.chain(self.mlp0.parameters(), self.mlp1.parameters())

    def assign(self, x):
        import torch
        import torch.nn.functional as F

        with torch.no_grad():
            logit0 = self.mlp0(x)
            role0  = F.one_hot(logit0.argmax(dim=-1), 4).float()
            logit1 = self.mlp1(torch.cat([x, role0], dim=-1))
            role1  = F.one_hot(logit1.argmax(dim=-1), 4).float()
        return role0, role1


# ---------------------------------------------------------------------------

class _MAPPOPolicy(_BasePolicy):
    algorithm = "mappo"
    obs_dim   = _BASE_OBS_DIM

    def __init__(self, ckpt: dict, config: dict, env: DogfightEnv, device,
                 own_ids=None, enemy_ids=None):
        from aircombat_bench.algorithms.mappo import MAPPOActor
        hidden      = int(config.get("training", {}).get("hidden_dim", 256))
        self._actor = MAPPOActor(_BASE_OBS_DIM, env.action_dim, hidden=hidden).to(device)
        self._actor.load_state_dict(ckpt["actor"], strict=False)
        self._actor.eval()
        self._own_ids = list(own_ids) if own_ids is not None else env.blue_ids
        self._device   = device
        self._MAPPOActor = MAPPOActor

    def get_blue_actions(self, obs_dict: dict, deterministic: bool) -> dict:
        import torch
        actions = {}
        with torch.no_grad():
            for aid in self._own_ids:
                o     = obs_dict.get(aid, np.zeros(self.obs_dim, np.float32))
                obs_t = torch.FloatTensor(o[:self.obs_dim]).unsqueeze(0).to(self._device)
                raw, _ = self._actor.act(obs_t, deterministic=deterministic)
                actions[aid] = self._MAPPOActor.squash(raw.squeeze(0)).cpu().numpy()
        return actions


# ---------------------------------------------------------------------------

class _MAPPOGATPolicy(_BasePolicy):
    """MAPPO+GAT ve MAPPO+GAT+OM+Role için ortak politika sarmalayıcısı."""

    def __init__(self, algorithm: str, ckpt: dict, config: dict,
                 env: DogfightEnv, device, own_ids=None, enemy_ids=None):
        import torch
        from aircombat_bench.algorithms.mappo import GATMAPPOActor, MAPPOActor
        from aircombat_bench.models.gat_comm import GATComm

        self.algorithm  = algorithm
        self.obs_dim    = 78
        self._own_ids = list(own_ids) if own_ids is not None else env.blue_ids
        self._enemy_ids = list(enemy_ids) if enemy_ids is not None else env.red_ids
        self._device    = device
        self._env       = env
        self._MAPPOActor = MAPPOActor

        comm_cfg   = config.get("communication", {})
        self._gat_node_dim  = int(comm_cfg.get("node_dim", 17))
        self._wez_range_max = float(config.get("weapons", {}).get("wez_range_max", 8000.0))

        hidden = int(config.get("training", {}).get("hidden_dim", 256))
        self._actor = GATMAPPOActor(
            old_obs_dim=50, ext_dim=18, intent_dim=6, role_dim=4,
            action_dim=env.action_dim, hidden=hidden,
        ).to(device)
        self._actor.load_state_dict(ckpt["actor"], strict=False)
        self._actor.eval()

        self._gat_comm = GATComm().to(device)
        if "gat_comm" in ckpt:
            self._gat_comm.load_state_dict(ckpt["gat_comm"])
        self._gat_comm.eval()

        self._use_om    = (algorithm == "mappo_gat_om")
        self._cent_om   = None
        self._cent_role = None
        self._enemy_hist = None

        if self._use_om:
            from aircombat_bench.models.om_net import (
                CentralizedOpponentModel, CentralizedRoleAssigner, EnemyHistoryBuffer,
            )
            self._cent_om = CentralizedOpponentModel().to(device)
            if "cent_om" in ckpt:
                self._cent_om.load_state_dict(ckpt["cent_om"])
            self._cent_om.eval()

            role_sd = ckpt.get("cent_role", {})
            if "mlp0.0.weight" in role_sd:
                self._cent_role = _LegacySequentialRoleAssigner().to(device)
            else:
                self._cent_role = CentralizedRoleAssigner().to(device)
            self._cent_role.load_state_dict(role_sd, strict=False)
            self._cent_role.eval()

            self._enemy_hist = EnemyHistoryBuffer()

        # Intent/role takibi (aggregate için)
        self._intent_steps: list = []
        self._role_steps: list   = [[], []]

    def reset(self) -> None:
        if self._enemy_hist is not None:
            self._enemy_hist.reset()

    def _build_edge_feats(self) -> np.ndarray:
        """Mavi takım kenar özellik matrisi (N×N×3)."""
        states = self._env.get_all_states()
        N      = len(self._own_ids)
        edge   = np.zeros((N, N, 3), dtype=np.float32)
        for i, ai in enumerate(self._own_ids):
            for j, aj in enumerate(self._own_ids):
                if i == j:
                    continue
                si = states.get(ai)
                sj = states.get(aj)
                if si is None or sj is None:
                    continue
                pi   = si[[STATE_X, STATE_Y, STATE_H]]
                pj   = sj[[STATE_X, STATE_Y, STATE_H]]
                dist = distance_3d(pi, pj)
                bear = bearing_angle(pi, pj)
                ts   = 0.0
                for eid in self._enemy_ids:
                    es = states.get(eid)
                    if es is not None and es[STATE_ALIVE] > 0.5:
                        d  = distance_3d(pj, es[[STATE_X, STATE_Y, STATE_H]])
                        ts = max(ts, float(np.clip(
                            1.0 - d / (self._wez_range_max + 1e-9), 0.0, 1.0
                        )))
                edge[i, j] = [
                    float(np.clip(dist / (self._env.map_size + 1e-9), 0.0, 1.0)),
                    float(np.clip(wrap_to_pi(bear) / np.pi, -1.0, 1.0)),
                    ts,
                ]
        return edge

    def _extend_obs(self, obs_dict: dict) -> dict:
        """50D base obs → 78D GAT(+OM) obs."""
        import torch

        states     = self._env.get_all_states()
        ego_list   = []
        alive_list = []
        for aid in self._own_ids:
            obs = obs_dict.get(aid, np.zeros(_BASE_OBS_DIM, np.float32))
            ego_list.append(obs[:self._gat_node_dim])
            s = states.get(aid)
            alive_list.append(float(s[STATE_ALIVE]) if s is not None else 0.0)

        messages = self._gat_comm.compute_messages(
            ego_list, self._build_edge_feats(), alive_list, self._device,
        )

        if self._use_om:
            from aircombat_bench.models.om_net import build_team_state
            base_arr = np.stack([
                obs_dict.get(aid, np.zeros(_BASE_OBS_DIM, np.float32))[:_BASE_OBS_DIM]
                for aid in self._own_ids
            ], axis=0)
            hist_960 = self._enemy_hist.update(base_arr)
            with torch.no_grad():
                hist_t    = torch.from_numpy(hist_960).unsqueeze(0).to(self._device)
                intent_np = self._cent_om.intent_flat(hist_t).squeeze(0).cpu().numpy()
            ts_np = build_team_state([states.get(bid) for bid in self._own_ids])
            with torch.no_grad():
                x_role = torch.from_numpy(
                    np.concatenate([intent_np, ts_np]).astype(np.float32)
                ).unsqueeze(0).to(self._device)
                role_0, role_1 = self._cent_role.assign(x_role)
            roles = [role_0.squeeze(0).cpu().numpy(), role_1.squeeze(0).cpu().numpy()]
            # Rol dağılımı ablasyondan ÖNCE kaydedilir — atayıcının ne
            # ürettiği ölçülmeye devam etsin, yalnızca aktöre giden vektör
            # sıfırlansın.
            self._intent_steps.append(intent_np.copy())
            self._role_steps[0].append(roles[0].copy())
            self._role_steps[1].append(roles[1].copy())
            roles = apply_role_mode(roles, self.role_mode, self.role_scale)
        else:
            intent_np = np.zeros(6,  dtype=np.float32)
            roles     = [np.zeros(4, dtype=np.float32)] * len(self._own_ids)

        extended = {}
        for i, aid in enumerate(self._own_ids):
            base  = obs_dict.get(aid, np.zeros(_BASE_OBS_DIM, np.float32))[:_BASE_OBS_DIM]
            ext18 = np.concatenate([np.zeros(2, dtype=np.float32), messages[i]])
            extended[aid] = np.concatenate([base, ext18, intent_np, roles[i]])   # 78D
        for aid in self._enemy_ids:
            extended[aid] = obs_dict.get(aid, np.zeros(_BASE_OBS_DIM, np.float32))
        return extended

    def get_blue_actions(self, obs_dict: dict, deterministic: bool) -> dict:
        import torch
        obs_ext = self._extend_obs(obs_dict)
        actions = {}
        with torch.no_grad():
            for aid in self._own_ids:
                o     = obs_ext.get(aid, np.zeros(self.obs_dim, np.float32))
                obs_t = torch.FloatTensor(o[:self.obs_dim]).unsqueeze(0).to(self._device)
                raw, _ = self._actor.act(obs_t, deterministic=deterministic)
                actions[aid] = self._MAPPOActor.squash(raw.squeeze(0)).cpu().numpy()
        return actions

    def get_intent_role_summary(self) -> dict:
        """Episode döngüsü sonrası intent/role istatistiklerini topla."""
        if not self._intent_steps:
            return {}
        out: dict = {
            "intent_mean": np.mean(self._intent_steps, axis=0).round(4).tolist(),
            "role0_mean":  np.mean(self._role_steps[0], axis=0).round(4).tolist(),
        }
        if self._role_steps[1]:
            out["role1_mean"] = np.mean(self._role_steps[1], axis=0).round(4).tolist()
        return out


# ---------------------------------------------------------------------------

class _QMIXPolicy(_BasePolicy):
    algorithm = "qmix"
    obs_dim   = _BASE_OBS_DIM

    def __init__(self, ckpt: dict, config: dict, env: DogfightEnv, device,
                 own_ids=None, enemy_ids=None):
        from aircombat_bench.models.qmix_net import AgentQNetwork, ActionMapper
        qcfg           = config.get("qmix", {})
        obs_dim        = int(qcfg.get("obs_dim",       50))
        n_actions      = int(qcfg.get("n_actions",    162))
        agent_hidden   = int(qcfg.get("agent_hidden", 128))
        self.obs_dim   = obs_dim
        self._agent_net = AgentQNetwork(obs_dim, n_actions, agent_hidden).to(device)
        self._agent_net.load_state_dict(ckpt["agent_net"])
        self._agent_net.eval()
        self._mapper    = ActionMapper()
        self._own_ids = list(own_ids) if own_ids is not None else env.blue_ids
        self._device    = device

    def get_blue_actions(self, obs_dict: dict, deterministic: bool) -> dict:
        import torch
        actions = {}
        with torch.no_grad():
            for aid in self._own_ids:
                o   = obs_dict.get(aid, np.zeros(self.obs_dim, np.float32))
                qt  = torch.FloatTensor(o[:self.obs_dim]).unsqueeze(0).to(self._device)
                idx = int(self._agent_net(qt).argmax(dim=-1).item())
                actions[aid] = self._mapper(idx)
        return actions


# ---------------------------------------------------------------------------

class _FACMACPolicy(_BasePolicy):
    algorithm = "facmac"
    obs_dim   = _BASE_OBS_DIM

    def __init__(self, ckpt: dict, config: dict, env: DogfightEnv, device,
                 own_ids=None, enemy_ids=None):
        from aircombat_bench.models.facmac_net import FACMACActor
        fcfg          = config.get("facmac", {})
        self._actor   = FACMACActor(
            obs_dim=_BASE_OBS_DIM, hidden=int(fcfg.get("hidden", 256))
        ).to(device)
        self._actor.load_state_dict(ckpt["actor"], strict=False)
        self._actor.eval()
        self._own_ids = list(own_ids) if own_ids is not None else env.blue_ids
        self._enemy_ids = list(enemy_ids) if enemy_ids is not None else env.red_ids
        self._device   = device
        self._env      = env

    def get_blue_actions(self, obs_dict: dict, deterministic: bool) -> dict:
        import torch
        states = self._env.get_all_states()
        ctrl: dict = {}
        with torch.no_grad():
            for aid in self._own_ids:
                o     = obs_dict.get(aid, np.zeros(self.obs_dim, np.float32))
                obs_t = torch.FloatTensor(o[:self.obs_dim]).unsqueeze(0).to(self._device)
                c, _  = self._actor.act(obs_t, deterministic=deterministic)
                ctrl[aid] = c.squeeze(0).cpu().numpy()   # 4D
        fire = _rule_based_fire(self._own_ids, self._enemy_ids, obs_dict, states)
        return {aid: np.append(ctrl[aid], fire[aid]) for aid in self._own_ids}


# ---------------------------------------------------------------------------

class _FACMACOMPolicy(_BasePolicy):
    """FACMAC+OM+REINFORCE (FACMACActorOM + CentralizedOpponentModel)."""

    algorithm = "facmac_omr"
    obs_dim   = 60

    def __init__(self, ckpt: dict, config: dict, env: DogfightEnv, device,
                 own_ids=None, enemy_ids=None):
        import torch
        from aircombat_bench.models.facmac_net import FACMACActorOM
        from aircombat_bench.models.om_net import (
            CentralizedOpponentModel, CentralizedRoleAssigner, EnemyHistoryBuffer,
        )

        fcfg       = config.get("facmac", {})
        self._actor = FACMACActorOM(
            base_obs_dim=_BASE_OBS_DIM, hidden=int(fcfg.get("hidden", 256))
        ).to(device)
        self._actor.load_state_dict(ckpt["actor"], strict=False)
        self._actor.eval()

        self._cent_om = CentralizedOpponentModel().to(device)
        if "cent_om" in ckpt:
            self._cent_om.load_state_dict(ckpt["cent_om"])
        self._cent_om.eval()

        self._cent_role = CentralizedRoleAssigner().to(device)
        if "cent_role" in ckpt:
            self._cent_role.load_state_dict(ckpt["cent_role"], strict=False)
        self._cent_role.eval()

        self._enemy_hist = EnemyHistoryBuffer()
        self._own_ids = list(own_ids) if own_ids is not None else env.blue_ids
        self._enemy_ids = list(enemy_ids) if enemy_ids is not None else env.red_ids
        self._device     = device
        self._env        = env

        # Intent / role takibi
        self._intent_steps: list  = []
        self._role_steps: list    = [[], []]

    def reset(self) -> None:
        self._enemy_hist.reset()

    def get_blue_actions(self, obs_dict: dict, deterministic: bool) -> dict:
        import torch
        from aircombat_bench.models.om_net import build_team_state

        states   = self._env.get_all_states()
        base_arr = np.stack([
            obs_dict.get(aid, np.zeros(_BASE_OBS_DIM, np.float32))[:_BASE_OBS_DIM]
            for aid in self._own_ids
        ], axis=0)
        hist_960 = self._enemy_hist.update(base_arr)
        with torch.no_grad():
            hist_t    = torch.from_numpy(hist_960).unsqueeze(0).to(self._device)
            intent_np = self._cent_om.intent_flat(hist_t).squeeze(0).cpu().numpy()
        ts_np = build_team_state([states.get(bid) for bid in self._own_ids])
        with torch.no_grad():
            x_role = torch.from_numpy(
                np.concatenate([intent_np, ts_np]).astype(np.float32)
            ).unsqueeze(0).to(self._device)
            role_0, role_1 = self._cent_role.assign(x_role)
        roles = [role_0.squeeze(0).cpu().numpy(), role_1.squeeze(0).cpu().numpy()]

        # Takip — ablasyondan ÖNCE kaydedilir (atayıcının çıktısı ölçülmeye
        # devam etsin; yalnızca aktöre giden vektör sıfırlansın).
        self._intent_steps.append(intent_np.copy())
        self._role_steps[0].append(roles[0].copy())
        self._role_steps[1].append(roles[1].copy())

        roles = apply_role_mode(roles, self.role_mode, self.role_scale)

        obs_60d = {
            aid: np.concatenate([
                obs_dict.get(aid, np.zeros(_BASE_OBS_DIM, np.float32))[:_BASE_OBS_DIM],
                intent_np, roles[i],
            ])
            for i, aid in enumerate(self._own_ids)
        }
        ctrl: dict = {}
        with torch.no_grad():
            for aid in self._own_ids:
                o     = obs_60d[aid]
                obs_t = torch.FloatTensor(o[:self.obs_dim]).unsqueeze(0).to(self._device)
                c, _  = self._actor.act(obs_t, deterministic=deterministic)
                ctrl[aid] = c.squeeze(0).cpu().numpy()   # 4D

        fire = _rule_based_fire(self._own_ids, self._enemy_ids, obs_dict, states)
        return {aid: np.append(ctrl[aid], fire[aid]) for aid in self._own_ids}

    def get_intent_role_summary(self) -> dict:
        """Episode döngüsü sonrası intent/role istatistiklerini topla."""
        if not self._intent_steps:
            return {}
        return {
            "intent_mean": np.mean(self._intent_steps, axis=0).round(4).tolist(),
            "role0_mean":  np.mean(self._role_steps[0], axis=0).round(4).tolist(),
            "role1_mean":  np.mean(self._role_steps[1], axis=0).round(4).tolist(),
        }


# ===========================================================================
# Policy factory
# ===========================================================================

def _make_policy(algorithm: str, ckpt: dict, config: dict,
                 env: DogfightEnv, device,
                 own_ids=None, enemy_ids=None,
                 ablate_role: bool = False,
                 role_mode: str = ROLE_MODE_LEARNED,
                 role_scale: float = 1.0) -> _BasePolicy:
    """Algoritma etiketine göre uygun politika sarmalayıcısını oluştur.

    own_ids / enemy_ids verilmezse politika Blue takımını sürer (varsayılan).
    Red tarafını süren bir politika için ikisini de takas ederek geçir
    (agent-vs-agent değerlendirmesi, eval 07).
    """
    kw = {"own_ids": own_ids, "enemy_ids": enemy_ids}
    if algorithm == "mappo":
        pol = _MAPPOPolicy(ckpt, config, env, device, **kw)
    elif algorithm in ("mappo_gat", "mappo_gat_om"):
        pol = _MAPPOGATPolicy(algorithm, ckpt, config, env, device, **kw)
    elif algorithm == "qmix":
        pol = _QMIXPolicy(ckpt, config, env, device, **kw)
    elif algorithm == "facmac":
        pol = _FACMACPolicy(ckpt, config, env, device, **kw)
    elif algorithm == "facmac_omr":
        pol = _FACMACOMPolicy(ckpt, config, env, device, **kw)
    else:
        pol = None
    if pol is not None:
        # ablate_role, role_mode'un geriye dönük uyumlu kısayolu.
        if ablate_role and role_mode == ROLE_MODE_LEARNED:
            role_mode = ROLE_MODE_ZEROED
        if role_mode not in ROLE_MODES:
            raise ValueError(
                f"Bilinmeyen role_mode: {role_mode!r}. Geçerli: {ROLE_MODES}")
        pol.role_mode = role_mode
        pol.role_scale = float(role_scale)
        pol.ablate_role = (role_mode == ROLE_MODE_ZEROED)
        return pol
    raise ValueError(
        f"Bilinmeyen algoritma: {algorithm!r}. "
        "Geçerli değerler: mappo, mappo_gat, mappo_gat_om, qmix, facmac, facmac_omr"
    )


# ===========================================================================
# Main evaluation loop
# ===========================================================================

def run_eval(
    config: dict,
    checkpoint_path: str,
    n_episodes: int,
    seed: int,
    device_str: str = "auto",
    deterministic: bool = True,
    scenario: str = DEFAULT_SCENARIO,
    verbose: bool = True,
    seeding: str = "legacy",
    ablate_role: bool = False,
    role_mode: str = ROLE_MODE_LEARNED,
    role_scale: float = 1.0,
    opponent_checkpoint: Optional[str] = None,
) -> dict:
    """Schema v1 değerlendirme çıktısını döndür.

    Parameters
    ----------
    config          : config.yaml içeriği (dict)
    checkpoint_path : .pt dosyasının mutlak veya göreceli yolu
    n_episodes      : episode sayısı
    seed            : başlangıç seed'i
    device_str      : "auto" | "cuda" | "cpu"
    deterministic   : True → greedy aksiyon, False → stokastik örnekleme
    scenario        : senaryo etiketi (yalnızca meta-veri; şimdilik 2v2_normal)
    verbose         : her 50 episodeda konsol özeti
    seeding         : "legacy"      → döngü öncesi tek env.seed(seed); tüm
                                      episode'lar tek sürekli RNG akışından
                                      çekilir.  Yayımlanan tüm baseline
                                      sayılarının ve eski eval script'lerinin
                                      şeması budur — varsayılan.
                      "per_episode" → her episode öncesi env.seed(seed + ep).
                                      Tek episode'u izole tekrar oynatmayı
                                      mümkün kılar (T2.3); legacy ile
                                      bit-exact eşleşmez, farklı RNG dizisi.
    ablate_role     : True → rol vektörü aktöre sıfır olarak verilir (yapısal
                      ablasyon, eval 08–09).  Rol atayıcının çıktısı yine
                      kaydedilir; yalnızca aktöre giden vektör sıfırlanır.
                      OM+Role olmayan modellerde etkisizdir.
    opponent_checkpoint : Verilirse Red takımı heuristic yerine bu
                      checkpoint'in politikasıyla oynar (agent-vs-agent,
                      eval 07).  None → `heuristic-v1`.

    Returns
    -------
    Schema v1 dict — JSON'a serileştirilebilir.
    """
    import torch

    pin_threads()                     # kanonik: tek thread (T2.5 Step 4b)
    device = _resolve_device(device_str)
    rt     = runtime_block(device)

    # Provenance koşunun BAŞINDA yakalanır, çıktı yazılırken değil.  Uzun bir
    # koşu sırasında depoda yapılan (ve o koşuyu etkilemeyen) bir düzenleme,
    # temiz bir ağaçta yürütülmüş sonucu kirli damgalamamalıdır.
    git_commit_at_start = _git_commit()
    git_dirty_files_at_start = _git_dirty_files()
    git_dirty_at_start = (True if git_dirty_files_at_start is None
                          else bool(git_dirty_files_at_start))

    # ── Meta-veri ─────────────────────────────────────────────────────────
    ckpt_abs = str(Path(checkpoint_path).resolve())
    ckpt     = torch.load(ckpt_abs, map_location=device, weights_only=False)

    algorithm = detect_algorithm(ckpt)
    ep_ckpt   = ckpt.get("episode", ckpt.get("global_episode", "?"))
    step_ckpt = ckpt.get("total_steps", ckpt.get("global_step", "?"))
    sha256    = _sha256(ckpt_abs)

    print(f"[Runner] Checkpoint  : {checkpoint_path}")
    print(f"         Algorithm   : {algorithm}  "
          f"(ep={ep_ckpt}, step={step_ckpt:,})" if isinstance(step_ckpt, int)
          else f"         Algorithm   : {algorithm}  (ep={ep_ckpt}, step={step_ckpt})")
    print(f"         SHA-256     : {sha256[:16]}…")
    print(f"         Device      : {device}  |  ENV_VERSION={ENV_VERSION}")
    print(f"         Runtime     : torch {rt['torch_version']}  "
          f"{rt['blas_backend']}  threads={rt['num_threads']}"
          + (f"  gpu={rt['gpu_model']}" if rt["device"] == "cuda" else ""))
    print(f"         Episodes    : {n_episodes}  seed={seed}  "
          f"deterministic={deterministic}  seeding={seeding}")

    # ── Ortam ─────────────────────────────────────────────────────────────
    env = DogfightEnv(config)
    env.set_curriculum_phase(4)   # 2v2 normal spawn (eval her zaman Faz-3)
    blue_ids = env.blue_ids
    red_ids  = env.red_ids
    all_ids  = blue_ids + red_ids
    team_map = {a: ("blue" if "blue" in a else "red") for a in all_ids}

    # ── Politika ve rakip ─────────────────────────────────────────────────
    policy = _make_policy(algorithm, ckpt, config, env, device,
                          ablate_role=ablate_role, role_mode=role_mode,
                          role_scale=role_scale)

    # Rakip: varsayılan heuristic-v1; opponent_checkpoint verilirse Red
    # takımını o checkpoint'in politikası sürer (agent-vs-agent, eval 07).
    opp_policy = None
    opp_label  = HEURISTIC_VERSION
    opp_meta: Optional[Dict[str, Any]] = None
    if opponent_checkpoint is None:
        opp = MultiHeuristicPolicy(config, all_ids, team_map)
    else:
        opp = None
        opp_abs  = str(Path(opponent_checkpoint).resolve())
        opp_ckpt = torch.load(opp_abs, map_location=device, weights_only=False)
        opp_algo = detect_algorithm(opp_ckpt)
        opp_sha  = _sha256(opp_abs)
        # Red tarafı: own/enemy takas edilir.
        opp_policy = _make_policy(opp_algo, opp_ckpt, config, env, device,
                                  own_ids=red_ids, enemy_ids=blue_ids)
        opp_label = f"checkpoint:{opp_algo}"
        opp_meta  = {"path": _repo_relative_path(opponent_checkpoint),
                     "sha256": opp_sha, "algorithm": opp_algo}
        print(f"         Opponent    : {opp_algo}  (Red)  "
              f"SHA-256 {opp_sha[:16]}…")

    # ── Determinizm: tüm rasgele kaynaklarını seed'le ────────────────────
    #
    # T2.3 uyumlu seeding:
    #   • Python random  — seed ile deterministik
    #   • numpy global   — seed ile deterministik (env.rng bağımsız, aşağıda)
    #   • PyTorch CPU    — seed ile deterministik
    #   • CUDA           — seed ile deterministik (best-effort; cublas ops için
    #                       torch.backends.cudnn.deterministic=True gerekebilir)
    #   • env.rng        — env.seed(seed) ile ayrıca seed'lenir
    #
    # Seeding şeması (seeding parametresi):
    #   "legacy"      → aşağıdaki tek env.seed(seed); episode döngüsü içinde
    #                   yeniden seed'lenmez.  Eski eval script'lerinin
    #                   (evaluation/eval.py, eval_facmac.py, eval_qmix.py —
    #                   hepsi 25050fb'den bugüne) şeması ile aynıdır.
    #   "per_episode" → ek olarak her episode başında env.seed(seed + ep).
    #                   Tek episode izole tekrar oynatılabilir; legacy ile
    #                   bit-exact eşleşmez (farklı RNG dizisi).
    #
    # CUDA non-determinizm notu:
    #   CUDA atomik operasyonlar (gather, scatter) GPU'da geri dönüşsüz olarak
    #   non-deterministik olabilir.  Tam bit-exactness için:
    #     torch.use_deterministic_algorithms(True)
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False
    #   CPU-only inference bu sorundan etkilenmez.
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    try:
        import torch as _torch
        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    if seeding not in ("legacy", "per_episode"):
        raise ValueError(
            f"seeding must be 'legacy' or 'per_episode', got {seeding!r}"
        )

    env.seed(seed)   # env.rng (np.random.default_rng) ayrıca seed'lenir

    wins = 0; losses = 0; draws = 0
    kills_list    : List[int]   = []
    lengths_list  : List[int]   = []
    survival_list : List[float] = []
    oob_list      : List[float] = []
    ep_records    : List[dict]  = []

    for ep in range(n_episodes):
        if seeding == "per_episode":
            env.seed(seed + ep)
        obs_dict = env.reset()
        if opp is not None:
            opp.reset()
        else:
            opp_policy.reset()
        policy.reset()

        ep_kills = 0
        ep_oob   = 0
        # Ajan bazlı kill — kill imbalance (T2.6.2) bunsuz hesaplanamaz.
        ep_kills_by_agent = {aid: 0 for aid in blue_ids}
        done     = {"__all__": False}
        info_dict: dict = {}

        while not done["__all__"]:
            blue_actions = policy.get_blue_actions(obs_dict, deterministic)
            states       = env.get_all_states()
            if opp is not None:
                opp_raw     = opp.act(states)
                red_actions = {k: v for k, v in opp_raw.items() if k in set(red_ids)}
            else:
                # Checkpoint rakip: kendi own_ids'i red_ids olduğu için
                # get_blue_actions doğrudan Red aksiyonlarını döndürür.
                red_actions = opp_policy.get_blue_actions(obs_dict, deterministic)

            obs_dict, rew_dict, done, info_dict = env.step(
                {**blue_actions, **red_actions}
            )

            for aid in blue_ids:
                if info_dict[aid].get("r_kill", 0.0) > 0.5:
                    ep_kills += 1
                    ep_kills_by_agent[aid] += 1

        # ── Episode sonucu ─────────────────────────────────────────────
        winner = done.get("winner", "draw")
        if winner == BLUE:
            wins   += 1
            outcome = "win"
        elif winner == RED:
            losses += 1
            outcome = "loss"
        else:
            draws  += 1
            outcome = "draw"

        n_survived = len(env.blue_alive)
        survival_list.append(n_survived / max(len(blue_ids), 1))

        for aid in blue_ids:
            ep_summary = info_dict.get(aid, {}).get("episode", {})
            if ep_summary.get("mean/r_penalty", 0.0) < -0.5:
                ep_oob += 1
        oob_list.append(ep_oob / max(len(blue_ids), 1))

        ep_len = env.get_step_count()
        kills_list.append(ep_kills)
        lengths_list.append(ep_len)

        ep_records.append({
            "idx":     ep,
            # T2.6.4 — ep_seed yalnızca gerçekten kullanıldığında yazılır.
            #
            # legacy modunda env döngü öncesi bir kez seed'lenir; episode i'nin
            # başlangıç koşulu kendisinden önceki i episode'un RNG çekilişlerine
            # bağlıdır.  `seed + i` değeri o episode'u İZOLE olarak yeniden
            # üretmez (ölçüldü: env.seed(47) ile legacy akışındaki 5. episode
            # spawn'ı 20 km'den fazla ayrışıyor).  Bu yüzden legacy'de ep_seed
            # None'dır ve episode'u tanımlayan şey akış içindeki konumudur.
            #
            # İzole tek-episode replay için seeding="per_episode" kullanılmalı.
            "ep_seed": (seed + ep) if seeding == "per_episode" else None,
            "stream_position": ep,
            "outcome": outcome,
            "kills":   ep_kills,
            # Ajan bazlı kill dağılımı — rol uzmanlaşması ablasyonunda
            # (T2.6.2) taşıyıcı nicelik olan kill imbalance buradan türetilir.
            "kills_by_agent": [ep_kills_by_agent[a] for a in blue_ids],
            "kill_imbalance": (
                abs(ep_kills_by_agent[blue_ids[0]] - ep_kills_by_agent[blue_ids[1]])
                if len(blue_ids) == 2 else 0
            ),
            "steps":   ep_len,
            "survived": n_survived,
        })

        if verbose and (ep + 1) % 50 == 0:
            print(
                f"  ep {ep+1:4d}/{n_episodes} | "
                f"W={wins/(ep+1):.3f}  K/ep={np.mean(kills_list):.2f}  "
                f"D={draws/(ep+1):.3f}",
                flush=True,
            )

    # ── Toplu metrikler ────────────────────────────────────────────────────
    n         = n_episodes
    win_rate  = wins / n
    ci_lo, ci_hi = wilson_ci(wins, n)
    kill_mean = float(np.mean(kills_list))
    kill_std  = float(np.std(kills_list))

    # one_kill_stall_rate: tam bir kill ama kazanamadı
    one_kill_stall = sum(
        1 for r in ep_records if r["kills"] == 1 and r["outcome"] != "win"
    )

    metrics = {
        "win_rate":             round(win_rate, 6),
        "win_ci95":             [round(ci_lo, 6), round(ci_hi, 6)],
        "loss_rate":            round(losses / n, 6),
        "draw_rate":            round(draws  / n, 6),
        "kill_per_ep":          round(kill_mean, 6),
        "kill_std":             round(kill_std,  6),
        "mean_ep_len":          round(float(np.mean(lengths_list)),  2),
        "survival_rate":        round(float(np.mean(survival_list)), 6),
        "oob_rate":             round(float(np.mean(oob_list)),      6),
        "one_kill_stall_rate":  round(one_kill_stall / n,            6),
    }

    # Intent / role istatistikleri (yalnızca OM modelleri)
    if hasattr(policy, "get_intent_role_summary"):
        metrics.update(policy.get_intent_role_summary())

    # ── Schema v1 çıktısı ──────────────────────────────────────────────────
    output = {
        "schema_version":  SCHEMA_VERSION,
        "package_version": _PKG_VERSION,
        "env_version":     ENV_VERSION,
        # Koşunun BAŞINDAKİ git durumu (yukarıya bakınız).
        "git_commit":      git_commit_at_start,
        "git_dirty":       git_dirty_at_start,
        # Hangi dosyaların kirli olduğu — değerlendirme yolunu etkileyen bir
        # kirlilik ile yalnızca dokümantasyon kirliliğini ayırt etmek için.
        "git_dirty_files": (git_dirty_files_at_start or [])[:50],
        "timestamp_utc":   datetime.now(timezone.utc).isoformat(),
        "scenario":        scenario,
        "opponent":        opp_label,
        **({"opponent_checkpoint": opp_meta} if opp_meta else {}),
        # T2.7.3 — ortam yan-simetrik DEĞİL: _critical_recovery_blue yalnızca
        # blue_ids'e uygulanır (dogfight_env.py:308).  Scripted red kendi
        # kurtarmasına sahiptir, ama checkpoint sürülen red hiçbirini almaz.
        # Bu yüzden tek yönlü bir agent-vs-agent sonucu YANLI'dır ve tek
        # başına raporlanmamalıdır.
        **({"side_bias": {
            "single_sided": True,
            "evaluated_side": "blue",
            "mirror_required": True,
            "reason": ("ortam yan-simetrik değil: OOB/zemin kurtarma yalnızca "
                       "blue tarafına uygulanır (dogfight_env.py:308); "
                       "checkpoint rakip kurtarma almaz"),
            "note": ("Bu dosya tek yönlü bir sonuçtur.  Raporlanabilir sayı, "
                     "aynalanmış koşuyla birlikte iki-yönlü ortalamadır — "
                     "aircombat_bench.eval.compare.combine_head_to_head()"),
        }} if opp_meta else {}),
        "checkpoint": {
            "path":   _repo_relative_path(checkpoint_path),
            "sha256": sha256,
        },
        "algorithm":      algorithm,
        "n_episodes":     n_episodes,
        "seed":           seed,
        "seeding":        seeding,
        "deterministic":  deterministic,
        "ablate_role":    ablate_role or (role_mode == ROLE_MODE_ZEROED),
        "role_mode":      getattr(policy, "role_mode", role_mode),
        "role_scale":     float(role_scale),
        "runtime":        rt,
        "metrics":        metrics,
        "wins":           wins,
        "losses":         losses,
        "draws":          draws,
        "episodes":       ep_records,
    }

    return output


# ===========================================================================
# Schema validation helper
# ===========================================================================

SCHEMA_V1_REQUIRED_KEYS = {
    "schema_version", "package_version", "env_version", "git_commit",
    "git_dirty", "scenario", "opponent", "checkpoint", "algorithm",
    "n_episodes", "seed", "deterministic", "metrics", "episodes",
    "runtime",
}

# `runtime` bloğunda her zaman bulunması gereken alanlar.  CUDA koşularında
# ayrıca gpu_model ve cuda_version bulunur.
SCHEMA_V1_RUNTIME_REQUIRED = {
    "device", "torch_version", "blas_backend", "num_threads",
    "cpu_model", "os", "python_version",
}

SCHEMA_V1_METRICS_REQUIRED = {
    "win_rate", "win_ci95", "loss_rate", "draw_rate",
    "kill_per_ep", "kill_std", "mean_ep_len",
    "survival_rate", "oob_rate", "one_kill_stall_rate",
}


def validate_schema_v1(data: dict) -> Tuple[bool, List[str]]:
    """Schema v1 uyumluluğunu doğrula. (ok, errors) döndürür."""
    errors: List[str] = []
    missing_top = SCHEMA_V1_REQUIRED_KEYS - set(data)
    if missing_top:
        errors.append(f"Eksik üst-düzey anahtarlar: {sorted(missing_top)}")
    metrics = data.get("metrics", {})
    missing_metrics = SCHEMA_V1_METRICS_REQUIRED - set(metrics)
    if missing_metrics:
        errors.append(f"Eksik metrics anahtarları: {sorted(missing_metrics)}")
    # T2.6.3 — taşınabilir yol zorunlu
    ck_path = data.get("checkpoint", {}).get("path", "")
    if not path_is_portable(ck_path):
        errors.append(
            f"checkpoint.path taşınabilir değil (mutlak yol ya da ev dizini): "
            f"{ck_path!r}"
        )
    opp_ck = data.get("opponent_checkpoint")
    if isinstance(opp_ck, dict) and not path_is_portable(opp_ck.get("path", "")):
        errors.append(
            f"opponent_checkpoint.path taşınabilir değil: {opp_ck.get('path')!r}"
        )

    # T2.6.6 — kirli ağaçta üretilmiş kanonik dosya reddedilir
    if data.get("git_dirty") is True:
        errors.append(
            "git_dirty=true — commit'lenmemiş çalışma ağacında üretilmiş; "
            "kaydedilen commit bu çıktıyı yeniden üretemez"
        )

    # T2.6.4 — legacy seeding'de ep_seed izole replay vaat edemez
    eps_ = data.get("episodes", [])
    if isinstance(eps_, list) and eps_:
        e0 = eps_[0]
        if "stream_position" not in e0:
            errors.append("episode kayıtlarında stream_position eksik")
        if data.get("seeding") == "legacy" and e0.get("ep_seed") is not None:
            errors.append(
                "seeding=legacy iken ep_seed None olmalı — legacy akışında "
                "seed+idx tek bir episode'u izole yeniden üretmez"
            )

    rt = data.get("runtime", {})
    if not isinstance(rt, dict):
        errors.append("runtime bir sözlük olmalı")
    else:
        missing_rt = SCHEMA_V1_RUNTIME_REQUIRED - set(rt)
        if missing_rt:
            errors.append(f"Eksik runtime anahtarları: {sorted(missing_rt)}")
        if rt.get("device") == "cuda":
            missing_gpu = {"gpu_model", "cuda_version"} - set(rt)
            if missing_gpu:
                errors.append(
                    f"CUDA koşusunda eksik runtime anahtarları: {sorted(missing_gpu)}"
                )
    if data.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"schema_version beklenen {SCHEMA_VERSION!r}, "
            f"bulunan {data.get('schema_version')!r}"
        )
    eps = data.get("episodes", [])
    if not isinstance(eps, list):
        errors.append("episodes bir liste olmalı")
    elif eps:
        ep0 = eps[0]
        for k in ("idx", "outcome", "kills", "steps"):
            if k not in ep0:
                errors.append(f"episodes[0] eksik: {k!r}")
    return len(errors) == 0, errors


# ===========================================================================
# Print summary
# ===========================================================================

def print_summary(output: dict) -> None:
    """Schema v1 çıktısını konsola yaz."""
    m  = output["metrics"]
    ci = m["win_ci95"]
    print("\n" + "=" * 62)
    print(f"  EVAL — {output['algorithm'].upper()}  |  "
          f"{output['n_episodes']} ep  seed={output['seed']}  "
          f"det={output['deterministic']}")
    print(f"  ENV_VERSION={output['env_version']}  |  "
          f"pkg={output['package_version']}  |  commit={output['git_commit']}")
    print("=" * 62)
    print(f"  Win  rate       : {m['win_rate']:.1%}  "
          f"[{ci[0]:.1%} – {ci[1]:.1%}]  (95% CI, Wilson)")
    print(f"  Loss rate       : {m['loss_rate']:.1%}")
    print(f"  Draw rate       : {m['draw_rate']:.1%}")
    print(f"  Kill / ep       : {m['kill_per_ep']:.3f} ± {m['kill_std']:.3f}")
    print(f"  1-kill stall    : {m['one_kill_stall_rate']:.1%}")
    print(f"  Mean ep len     : {m['mean_ep_len']:.0f} steps")
    print(f"  Survival rate   : {m['survival_rate']:.1%}")
    print(f"  OOB rate        : {m['oob_rate']:.1%}")
    print(f"  W/D/L           : {output['wins']}/{output['draws']}/{output['losses']}")
    if "intent_mean" in m:
        im = m["intent_mean"]
        r0 = m.get("role0_mean", [])
        r1 = m.get("role1_mean", [])
        print()
        print(f"  Intent (red_0)  : agg={im[0]:.3f} def={im[1]:.3f} eva={im[2]:.3f}")
        print(f"  Intent (red_1)  : agg={im[3]:.3f} def={im[4]:.3f} eva={im[5]:.3f}")
        if r0:
            print(f"  Role (blue_0)   : sniper={r0[0]:.3f} pursuit={r0[1]:.3f} "
                  f"def={r0[2]:.3f} sup={r0[3]:.3f}")
        if r1:
            print(f"  Role (blue_1)   : sniper={r1[0]:.3f} pursuit={r1[1]:.3f} "
                  f"def={r1[2]:.3f} sup={r1[3]:.3f}")
    print("=" * 62)


# ===========================================================================
# CLI entry point
# ===========================================================================

def main(argv: Optional[List[str]] = None) -> None:
    """aircombat-eval komut satırı giriş noktası."""
    p = argparse.ArgumentParser(
        description="aircombat-bench unified evaluation runner (schema v1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint", required=True,
        help="Değerlendirilecek .pt dosyası yolu",
    )
    p.add_argument(
        "--config", default="configs/config.yaml",
        help="YAML config dosyası yolu",
    )
    p.add_argument("--episodes",     type=int,  default=500,   help="Episode sayısı")
    p.add_argument("--seed",         type=int,  default=42,    help="Başlangıç seed'i")
    p.add_argument(
        "--ablate-role", action="store_true", dest="ablate_role",
        help="Rol vektörünü aktöre sıfır olarak ver (yapısal ablasyon, "
             "eval 08–09). OM+Role olmayan modellerde etkisizdir.",
    )
    p.add_argument(
        "--role-mode", dest="role_mode", choices=list(ROLE_MODES),
        default=ROLE_MODE_LEARNED,
        help="Rol vektörüne çıkarım-anı müdahalesi (T2.9): learned=atayıcı "
             "çıktısı, zeroed=sıfır, fixed_distinct=ajana bağlı sabit, "
             "fixed_identical=ikisine aynı sabit, shuffled=ajanlar arası takas.",
    )
    p.add_argument(
        "--role-scale", dest="role_scale", type=float, default=1.0,
        help="Rol vektörünü bu katsayıyla çarp (T2.10 C6). Eğitim dağılımı "
             "L2 normu 1.0 olan one-hot'lardan oluşur; <1 biçimi korur, "
             "büyüklüğü düşürür.",
    )
    p.add_argument(
        "--opponent-checkpoint", dest="opponent_checkpoint", default=None,
        help="Red takımını heuristic yerine bu checkpoint sürsün "
             "(agent-vs-agent, eval 07).",
    )
    p.add_argument(
        "--seeding", choices=["legacy", "per_episode"], default="legacy",
        help="legacy: döngü öncesi tek env.seed(seed) — yayımlanan sayıların "
             "ve eski eval script'lerinin şeması (varsayılan). "
             "per_episode: her episode öncesi env.seed(seed+ep) — episode "
             "replay için; legacy ile bit-exact eşleşmez.",
    )
    p.add_argument("--device",       default="auto",           help="cuda | cpu | auto")
    p.add_argument(
        "--output",
        help="JSON çıktısı için dosya yolu (varsayılan: logs/eval_<algorithm>_<n>ep.json)",
    )
    p.add_argument(
        "--stochastic", action="store_true",
        help="Stokastik aksiyon örneklemesi (varsayılan: deterministic)",
    )
    p.add_argument(
        "--scenario", default=DEFAULT_SCENARIO,
        help="Senaryo etiketi (meta-veri)",
    )
    args = p.parse_args(argv)

    cfg_path = ROOT / args.config
    with open(cfg_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    output = run_eval(
        config        = config,
        checkpoint_path = str(ROOT / args.checkpoint),
        n_episodes    = args.episodes,
        seed          = args.seed,
        device_str    = args.device,
        deterministic = not args.stochastic,
        scenario      = args.scenario,
        verbose       = True,
        seeding       = args.seeding,
        ablate_role   = args.ablate_role,
        role_mode     = args.role_mode,
        role_scale    = args.role_scale,
        opponent_checkpoint = (
            str(ROOT / args.opponent_checkpoint) if args.opponent_checkpoint else None
        ),
    )

    print_summary(output)

    # Validate schema
    ok, errs = validate_schema_v1(output)
    if not ok:
        print(f"\n⚠️  Schema doğrulama hatası:\n  " + "\n  ".join(errs), file=sys.stderr)

    # Kaydet
    if args.output:
        out_path = Path(args.output)
    else:
        algo = output["algorithm"]
        out_path = ROOT / "logs" / f"eval_{algo}_{args.episodes}ep.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[Runner] Kaydedildi: {out_path}")


# ---------------------------------------------------------------------------
# Backward-compat re-exports  (test_eval.py ve legacy eval/ scripts için)
# ---------------------------------------------------------------------------
try:
    from evaluation.eval import (   # noqa: F401
        Evaluator,
        eval_heuristic_baseline,
        PHASE1_CRITERIA,
    )
except Exception:
    pass   # evaluation/ yoksa sessizce atla — test_eval.py bunu halleder


if __name__ == "__main__":
    main()

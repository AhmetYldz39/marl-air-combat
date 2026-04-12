"""
opponent_pool.py
================
Fictitious self-play için geçmiş Blue checkpoint'lerinden oluşan rakip pool.

Her episode başında pool'dan rastgele bir checkpoint seçilir ve Red takımı
için policy olarak kullanılır. Pool boşsa heuristic fallback devreye girer.

Referans: Heinrich & Silver (2015) "Fictitious Self-Play in Extensive-Form Games"

Kullanım:
    pool = OpponentPool(config, red_ids, obs_dim, action_dim, device, fallback)
    pool.add_checkpoint("checkpoints/pool_actor_ep200.pt")
    pool.reset()          # episode başı — checkpoint seç ve yükle
    acts = pool.act(obs_dict, state_dict)   # rollout içi

Bu dosya değişirse etkilenen dosyalar:
    - training/train_mappo.py
    - tests/test_opponent_pool.py
"""

import numpy as np
import torch
from collections import deque
from pathlib import Path


class OpponentPool:
    """
    Fictitious self-play için checkpoint tabanlı rakip pool.

    Pool'daki checkpoint'ler Blue ajanının geçmiş ağırlıklarıdır.
    Red takımı bu ağırlıkları kendi perspektifinden (ego=red, düşman=blue)
    üretilmiş normalize obs ile kullanır — obs_dict bu perspektifi zaten
    içerir (DogfightEnv._build_obs_dict tüm ajanlar için üretir).

    Pool boşken heuristic fallback kullanılır; yeni checkpoint eklendikçe
    self-play oranı artar.

    Parameters
    ----------
    config        : global config dict
    red_ids       : Red takımı ajan ID listesi  (["red_0", "red_1"])
    obs_dim       : tek ajan observation boyutu
    action_dim    : aksiyon boyutu (5)
    device        : torch.device
    fallback      : MultiHeuristicPolicy — pool boşken kullanılır
    max_pool_size : ring buffer kapasitesi (eski checkpoint'ler silinir)
    """

    def __init__(
        self,
        config: dict,
        red_ids: list,
        obs_dim: int,
        action_dim: int,
        device,
        fallback,
        max_pool_size: int = 20,
    ):
        self.config        = config
        self.red_ids       = red_ids
        self.obs_dim       = obs_dim
        self.action_dim    = action_dim
        self.device        = device
        self.fallback      = fallback
        self.max_pool_size = max_pool_size

        # Checkpoint path pool (ring buffer)
        self._pool: deque = deque(maxlen=max_pool_size)

        # Aktif episode için yüklü actor
        self._current_actor = None  # MAPPOActor | None
        self._use_fallback  = True  # başlangıçta fallback aktif

    # -----------------------------------------------------------------------
    # Pool Yönetimi
    # -----------------------------------------------------------------------

    def add_checkpoint(self, path: str) -> None:
        """Yeni checkpoint path'ini pool'a ekle."""
        self._pool.append(str(path))

    @property
    def size(self) -> int:
        """Mevcut pool boyutu."""
        return len(self._pool)

    # -----------------------------------------------------------------------
    # Episode Başı: Checkpoint Seçimi
    # -----------------------------------------------------------------------

    def reset(self) -> None:
        """
        Episode başında çağrılır.

        Pool boşsa heuristic fallback aktif edilir.
        Değilse pool'dan rastgele bir checkpoint yükler.
        """
        self.fallback.reset()

        if len(self._pool) == 0:
            self._use_fallback  = True
            self._current_actor = None
            return

        path = str(np.random.choice(list(self._pool)))
        self._use_fallback = False

        try:
            self._current_actor = self._load_actor(path)
        except Exception as e:
            print(
                f"[OpponentPool] Checkpoint yüklenemedi "
                f"({Path(path).name}): {e} → heuristic fallback"
            )
            self._use_fallback  = True
            self._current_actor = None

    def _load_actor(self, path: str):
        """Checkpoint dosyasından MAPPOActor yükle."""
        # Lazy import: train_mappo → opponent_pool döngüsel bağımlılığını önler
        from training.train_mappo import MAPPOActor

        ckpt   = torch.load(path, map_location=self.device)
        tr     = self.config["training"]
        hidden = int(tr.get("hidden_dim", 256))
        actor  = MAPPOActor(self.obs_dim, self.action_dim, hidden=hidden)
        actor.load_state_dict(ckpt["actor"])
        actor.to(self.device)
        actor.eval()
        return actor

    # -----------------------------------------------------------------------
    # Aksiyon Üretimi
    # -----------------------------------------------------------------------

    def act(self, obs_dict: dict, state_dict: dict = None) -> dict:
        """
        Red takımı için aksiyon üret.

        Fallback aktifse heuristic policy kullanılır.
        Değilse yüklü actor ile neural policy çalışır.

        Parameters
        ----------
        obs_dict   : normalize edilmiş obs (tüm ajanlar — Red obs'u da içerir)
        state_dict : ham state dict (fallback için gerekli)

        Returns
        -------
        dict[str, np.ndarray]  — her red_id için ACTION_DIM aksiyon
        """
        if self._use_fallback or self._current_actor is None:
            if state_dict is None:
                raise ValueError(
                    "OpponentPool: fallback için state_dict zorunlu."
                )
            return self.fallback.act(state_dict)

        # Neural policy: Red ajanları obs_dict'ten alınır
        from training.train_mappo import MAPPOActor

        actions = {}
        with torch.no_grad():
            for rid in self.red_ids:
                obs = obs_dict.get(rid)
                if obs is None or not np.all(np.isfinite(obs)):
                    # NaN/Inf veya eksik obs → güvenli sıfır aksiyon
                    actions[rid] = np.zeros(self.action_dim, dtype=np.float32)
                    continue
                obs_t    = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                raw, _   = self._current_actor.act(obs_t, deterministic=False)
                squashed = MAPPOActor.squash(raw.squeeze(0)).cpu().numpy()
                actions[rid] = squashed

        return actions

"""
opponent_pool.py
================
Fictitious self-play için geçmiş Blue checkpoint'lerinden oluşan adaptif rakip pool.

Her episode başında pool'dan adaptif checkpoint seçilir:
- Blue win_rate 0.4–0.6 arasında tutulacak şekilde zorluk ayarlanır
- win_rate < 0.2  → en zayıf checkpoint (daha kolay rakip)
- win_rate > 0.6  → en güçlü checkpoint (daha zor rakip)
- 0.2–0.6 arası  → win_rate'i 0.5'e çekecek checkpoint seçilir
- Pool boşsa heuristic fallback devreye girer

Her checkpoint için son 20 episode'daki win_rate takip edilir.

Referans: Heinrich & Silver (2015) "Fictitious Self-Play in Extensive-Form Games"

Kullanım:
    pool = OpponentPool(config, red_ids, obs_dim, action_dim, device, fallback)
    pool.add_checkpoint("checkpoints/pool_actor_ep200.pt")
    pool.reset()                        # episode başı — checkpoint seç ve yükle
    acts = pool.act(obs_dict, state_dict)   # rollout içi
    pool.record_outcome(is_win=1)       # episode sonu — win_rate güncelle

Bu dosya değişirse etkilenen dosyalar:
    - training/train_mappo.py
    - tests/test_opponent_pool.py

MİMARİ DEĞİŞİKLİK NOTU:
    Blue actor'e yeni modül eklendiğinde (role_selector, opponent_model, vb.)
    _load_actor() içindeki SKIP_PREFIXES listesini güncelle.
    Red pool actor'ü her zaman base MAPPOActor(obs_dim=50) kullanır;
    ekstra modüller strict=False ile yok sayılır.
"""

import numpy as np
import torch
from collections import deque
from pathlib import Path


class OpponentPool:
    """
    Adaptif fictitious self-play için checkpoint tabanlı rakip pool.

    Pool'daki her checkpoint için son WIN_WINDOW episode'daki win_rate
    tutulur. Episode başında Blue'nun hedef win_rate'ini (0.4–0.6) koruyacak
    şekilde checkpoint seçilir.

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

    WIN_WINDOW    = 20    # her checkpoint için takip edilen episode sayısı
    TARGET_LOW    = 0.4   # hedef win_rate alt sınırı
    TARGET_HIGH   = 0.6   # hedef win_rate üst sınırı
    WEAK_THRESH   = 0.2   # bu altında en zayıf checkpoint'e dön
    STRONG_THRESH = 0.6   # bu üstünde en güçlü checkpoint'i seç
    MIN_MATCHES   = 5     # global_wr hesabına dahil edilmek için minimum episode sayısı

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

        # Checkpoint path pool (ekleme sıralı liste — ring buffer değil,
        # sıralamayı korumak için deque yerine list kullanıyoruz)
        self._pool: list = []

        # Her checkpoint için win geçmişi: path → deque(maxlen=WIN_WINDOW)
        self._win_history: dict = {}

        # Aktif episode için seçilen checkpoint path ve actor
        self._current_path     = None
        self._current_actor    = None
        self._current_obs_dim  = self.obs_dim  # yüklü actor'ün obs boyutu
        self._use_fallback     = True

    # -----------------------------------------------------------------------
    # Pool Yönetimi
    # -----------------------------------------------------------------------

    def add_checkpoint(self, path: str) -> None:
        """Yeni checkpoint path'ini pool'a ekle. Max kapasiteyi aşarsa en eskiyi çıkar."""
        path = str(path)
        if path not in self._win_history:
            self._win_history[path] = deque(maxlen=self.WIN_WINDOW)
        if path not in self._pool:
            self._pool.append(path)
        # Ring buffer: max kapasiteyi aşarsa en eski checkpoint'i çıkar
        if len(self._pool) > self.max_pool_size:
            removed = self._pool.pop(0)
            self._win_history.pop(removed, None)

    @property
    def size(self) -> int:
        """Mevcut pool boyutu."""
        return len(self._pool)

    def win_rates(self) -> dict:
        """Her checkpoint için mevcut win_rate'i döndür. {path: float}"""
        result = {}
        for path in self._pool:
            hist = self._win_history.get(path, deque())
            result[path] = float(np.mean(hist)) if hist else 0.0
        return result

    # -----------------------------------------------------------------------
    # Episode Başı: Adaptif Checkpoint Seçimi
    # -----------------------------------------------------------------------

    def reset(self) -> None:
        """
        Episode başında çağrılır.

        Pool boşsa heuristic fallback aktif edilir.
        Değilse son win_rate'e göre uygun checkpoint seçilir.
        """
        self.fallback.reset()

        if len(self._pool) == 0:
            self._use_fallback  = True
            self._current_path  = None
            self._current_actor = None
            return

        path = self._select_checkpoint()
        self._current_path = path
        self._use_fallback = False

        try:
            actor, obs_dim = self._load_actor(path)
            self._current_actor   = actor
            self._current_obs_dim = obs_dim
        except Exception as e:
            print(
                f"[OpponentPool] Checkpoint yüklenemedi "
                f"({Path(path).name}): {e} → heuristic fallback"
            )
            self._use_fallback    = True
            self._current_path    = None
            self._current_actor   = None
            self._current_obs_dim = self.obs_dim

    def _select_checkpoint(self) -> str:
        """
        Adaptif checkpoint seçimi.

        - Genel win_rate < WEAK_THRESH   → en zayıf (en eski) checkpoint
        - Genel win_rate > STRONG_THRESH → en güçlü (en yeni) checkpoint
        - Arada                          → win_rate'i 0.5'e çekecek şekilde
                                           weighted random seçim
        """
        rates  = self.win_rates()
        paths  = list(rates.keys())
        values = [rates[p] for p in paths]

        # Genel win_rate: yalnızca MIN_MATCHES eşiğini geçen checkpoint'lerin geçmişi
        all_hist = []
        for h in self._win_history.values():
            if len(h) >= self.MIN_MATCHES:
                all_hist.extend(h)
        global_wr = float(np.mean(all_hist)) if all_hist else 0.5

        if global_wr < self.WEAK_THRESH:
            # En zayıf = en eski checkpoint (en düşük index)
            return paths[0]

        if global_wr > self.STRONG_THRESH:
            # En güçlü = en yeni checkpoint (en yüksek index)
            return paths[-1]

        # Hedef bölge: win_rate'i 0.5'e çekecek checkpoint seç.
        # Mevcut win_rate yüksekse daha güçlü rakip (yüksek index tercih edilir),
        # düşükse daha zayıf rakip.
        # Ağırlık: rakibin win_rate'i 0.5'e ne kadar yakın olursa o kadar yüksek
        # (karşılıklı — Blue win_rate yüksekse güçlü rakip ister, tersi de geçerli)
        if global_wr >= 0.5:
            # Daha güçlü rakip iste: yüksek indeksleri tercih et
            weights = np.array([i + 1 for i in range(len(paths))], dtype=float)
        else:
            # Daha zayıf rakip iste: düşük indeksleri tercih et
            weights = np.array([len(paths) - i for i in range(len(paths))], dtype=float)

        weights /= weights.sum()
        idx = np.random.choice(len(paths), p=weights)
        return paths[idx]

    # Blue actor'e yeni modül eklenince buraya prefix ekle
    _SKIP_PREFIXES = ("fc1_new", "role_selector", "gat_comm", "opponent_model")

    def _load_actor(self, path: str):
        """
        Checkpoint dosyasından MAPPOActor yükle.

        GAT (Faz-2+) snapshot'larında:
          - fc1_old/fc2 keyleri net.0/net.2'ye remap edilir
          - fc1_new, role_selector, gat_comm, opponent_model keyleri atlanır
          - strict=False: kalan beklenmedik keyler de crash yapmaz

        Red opponent base 50D policy üzerinden çalışır; Blue'ya özgü
        modüller (role_selector vb.) yüklenmez.

        Returns
        -------
        (actor, obs_dim) : yüklü actor ve girdi obs boyutu
        """
        from training.train_mappo import MAPPOActor

        ckpt      = torch.load(path, map_location=self.device, weights_only=False)
        tr        = self.config["training"]
        hidden    = int(tr.get("hidden_dim", 256))
        actor_sd  = ckpt["actor"]

        # GAT checkpoint tespiti: fc1_old.weight varsa remap gerekli
        if "fc1_old.weight" in actor_sd:
            key_map = {
                "fc1_old.weight": "net.0.weight",
                "fc1_old.bias":   "net.0.bias",
                "fc2.weight":     "net.2.weight",
                "fc2.bias":       "net.2.bias",
            }
            remapped = {}
            for k, v in actor_sd.items():
                if any(k.startswith(p) for p in self._SKIP_PREFIXES):
                    continue          # Blue'ya özgü modülleri atla
                remapped[key_map.get(k, k)] = v
            actor_sd  = remapped
            obs_dim   = 50            # base obs (GAT/role msg olmadan)
        else:
            obs_dim   = self.obs_dim

        actor = MAPPOActor(obs_dim, self.action_dim, hidden=hidden)
        actor.load_state_dict(actor_sd, strict=False)   # beklenmedik key → crash yok
        actor.to(self.device)
        actor.eval()
        return actor, obs_dim

    # -----------------------------------------------------------------------
    # Episode Sonu: Sonuç Kaydı
    # -----------------------------------------------------------------------

    def record_outcome(self, is_win: int) -> None:
        """
        Episode sonunda çağrılır. Seçilen checkpoint'in win geçmişini günceller.

        Parameters
        ----------
        is_win : 1 = Blue kazandı, 0 = kaybetti veya beraberlik
        """
        if self._current_path is not None and not self._use_fallback:
            self._win_history[self._current_path].append(float(is_win))

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------

    def log_win_rate_distribution(self) -> str:
        """
        Her checkpoint için win_rate dağılımını string olarak döndürür.
        Her 100 episode'da train_mappo.py tarafından çağrılır.
        """
        if not self._pool:
            return "[OpponentPool] Pool boş — heuristic fallback aktif"

        lines = ["[OpponentPool] Checkpoint win_rate dağılımı:"]
        rates = self.win_rates()
        for path, wr in rates.items():
            name  = Path(path).name
            hist  = self._win_history.get(path, deque())
            n     = len(hist)
            bar   = "#" * int(wr * 20)
            lines.append(f"  {name:<30} wr={wr:.2f} (n={n:>2}) |{bar:<20}|")

        all_hist = []
        for h in self._win_history.values():
            all_hist.extend(h)
        global_wr = float(np.mean(all_hist)) if all_hist else float("nan")
        lines.append(f"  {'GENEL':30} wr={global_wr:.2f}")
        return "\n".join(lines)

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

        from training.train_mappo import MAPPOActor

        actions = {}
        with torch.no_grad():
            for rid in self.red_ids:
                obs = obs_dict.get(rid)
                if obs is None or not np.all(np.isfinite(obs)):
                    actions[rid] = np.zeros(self.action_dim, dtype=np.float32)
                    continue
                obs_t    = torch.FloatTensor(obs[:self._current_obs_dim]).unsqueeze(0).to(self.device)
                raw, _   = self._current_actor.act(obs_t, deterministic=False)
                squashed = MAPPOActor.squash(raw.squeeze(0)).cpu().numpy()
                actions[rid] = squashed

        return actions

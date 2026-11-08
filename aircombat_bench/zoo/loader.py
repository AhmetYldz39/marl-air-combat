"""
aircombat_bench/zoo/loader.py
==============================
Model zoo: load pre-trained baselines and run them.

Quickstart (five lines)
-----------------------
>>> from aircombat_bench.envs import make_env
>>> from aircombat_bench.zoo import load_baseline
>>> policy = load_baseline("facmac_omr")
>>> env = make_env("2v2_normal")
>>> obs, _ = env.reset(seed=42)
>>> while env.agents:
...     actions = policy.act(obs, env.agents)
...     obs, rew, term, trunc, info = env.step(actions)

Supported baseline IDs
-----------------------
  mappo          MAPPO baseline                    Win=46.4%
  mappo_gat      MAPPO + GAT communication         Win=69.8%
  mappo_gat_om   MAPPO + GAT + OM + Role           Win=70.8%
  qmix           QMIX (162 discrete actions)       Win=65.0%
  facmac         Pure FACMAC                       Win=82.0%
  facmac_omr     FACMAC + OM + REINFORCE  (best)   Win=84.2%

All figures: 500 ep, seed=42, deterministic, ENV_VERSION=1.0.0.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Paket kök dizini — loader.py'nin iki üstünde
_PKG_DIR  = Path(__file__).resolve().parent.parent     # aircombat_bench/
_REPO_DIR = _PKG_DIR.parent                            # repo kökü


def _load_manifest() -> dict:
    """zoo/manifest.json dosyasını yükle ve döndür."""
    mf_path = _PKG_DIR / "zoo" / "manifest.json"
    if not mf_path.exists():
        raise FileNotFoundError(f"manifest.json bulunamadı: {mf_path}")
    with open(mf_path, encoding="utf-8") as fh:
        return json.load(fh)


def manifest() -> dict:
    """Model zoo manifest bilgisini döndür.

    Returns
    -------
    dict
        schema_version, env_version, baselines listesi.

    Examples
    --------
    >>> from aircombat_bench.zoo import manifest
    >>> mf = manifest()
    >>> [b["id"] for b in mf["baselines"]]
    ['mappo', 'mappo_gat', 'mappo_gat_om', 'qmix', 'facmac', 'facmac_omr']
    """
    return _load_manifest()


def _verify_sha256(path: Path, expected: str) -> None:
    """Dosyanın SHA-256 hash'ini doğrula; uyuşmazsa ValueError."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected:
        raise ValueError(
            f"SHA-256 doğrulama hatası: {path.name}\n"
            f"  beklenen : {expected}\n"
            f"  hesaplanan: {actual}\n"
            "Checkpoint dosyası bozuk veya yanlış dosya kullanılıyor."
        )


class BaselinePolicy:
    """Yüklenmiş bir baseline checkpoint'i için çıkarım sarmalayıcısı.

    act(obs_dict, agents) çağrısı PettingZoo observation dict'ini alır ve
    her mevcut ajan için 5D aksiyon döndürür.

    Parameters
    ----------
    baseline_id : str
        Manifest'teki baseline kimliği (örn. ``"facmac_omr"``).
    algorithm   : str
        Otomatik algılanan algoritma tipi.

    Examples
    --------
    >>> from aircombat_bench.zoo import load_baseline
    >>> policy = load_baseline("facmac_omr")
    >>> policy.baseline_id
    'facmac_omr'
    >>> policy.algorithm
    'facmac_omr'
    """

    def __init__(self, baseline_id: str, algorithm: str, _impl):
        self.baseline_id = baseline_id
        self.algorithm   = algorithm
        self._impl       = _impl   # _BasePolicy benzeri nesne

    def act(
        self,
        obs_dict: Dict[str, np.ndarray],
        agents:   Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """Gözlemlerden aksiyonları hesapla.

        Parameters
        ----------
        obs_dict : {agent_id: obs_array}
            PettingZoo reset() / step() çıktısıyla aynı format.
        agents   : list[str] | None
            Aksiyon üretilecek ajan listesi; None ise obs_dict.keys() kullanılır.

        Returns
        -------
        dict[str, np.ndarray]
            {agent_id: 5D_action_array}  — her ajan için sürekli aksiyon.
        """
        if agents is None:
            agents = list(obs_dict.keys())
        return self._impl.act_batch(obs_dict, agents)

    def __repr__(self) -> str:
        return f"BaselinePolicy(id={self.baseline_id!r}, alg={self.algorithm!r})"


def load_baseline(
    baseline_id: str,
    checkpoint_dir: Optional[str] = None,
    verify_hash: bool = True,
    device: str = "cpu",
) -> "BaselinePolicy":
    """Bir baseline checkpoint'i yükle ve çıkarım için hazır policy döndür.

    Parameters
    ----------
    baseline_id     : str
        Manifest'teki ID: "mappo" | "mappo_gat" | "mappo_gat_om" |
        "qmix" | "facmac" | "facmac_omr".
    checkpoint_dir  : str | None
        Checkpoint dosyalarının arandığı dizin.
        None → repo kökündeki ``checkpoints/`` dizini.
    verify_hash     : bool
        True (varsayılan): SHA-256 doğrulaması yap.
    device          : str
        PyTorch cihazı: "cpu" | "cuda" | "cuda:0".

    Returns
    -------
    BaselinePolicy
        ``policy.act(obs_dict, agents)`` arayüzüne sahip policy nesnesi.

    Raises
    ------
    ValueError
        Bilinmeyen baseline_id veya SHA-256 uyuşmazlığı.
    FileNotFoundError
        Checkpoint dosyası bulunamazsa.

    Examples
    --------
    >>> from aircombat_bench.zoo import load_baseline
    >>> policy = load_baseline("facmac_omr")
    >>> policy.algorithm
    'facmac_omr'
    """
    import torch

    mf = _load_manifest()
    entry = next((b for b in mf["baselines"] if b["id"] == baseline_id), None)
    if entry is None:
        valid = [b["id"] for b in mf["baselines"]]
        raise ValueError(
            f"Bilinmeyen baseline: {baseline_id!r}. "
            f"Geçerliler: {valid}"
        )

    # Checkpoint yolu çözümle
    ckpt_rel  = Path(entry["checkpoint"])          # "checkpoints/foo.pt"
    if checkpoint_dir is not None:
        ckpt_path = Path(checkpoint_dir) / ckpt_rel.name
    else:
        ckpt_path = _REPO_DIR / ckpt_rel

    if not ckpt_path.exists():
        # Paket içi checkpoints/ deneyin (geliştirici kurulumu)
        alt = _PKG_DIR.parent / ckpt_rel
        if alt.exists():
            ckpt_path = alt
        else:
            raise FileNotFoundError(
                f"Checkpoint bulunamadı: {ckpt_path}\n"
                f"Checkpoint'leri indirmek için GitHub Releases sayfasına bakın."
            )

    # SHA-256 doğrula
    if verify_hash:
        _verify_sha256(ckpt_path, entry["sha256"])

    # Checkpoint yükle
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Runner'dan algoritma tespiti ve policy oluştur
    from aircombat_bench.eval.runner import detect_algorithm, _make_policy
    import yaml

    config_path = _REPO_DIR / "configs" / "config.yaml"
    with open(config_path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    algorithm = detect_algorithm(ckpt)

    # Env sadece boyutları almak için kullanılır — kısa ömürlü
    from aircombat_bench.envs.dogfight_env import DogfightEnv
    _tmp_env = DogfightEnv(config)
    _tmp_env.set_curriculum_phase(4)

    raw_policy = _make_policy(algorithm, ckpt, config, _tmp_env, device)

    return BaselinePolicy(
        baseline_id=baseline_id,
        algorithm=algorithm,
        _impl=_ZooPolicy(raw_policy),
    )


# ---------------------------------------------------------------------------
# Dahili sarmalayıcı: runner._BasePolicy → BaselinePolicy._impl
# ---------------------------------------------------------------------------

class _ZooPolicy:
    """runner._BasePolicy nesnesini zoo arayüzüne uyarlar."""

    def __init__(self, raw_policy):
        self._p = raw_policy

    def act_batch(
        self,
        obs_dict: Dict[str, np.ndarray],
        agents:   List[str],
    ) -> Dict[str, np.ndarray]:
        """Batch aksiyon hesapla (deterministik).

        runner._BasePolicy.get_blue_actions(obs_dict, deterministic) kullanır.
        """
        # Yalnızca istenen ajanların obs'ını aktar
        filtered = {a: obs_dict[a] for a in agents if a in obs_dict}
        return self._p.get_blue_actions(filtered, deterministic=True)

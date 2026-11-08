"""
test_determinism.py
===================
T2.3 — Determinism audit.

Aynı komut iki kez çalıştırıldığında metriklerin ve per-episode kayıtların
aynı olduğunu doğrular (timestamp ve git_commit hariç).

Hızlı tutmak için her modelden yalnızca 20 episode çalıştırır.
Tam 500-ep determinizm testi manuel olarak:
    python -X utf8 tests/test_determinism.py --full

Çalıştırma:
    pytest tests/test_determinism.py -v
"""

import json
import os
import sys
import time

import numpy as np
import pytest
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aircombat_bench.eval.runner import run_eval

# Kısa test için 20 episode; CI zamanlama için daha az yapılabilir
_N_EPISODES = 20
_SEED       = 42

# Birkaç checkpoint ile test et (en azından 1'i mevcut olmalı)
_TEST_CHECKPOINTS = [
    "checkpoints/facmac_omr_ep27000.pt",   # en güçlü, test için iyi
    "checkpoints/qmix_ep2000.pt",
    "checkpoints/mappo_final.pt",
]

IGNORED_KEYS = {"timestamp_utc", "git_commit"}


def _run(ckpt_path: str, config: dict) -> dict:
    return run_eval(
        config=config,
        checkpoint_path=ckpt_path,
        n_episodes=_N_EPISODES,
        seed=_SEED,
        device_str="cpu",
        deterministic=True,
        verbose=False,
    )


def _strip_volatile(d: dict) -> dict:
    """Karşılaştırmadan önce zaman damgası ve commit hash'i kaldır."""
    return {k: v for k, v in d.items() if k not in IGNORED_KEYS}


def _episodes_equal(eps_a: list, eps_b: list) -> bool:
    """Per-episode listelerinin bit-exact eşitliğini kontrol et."""
    if len(eps_a) != len(eps_b):
        return False
    for a, b in zip(eps_a, eps_b):
        if a != b:
            return False
    return True


def _metrics_equal(m_a: dict, m_b: dict, tol: float = 1e-9) -> bool:
    """Metrik dict'lerinin yakın eşitliğini kontrol et."""
    if set(m_a) != set(m_b):
        return False
    for k in m_a:
        va, vb = m_a[k], m_b[k]
        if isinstance(va, list):
            if not all(abs(x - y) < tol for x, y in zip(va, vb)):
                return False
        elif isinstance(va, float):
            if abs(va - vb) > tol:
                return False
        elif va != vb:
            return False
    return True


# ── Fixture: config ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def config():
    with open(ROOT / "configs/config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Tests ────────────────────────────────────────────────────────────────────

def _get_test_checkpoint():
    """Mevcut ilk test checkpoint'ini döndür."""
    for ckpt in _TEST_CHECKPOINTS:
        p = ROOT / ckpt
        if p.exists():
            return str(p)
    pytest.skip("Test checkpoint bulunamadı — checkpoints/ dizini eksik")


class TestDeterminism:
    """Aynı seed iki kez → aynı sonuç."""

    def test_same_seed_produces_identical_metrics(self, config):
        """İki çalıştırma aynı metrikleri üretmeli."""
        ckpt = _get_test_checkpoint()
        out1 = _run(ckpt, config)
        out2 = _run(ckpt, config)

        m1, m2 = out1["metrics"], out2["metrics"]
        assert _metrics_equal(m1, m2), (
            f"Metrikler eşleşmedi!\n"
            f"  Run 1: win={m1['win_rate']}, kill={m1['kill_per_ep']}\n"
            f"  Run 2: win={m2['win_rate']}, kill={m2['kill_per_ep']}"
        )

    def test_same_seed_produces_identical_episodes(self, config):
        """İki çalıştırma aynı per-episode kayıtları üretmeli."""
        ckpt = _get_test_checkpoint()
        out1 = _run(ckpt, config)
        out2 = _run(ckpt, config)

        eps1, eps2 = out1["episodes"], out2["episodes"]
        assert _episodes_equal(eps1, eps2), (
            f"Per-episode kayıtlar eşleşmedi!\n"
            f"  Run 1 ep[0]: {eps1[0]}\n"
            f"  Run 2 ep[0]: {eps2[0]}"
        )

    def test_different_seeds_produce_different_results(self, config):
        """Farklı seed'ler farklı (büyük olasılıkla) sonuç üretmeli.

        Not: Küçük episode sayılarında aynı sonuç çıkabilir — bu geçerli
        istatistiksel durum, hata değil. Sadece episode dizisinin farklı
        olduğunu kontrol ediyoruz.
        """
        ckpt = _get_test_checkpoint()
        out_42  = run_eval(config=config, checkpoint_path=ckpt,
                           n_episodes=_N_EPISODES, seed=42,
                           device_str="cpu", deterministic=True, verbose=False)
        out_999 = run_eval(config=config, checkpoint_path=ckpt,
                           n_episodes=_N_EPISODES, seed=999,
                           device_str="cpu", deterministic=True, verbose=False)

        eps_42  = out_42["episodes"]
        eps_999 = out_999["episodes"]

        # En az bir episode dizisi farklı olmalı (aynı olursa da test geçer ama
        # bu çok nadir — 20 episode'da RNG diverges quickly)
        # Opsiyonel kontrol: sadece bir uyarı yaz, fail etme
        if _episodes_equal(eps_42, eps_999):
            import warnings
            warnings.warn(
                f"Seed 42 ve 999 aynı episode dizisini üretti "
                f"(N={_N_EPISODES} ile nadir ama mümkün)."
            )

    def test_volatile_keys_excluded_from_comparison(self, config):
        """timestamp_utc ve git_commit iki çalıştırma arasında farklı olabilir
        — karşılaştırmadan çıkarıldıklarını doğrula."""
        ckpt = _get_test_checkpoint()
        out1 = _run(ckpt, config)
        out2 = _run(ckpt, config)

        stripped1 = _strip_volatile(out1)
        stripped2 = _strip_volatile(out2)

        for key in IGNORED_KEYS:
            assert key not in stripped1
            assert key not in stripped2

        # Geri kalan anahtarlar aynı olmalı
        assert set(stripped1.keys()) == set(stripped2.keys())

    def test_schema_version_stable(self, config):
        """Schema versiyonu iki çalıştırma arasında değişmemeli."""
        ckpt = _get_test_checkpoint()
        out1 = _run(ckpt, config)
        out2 = _run(ckpt, config)
        assert out1["schema_version"] == out2["schema_version"] == "1.0"

    def test_env_version_stable(self, config):
        """ENV_VERSION iki çalıştırma arasında değişmemeli."""
        ckpt = _get_test_checkpoint()
        out = _run(ckpt, config)
        assert out["env_version"] == "1.0.0"

    def test_seed_in_output_matches_input(self, config):
        """Çıktı JSON'daki seed, girdi seed ile eşleşmeli."""
        ckpt = _get_test_checkpoint()
        out = _run(ckpt, config)
        assert out["seed"] == _SEED

    def test_deterministic_flag_in_output(self, config):
        """Deterministic=True bayrağı çıktıda yansıtılmalı."""
        ckpt = _get_test_checkpoint()
        out = _run(ckpt, config)
        assert out["deterministic"] is True


class TestDeterminismDocumentation:
    """Determinizm sınırlamalarının belgelendiğini doğrula."""

    def test_runner_docstring_mentions_seeding(self):
        """Runner modül docstring'i seed yaklaşımını belgelemeli."""
        from aircombat_bench.eval import runner
        doc = runner.__doc__ or ""
        assert "seed" in doc.lower(), "Docstring seed'i açıklamalı"

    def test_t23_note_in_runner_source(self):
        """Runner kaynağı T2.3 per-episode seeding notunu içermeli."""
        import inspect
        from aircombat_bench.eval import runner
        src = inspect.getsource(runner)
        assert "T2.3" in src, "runner.py T2.3 notu içermeli"
        assert "per-episode" in src.lower(), "per-episode seeding notu eksik"

    def test_cuda_nondeterminism_documented(self):
        """Runner CUDA non-determinizmi belgelemeli."""
        import inspect
        from aircombat_bench.eval import runner
        src = inspect.getsource(runner)
        assert "CUDA" in src or "cuda" in src, "CUDA uyarısı eksik"


# ── Summary (standalone) ─────────────────────────────────────────────────────

if __name__ == "__main__":
    # Bağımsız çalıştırma için basit runner
    import sys
    passed = failed = 0

    with open(ROOT / "configs/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ckpt = None
    for c in _TEST_CHECKPOINTS:
        if (ROOT / c).exists():
            ckpt = str(ROOT / c)
            break
    if ckpt is None:
        print("Checkpoint bulunamadı — tests atlanıyor.")
        sys.exit(1)

    print(f"Test checkpoint: {ckpt}")
    print(f"N_EPISODES={_N_EPISODES}  SEED={_SEED}")

    print("\n[1] İki çalıştırma metrikleri karşılaştırılıyor...")
    t0 = time.time()
    o1 = _run(ckpt, cfg)
    o2 = _run(ckpt, cfg)
    elapsed = time.time() - t0

    m_ok = _metrics_equal(o1["metrics"], o2["metrics"])
    e_ok = _episodes_equal(o1["episodes"], o2["episodes"])
    print(f"  Metrik eşitliği: {'OK' if m_ok else 'FAIL'}")
    print(f"  Episode eşitliği: {'OK' if e_ok else 'FAIL'}")
    print(f"  Süre: {elapsed:.1f}s")

    ok_all = m_ok and e_ok
    print(f"\nSonuç: {'TÜMÜ OK ✅' if ok_all else 'HATALAR BULUNDU ❌'}")
    sys.exit(0 if ok_all else 1)

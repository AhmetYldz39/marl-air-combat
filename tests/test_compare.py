"""
tests/test_compare.py — T2.6.1

`aircombat_bench.eval.compare` — eşleştirilmiş model karşılaştırması.

Kapsam:
  • önkoşulların uyarı değil HATA olarak uygulanması
  • McNemar / işaret testi / paired bootstrap doğruluğu
  • kanonik dosyalarda §2 rakamlarının yeniden üretilmesi
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from aircombat_bench.eval.compare import (
    ComparabilityError, compare, compare_files, holm, mcnemar,
    paired_bootstrap_ci, sign_test,
)

ROOT = Path(__file__).resolve().parent.parent
V1 = ROOT / "docs" / "evals" / "v1.0"


# ---------------------------------------------------------------------------
# Yardımcı: yapay eval çıktısı
# ---------------------------------------------------------------------------

def _make(outcomes, kills=None, **over):
    n = len(outcomes)
    kills = kills if kills is not None else [0] * n
    d = {
        "schema_version": "1.0", "env_version": "1.0.0",
        "scenario": "2v2_normal", "opponent": "heuristic-v1",
        "n_episodes": n, "seed": 42, "seeding": "legacy",
        "runtime": {"device": "cpu"},
        "episodes": [
            {"idx": i, "ep_seed": None, "stream_position": i,
             "outcome": o, "kills": k, "steps": 100}
            for i, (o, k) in enumerate(zip(outcomes, kills))
        ],
    }
    d.update(over)
    return d


# ---------------------------------------------------------------------------
# Önkoşullar — uyarı değil hata
# ---------------------------------------------------------------------------

def test_device_mismatch_is_hard_error():
    """CPU dosyası ile CUDA dosyası asla karşılaştırılmamalı."""
    a = _make(["win", "loss"])
    b = _make(["win", "win"])
    b["runtime"] = {"device": "cuda"}
    with pytest.raises(ComparabilityError, match="device"):
        compare(a, b)


@pytest.mark.parametrize("field,value", [
    ("scenario", "1v1"), ("opponent", "other"), ("seed", 7),
    ("seeding", "per_episode"), ("env_version", "2.0.0"),
])
def test_field_mismatch_is_hard_error(field, value):
    a = _make(["win", "loss"])
    b = _make(["win", "win"])
    b[field] = value
    with pytest.raises(ComparabilityError, match=field):
        compare(a, b)


def test_episode_count_mismatch_is_hard_error():
    with pytest.raises(ComparabilityError):
        compare(_make(["win", "loss"]), _make(["win", "loss", "win"]))


def test_stream_position_mismatch_is_hard_error():
    a = _make(["win", "loss"])
    b = _make(["win", "loss"])
    b["episodes"][1]["stream_position"] = 99
    with pytest.raises(ComparabilityError, match="stream_position"):
        compare(a, b)


def test_ep_seed_mismatch_is_hard_error():
    """ep_seed dolu dosyalarda dizi eleman eleman karşılaştırılır."""
    a = _make(["win", "loss"], seeding="per_episode")
    b = _make(["win", "loss"], seeding="per_episode")
    for i, r in enumerate(a["episodes"]):
        r["ep_seed"] = 42 + i
    for i, r in enumerate(b["episodes"]):
        r["ep_seed"] = 1000 + i
    with pytest.raises(ComparabilityError, match="ep_seed"):
        compare(a, b)


def test_comparable_files_do_not_raise():
    compare(_make(["win", "loss"]), _make(["win", "win"]))


# ---------------------------------------------------------------------------
# --device-comparison modu (T2.8.1)
# ---------------------------------------------------------------------------

def _dev_pair():
    """Aynı checkpoint, farklı cihaz — geçerli bir cihaz karşılaştırması."""
    a = _make(["win", "loss", "win"])
    b = _make(["win", "win", "win"])
    for d in (a, b):
        d["checkpoint"] = {"path": "checkpoints/x.pt", "sha256": "abc123"}
    b["runtime"] = {"device": "cuda"}
    return a, b


def test_device_comparison_permits_differing_device():
    a, b = _dev_pair()
    with pytest.raises(ComparabilityError):
        compare(a, b)                                   # opt-in olmadan yasak
    r = compare(a, b, device_comparison=True)           # opt-in ile serbest
    assert r["comparison_type"] == "device"


def test_device_comparison_labels_output():
    a, b = _dev_pair()
    r = compare(a, b, device_comparison=True)
    assert r["devices"] == {"a": "cpu", "b": "cuda"}
    assert "device_comparison_note" in r
    from aircombat_bench.eval.compare import format_report
    assert "CİHAZ KARŞILAŞTIRMASI" in format_report(r)


def test_method_comparison_is_not_labelled_as_device():
    r = compare(_make(["win", "loss"]), _make(["win", "win"]))
    assert r["comparison_type"] == "method"
    assert "devices" not in r


def test_device_comparison_requires_same_checkpoint():
    """Farklı checkpoint → cihaz etkisi model etkisinden ayrıştırılamaz."""
    a, b = _dev_pair()
    b["checkpoint"] = {"path": "checkpoints/y.pt", "sha256": "different"}
    with pytest.raises(ComparabilityError, match="checkpoint"):
        compare(a, b, device_comparison=True)


def test_device_comparison_requires_devices_to_differ():
    a = _make(["win", "loss"])
    b = _make(["win", "win"])
    for d in (a, b):
        d["checkpoint"] = {"path": "checkpoints/x.pt", "sha256": "abc123"}
    with pytest.raises(ComparabilityError, match="cihazlar farklı olmalı"):
        compare(a, b, device_comparison=True)


@pytest.mark.parametrize("field,value", [
    ("scenario", "1v1"), ("seed", 7), ("seeding", "per_episode"),
    ("env_version", "2.0.0"), ("opponent", "other"),
])
def test_device_comparison_still_enforces_other_preconditions(field, value):
    """Cihaz dışındaki her önkoşul bu modda da geçerli kalır."""
    a, b = _dev_pair()
    b[field] = value
    with pytest.raises(ComparabilityError, match=field):
        compare(a, b, device_comparison=True)


def test_device_comparison_still_enforces_episode_identity():
    a, b = _dev_pair()
    b["episodes"][1]["stream_position"] = 99
    with pytest.raises(ComparabilityError, match="stream_position"):
        compare(a, b, device_comparison=True)


# ---------------------------------------------------------------------------
# Çokluk aileleri (T2.10 §3)
# ---------------------------------------------------------------------------

def test_family_recorded_in_output():
    from aircombat_bench.eval.compare import FAMILY_CONFIRMATORY, FAMILY_EXPLORATORY
    a, b = _make(["win", "loss"]), _make(["win", "win"])
    assert compare(a, b, family=FAMILY_CONFIRMATORY)["family"] == "confirmatory"
    assert compare(a, b, family=FAMILY_EXPLORATORY)["family"] == "exploratory"


def test_family_defaults_to_unspecified():
    """Aile belirtilmezse 'unspecified' — Holm ailesine sessizce giremez."""
    r = compare(_make(["win", "loss"]), _make(["win", "win"]))
    assert r["family"] == "unspecified"


def test_unknown_family_rejected():
    from aircombat_bench.eval.compare import FamilyError
    with pytest.raises(FamilyError, match="Bilinmeyen aile"):
        compare(_make(["win"]), _make(["loss"]), family="made_up")


def test_device_comparison_family_defaults_to_device():
    a, b = _dev_pair()
    assert compare(a, b, device_comparison=True)["family"] == "device"


def test_holm_confirmatory_accepts_only_family_members():
    from aircombat_bench.eval.compare import (
        holm_confirmatory, CONFIRMATORY_FAMILY, FamilyError,
    )
    ok = {name: 0.01 for name in CONFIRMATORY_FAMILY}
    adj = holm_confirmatory(ok)
    assert set(adj) == set(CONFIRMATORY_FAMILY)
    with pytest.raises(FamilyError, match="Doğrulayıcı ailede olmayan"):
        holm_confirmatory({"C0 ↔ C1": 0.014})


def test_confirmatory_family_excludes_ablations():
    """Ablasyonlar ve rol serisi doğrulayıcı ailede olmamalı — bu, aynı
    karşılaştırmanın iki Holm hükmü almasını kaynağında engeller."""
    from aircombat_bench.eval.compare import CONFIRMATORY_FAMILY
    joined = " ".join(CONFIRMATORY_FAMILY).lower()
    for forbidden in ("zero", "sıfır", "role_zeroed", "c0", "c1", "ablat"):
        assert forbidden not in joined
    assert len(CONFIRMATORY_FAMILY) == 5


def test_confirmatory_family_membership_is_fixed():
    """Üyelik sabittir; değişirse yayımlanmış Holm değerleri geçersiz olur."""
    from aircombat_bench.eval.compare import CONFIRMATORY_FAMILY
    assert CONFIRMATORY_FAMILY == (
        "MAPPO → MAPPO+GAT",
        "MAPPO+GAT → MAPPO+GAT+OM+Role",
        "QMIX → MAPPO+GAT",
        "FACMAC → FACMAC+OM+R",
        "MAPPO+GAT → FACMAC",
    )


def test_outcome_flips_counted():
    a = _make(["win", "win", "loss", "draw"])
    b = _make(["win", "loss", "win", "draw"])
    r = compare(a, b)
    assert r["outcome_flips"] == 2


# ---------------------------------------------------------------------------
# İstatistik doğruluğu
# ---------------------------------------------------------------------------

def test_mcnemar_known_value():
    """Süreklilik düzeltmeli McNemar: (|b01-b10|-1)^2 / (b01+b10)."""
    chi2, p = mcnemar(175, 122)
    assert chi2 == pytest.approx((abs(175 - 122) - 1) ** 2 / (175 + 122), rel=1e-12)
    assert chi2 == pytest.approx(9.1044, abs=1e-3)   # 52² / 297
    assert 0.002 < p < 0.003


def test_mcnemar_symmetric_gives_null():
    chi2, p = mcnemar(100, 100)
    assert chi2 == pytest.approx(0.0)
    assert p == pytest.approx(1.0)


def test_mcnemar_no_discordant_pairs():
    assert mcnemar(0, 0) == (0.0, 1.0)


def test_sign_test_agrees_with_mcnemar_direction():
    """İki test aynı yönde ve benzer büyüklükte p vermeli."""
    for b01, b10 in [(175, 122), (170, 124), (195, 101)]:
        _, p_mc = mcnemar(b01, b10)
        p_sg = sign_test(b01, b10)
        assert (p_mc < 0.05) == (p_sg < 0.05)
        assert abs(math.log10(p_mc) - math.log10(p_sg)) < 0.5


def test_sign_test_symmetric():
    assert sign_test(50, 50) == pytest.approx(1.0)


def test_paired_bootstrap_is_deterministic():
    d = np.random.default_rng(0).normal(0.2, 1.0, 500)
    assert paired_bootstrap_ci(d, 2000, seed=42) == paired_bootstrap_ci(d, 2000, seed=42)


def test_paired_bootstrap_brackets_the_mean():
    d = np.full(400, 0.05)
    d[:20] = -0.3
    lo, hi = paired_bootstrap_ci(d, 4000, seed=1)
    assert lo <= float(d.mean()) <= hi


def test_holm_is_monotone_and_bounded():
    adj = holm({"a": 0.001, "b": 0.02, "c": 0.6})
    assert adj["a"] <= adj["b"] <= adj["c"]
    assert all(0.0 <= v <= 1.0 for v in adj.values())
    assert adj["a"] == pytest.approx(0.003)


def test_holm_single_comparison_unchanged():
    assert holm({"only": 0.03})["only"] == pytest.approx(0.03)


# ---------------------------------------------------------------------------
# Uçtan uca — eşleştirme mantığı
# ---------------------------------------------------------------------------

def test_matched_table_counts():
    a = _make(["win", "win", "loss", "draw"])
    b = _make(["win", "loss", "win", "draw"])
    r = compare(a, b)
    t = r["matched_table"]
    assert t["b11"] == 1        # ep0: ikisi de kazandı
    assert t["b10"] == 1        # ep1: yalnızca A
    assert t["b01"] == 1        # ep2: yalnızca B
    assert t["b00"] == 1        # ep3: ikisi de değil
    assert t["b11"] + t["b10"] + t["b01"] + t["b00"] == r["n_episodes"]


def test_agreement_reported_against_independence():
    a = _make(["win"] * 8 + ["loss"] * 2)
    b = _make(["win"] * 8 + ["loss"] * 2)
    r = compare(a, b)
    assert r["agreement"]["observed"] == pytest.approx(1.0)
    assert r["agreement"]["expected_under_independence"] < 1.0


def test_win_rate_diff_sign_is_b_minus_a():
    a = _make(["loss"] * 10)
    b = _make(["win"] * 10)
    r = compare(a, b)
    assert r["win_rate_diff"]["estimate"] == pytest.approx(1.0)


def test_kill_diff_paired():
    a = _make(["win"] * 4, kills=[1, 1, 1, 1])
    b = _make(["win"] * 4, kills=[2, 2, 2, 2])
    r = compare(a, b)
    assert r["kill_per_ep_diff"]["estimate"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Kanonik dosyalarda §2 rakamlarının yeniden üretimi
# ---------------------------------------------------------------------------

_CANON = [
    ("02_mappo_gat", "03_mappo_gat_om_role", 175, 122, 9.10),
    ("05_facmac_base", "06_facmac_omr", 170, 124, 6.89),
]


@pytest.mark.parametrize("a,b,b01,b10,chi2", _CANON)
def test_canonical_matched_figures(a, b, b01, b10, chi2):
    """T2.6 §2 tablosunun kanonik dosyalardan bağımsız doğrulanması."""
    fa = V1 / f"{a}_2000ep.json"
    fb = V1 / f"{b}_2000ep.json"
    if not (fa.exists() and fb.exists()):
        pytest.skip("kanonik n=2000 dosyaları henüz yok")
    r = compare_files(str(fa), str(fb))
    assert r["matched_table"]["b01"] == b01
    assert r["matched_table"]["b10"] == b10
    assert r["mcnemar"]["chi2"] == pytest.approx(chi2, abs=0.01)
    assert r["mcnemar"]["p_value"] < 0.01
    # Eşleştirmenin neden gerektiğini gösteren fark
    assert (r["agreement"]["observed"]
            > r["agreement"]["expected_under_independence"] + 0.05)

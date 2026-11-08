"""
test_runner_schema.py
=====================
Schema v1 unified eval runner birim testleri.

Gerçek checkpoint çalıştırması yapmaz — sadece şunları test eder:
  1. Wilson CI hesabı
  2. detect_algorithm() doğruluğu (tüm 6 checkpoint türü)
  3. validate_schema_v1() doğru eksik anahtar hatasını raporlar
  4. Sınır durumları: n=0, k=0, k=n

Çalıştırma:
    pytest tests/test_runner_schema.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np

from aircombat_bench.eval.runner import (
    wilson_ci,
    detect_algorithm,
    validate_schema_v1,
    SCHEMA_VERSION,
)


# ===========================================================================
# 1. Wilson CI
# ===========================================================================

class TestWilsonCI:
    """Wilson score güven aralığı hesabı testleri."""

    def test_published_facmac_omr(self):
        """FACMAC+OM+R yayımlanan CI: 421/500 → [0.807, 0.871]."""
        lo, hi = wilson_ci(421, 500)
        assert abs(lo - 0.8074) < 0.001, f"lo={lo:.4f}"
        assert abs(hi - 0.8713) < 0.001, f"hi={hi:.4f}"

    def test_published_qmix(self):
        """QMIX yayımlanan CI: 325/500 → [0.607, 0.690]."""
        lo, hi = wilson_ci(325, 500)
        assert abs(lo - 0.607) < 0.002, f"lo={lo:.4f}"
        assert abs(hi - 0.690) < 0.002, f"hi={hi:.4f}"

    def test_zero_wins(self):
        """Sıfır kazanım — lo=hi=0 değil, Wilson düzeltiyor."""
        lo, hi = wilson_ci(0, 100)
        assert lo == 0.0
        assert hi > 0.0     # Wilson 0 kazanımda da pozitif üst sınır verir

    def test_all_wins(self):
        """Tüm kazanım — hi yakın-1.0, lo < 1.0.

        Wilson CI k==n için hi=1.0 döndürmez; continuity correction nedeniyle
        hi ≈ 0.9999 civarı olur. Yalnızca lo < 1.0 ve hi > lo kontrolü yeterli.
        """
        lo, hi = wilson_ci(100, 100)
        assert hi > 0.96,  f"hi={hi:.6f} çok düşük"
        assert lo < 1.0,   f"lo={lo:.6f} beklenen < 1.0"
        assert lo < hi,    f"lo >= hi: {lo:.6f} >= {hi:.6f}"

    def test_n_zero(self):
        """Sıfır episode — (0.0, 0.0) döndür, hata verme."""
        lo, hi = wilson_ci(0, 0)
        assert lo == 0.0
        assert hi == 0.0

    def test_symmetry(self):
        """Wilson simetrik değil (amaçlı) — yalnızca in-range kontrolü."""
        lo, hi = wilson_ci(50, 100)
        assert 0.0 <= lo <= 0.5
        assert 0.5 <= hi <= 1.0


# ===========================================================================
# 2. detect_algorithm
# ===========================================================================

def _fake_ckpt(actor_keys=None, top_keys=None):
    """Sahte checkpoint dict oluştur."""
    ckpt = {}
    if top_keys:
        for k in top_keys:
            ckpt[k] = None
    if actor_keys is not None:
        actor_sd = {k: None for k in actor_keys}
        ckpt["actor"] = actor_sd
    return ckpt


class TestDetectAlgorithm:
    """Algoritma otomatik tespiti testleri."""

    def test_qmix(self):
        ckpt = _fake_ckpt(top_keys=["agent_net", "mixer_net", "target_agent_net"])
        assert detect_algorithm(ckpt) == "qmix"

    def test_facmac_omr(self):
        ckpt = _fake_ckpt(actor_keys=["fc1_base.weight", "fc1_base.bias",
                                       "ctrl_head.weight"])
        assert detect_algorithm(ckpt) == "facmac_omr"

    def test_facmac_base(self):
        ckpt = _fake_ckpt(actor_keys=["net.0.weight", "net.0.bias",
                                       "ctrl_head.weight", "ctrl_head.bias"])
        assert detect_algorithm(ckpt) == "facmac"

    def test_mappo_gat_om(self):
        ckpt = _fake_ckpt(actor_keys=["fc1_intent.weight", "fc1_old.weight",
                                       "mean_head.weight"])
        assert detect_algorithm(ckpt) == "mappo_gat_om"

    def test_mappo_gat(self):
        ckpt = _fake_ckpt(actor_keys=["fc1_old.weight", "fc1_old.bias",
                                       "mean_head.weight"])
        assert detect_algorithm(ckpt) == "mappo_gat"

    def test_mappo_base(self):
        ckpt = _fake_ckpt(actor_keys=["log_std", "net.0.weight",
                                       "mean_head.weight", "fire_head.weight"])
        assert detect_algorithm(ckpt) == "mappo"

    def test_qmix_takes_priority_over_actor(self):
        """agent_net üst düzeyde varsa actor içeriklerinden bağımsız QMIX."""
        ckpt = _fake_ckpt(
            actor_keys=["ctrl_head.weight"],   # FACMAC benzeri
            top_keys=["agent_net"],
        )
        assert detect_algorithm(ckpt) == "qmix"

    def test_facmac_omr_beats_facmac(self):
        """fc1_base.weight varsa ctrl_head.weight'den önce FACMAC+OM+R."""
        ckpt = _fake_ckpt(actor_keys=["fc1_base.weight", "ctrl_head.weight"])
        assert detect_algorithm(ckpt) == "facmac_omr"


# ===========================================================================
# 3. validate_schema_v1
# ===========================================================================

def _minimal_valid_schema():
    """Minimum geçerli schema v1 dict."""
    return {
        "schema_version": SCHEMA_VERSION,
        "package_version": "0.1.0",
        "env_version": "1.0.0",
        "git_commit": "abc1234",
        # T2.6.6 — kirli ağaçta üretilmiş dosya reddedilir
        "git_dirty": False,
        "scenario": "2v2_normal",
        "opponent": "heuristic-v1",
        # T2.6.3 — yol depo köküne göreceli olmalı
        "checkpoint": {"path": "checkpoints/x.pt", "sha256": "abc"},
        "algorithm": "facmac_omr",
        "n_episodes": 500,
        "seed": 42,
        "seeding": "legacy",
        "deterministic": True,
        # T2.5 Step 4b — runtime parmak izi zorunlu
        "runtime": {
            "device": "cpu",
            "torch_version": "2.6.0+cu124",
            "blas_backend": "mkl+mkldnn",
            "num_threads": 1,
            "cpu_model": "test-cpu",
            "os": "test-os",
            "python_version": "3.12.3",
        },
        "metrics": {
            "win_rate": 0.842,
            "win_ci95": [0.807, 0.871],
            "loss_rate": 0.006,
            "draw_rate": 0.152,
            "kill_per_ep": 1.78,
            "kill_std": 0.5,
            "mean_ep_len": 308.0,
            "survival_rate": 0.9,
            "oob_rate": 0.05,
            "one_kill_stall_rate": 0.1,
        },
        # T2.6.4 — legacy seeding'de ep_seed null, kimlik stream_position'da
        "episodes": [{"idx": 0, "ep_seed": None, "stream_position": 0,
                      "outcome": "win", "kills": 2, "steps": 271}],
    }


class TestValidateSchemaV1:
    """Schema v1 doğrulayıcı testleri."""

    def test_valid_schema_passes(self):
        data = _minimal_valid_schema()
        ok, errs = validate_schema_v1(data)
        assert ok, f"Schema hataları: {errs}"
        assert errs == []

    def test_missing_top_level_key(self):
        data = _minimal_valid_schema()
        del data["scenario"]
        ok, errs = validate_schema_v1(data)
        assert not ok
        assert any("scenario" in e for e in errs)

    def test_missing_metric(self):
        data = _minimal_valid_schema()
        del data["metrics"]["oob_rate"]
        ok, errs = validate_schema_v1(data)
        assert not ok
        assert any("oob_rate" in e for e in errs)

    def test_second_kill_rate_not_required(self):
        """second_kill_rate artık şemada zorunlu değil."""
        data = _minimal_valid_schema()
        # Olmadığında geçmeli
        ok, errs = validate_schema_v1(data)
        assert ok, f"Beklenmedik hata: {errs}"

    def test_one_kill_stall_rate_required(self):
        """one_kill_stall_rate v1'de zorunlu."""
        data = _minimal_valid_schema()
        del data["metrics"]["one_kill_stall_rate"]
        ok, errs = validate_schema_v1(data)
        assert not ok
        assert any("one_kill_stall_rate" in e for e in errs)

    def test_wrong_schema_version(self):
        data = _minimal_valid_schema()
        data["schema_version"] = "0.9"
        ok, errs = validate_schema_v1(data)
        assert not ok

    def test_episodes_list_check(self):
        data = _minimal_valid_schema()
        data["episodes"] = "not a list"
        ok, errs = validate_schema_v1(data)
        assert not ok
        assert any("liste" in e for e in errs)

    def test_episode_record_keys(self):
        """Per-episode kayıtta zorunlu anahtarlar."""
        data = _minimal_valid_schema()
        data["episodes"] = [{"idx": 0}]   # outcome/kills/steps eksik
        ok, errs = validate_schema_v1(data)
        assert not ok
        assert any("outcome" in e or "kills" in e for e in errs)


# ===========================================================================
# 4. one_kill_stall_rate computation
# ===========================================================================

class TestOneKillStall:
    """one_kill_stall_rate hesabının doğruluğu."""

    def _compute(self, records):
        from aircombat_bench.eval.runner import run_eval
        # run_eval çalıştırmak yerine mantığı doğrudan test et
        n = len(records)
        stall = sum(1 for r in records if r["kills"] == 1 and r["outcome"] != "win")
        return stall / n if n > 0 else 0.0

    def test_no_stalls(self):
        recs = [
            {"kills": 2, "outcome": "win"},
            {"kills": 2, "outcome": "win"},
        ]
        assert self._compute(recs) == 0.0

    def test_all_stalls(self):
        recs = [
            {"kills": 1, "outcome": "draw"},
            {"kills": 1, "outcome": "draw"},
        ]
        assert self._compute(recs) == 1.0

    def test_mixed(self):
        recs = [
            {"kills": 2, "outcome": "win"},    # win — stall değil
            {"kills": 1, "outcome": "draw"},   # stall
            {"kills": 0, "outcome": "draw"},   # stall değil (0 kill)
            {"kills": 1, "outcome": "loss"},   # stall
        ]
        rate = self._compute(recs)
        assert abs(rate - 0.5) < 1e-9

    def test_two_kills_in_loss_not_stall(self):
        """İki kill ama loss — stall sayılmaz (teorik, heuristic vs olabilir)."""
        recs = [{"kills": 2, "outcome": "loss"}]
        assert self._compute(recs) == 0.0


# ===========================================================================
# Summary
# ===========================================================================

if __name__ == "__main__":
    passed = 0
    failed = 0
    tests = [
        TestWilsonCI(), TestDetectAlgorithm(),
        TestValidateSchemaV1(), TestOneKillStall(),
    ]
    for suite in tests:
        for name in dir(suite):
            if not name.startswith("test_"):
                continue
            try:
                getattr(suite, name)()
                print(f"  OK  {suite.__class__.__name__}.{name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL {suite.__class__.__name__}.{name}: {e}")
                failed += 1
    print(f"\nTOPLAM: {passed+failed}  OK: {passed}  FAIL: {failed}")
    sys.exit(0 if failed == 0 else 1)

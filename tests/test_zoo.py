"""
test_zoo.py
===========
T3.3 — Model zoo testleri: manifest, load_baseline, five-line quickstart.

Çalıştırma:
    pytest tests/test_zoo.py -v

Not: SHA-256 doğrulaması yapıldığından checkpoints/ dizini gereklidir.
checkpoints yoksa testler atlanır.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "checkpoints"

# Checkpoint dizini yoksa zoo testlerini atla
_HAS_CHECKPOINTS = CKPT_DIR.exists() and any(CKPT_DIR.glob("*.pt"))

pytestmark = pytest.mark.skipif(
    not _HAS_CHECKPOINTS,
    reason="checkpoints/ dizini bulunamadı — zoo testleri checkpoint gerektirir",
)


@pytest.fixture(scope="module")
def mf():
    from aircombat_bench.zoo import manifest
    return manifest()


# ===========================================================================
# 1. Manifest testleri
# ===========================================================================

class TestManifest:
    """manifest() içerik ve bütünlük testleri."""

    EXPECTED_IDS = ["mappo", "mappo_gat", "mappo_gat_om", "qmix", "facmac", "facmac_omr"]

    def test_manifest_loads(self, mf):
        """manifest() sözlük döndürmeli."""
        assert isinstance(mf, dict)

    def test_manifest_has_schema_version(self, mf):
        """manifest schema_version alanı içermeli."""
        assert "schema_version" in mf
        assert mf["schema_version"] == "1.0"

    def test_manifest_has_env_version(self, mf):
        """manifest env_version alanı içermeli."""
        assert mf.get("env_version") == "1.0.0"

    def test_manifest_has_six_baselines(self, mf):
        """manifest tam olarak 6 baseline içermeli."""
        assert len(mf["baselines"]) == 6

    def test_manifest_baseline_ids(self, mf):
        """manifest baseline ID'leri beklenen sırayla olmalı."""
        ids = [b["id"] for b in mf["baselines"]]
        assert ids == self.EXPECTED_IDS

    def test_manifest_each_baseline_has_required_fields(self, mf):
        """Her baseline gerekli alanları içermeli."""
        required = {"id", "algorithm", "checkpoint", "sha256", "training", "eval"}
        for b in mf["baselines"]:
            missing = required - set(b.keys())
            assert not missing, f"{b['id']}: eksik alanlar: {missing}"

    def test_manifest_eval_fields(self, mf):
        """Her baseline eval bloğu gerekli metrikleri içermeli."""
        eval_required = {"win_pct", "win_ci95", "kill_per_ep", "draw_pct"}
        for b in mf["baselines"]:
            missing = eval_required - set(b["eval"].keys())
            assert not missing, f"{b['id']}: eksik eval alanlar: {missing}"

    def test_manifest_win_ci95_format(self, mf):
        """win_ci95 [lo, hi] formatında, yüzde ölçeğinde (0–100) olmalı."""
        for b in mf["baselines"]:
            ci = b["eval"]["win_ci95"]
            assert len(ci) == 2, f"{b['id']}: win_ci95 {ci}"
            lo, hi = ci
            assert 0.0 <= lo < hi <= 100.0, f"{b['id']}: geçersiz CI [{lo},{hi}]"
            win = b["eval"]["win_pct"]
            assert lo <= win <= hi, f"{b['id']}: win {win}% CI dışında [{lo},{hi}]"

    def test_manifest_sha256_format(self, mf):
        """SHA-256 hash'leri 64 hex karakter olmalı."""
        import re
        pattern = re.compile(r"^[0-9a-f]{64}$")
        for b in mf["baselines"]:
            assert pattern.match(b["sha256"]), f"{b['id']}: geçersiz SHA-256"

    def test_manifest_training_budget(self, mf):
        """Eğitim bütçesi pozitif olmalı."""
        for b in mf["baselines"]:
            t = b["training"]
            assert t["episodes"] > 0, f"{b['id']}: sıfır episode"
            assert t["env_steps"] > 0, f"{b['id']}: sıfır adım"

    def test_manifest_facmac_omr_is_best(self, mf):
        """facmac_omr en yüksek win% ile lider olmalı."""
        wins = {b["id"]: b["eval"]["win_pct"] for b in mf["baselines"]}
        best = max(wins, key=wins.get)
        assert best == "facmac_omr", f"En iyi baseline: {best}"


# ===========================================================================
# 2. load_baseline testleri
# ===========================================================================

class TestLoadBaseline:
    """load_baseline() işlev testleri."""

    VALID_IDS = ["mappo", "mappo_gat", "mappo_gat_om", "qmix", "facmac", "facmac_omr"]

    def test_load_invalid_raises_value_error(self):
        """Bilinmeyen baseline ValueError üretmeli."""
        from aircombat_bench.zoo import load_baseline
        with pytest.raises(ValueError, match="Bilinmeyen"):
            load_baseline("nonexistent_model")

    @pytest.mark.parametrize("bid", VALID_IDS)
    def test_load_baseline_returns_policy(self, bid):
        """Her baseline yüklenebilmeli ve BaselinePolicy döndürmeli."""
        from aircombat_bench.zoo import load_baseline, BaselinePolicy
        policy = load_baseline(bid, verify_hash=True)
        assert isinstance(policy, BaselinePolicy)

    @pytest.mark.parametrize("bid", VALID_IDS)
    def test_baseline_id_matches(self, bid):
        """policy.baseline_id argümanla eşleşmeli."""
        from aircombat_bench.zoo import load_baseline
        policy = load_baseline(bid, verify_hash=False)   # Hızlı test
        assert policy.baseline_id == bid

    @pytest.mark.parametrize("bid", VALID_IDS)
    def test_algorithm_detected(self, bid):
        """policy.algorithm dolu ve bilinen bir değer olmalı."""
        from aircombat_bench.zoo import load_baseline
        policy = load_baseline(bid, verify_hash=False)
        known = {"mappo", "mappo_gat", "mappo_gat_om", "qmix", "facmac", "facmac_omr"}
        assert policy.algorithm in known, f"Bilinmeyen algoritma: {policy.algorithm}"

    @pytest.mark.parametrize("bid", VALID_IDS)
    def test_act_returns_correct_shape(self, bid):
        """policy.act() her ajan için 5D aksiyon döndürmeli."""
        import yaml
        from aircombat_bench.zoo import load_baseline
        from aircombat_bench.envs.dogfight_env import DogfightEnv

        policy = load_baseline(bid, verify_hash=False)

        with open(ROOT / "configs/config.yaml", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        env = DogfightEnv(config)
        env.set_curriculum_phase(4)
        all_obs = env.reset()
        blue_ids = env.blue_ids

        obs_dict = {a: all_obs[a] for a in blue_ids}
        actions = policy.act(obs_dict, blue_ids)

        assert set(actions.keys()) == set(blue_ids)
        for aid, act in actions.items():
            assert act.shape == (5,), f"{bid}/{aid}: {act.shape}"
            assert act.dtype == np.float32 or np.issubdtype(act.dtype, np.floating)


# ===========================================================================
# 3. Five-line quickstart test
# ===========================================================================

class TestQuickstart:
    """Beş satırlık quickstart örneğinin çalıştığını doğrular."""

    def test_quickstart_10_steps(self):
        """Quickstart 10 adım PettingZoo döngüsü hatasız tamamlanmalı."""
        from aircombat_bench.envs import make_env
        from aircombat_bench.zoo import load_baseline

        policy = load_baseline("facmac_omr", verify_hash=False)
        env = make_env("2v2_normal")
        obs, _ = env.reset(seed=42)

        steps = 0
        while env.agents and steps < 10:
            actions = policy.act(obs, env.agents)

            # Aksiyon değerleri [-1,1]×[-1,1]×[-1,1]×[0,1]×[0,1] aralığında olmalı
            for aid, act in actions.items():
                assert not np.any(np.isnan(act)), f"NaN action for {aid}"
                assert act.shape == (5,)

            obs, rew, term, trunc, info = env.step(actions)
            steps += 1

        env.close()
        assert steps > 0, "En az bir adım atılmalı"

    def test_quickstart_all_algorithms(self):
        """Tüm algoritmalar 3 adım döngü hatasız tamamlanmalı."""
        from aircombat_bench.envs import make_env
        from aircombat_bench.zoo import load_baseline

        baseline_ids = ["mappo", "mappo_gat", "mappo_gat_om", "qmix", "facmac", "facmac_omr"]
        for bid in baseline_ids:
            policy = load_baseline(bid, verify_hash=False)
            env = make_env("2v2_normal")
            obs, _ = env.reset(seed=7)
            for _ in range(3):
                if not env.agents:
                    break
                actions = policy.act(obs, env.agents)
                obs, _, _, _, _ = env.step(actions)
            env.close()
            # Buraya ulaşıldıysa crash yok

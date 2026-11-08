"""
test_eval_pipeline.py
=====================
T2.5 Step 3 — Hardened evaluation pipeline determinism tests.

Dört test kategorisi (T2.5 spec §5):

  1. Pipeline determinism  : aynı komut iki kez → byte-identical JSON
  2. Episode replay        : episode-N'i yeniden çalıştırmak aynı sonucu üretmeli
  3. Seed sensitivity      : farklı seed → farklı episode dizisi (assert)
  4. Cross-platform not    : CPU vs GPU farkları için test marker'ları

NOT — Per-episode seeding (T2.3):
    Şu an tüm run tek seed kullanıyor (env.seed(base_seed) + runner içinde tek
    seferlik seeding).  Episode-N'i izole çalıştırmak için T2.3 tamamlandıktan
    sonra per-episode seeding uygulanacak.  "Episode replay" testi şimdilik
    tam 20-episode run'u tekrar ederek episode[N]'i karşılaştırıyor — bu da
    pipeline determinizmini kanıtlamak için yeterli.

Çalıştırma:
    pytest tests/test_eval_pipeline.py -v
    pytest tests/test_eval_pipeline.py -v -k "not slow"   # CPU-only hızlı testler

Checkpoint gerektiren testler skip edilir eğer checkpoints/ dizini yoksa.
"""

import json
import os
import sys
import pytest
import yaml
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aircombat_bench.eval.runner import run_eval

# ---------------------------------------------------------------------------
# Ayarlar
# ---------------------------------------------------------------------------

_SEED       = 42
_ALT_SEED   = 999   # Seed sensitivity için farklı seed
_N_EPISODES = 20    # Hızlı testler için; tam test --full ile
_DEVICE     = "cpu" # CPU → CUDA non-determinizmi yok, her platformda tekrarlanabilir

# Tüm 6 baseline checkpoint + algoritma kimliği
_ALL_BASELINES = [
    ("mappo",        "checkpoints/mappo_final.pt"),
    ("mappo_gat",    "checkpoints/mappo_gat_ep37000.pt"),
    ("mappo_gat_om", "checkpoints/mappo_gat_ep44000.pt"),
    ("qmix",         "checkpoints/qmix_ep2000.pt"),
    ("facmac",       "checkpoints/facmac_base_ep2000.pt"),
    ("facmac_omr",   "checkpoints/facmac_omr_ep27000.pt"),
]

_HAS_CHECKPOINTS = (ROOT / "checkpoints").exists() and any(
    (ROOT / ckpt).exists() for _, ckpt in _ALL_BASELINES
)


def _available_baselines():
    """Mevcut checkpoint'lere sahip baseline (id, path) listesi döndür."""
    out = []
    for bid, rel in _ALL_BASELINES:
        p = ROOT / rel
        if p.exists():
            out.append((bid, str(p)))
    return out


def _baseline_ids():
    return [bid for bid, _ in _available_baselines()]


def _baseline_params():
    return [(bid, path) for bid, path in _available_baselines()]


# ---------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def config():
    with open(ROOT / "configs/config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _run(checkpoint_path: str, config: dict,
         seed: int = _SEED, n: int = _N_EPISODES,
         seeding: str = "legacy") -> dict:
    return run_eval(
        config          = config,
        checkpoint_path = checkpoint_path,
        n_episodes      = n,
        seed            = seed,
        device_str      = _DEVICE,
        deterministic   = True,
        verbose         = False,
        seeding         = seeding,
    )


VOLATILE = {"timestamp_utc", "git_commit", "wall_seconds"}


def _strip(d: dict) -> dict:
    return {k: v for k, v in d.items() if k not in VOLATILE}


def _metrics_bit_equal(m1: dict, m2: dict) -> bool:
    """Metrik dict'lerinin tam eşitliği (float toleransı yok)."""
    if set(m1) != set(m2):
        return False
    for k in m1:
        v1, v2 = m1[k], m2[k]
        if isinstance(v1, list):
            if v1 != v2:
                return False
        elif v1 != v2:
            return False
    return True


def _episodes_equal(eps1: list, eps2: list) -> bool:
    if len(eps1) != len(eps2):
        return False
    for a, b in zip(eps1, eps2):
        if a != b:
            return False
    return True


# ---------------------------------------------------------------------------
# 1. Pipeline Determinism — her baseline için ayrı test
# ---------------------------------------------------------------------------

class TestPipelineDeterminism:
    """
    T2.5 Step 3, Madde 1: Aynı eval komutu iki kez → byte-identical JSON.

    Neden önemli:
        Bir eval metodolojisi iddiası ancak tekrarlanabilirse anlamlıdır.
        Bu testler benchmarkın yayım öncesi geçmesi gereken minimum çıta.

    CPU parametresi:
        device_str="cpu" — CUDA'nın atomik operasyon non-determinizmi
        olmadan tutarlı bit-exact sonuçlar üretir.
        GPU çalıştırması çoğu mimaride deterministik olsa da bu testler
        bunu varsaymaz.  Yayımlanan sayılar CPU ile üretilmiştir.
    """

    @pytest.mark.parametrize("bid,ckpt", _baseline_params(),
                             ids=_baseline_ids())
    def test_metrics_bit_identical(self, bid, ckpt, config):
        """Aynı seed × aynı checkpoint → metrikler bit-exact eşit."""
        out1 = _run(ckpt, config)
        out2 = _run(ckpt, config)
        m1, m2 = out1["metrics"], out2["metrics"]
        assert _metrics_bit_equal(m1, m2), (
            f"{bid}: metrikler iki run arasında farklı!\n"
            f"  Run 1 win_rate={m1['win_rate']}, kill_per_ep={m1['kill_per_ep']}\n"
            f"  Run 2 win_rate={m2['win_rate']}, kill_per_ep={m2['kill_per_ep']}"
        )

    @pytest.mark.parametrize("bid,ckpt", _baseline_params(),
                             ids=_baseline_ids())
    def test_episode_records_bit_identical(self, bid, ckpt, config):
        """Aynı seed × aynı checkpoint → per-episode kayıtlar bit-exact eşit."""
        out1 = _run(ckpt, config)
        out2 = _run(ckpt, config)
        eps1, eps2 = out1["episodes"], out2["episodes"]
        assert _episodes_equal(eps1, eps2), (
            f"{bid}: per-episode kayıtlar eşleşmedi!\n"
            f"  İlk farklı episode:\n"
            f"  Run 1: {next((e1 for e1,e2 in zip(eps1,eps2) if e1!=e2), None)}\n"
            f"  Run 2: {next((e2 for e1,e2 in zip(eps1,eps2) if e1!=e2), None)}"
        )

    @pytest.mark.parametrize("bid,ckpt", _baseline_params(),
                             ids=_baseline_ids())
    def test_json_serializable_and_stable(self, bid, ckpt, config):
        """Çıktı JSON serialize edilebilmeli ve round-trip sonrası değişmemeli."""
        out = _run(ckpt, config)
        serialized   = json.dumps(out, ensure_ascii=False)
        deserialized = json.loads(serialized)

        # Metrikler round-trip'te değişmemeli
        m_orig  = out["metrics"]
        m_rt    = deserialized["metrics"]
        assert _metrics_bit_equal(m_orig, m_rt), (
            f"{bid}: JSON round-trip sonrası metrikler değişti"
        )


# ---------------------------------------------------------------------------
# 2. Episode Replay (T2.3 öncesi basitleştirilmiş versiyon)
# ---------------------------------------------------------------------------

class TestEpisodeReplay:
    """
    T2.5 Step 3, Madde 2: Episode-N izole çalıştırması aynı sonucu üretmeli.

    Şu an uygulaması: tam N-episode run iki kez çalıştırılıp episode[idx]
    kayıtları karşılaştırılıyor.  Bu pipeline determinizmini kanıtlar ama
    "izole episode" çalıştırmasını kanıtlamaz.

    T2.3 (per-episode seeding) tamamlandıktan sonra bu test şu şekilde
    güçlendirilmeli:
        env.seed(base_seed + episode_idx) — her episode bağımsız seeded
        → episode[idx] yalnızca o seed ile tam olarak yeniden üretilebilir

    Test etiketi: "replay"  — ayrıca çalıştırmak için:
        pytest tests/test_eval_pipeline.py -v -k "replay"
    """

    # Hangi episode index'i test edelim
    _REPLAY_IDX = 3

    @pytest.mark.parametrize("bid,ckpt", _baseline_params(),
                             ids=_baseline_ids())
    def test_episode_replay_consistent(self, bid, ckpt, config):
        """Episode[3] iki tam run arasında aynı outcome/kills/steps içermeli."""
        idx = self._REPLAY_IDX
        out1 = _run(ckpt, config, n=max(_N_EPISODES, idx + 1))
        out2 = _run(ckpt, config, n=max(_N_EPISODES, idx + 1))

        ep1 = out1["episodes"][idx]
        ep2 = out2["episodes"][idx]

        assert ep1["outcome"] == ep2["outcome"], (
            f"{bid} ep[{idx}] outcome: {ep1['outcome']} ≠ {ep2['outcome']}"
        )
        assert ep1["kills"] == ep2["kills"], (
            f"{bid} ep[{idx}] kills: {ep1['kills']} ≠ {ep2['kills']}"
        )
        assert ep1["steps"] == ep2["steps"], (
            f"{bid} ep[{idx}] steps: {ep1['steps']} ≠ {ep2['steps']}"
        )

    @pytest.mark.parametrize("bid,ckpt", _baseline_params(),
                             ids=_baseline_ids())
    def test_all_episode_seeds_recorded(self, bid, ckpt, config):
        """Episode kimliği kaydedilmeli — legacy'de konum, per_episode'da seed.

        T2.6.4: legacy seeding'de `seed + idx` bir seed DEĞİLDİR; o episode'u
        izole yeniden üretmez.  Bu yüzden legacy'de ep_seed None olmalı ve
        episode'u tanımlayan alan stream_position olmalıdır.
        """
        out = _run(ckpt, config)
        assert out["seeding"] == "legacy", "varsayılan legacy bekleniyordu"
        for i, rec in enumerate(out["episodes"]):
            assert "ep_seed" in rec, f"{bid}: ep_seed alanı eksik: {rec}"
            assert rec["ep_seed"] is None, (
                f"{bid}: seeding=legacy iken ep[{i}].ep_seed="
                f"{rec['ep_seed']!r} — tutulamayacak bir replay sözü"
            )
            assert rec["stream_position"] == i, (
                f"{bid}: ep[{i}].stream_position={rec.get('stream_position')} ≠ {i}"
            )

    @pytest.mark.parametrize("bid,ckpt", _baseline_params()[:1],
                             ids=_baseline_ids()[:1])
    def test_per_episode_seeding_records_real_seed(self, bid, ckpt, config):
        """per_episode modunda ep_seed env'e fiilen verilen seed'dir."""
        out = _run(ckpt, config, seeding="per_episode")
        assert out["seeding"] == "per_episode"
        for i, rec in enumerate(out["episodes"]):
            assert rec["ep_seed"] == _SEED + i, (
                f"{bid}: ep[{i}].ep_seed={rec['ep_seed']} ≠ {_SEED + i}"
            )


# ---------------------------------------------------------------------------
# 3. Seed Sensitivity
# ---------------------------------------------------------------------------

class TestSeedSensitivity:
    """
    T2.5 Step 3, Madde 3: Farklı seed farklı sonuç üretmeli.

    Bunu test etmenin amacı: runner'ın seed'i gerçekten kullandığını ve
    seed parametresini tamamen görmezden gelmediğini kanıtlamak.

    NOT: Küçük N'lerde aynı win_rate ortaya çıkabilir (istatistiksel tesadüf).
    Episode dizisi farklı olmalı — bu daha güvenilir bir test.
    """

    @pytest.mark.parametrize("bid,ckpt", _baseline_params(),
                             ids=_baseline_ids())
    def test_different_seed_produces_different_episodes(self, bid, ckpt, config):
        """Seed 42 ve seed 999 farklı episode dizisi üretmeli."""
        out_42  = _run(ckpt, config, seed=42)
        out_999 = _run(ckpt, config, seed=999)

        eps_42  = out_42["episodes"]
        eps_999 = out_999["episodes"]

        # En az 1 episode'un outcome veya steps'i farklı olmalı
        any_diff = any(
            e42["outcome"] != e999["outcome"] or e42["steps"] != e999["steps"]
            for e42, e999 in zip(eps_42, eps_999)
        )
        assert any_diff, (
            f"{bid}: Seed 42 ve 999 aynı episode dizisini üretti "
            f"(N={_N_EPISODES}). Seed parametresi kullanılmıyor olabilir!"
        )

    @pytest.mark.parametrize("bid,ckpt", _baseline_params(),
                             ids=_baseline_ids())
    def test_seed_42_win_rate_stable_across_calls(self, bid, ckpt, config):
        """Seed 42 win_rate her çalıştırmada aynı olmalı (determinizm kanıtı)."""
        out1 = _run(ckpt, config, seed=42)
        out2 = _run(ckpt, config, seed=42)
        assert out1["metrics"]["win_rate"] == out2["metrics"]["win_rate"], (
            f"{bid}: Seed=42 iki çalıştırmasında win_rate farklı! "
            f"{out1['metrics']['win_rate']} ≠ {out2['metrics']['win_rate']}"
        )


# ---------------------------------------------------------------------------
# 4. Cross-Platform Notları
# ---------------------------------------------------------------------------

class TestCrossPlatformNotes:
    """
    T2.5 Step 3, Madde 4: CPU vs GPU farkları için belgeler + kontroller.

    Canonical baseline sayılar CPU'da üretildi:
      device_str = "cpu"
      torch.use_deterministic_algorithms() çağrılmamış
      torch.backends.cudnn.benchmark = False ayarlanmamış

    GPU çalıştırması için uyarı:
      CUDA atomik operasyonlar (scatter/gather) non-deterministik olabilir.
      Bit-exact yeniden üretim için:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
      Bu ayarlar CPU inference'ı etkilemez.
    """

    def test_cpu_device_produces_deterministic_results(self, config):
        """CPU inference deterministik olmalı (platforma bağımsız)."""
        available = _available_baselines()
        if not available:
            pytest.skip("Checkpoint bulunamadı")

        bid, ckpt = available[0]   # İlk mevcut baseline
        out1 = _run(ckpt, config)
        out2 = _run(ckpt, config)
        assert out1["metrics"] == out2["metrics"], (
            f"CPU inference non-deterministik: {bid}"
        )

    def test_runner_uses_cpu_for_canonical_evals(self):
        """Eval runner'ı CPU'da bit-exact ürettiğini belgeliyor.

        Bu test 'geçer' ibaresi runner'ın CPU desteğinin varlığını kanıtlar.
        GPU kullanımında aynı bit-exact garantisi verilmez.
        """
        import inspect
        from aircombat_bench.eval import runner
        src = inspect.getsource(runner)
        assert "cpu" in src, "runner.py 'cpu' device desteği içermeli"

    @pytest.mark.skipif(
        not __import__("torch").cuda.is_available(),
        reason="CUDA mevcut değil"
    )
    def test_cuda_results_match_cpu_within_tolerance(self, config):
        """CUDA ve CPU sonuçları birbirine yakın olmalı (bit-exact garanti yok).

        Bu test eğer CUDA mevcut ise CPU ile CUDA sonuçlarını karşılaştırır.
        Beklenen: win_rate ±0.02 (olası CUDA floating-point farkı).
        """
        available = _available_baselines()
        if not available:
            pytest.skip("Checkpoint bulunamadı")

        bid, ckpt = available[0]
        out_cpu  = _run(ckpt, config)
        out_cuda = run_eval(
            config          = config,
            checkpoint_path = ckpt,
            n_episodes      = _N_EPISODES,
            seed            = _SEED,
            device_str      = "cuda",
            deterministic   = True,
            verbose         = False,
        )
        diff = abs(out_cpu["metrics"]["win_rate"] - out_cuda["metrics"]["win_rate"])
        assert diff <= 0.02, (
            f"{bid}: CPU({out_cpu['metrics']['win_rate']:.4f}) vs "
            f"CUDA({out_cuda['metrics']['win_rate']:.4f}) fark={diff:.4f} > 0.02\n"
            "CUDA non-determinizmi veya farklı hesap yolu işaret ediyor."
        )


# ---------------------------------------------------------------------------
# Schema ve ENV_VERSION kontrolleri
# ---------------------------------------------------------------------------

class TestSchemaAndEnvVersion:
    """Eval çıktısının schema v1 uyumunu kontrol eder."""

    @pytest.mark.parametrize("bid,ckpt", _baseline_params(),
                             ids=_baseline_ids())
    def test_schema_v1_fields_present(self, bid, ckpt, config):
        """Schema v1 zorunlu alanları çıktıda bulunmalı."""
        required = {
            "schema_version", "env_version", "seed", "deterministic",
            "metrics", "episodes", "checkpoint",
        }
        out = _run(ckpt, config)
        missing = required - set(out.keys())
        assert not missing, f"{bid}: Eksik alanlar: {missing}"

    @pytest.mark.parametrize("bid,ckpt", _baseline_params(),
                             ids=_baseline_ids())
    def test_env_version_is_1_0_0(self, bid, ckpt, config):
        """ENV_VERSION 1.0.0 olmalı — T2.5 kapsamındaki değişiklikler env'i etkilemedi."""
        out = _run(ckpt, config)
        assert out["env_version"] == "1.0.0", (
            f"{bid}: env_version={out['env_version']!r} ≠ '1.0.0'\n"
            "ENV_VERSION değiştiyse T2.5 kapsamı dışına çıkılmış demektir."
        )

    @pytest.mark.parametrize("bid,ckpt", _baseline_params(),
                             ids=_baseline_ids())
    def test_metrics_has_wilson_ci(self, bid, ckpt, config):
        """win_ci95 [lo, hi] formatında metrics içinde bulunmalı."""
        out = _run(ckpt, config)
        ci  = out["metrics"]["win_ci95"]
        assert isinstance(ci, list) and len(ci) == 2, (
            f"{bid}: win_ci95 eksik veya yanlış format: {ci}"
        )
        lo, hi = ci
        assert 0.0 <= lo < hi <= 1.0, (
            f"{bid}: win_ci95={ci} geçersiz aralık"
        )
        wr = out["metrics"]["win_rate"]
        assert lo <= wr <= hi, (
            f"{bid}: win_rate={wr:.4f} CI=[{lo:.4f},{hi:.4f}] dışında"
        )

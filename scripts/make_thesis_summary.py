"""
make_thesis_summary.py
======================
Kanonik eval JSON'larından tez için kompakt bir özet üretir:
``docs/evals/v1.0/thesis_summary.json``.

Kurallar
--------
* Her sayı bir schema v1 JSON dosyasından okunur ya da onlardan
  ``aircombat_bench.eval.compare`` ile hesaplanır.  Elle girilmiş sayı yoktur.
* Her blok ``source`` alanında hangi dosyalardan geldiğini taşır; üst düzey
  ``provenance`` bloğu her kaynak dosyanın SHA-256 özetini verir.
* İstatistik yöntemleri ``compare.py`` ile birebir aynıdır (McNemar süreklilik
  düzeltmeli, paired bootstrap 10 000 örnek / seed 42, Holm yalnızca
  doğrulayıcı ailede).

Kullanım
--------
    python -X utf8 scripts/make_thesis_summary.py
    python -X utf8 scripts/make_thesis_summary.py --out başka/yol.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aircombat_bench.eval import compare as C  # noqa: E402

EVALS = ROOT / "docs" / "evals"
V10 = EVALS / "v1.0"
CUDA = EVALS / "robustness_cuda"
ROLE = EVALS / "role_experiment"

N_BOOT = 10_000
BOOT_SEED = 42

# ---------------------------------------------------------------------------
# Dosya eşlemesi — yalnızca etiket/yol; sayı içermez
# ---------------------------------------------------------------------------

MODELS = [
    ("01", "MAPPO baseline",     "01_mappo_baseline_2000ep.json"),
    ("02", "MAPPO+GAT",          "02_mappo_gat_2000ep.json"),
    ("03", "MAPPO+GAT+OM+Role",  "03_mappo_gat_om_role_2000ep.json"),
    ("04", "QMIX",               "04_qmix_ep2000_2000ep.json"),
    ("05", "Pure FACMAC",        "05_facmac_base_2000ep.json"),
    ("06", "FACMAC+OM+R",        "06_facmac_omr_2000ep.json"),
]

# compare.CONFIRMATORY_FAMILY sırasıyla eşleşen (a, b) çiftleri; fark b − a.
CONFIRMATORY_PAIRS = {
    "MAPPO → MAPPO+GAT":             ("01", "02"),
    "MAPPO+GAT → MAPPO+GAT+OM+Role": ("02", "03"),
    "QMIX → MAPPO+GAT":              ("04", "02"),
    "FACMAC → FACMAC+OM+R":          ("05", "06"),
    "MAPPO+GAT → FACMAC":            ("02", "05"),
}

EXPLORATORY_STRUCTURAL = [
    # (ad, a dosyası, b dosyası)
    ("FACMAC+OM+R → rol sıfırlanmış (06→08)",
     V10 / "06_facmac_omr_2000ep.json",
     V10 / "08_structural_role_zeroed_facmac_2000ep.json"),
    ("MAPPO+GAT+OM+Role → rol sıfırlanmış (03→09)",
     V10 / "03_mappo_gat_om_role_2000ep.json",
     V10 / "09_structural_role_zeroed_mappo_2000ep.json"),
    ("Pure FACMAC → FACMAC+OM+R rol sıfırlanmış (05→08)",
     V10 / "05_facmac_base_2000ep.json",
     V10 / "08_structural_role_zeroed_facmac_2000ep.json"),
]

# Rol müdahale serisi.  Tanımlar runner'daki role_mode etiketlerinin metin
# karşılığıdır; ε değerleri JSON'daki role_scale alanından okunur.
ROLE_MODE_DEF = {
    "learned":         "atayıcı çıktısı (one-hot)",
    "zeroed":          "sıfır vektör",
    "fixed_distinct":  "sabit, ajana göre farklı (a0=support, a1=pursuit)",
    "fixed_identical": "sabit, iki ajana aynı (pursuit)",
    "shuffled":        "atayıcı çıktısı ajanlar arasında takas",
    "uniform":         "uniform [0.25]*4, iki ajana aynı",
}
FACMAC_ROLE_SERIES = [
    # (kod, dosya, ön-kayıt belgesi)
    ("C0", ROLE / "C_learned_2000ep.json",         "T2.9"),
    ("C1", ROLE / "C_zeroed_2000ep.json",          "T2.9"),
    ("C2", ROLE / "C_fixed_distinct_2000ep.json",  "T2.9"),
    ("C3", ROLE / "C_fixed_identical_2000ep.json", "T2.9"),
    ("C4", ROLE / "C_shuffled_2000ep.json",        "T2.9"),
    ("C5", ROLE / "C5_uniform_2000ep.json",        "T2.10"),
    ("C6", ROLE / "C6_scaled_eps0.5_2000ep.json",  "T2.10"),
    ("C6", ROLE / "C6_scaled_eps0.1_2000ep.json",  "T2.10"),
    ("C6", ROLE / "C6_scaled_eps0.01_2000ep.json", "T2.10"),
]
MAPPO_ROLE_SERIES = [
    ("C0", V10 / "03_mappo_gat_om_role_2000ep.json",          "T2.9"),
    ("C1", V10 / "09_structural_role_zeroed_mappo_2000ep.json", "T2.9"),
]
PREREG_DOCS = {
    "T2.9":  ("T2.9 için ayrı bir ön-kayıt belgesi yok; C2/C3 tasarım "
              "seçimleri koşulardan önce commit 53d7a52 mesajında "
              "sabitlenmiştir (sonuç tahmini içermez)."),
    "T2.10": "docs/PREREGISTRATION_T2.10.md (commit a94b944)",
}

# Rol serisinde C0'a karşı (ve ön-kayıttaki ek) eşleştirilmiş karşılaştırmalar
ROLE_CONTRASTS = [
    ("C0 → C1", "C0", "C1"), ("C0 → C2", "C0", "C2"), ("C0 → C3", "C0", "C3"),
    ("C0 → C4", "C0", "C4"), ("C0 → C5", "C0", "C5"),
    ("C0 → C6@0.5", "C0", "C6@0.5"), ("C0 → C6@0.1", "C0", "C6@0.1"),
    ("C0 → C6@0.01", "C0", "C6@0.01"),
    ("C3 → C5", "C3", "C5"), ("C1 → C5", "C1", "C5"),
    ("C5 → C6@0.5 (L2-eşleşmiş)", "C5", "C6@0.5"),
]

# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------

_sources: Dict[str, str] = {}


def rel(p: Path) -> str:
    return p.resolve().relative_to(ROOT).as_posix()


def load(p: Path) -> dict:
    raw = p.read_bytes()
    _sources[rel(p)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw.decode("utf-8"))


def r(x: Optional[float], nd: int = 6) -> Optional[float]:
    return None if x is None else round(float(x), nd)


def wilson(k: int, n: int, z: float = 1.959963984540054) -> List[float]:
    """Wilson skor aralığı (yalnızca JSON'daki değeri doğrulamak için)."""
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [c - h, c + h]


def asym_ci(d: dict) -> List[float]:
    """Tek koşulun ajan-düzeyi asimetrisi için bootstrap CI95.

    compare.paired_asymmetry_diff ile aynı yeniden örnekleme şeması
    (episode indeksleri, 10 000 örnek, seed 42).
    """
    k = np.array([e["kills_by_agent"] for e in d["episodes"]], dtype=np.float64)
    n = k.shape[0]
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    a = np.abs(k[:, 0][idx].mean(1) - k[:, 1][idx].mean(1))
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def kill_share(d: dict) -> Dict[str, Any]:
    k = np.array([e["kills_by_agent"] for e in d["episodes"]], dtype=np.float64)
    tot = k.sum()
    return {"agent0": r(k[:, 0].sum() / tot), "agent1": r(k[:, 1].sum() / tot),
            "kills_agent0": int(k[:, 0].sum()), "kills_agent1": int(k[:, 1].sum())}


def model_block(d: dict, path: Path) -> Dict[str, Any]:
    m = d["metrics"]
    rt = d.get("runtime", {})
    n = d["n_episodes"]
    out = {
        "algorithm": d.get("algorithm"),
        "n": n,
        "seed": d.get("seed"),
        "seeding": d.get("seeding"),
        "deterministic": d.get("deterministic"),
        "win_rate": m["win_rate"],
        "win_ci95_wilson": m["win_ci95"],
        "W": d["wins"], "D": d["draws"], "L": d["losses"],
        "kill_per_ep": m["kill_per_ep"],
        "draw_rate": m["draw_rate"],
        "loss_rate": m["loss_rate"],
        "oob_rate": m.get("oob_rate"),
        "runtime": {"device": rt.get("device"),
                    "num_threads": rt.get("num_threads"),
                    "torch_version": rt.get("torch_version"),
                    "reconstructed": rt.get("reconstructed", False)},
        "env_version": d.get("env_version"),
        "opponent": d.get("opponent"),
        "git_commit": d.get("git_commit"),
        "git_dirty": d.get("git_dirty"),
        "checkpoint": d.get("checkpoint"),
        "source": rel(path),
    }
    for key in ("role0_mean", "role1_mean", "intent_mean"):
        if key in m:
            out[key + "_stepwise"] = m[key]
    return out


def cmp_block(res: Dict[str, Any], with_mcnemar: bool = True) -> Dict[str, Any]:
    out = {
        "win_rate_a": r(res["win_rate_a"]),
        "win_rate_b": r(res["win_rate_b"]),
        "delta_win": r(res["win_rate_diff"]["estimate"]),
        "delta_win_ci95_paired_bootstrap":
            [r(x) for x in res["win_rate_diff"]["ci95_paired_bootstrap"]],
        "matched_table": res["matched_table"],
        "outcome_flips": res["outcome_flips"],
    }
    if with_mcnemar:
        out["mcnemar_chi2"] = r(res["mcnemar"]["chi2"], 4)
        out["p_raw"] = res["mcnemar"]["p_value"]
    return out


def git_head(out: Path) -> Dict[str, Any]:
    """Üretim anındaki commit ve girdi dizinlerinin temizliği.

    Çıktı dosyasının kendisi (``out``) dirty sayılmaz — aksi hâlde yeniden
    üretim her zaman dirty görünürdü.
    """
    def run(*a: str) -> Optional[str]:
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True,
                                  text=True, check=True).stdout
        except Exception:
            return None
    commit = run("rev-parse", "--short", "HEAD")
    status = run("status", "--porcelain", "--", "docs/evals",
                 "aircombat_bench", "scripts")
    try:
        out_rel = out.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        out_rel = None
    lines = [] if status is None else [
        ln for ln in status.splitlines()
        if ln.strip() and (out_rel is None or not ln.endswith(out_rel))]
    return {"commit": commit.strip() if commit else "unknown",
            "dirty": None if status is None else bool(lines),
            "dirty_files": [ln[3:] for ln in lines]}


# ---------------------------------------------------------------------------
# Bloklar
# ---------------------------------------------------------------------------

def build(out: Path) -> Dict[str, Any]:
    canon = {code: load(V10 / fn) for code, _, fn in MODELS}
    names = {code: nm for code, nm, _ in MODELS}

    # 1) Modeller
    models = {}
    for code, nm, fn in MODELS:
        blk = model_block(canon[code], V10 / fn)
        # Wilson aralığını W ve n'den bağımsız olarak doğrula
        w_lo, w_hi = wilson(blk["W"], blk["n"])
        blk["wilson_recomputed_matches_json"] = (
            abs(w_lo - blk["win_ci95_wilson"][0]) < 5e-6
            and abs(w_hi - blk["win_ci95_wilson"][1]) < 5e-6)
        models[code] = {"name": nm, **blk}

    # 2) Doğrulayıcı aile
    conf_raw, conf = {}, {}
    for label, (a, b) in CONFIRMATORY_PAIRS.items():
        res = C.compare(canon[a], canon[b], names[a], names[b],
                        n_boot=N_BOOT, bootstrap_seed=BOOT_SEED,
                        family=C.FAMILY_CONFIRMATORY)
        conf[label] = {"a": a, "b": b, **cmp_block(res),
                       "source": [rel(V10 / dict((c, f) for c, _, f in MODELS)[a]),
                                  rel(V10 / dict((c, f) for c, _, f in MODELS)[b])]}
        conf_raw[label] = res["mcnemar"]["p_value"]
    holm = C.holm_confirmatory(conf_raw)
    for label in conf:
        conf[label]["p_holm"] = holm[label]
    assert tuple(conf) == tuple(C.CONFIRMATORY_FAMILY), "aile üyeliği uyuşmuyor"

    # 3) Keşifsel yapısal ablasyonlar (Holm yok)
    expl = {}
    for label, pa, pb in EXPLORATORY_STRUCTURAL:
        da, db = load(pa), load(pb)
        res = C.compare(da, db, pa.stem, pb.stem, n_boot=N_BOOT,
                        bootstrap_seed=BOOT_SEED, family=C.FAMILY_EXPLORATORY)
        asy = C.paired_asymmetry_diff(da, db, n_boot=N_BOOT,
                                      bootstrap_seed=BOOT_SEED)
        expl[label] = {**cmp_block(res),
                       "asymmetry_a": r(asy["asymmetry_a"]),
                       "asymmetry_b": r(asy["asymmetry_b"]),
                       "delta_asymmetry": r(asy["estimate"]),
                       "delta_asymmetry_ci95_paired_bootstrap":
                           [r(x) for x in asy["ci95_paired_bootstrap"]],
                       "holm": "uygulanmaz (keşifsel aile)",
                       "source": [rel(pa), rel(pb)]}

    # 4) Rol müdahale serisi
    def role_rows(series, model_name):
        rows, docs = [], {}
        for code, path, prereg in series:
            d = load(path)
            mode = d.get("role_mode") or ("zeroed" if d.get("ablate_role") else "learned")
            scale = d.get("role_scale", 1.0)
            key = code if code != "C6" else f"C6@{scale:g}"
            definition = ROLE_MODE_DEF.get(mode, mode)
            if code == "C6":
                definition = f"{ROLE_MODE_DEF[mode]} × ε (ε={scale:g})"
            row = {
                "code": code, "key": key,
                "definition": definition,
                "role_mode_json": mode, "role_scale_json": scale,
                "model": model_name,
                "file": rel(path),
                "n": d["n_episodes"],
                "win_rate": d["metrics"]["win_rate"],
                "win_ci95_wilson": d["metrics"]["win_ci95"],
                "W": d["wins"], "D": d["draws"], "L": d["losses"],
                "kill_per_ep": d["metrics"]["kill_per_ep"],
                "asymmetry": r(C.agent_level_asymmetry(d)),
                "asymmetry_ci95_bootstrap": [r(x) for x in asym_ci(d)],
                "kill_share": kill_share(d),
                "preregistration": prereg,
                "git_commit": d.get("git_commit"),
                "git_dirty": d.get("git_dirty"),
                "device": d.get("runtime", {}).get("device"),
            }
            rows.append(row)
            docs[key] = d
        return rows, docs

    f_rows, f_docs = role_rows(FACMAC_ROLE_SERIES, "FACMAC+OM+R")
    m_rows, m_docs = role_rows(MAPPO_ROLE_SERIES, "MAPPO+GAT+OM+Role")
    contrasts = {}
    for label, ka, kb in ROLE_CONTRASTS:
        da, db = f_docs[ka], f_docs[kb]
        res = C.compare(da, db, ka, kb, n_boot=N_BOOT, bootstrap_seed=BOOT_SEED,
                        family=C.FAMILY_EXPLORATORY)
        asy = C.paired_asymmetry_diff(da, db, n_boot=N_BOOT,
                                      bootstrap_seed=BOOT_SEED)
        contrasts[label] = {**cmp_block(res),
                            "delta_asymmetry": r(asy["estimate"]),
                            "delta_asymmetry_ci95_paired_bootstrap":
                                [r(x) for x in asy["ci95_paired_bootstrap"]]}
    res_m = C.compare(m_docs["C0"], m_docs["C1"], "C0", "C1", n_boot=N_BOOT,
                      bootstrap_seed=BOOT_SEED, family=C.FAMILY_EXPLORATORY)
    asy_m = C.paired_asymmetry_diff(m_docs["C0"], m_docs["C1"], n_boot=N_BOOT,
                                    bootstrap_seed=BOOT_SEED)
    contrasts["MAPPO C0 → C1"] = {**cmp_block(res_m),
                                  "delta_asymmetry": r(asy_m["estimate"]),
                                  "delta_asymmetry_ci95_paired_bootstrap":
                                      [r(x) for x in asy_m["ci95_paired_bootstrap"]]}

    # Paper'daki "six conditions" eşlemesi — etiket düzeyinde
    six = {"learned": "C0", "zeroed": "C1", "fixed distinct": "C2",
           "fixed identical": "C3", "permuted": "C4",
           "magnitude-scaled": "C6 (ε=0.5, 0.1, 0.01)"}

    role_series = {
        "family": "keşifsel (Holm uygulanmaz)",
        "facmac_rows": f_rows,
        "mappo_rows": m_rows,
        "contrasts": contrasts,
        "paper_six_conditions_map": six,
        "not_in_paper_six": ["C5 (uniform) — metinde ayrıca anılıyor"],
        "preregistration_documents": PREREG_DOCS,
    }

    # 5) Head-to-head
    p7a = V10 / "07a_facmac_blue_vs_mappo_2000ep.json"
    p7b = V10 / "07b_mappo_blue_vs_facmac_2000ep.json"
    d7a, d7b = load(p7a), load(p7b)
    h2h = C.combine_head_to_head(d7a, d7b, "FACMAC+OM+R", "MAPPO+GAT")
    head_to_head = {
        "n_total": h2h["n_total"],
        "FACMAC+OM+R": {k: r(v) if isinstance(v, float) else v
                        for k, v in h2h["model_x"].items() if k != "name"},
        "MAPPO+GAT": {k: r(v) if isinstance(v, float) else v
                      for k, v in h2h["model_y"].items() if k != "name"},
        "draw_rate_two_sided": r(h2h["draw_rate"]),
        "07a": {"W": d7a["wins"], "D": d7a["draws"], "L": d7a["losses"],
                "blue": d7a["algorithm"], "red": d7a["opponent"]},
        "07b": {"W": d7b["wins"], "D": d7b["draws"], "L": d7b["losses"],
                "blue": d7b["algorithm"], "red": d7b["opponent"]},
        "device": d7a.get("runtime", {}).get("device"),
        "source": [rel(p7a), rel(p7b)],
    }

    # 6) CPU–CUDA
    device = {}
    for code, nm, fn in MODELS:
        dc = load(CUDA / fn)
        res = C.compare(canon[code], dc, "cpu", "cuda", n_boot=N_BOOT,
                        bootstrap_seed=BOOT_SEED, device_comparison=True,
                        family=C.FAMILY_DEVICE)
        device[code] = {"name": nm,
                        "win_cpu": r(res["win_rate_a"]),
                        "win_cuda": r(res["win_rate_b"]),
                        "delta_cuda_minus_cpu": r(res["win_rate_diff"]["estimate"]),
                        "delta_ci95_paired_bootstrap":
                            [r(x) for x in res["win_rate_diff"]["ci95_paired_bootstrap"]],
                        "mcnemar_p": res["mcnemar"]["p_value"],
                        "outcome_flips": res["outcome_flips"],
                        "cuda_git_commit": dc.get("git_commit"),
                        "cuda_git_dirty": dc.get("git_dirty"),
                        "source": [rel(V10 / fn), rel(CUDA / fn)]}

    # 7) Mekanizma sayıları
    d06 = canon["06"]
    d08 = load(V10 / "08_structural_role_zeroed_facmac_2000ep.json")
    mechanism = {
        "kill_share_learned_06": {**kill_share(d06), "source": rel(V10 / "06_facmac_omr_2000ep.json")},
        "kill_share_zeroed_08": {**kill_share(d08),
                                 "source": rel(V10 / "08_structural_role_zeroed_facmac_2000ep.json")},
        "oob_rate": {code: {"name": names[code],
                            "oob_rate": canon[code]["metrics"].get("oob_rate")}
                     for code in canon},
        "role_predictive_power": {
            "status": "kanonik JSON'larda yok",
            "note": ("Pursuit/defensive öngörü gücü değerleri yalnızca "
                     "PAPER_DELTA_v0.1.md §A2c metninde geçiyor; kaynak "
                     "enstrümanlı koşular logs/role_attr_06_normal.json ve "
                     "logs/role_attr_08_ablate.json (git-ignored). Tanım "
                     "belgelenmediği için bu script sayı üretmez."),
        },
    }

    return {
        "schema": "thesis_summary/1",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "scripts/make_thesis_summary.py",
        "generator_git": git_head(out),
        "conventions": {
            "rates": "oran (0–1); yüzde değil",
            "delta": "b − a",
            "mcnemar": "süreklilik düzeltmeli (Edwards), compare.mcnemar",
            "bootstrap": f"paired, episode indeksleri, {N_BOOT} örnek, seed {BOOT_SEED}",
            "holm": "yalnızca compare.CONFIRMATORY_FAMILY (5 karşılaştırma)",
            "asymmetry": "|mean(k0) − mean(k1)| (compare.agent_level_asymmetry)",
            "role_mean_stepwise": "adım-düzeyi ortalama rol vektörü; episode oranı değil",
        },
        "canonical_config": {
            "n": canon["06"]["n_episodes"], "seed": canon["06"]["seed"],
            "device": canon["06"]["runtime"]["device"],
            "num_threads": canon["06"]["runtime"]["num_threads"],
            "env_version": canon["06"]["env_version"],
            "opponent": canon["06"]["opponent"],
            "git_commit": canon["06"]["git_commit"],
        },
        "models": models,
        "confirmatory_family": conf,
        "exploratory_structural": expl,
        "role_intervention_series": role_series,
        "head_to_head": head_to_head,
        "cpu_vs_cuda": device,
        "mechanism": mechanism,
        "provenance": {"files_sha256": dict(sorted(_sources.items()))},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", default=str(V10 / "thesis_summary.json"))
    a = ap.parse_args(argv)
    out = Path(a.out)
    summary = build(out)
    text = json.dumps(summary, ensure_ascii=False, indent=1)
    out.write_text(text + "\n", encoding="utf-8")
    kb = out.stat().st_size / 1024
    print(f"[thesis_summary] yazıldı: {out}  ({kb:.1f} KB)")
    if kb >= 200:
        print("[thesis_summary] UYARI: hedef 200 KB'ın üzerinde", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

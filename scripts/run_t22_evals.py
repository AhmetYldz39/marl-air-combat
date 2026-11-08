"""
scripts/run_t22_evals.py
========================
T2.2 — Unified runner ile tüm 6 baseline evalue yeniden çalıştır.
Her model için 500 ep, seed=42, deterministic, 2v2 vs heuristic.

Sonuçlar docs/evals/v0.1/ altına kaydedilir ve yayımlanan CI ile karşılaştırılır.
CI dışına çıkan herhangi bir model rapor edilir ve o noktada durulur.

Kullanım:
    python -X utf8 scripts/run_t22_evals.py
    python -X utf8 scripts/run_t22_evals.py --episodes 500 --skip qmix
"""

import argparse
import json
import sys
import time
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aircombat_bench.eval.runner import run_eval, validate_schema_v1, print_summary

# ── Authoritative baselines (benchmark spec §1) ──────────────────────────────
PUBLISHED = {
    "mappo": {
        "checkpoint": "checkpoints/mappo_final.pt",
        "out":        "docs/evals/v0.1/01_mappo_baseline_500ep.json",
        "win_ci95":   (0.421, 0.508),
        "win_rate":   0.464,
    },
    "mappo_gat": {
        "checkpoint": "checkpoints/mappo_gat_ep37000.pt",
        "out":        "docs/evals/v0.1/02_mappo_gat_500ep.json",
        "win_ci95":   (0.656, 0.737),
        "win_rate":   0.698,
    },
    "mappo_gat_om": {
        "checkpoint": "checkpoints/mappo_gat_ep44000.pt",
        "out":        "docs/evals/v0.1/03_mappo_gat_om_role_500ep.json",
        "win_ci95":   (0.667, 0.746),
        "win_rate":   0.708,
    },
    "qmix": {
        "checkpoint": "checkpoints/qmix_ep2000.pt",
        "out":        "docs/evals/v0.1/04_qmix_ep2000_500ep.json",
        "win_ci95":   (0.607, 0.690),
        "win_rate":   0.650,
    },
    "facmac": {
        "checkpoint": "checkpoints/facmac_base_ep2000.pt",
        "out":        "docs/evals/v0.1/05_facmac_base_500ep.json",
        "win_ci95":   (0.784, 0.851),
        "win_rate":   0.820,
    },
    "facmac_omr": {
        "checkpoint": "checkpoints/facmac_omr_ep27000.pt",
        "out":        "docs/evals/v0.1/06_facmac_omr_500ep.json",
        "win_ci95":   (0.807, 0.871),
        "win_rate":   0.842,
    },
}


def check_ci(out: dict, pub: dict) -> tuple:
    """Sonuç yayımlanan CI içinde mi?"""
    win  = out["metrics"]["win_rate"]
    ci   = out["metrics"]["win_ci95"]
    lo_pub, hi_pub = pub["win_ci95"]
    within = lo_pub - 0.001 <= win <= hi_pub + 0.001
    return within, win, ci, (lo_pub, hi_pub)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--device",   default="auto")
    p.add_argument("--skip",     nargs="*", default=[],
                   help="Atlanacak model etiketleri (örn. --skip qmix mappo)")
    args = p.parse_args()

    with open(ROOT / "configs/config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    out_dir = ROOT / "docs/evals/v0.1"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"T2.2 — Unified Runner Baseline Eval")
    print(f"        episodes={args.episodes}  seed={args.seed}  device={args.device}")
    print("=" * 70)

    results_summary = []

    for alg, pub in PUBLISHED.items():
        if alg in args.skip:
            print(f"\n[SKIP] {alg}")
            continue

        ckpt_path = ROOT / pub["checkpoint"]
        if not ckpt_path.exists():
            print(f"\n[MISS] {alg} — checkpoint not found: {ckpt_path}")
            continue

        out_path = ROOT / pub["out"]
        if out_path.exists():
            print(f"\n[SKIP] {alg} — output already exists: {out_path}")
            print("       Sil veya --skip ile atla")
            continue

        print(f"\n{'─'*70}")
        print(f"[RUN] {alg}  |  {ckpt_path.name}")
        t0 = time.time()

        output = run_eval(
            config=config,
            checkpoint_path=str(ckpt_path),
            n_episodes=args.episodes,
            seed=args.seed,
            device_str=args.device,
            deterministic=True,
            verbose=True,
        )

        elapsed = time.time() - t0
        print_summary(output)

        # ── CI kontrolü ───────────────────────────────────────────────────────
        ok_schema, schema_errs = validate_schema_v1(output)
        within, win, ci, (lo_pub, hi_pub) = check_ci(output, pub)

        ci_status = "IN CI" if within else "OUT OF CI"
        print(f"\n[CI CHECK] {alg}: win={win:.3f}  "
              f"computed=[{ci[0]:.3f},{ci[1]:.3f}]  "
              f"published=[{lo_pub:.3f},{hi_pub:.3f}]  → {ci_status}")
        print(f"[TIMING]   {elapsed:.0f}s ({elapsed/args.episodes:.2f}s/ep)")

        results_summary.append({
            "algorithm": alg,
            "win_rate":  win,
            "win_ci95":  ci,
            "published_ci": list(pub["win_ci95"]),
            "within_ci": within,
            "elapsed_s": round(elapsed, 1),
        })

        if not within:
            print(f"\n⛔  {alg} published CI dışında!")
            print("   Olası nedenler: env değişikliği, reward ağırlıkları, seed.")
            print("   T2.2 kuralı: durduruluyor — override yapma.")
            # Sonuçları kaydet (CI dışı olsa bile, inceleme için)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            print(f"   Kaydedildi (inceleme için): {out_path}")
            break

        if not ok_schema:
            print(f"⚠️  Schema hatası: {schema_errs}")

        # Kaydet
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"[SAVE] {out_path}")

    # ── Özet tablosu ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("T2.2 ÖZET")
    print("=" * 70)
    print(f"{'Model':<20} {'Win%':>6} {'Pub%':>6} {'Status':<12} {'Time':>8}")
    print("─" * 70)
    for r in results_summary:
        pub_win = PUBLISHED[r["algorithm"]]["win_rate"]
        status  = "✓ IN CI" if r["within_ci"] else "✗ OUT CI"
        print(f"{r['algorithm']:<20} {r['win_rate']:>6.1%} {pub_win:>6.1%} "
              f"{status:<12} {r['elapsed_s']:>7.0f}s")

    all_within = all(r["within_ci"] for r in results_summary)
    print("─" * 70)
    print(f"\nSonuç: {'TÜMDÜ BAŞARILI ✅' if all_within else 'BAZI MODELLER CI DIŞI ❌'}")


if __name__ == "__main__":
    main()

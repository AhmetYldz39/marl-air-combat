"""
aircombat_bench/eval/compare.py
===============================
Matched-pair (eşleştirilmiş) model karşılaştırması — T2.6.

Neden gerekli
-------------
Benchmark'taki her model **aynı episode tohumlarıyla** değerlendirilir: episode
*i* her modele aynı başlangıç koşulunu sunar.  Bu yüzden iki model arasındaki
karşılaştırma bağımsız iki örneklem değil, **eşleştirilmiş** gözlemlerdir.

İki-oran z-testi, her iki modelin aynı sonucu verdiği episode'ları — ki veri
setinin çoğu odur — bilgi olarak atar ve bu tasarımda ciddi biçimde
güçsüzdür (underpowered).  Eşleştirilmiş ikili sonuç için doğru test
**McNemar testidir**.

Wilson aralıklarının yerini almaz
---------------------------------
Eval JSON'larındaki Wilson aralıkları her modelin sabit rakibe karşı kendi
kazanma oranını tanımlar; bu ayrı ve hâlâ doğru bir niceliktir.  Bu modül
eşleştirilmiş analizi onun *yanına* ekler.

**Örtüşen Wilson aralıkları, eşleştirilmiş tasarımda bir fark testi
DEĞİLDİR.** İki modeli karşılaştırmak için bu modül kullanılmalıdır.

Kullanım
--------
    python -m aircombat_bench.eval.compare A.json B.json
    from aircombat_bench.eval.compare import compare_files
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "CONFIRMATORY_FAMILY",
    "FAMILIES",
    "FAMILY_CONFIRMATORY",
    "FAMILY_DEVICE",
    "FAMILY_EXPLORATORY",
    "FamilyError",
    "holm_confirmatory",
    "ComparabilityError",
    "SideBiasError",
    "compare_files",
    "compare",
    "combine_head_to_head",
    "agent_level_asymmetry",
    "mcnemar",
    "paired_bootstrap_ci",
    "sign_test",
    "holm",
    "format_report",
]


# ---------------------------------------------------------------------------
# Çokluk aileleri (T2.10 §3) — bir karşılaştırma TAM OLARAK bir aileye aittir
# ---------------------------------------------------------------------------

FAMILY_CONFIRMATORY = "confirmatory"
FAMILY_EXPLORATORY  = "exploratory"
FAMILY_DEVICE       = "device"
FAMILIES = (FAMILY_CONFIRMATORY, FAMILY_EXPLORATORY, FAMILY_DEVICE)

#: **DOĞRULAYICI AİLE — üyeliği sabittir.**
#: Bildirinin manşet iddialarını taşıyan yöntem karşılaştırmaları.  Holm
#: düzeltmesi YALNIZCA bu aile içinde uygulanır.
#:
#: Ablasyonlar (06↔08, 03↔09) ve rol müdahale serisi bu ailede DEĞİLDİR:
#: onlar mekanizma araştırmasıdır, bir performans iddiasını taşımazlar ve
#: keşifsel olarak raporlanırlar.  Bu ayrım, aynı karşılaştırmanın iki farklı
#: Holm hükmü alması sorununu kaynağında ortadan kaldırır.
CONFIRMATORY_FAMILY = (
    "MAPPO → MAPPO+GAT",
    "MAPPO+GAT → MAPPO+GAT+OM+Role",
    "QMIX → MAPPO+GAT",
    "FACMAC → FACMAC+OM+R",
    "MAPPO+GAT → FACMAC",
)


class FamilyError(ValueError):
    """Çokluk ailesi ihlali — bir karşılaştırma yanlış ailede düzeltiliyor."""


def holm_confirmatory(pvals: Dict[str, float]) -> Dict[str, float]:
    """Holm düzeltmesini YALNIZCA doğrulayıcı aile içinde uygula.

    Üyeliği doğrular: doğrulayıcı ailede olmayan bir karşılaştırma buraya
    girerse hata verir.  Keşifsel karşılaştırmalar düzeltilmez — ham p ve CI
    ile raporlanırlar (T2.10 §3).
    """
    unknown = set(pvals) - set(CONFIRMATORY_FAMILY)
    if unknown:
        raise FamilyError(
            f"Doğrulayıcı ailede olmayan karşılaştırma(lar): {sorted(unknown)}. "
            f"Keşifsel karşılaştırmalar Holm ile düzeltilmez; ham p ve CI ile "
            f"raporlanır.  Doğrulayıcı aile: {list(CONFIRMATORY_FAMILY)}"
        )
    return holm(pvals)


class ComparabilityError(ValueError):
    """İki eval dosyası eşleştirilmiş karşılaştırma için uyumlu değil.

    Uyarı değil hata: uyumsuz dosyaların karşılaştırılması sessizce yanlış
    bir istatistik üretir.
    """


class SideBiasError(ValueError):
    """Tek yönlü bir agent-vs-agent sonucu aynalanmış koşu olmadan raporlanamaz.

    Ortam yan-simetrik değildir: `_critical_recovery_blue` yalnızca blue
    tarafına uygulanır (`dogfight_env.py:308`).  Scripted red kendi
    kurtarmasına sahiptir; checkpoint sürülen red hiçbirini almaz.  Ölçülen
    yan etkisi ~3.2–3.5 pp'dir — raporlanan farkların mertebesinde.
    """


# ---------------------------------------------------------------------------
# Önkoşullar
# ---------------------------------------------------------------------------

#: Eşit olması gereken üst-düzey alanlar.
_MATCH_FIELDS = ("scenario", "opponent", "n_episodes", "seed", "seeding",
                 "env_version")


def _check_comparable(a: dict, b: dict, name_a: str, name_b: str,
                      device_comparison: bool = False) -> None:
    """Eşleştirilmiş karşılaştırma önkoşullarını doğrula; ihlalde hata fırlat.

    device_comparison=True ise cihazın FARKLI olmasına izin verilir — çünkü o
    modda cihaz karıştırıcı değil, ölçülen müdahaledir.  Bu durumda checkpoint
    hash'inin AYNI olması ayrıca zorunlu kılınır: aynı model, aynı episode
    tohumları, farklı donanım.
    """
    problems: List[str] = []

    for f in _MATCH_FIELDS:
        va, vb = a.get(f, "<yok>"), b.get(f, "<yok>")
        if va != vb:
            problems.append(f"{f}: {va!r} != {vb!r}")

    da = a.get("runtime", {}).get("device", "<yok>")
    db = b.get("runtime", {}).get("device", "<yok>")
    if device_comparison:
        # Cihaz müdahale olduğu için farklı OLMALI; buna karşılık model aynı
        # olmalı, yoksa cihaz etkisi ile model etkisi ayrıştırılamaz.
        if da == db:
            problems.append(
                f"device_comparison=True ama iki dosya da {da!r} — cihaz "
                "karşılaştırması için cihazlar farklı olmalı"
            )
        sa = a.get("checkpoint", {}).get("sha256")
        sb = b.get("checkpoint", {}).get("sha256")
        if sa != sb:
            problems.append(
                f"device_comparison=True iken checkpoint aynı olmalı: "
                f"{(sa or '?')[:12]}… != {(sb or '?')[:12]}… — aksi hâlde "
                "cihaz etkisi model etkisinden ayrıştırılamaz"
            )
    else:
        # Normal (yöntem) karşılaştırmasında cihaz bir karıştırıcıdır ve
        # ölçülen etkilerle aynı mertebededir (bkz. EVAL_PROTOCOL_AUDIT.md).
        if da != db:
            problems.append(
                f"runtime.device: {da!r} != {db!r} — cihaz etkisi ölçülen "
                f"etkilerle aynı mertebede; CPU ve CUDA dosyaları "
                f"karşılaştırılamaz.  Cihaz etkisini ÖLÇMEK istiyorsanız "
                f"device_comparison=True (CLI: --device-comparison) kullanın."
            )

    ea, eb = a.get("episodes", []), b.get("episodes", [])
    if len(ea) != len(eb):
        problems.append(f"episode sayısı: {len(ea)} != {len(eb)}")
    else:
        # ep_seed dizisi eleman eleman aynı olmalı.  T2.6.4 sonrası legacy
        # dosyalarda ep_seed None'dır; o durumda stream_position kullanılır.
        key = "ep_seed"
        if ea and ea[0].get("ep_seed") is None:
            key = "stream_position"
        sa = [r.get(key) for r in ea]
        sb = [r.get(key) for r in eb]
        if sa != sb:
            n_diff = sum(1 for x, y in zip(sa, sb) if x != y)
            problems.append(
                f"{key} dizisi aynı değil ({n_diff} konumda fark) — "
                f"episode'lar eşleşmiyor"
            )

    if problems:
        raise ComparabilityError(
            f"{name_a} ve {name_b} eşleştirilmiş karşılaştırma için uyumsuz:\n  - "
            + "\n  - ".join(problems)
        )


# ---------------------------------------------------------------------------
# İstatistikler
# ---------------------------------------------------------------------------

def mcnemar(b01: int, b10: int, continuity: bool = True) -> Tuple[float, float]:
    """McNemar χ² ve p-değeri.

    b01 : ikinci modelin kazandığı, birincinin kazanmadığı episode sayısı
    b10 : tersi

    Süreklilik düzeltmesi (Edwards) varsayılan olarak açıktır.
    """
    n = b01 + b10
    if n == 0:
        return 0.0, 1.0
    diff = abs(b01 - b10)
    if continuity:
        diff = max(0.0, diff - 1.0)
    chi2 = (diff * diff) / n
    # 1 serbestlik dereceli χ² üst kuyruk: erfc(sqrt(chi2/2))
    p = math.erfc(math.sqrt(chi2 / 2.0))
    return float(chi2), float(p)


def sign_test(b01: int, b10: int) -> float:
    """Uyuşmayan çiftler üzerinde iki-kuyruklu tam binom (işaret) testi."""
    n = b01 + b10
    if n == 0:
        return 1.0
    k = min(b01, b10)
    # P(X <= k) + P(X >= n-k)  with p=0.5  →  2 * P(X <= k), 1.0'a kırpılmış
    cum = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return float(min(1.0, 2.0 * cum))


def paired_bootstrap_ci(
    diff: Sequence[float],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    """Episode indeksleri üzerinden yeniden örnekleyerek paired bootstrap %95 CI.

    Yeniden örnekleme **episode indeksleri** üzerindedir, modeller üzerinde
    değil — eşleştirme bu sayede korunur.
    """
    d = np.asarray(diff, dtype=np.float64)
    n = d.size
    if n == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    """Holm-Bonferroni düzeltmesi. {ad: p} → {ad: düzeltilmiş p}."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out: Dict[str, float] = {}
    running = 0.0
    for i, (name, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)      # monotonluk zorunlu
        out[name] = running
    return out


# ---------------------------------------------------------------------------
# Ana karşılaştırma
# ---------------------------------------------------------------------------

def _wins(episodes: List[dict]) -> np.ndarray:
    return np.array([1 if r.get("outcome") == "win" else 0 for r in episodes],
                    dtype=np.int64)


def _kills(episodes: List[dict]) -> np.ndarray:
    return np.array([float(r.get("kills", 0)) for r in episodes],
                    dtype=np.float64)


def compare(a: dict, b: dict, name_a: str = "A", name_b: str = "B",
            n_boot: int = 10_000, bootstrap_seed: int = 42,
            device_comparison: bool = False,
            family: Optional[str] = None) -> Dict[str, Any]:
    """İki schema v1 eval çıktısını eşleştirilmiş olarak karşılaştır.

    b, "ikinci"/"yeni" modeldir; farklar b − a yönündedir.

    device_comparison=True → cihaz karşılaştırması modu (T2.8.1): aynı
    checkpoint, aynı episode tohumları, farklı donanım.  Opt-in'dir ve çıktı
    `comparison_type: "device"` ile etiketlenir; bir yöntem karşılaştırmasıyla
    karıştırılamaz.

    family → bu karşılaştırmanın çokluk ailesi (T2.10 §3).  Çıktıya kaydedilir
    ki hiçbir karşılaştırma sessizce iki kez düzeltilemesin.  Belirtilmezse
    cihaz modunda "device", aksi hâlde "unspecified" yazılır — ve
    "unspecified" bir karşılaştırma Holm ailesine sokulmamalıdır.
    """
    if family is not None and family not in FAMILIES:
        raise FamilyError(f"Bilinmeyen aile: {family!r}. Geçerli: {FAMILIES}")
    _check_comparable(a, b, name_a, name_b, device_comparison=device_comparison)

    ea, eb = a["episodes"], b["episodes"]
    wa, wb = _wins(ea), _wins(eb)
    ka, kb = _kills(ea), _kills(eb)
    n = wa.size

    # 2×2 eşleşmiş tablo
    b11 = int(np.sum((wa == 1) & (wb == 1)))   # ikisi de kazandı
    b10 = int(np.sum((wa == 1) & (wb == 0)))   # yalnızca A kazandı
    b01 = int(np.sum((wa == 0) & (wb == 1)))   # yalnızca B kazandı
    b00 = int(np.sum((wa == 0) & (wb == 0)))   # ikisi de kazanmadı

    chi2, p_mcnemar = mcnemar(b01, b10)
    p_sign = sign_test(b01, b10)

    p_a, p_b = float(wa.mean()), float(wb.mean())
    win_diff = p_b - p_a
    win_lo, win_hi = paired_bootstrap_ci(wb - wa, n_boot, seed=bootstrap_seed)

    kill_diff = float(kb.mean() - ka.mean())
    kill_lo, kill_hi = paired_bootstrap_ci(kb - ka, n_boot, seed=bootstrap_seed)

    agree_obs = float(np.mean(wa == wb))
    agree_indep = p_a * p_b + (1 - p_a) * (1 - p_b)

    return {
        # Karşılaştırmanın TÜRÜ — cihaz karşılaştırması bir yöntem
        # karşılaştırmasıyla karıştırılmamalı.
        "comparison_type": "device" if device_comparison else "method",
        # T2.10 §3 — bir karşılaştırma tam olarak bir aileye aittir.
        "family": family or (FAMILY_DEVICE if device_comparison else "unspecified"),
        **({"devices": {"a": a.get("runtime", {}).get("device"),
                        "b": b.get("runtime", {}).get("device")},
            "device_comparison_note": (
                "Aynı checkpoint ve aynı episode tohumları, farklı donanım. "
                "Bu bir YÖNTEM karşılaştırması DEĞİLDİR."
            )} if device_comparison else {}),
        "model_a": name_a,
        "model_b": name_b,
        "n_episodes": int(n),
        # Sonucu değişen (flip eden) episode sayısı
        "outcome_flips": int(np.sum(wa != wb)),
        "win_rate_a": p_a,
        "win_rate_b": p_b,
        "matched_table": {"b11": b11, "b10": b10, "b01": b01, "b00": b00},
        "mcnemar": {"chi2": chi2, "p_value": p_mcnemar,
                    "continuity_correction": True},
        "sign_test": {"p_value": p_sign},
        "win_rate_diff": {
            "estimate": win_diff,
            "ci95_paired_bootstrap": [win_lo, win_hi],
        },
        "kill_per_ep_diff": {
            "estimate": kill_diff,
            "ci95_paired_bootstrap": [kill_lo, kill_hi],
        },
        "agreement": {
            "observed": agree_obs,
            "expected_under_independence": agree_indep,
        },
        "bootstrap": {"n_resamples": int(n_boot), "seed": int(bootstrap_seed),
                      "resampled": "episode_indices"},
        # T2.7.3 — tek yönlü agent-vs-agent sonucu etiketlenmeden raporlanamaz.
        "side_biased": _is_head_to_head(a) or _is_head_to_head(b),
    }


def paired_metric_diff(a: dict, b: dict, values_a: Sequence[float],
                       values_b: Sequence[float], n_boot: int = 10_000,
                       bootstrap_seed: int = 42) -> Dict[str, Any]:
    """Keyfi bir per-episode büyüklük için eşleştirilmiş fark + bootstrap CI.

    Kill imbalance gibi, episode başına türetilen niceliklerde kullanılır.
    """
    va = np.asarray(values_a, dtype=np.float64)
    vb = np.asarray(values_b, dtype=np.float64)
    if va.size != vb.size:
        raise ComparabilityError(
            f"değer dizileri farklı uzunlukta: {va.size} != {vb.size}")
    lo, hi = paired_bootstrap_ci(vb - va, n_boot, seed=bootstrap_seed)
    return {
        "mean_a": float(va.mean()),
        "mean_b": float(vb.mean()),
        "estimate": float(vb.mean() - va.mean()),
        "ci95_paired_bootstrap": [lo, hi],
    }


def compare_files(path_a: str, path_b: str, **kw) -> Dict[str, Any]:
    """İki schema v1 JSON dosyasını yükle ve karşılaştır."""
    pa, pb = Path(path_a), Path(path_b)
    a = json.loads(pa.read_text(encoding="utf-8"))
    b = json.loads(pb.read_text(encoding="utf-8"))
    return compare(a, b, pa.stem, pb.stem, **kw)


# ---------------------------------------------------------------------------
# Ajan-düzeyi rol asimetrisi (T2.7 / yapısal ablasyon)
# ---------------------------------------------------------------------------

def agent_level_asymmetry(d: dict) -> float:
    """|mean(kills_agent0) − mean(kills_agent1)| — rol uzmanlaşması metriği.

    DİKKAT — bu, episode-içi yoğunlaşmadan (`mean(|k0 − k1|)`) FARKLI bir
    niceliktir ve ikisi karıştırılmamalıdır:

      * Bu fonksiyon iki ajanın kill oranlarının koşu boyunca ne kadar
        ayrıştığını ölçer.  Rol uzmanlaşmasının taşıyıcı metriği budur ve
        yayımlanan yapısal ablasyon (08/09) bu tanımı kullanır.
      * `mean(|k0 − k1|)` ise tek bir episode içinde kill'lerin bir ajanda
        toplanıp toplanmadığını ölçer.  Ajanlar ortalamada tamamen simetrik
        olsa bile bu büyük çıkar ve rol sinyaline duyarsızdır.
    """
    a = np.array([e["kills_by_agent"] for e in d["episodes"]], dtype=np.float64)
    return float(abs(a[:, 0].mean() - a[:, 1].mean()))


def paired_asymmetry_diff(a: dict, b: dict, n_boot: int = 10_000,
                          bootstrap_seed: int = 42) -> Dict[str, Any]:
    """İki koşul arasında ajan-düzeyi asimetri farkı + paired bootstrap CI."""
    _check_comparable(a, b, "A", "B")
    ka = np.array([e["kills_by_agent"] for e in a["episodes"]], dtype=np.float64)
    kb = np.array([e["kills_by_agent"] for e in b["episodes"]], dtype=np.float64)
    n = ka.shape[0]
    rng = np.random.default_rng(bootstrap_seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    da = np.abs(ka[:, 0][idx].mean(1) - ka[:, 1][idx].mean(1))
    db = np.abs(kb[:, 0][idx].mean(1) - kb[:, 1][idx].mean(1))
    diff = db - da
    return {
        "asymmetry_a": agent_level_asymmetry(a),
        "asymmetry_b": agent_level_asymmetry(b),
        "estimate": agent_level_asymmetry(b) - agent_level_asymmetry(a),
        "ci95_paired_bootstrap": [float(np.percentile(diff, 2.5)),
                                  float(np.percentile(diff, 97.5))],
    }


# ---------------------------------------------------------------------------
# Head-to-head — yan yanlılığına karşı koruma (T2.7.3)
# ---------------------------------------------------------------------------

def _is_head_to_head(d: dict) -> bool:
    return str(d.get("opponent", "")).startswith("checkpoint:")


def combine_head_to_head(a: dict, b: dict,
                         name_a: str = "A", name_b: str = "B") -> Dict[str, Any]:
    """İki aynalanmış agent-vs-agent koşusunu iki-yönlü ortalamaya indirger.

    `a` ve `b` birbirinin aynası olmalıdır: a'da blue olan model b'de red,
    a'da red olan model b'de blue.  Tek yönlü bir sonuç raporlanabilir
    değildir — ortam yan-simetrik olmadığı için (`SideBiasError` bkz.).

    Döndürülen sözlük hem birleşik oranları hem de her iki yanı ayrı ayrı
    verir; yan etkisi gizlenmez, ölçülüp raporlanır.
    """
    for d, nm in ((a, name_a), (b, name_b)):
        if not _is_head_to_head(d):
            raise SideBiasError(
                f"{nm}: agent-vs-agent koşusu değil "
                f"(opponent={d.get('opponent')!r}); combine_head_to_head "
                "yalnızca checkpoint rakipli koşular içindir"
            )

    a_blue = a["checkpoint"]["sha256"]
    a_red  = a.get("opponent_checkpoint", {}).get("sha256")
    b_blue = b["checkpoint"]["sha256"]
    b_red  = b.get("opponent_checkpoint", {}).get("sha256")
    if not (a_blue == b_red and a_red == b_blue):
        raise SideBiasError(
            f"{name_a} ve {name_b} birbirinin aynası değil: "
            f"{name_a} blue={a_blue[:12]}… red={(a_red or '?')[:12]}… / "
            f"{name_b} blue={b_blue[:12]}… red={(b_red or '?')[:12]}…  "
            "İki-yönlü ortalama yalnızca aynalanmış koşulardan üretilebilir."
        )
    if a["n_episodes"] != b["n_episodes"] or a["seed"] != b["seed"]:
        raise SideBiasError("aynalanmış koşular aynı n_episodes ve seed olmalı")
    da = a.get("runtime", {}).get("device")
    db = b.get("runtime", {}).get("device")
    if da != db:
        raise SideBiasError(f"cihazlar farklı: {da!r} != {db!r}")

    n = a["n_episodes"]
    # a: model-X blue.  b: model-X red (yani b'nin kayıpları X'in galibiyeti).
    x_blue_wins, x_red_wins = a["wins"], b["losses"]
    y_blue_wins, y_red_wins = b["wins"], a["losses"]
    draws = a["draws"] + b["draws"]
    total = 2 * n

    x_blue_rate, x_red_rate = x_blue_wins / n, x_red_wins / n
    y_blue_rate, y_red_rate = y_blue_wins / n, y_red_wins / n

    return {
        "n_total": total,
        "model_x": {
            "name": name_a,
            "checkpoint_sha256": a_blue,
            "win_rate_two_sided": (x_blue_wins + x_red_wins) / total,
            "win_rate_as_blue": x_blue_rate,
            "win_rate_as_red": x_red_rate,
            "side_effect_pp": (x_blue_rate - x_red_rate) * 100.0,
        },
        "model_y": {
            "name": name_b,
            "checkpoint_sha256": b_blue,
            "win_rate_two_sided": (y_blue_wins + y_red_wins) / total,
            "win_rate_as_blue": y_blue_rate,
            "win_rate_as_red": y_red_rate,
            "side_effect_pp": (y_blue_rate - y_red_rate) * 100.0,
        },
        "draw_rate": draws / total,
        "side_bias_note": (
            "Ortam yan-simetrik değildir (dogfight_env.py:308): OOB/zemin "
            "kurtarma yalnızca blue tarafına uygulanır ve checkpoint sürülen "
            "red hiç kurtarma almaz.  İki-yönlü ortalama tam da bunu telafi "
            "etmek için raporlanır; tek yönlü sayı yanlıdır."
        ),
    }


# ---------------------------------------------------------------------------
# Raporlama
# ---------------------------------------------------------------------------

def format_report(r: Dict[str, Any]) -> str:
    """İnsan okunur rapor metni."""
    t = r["matched_table"]
    L: List[str] = []
    if r.get("comparison_type") == "device":
        d = r.get("devices", {})
        L.append(f"CİHAZ KARŞILAŞTIRMASI (yöntem karşılaştırması DEĞİL): "
                 f"{d.get('a')} → {d.get('b')}")
        L.append(f"  model: {r['model_a']}  —  aynı checkpoint, aynı tohumlar")
    else:
        L.append(f"Eşleştirilmiş karşılaştırma: {r['model_a']}  →  {r['model_b']}")
    L.append(f"  n = {r['n_episodes']} eşleşmiş episode   "
             f"sonucu değişen: {r.get('outcome_flips', '?')}")
    fam = r.get("family", "unspecified")
    L.append(f"  çokluk ailesi: {fam}"
             + ("   (keşifsel — ham p ve CI ile raporlanır, Holm uygulanmaz)"
                if fam == FAMILY_EXPLORATORY else
                "   (doğrulayıcı — Holm bu aile içinde uygulanır)"
                if fam == FAMILY_CONFIRMATORY else
                "   ⚠ aile belirtilmedi — Holm ailesine sokulmamalı"
                if fam == "unspecified" else ""))
    L.append("")
    L.append(f"  win_rate  A = {r['win_rate_a']*100:.2f}%   "
             f"B = {r['win_rate_b']*100:.2f}%")
    d = r["win_rate_diff"]
    L.append(f"  fark (B−A) = {d['estimate']*100:+.2f} pp   "
             f"paired bootstrap CI95 = "
             f"[{d['ci95_paired_bootstrap'][0]*100:+.2f}, "
             f"{d['ci95_paired_bootstrap'][1]*100:+.2f}] pp")
    k = r["kill_per_ep_diff"]
    L.append(f"  kill/ep farkı = {k['estimate']:+.4f}   "
             f"CI95 = [{k['ci95_paired_bootstrap'][0]:+.4f}, "
             f"{k['ci95_paired_bootstrap'][1]:+.4f}]")
    L.append("")
    L.append("  2×2 eşleşmiş tablo (kazandı / kazanmadı):")
    L.append(f"      ikisi de kazandı      b11 = {t['b11']}")
    L.append(f"      yalnızca A kazandı    b10 = {t['b10']}")
    L.append(f"      yalnızca B kazandı    b01 = {t['b01']}")
    L.append(f"      ikisi de kazanmadı    b00 = {t['b00']}")
    L.append("")
    m = r["mcnemar"]
    L.append(f"  McNemar (süreklilik düzeltmeli): χ² = {m['chi2']:.2f}   "
             f"p = {m['p_value']:.3g}")
    L.append(f"  İşaret testi (tam binom)       : p = {r['sign_test']['p_value']:.3g}")
    a = r["agreement"]
    L.append("")
    L.append(f"  Sonuç uyuşması: gözlenen {a['observed']*100:.1f}%  vs  "
             f"bağımsızlık altında beklenen {a['expected_under_independence']*100:.1f}%")
    L.append("  (aradaki fark eşleştirmenin neden önemli olduğunu gösterir)")
    if r.get("side_biased"):
        L.append("")
        L.append("  ⚠ YAN YANLI — TEK YÖNLÜ AGENT-VS-AGENT SONUCU")
        L.append("    Ortam yan-simetrik değil (dogfight_env.py:308): kurtarma")
        L.append("    yalnızca blue'ya uygulanır, checkpoint red hiç almaz.")
        L.append("    Ölçülen yan etkisi ~3.2–3.5 pp.  Raporlanabilir sayı için")
        L.append("    aynalanmış koşuyla combine_head_to_head() kullanın.")
    return "\n".join(L)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Eşleştirilmiş (matched-pair) model karşılaştırması — "
                    "McNemar + paired bootstrap.",
    )
    p.add_argument("file_a", help="Temel model (schema v1 JSON)")
    p.add_argument("file_b", help="Karşılaştırılan model (schema v1 JSON)")
    p.add_argument("--json", action="store_true", help="Ham JSON çıktısı ver")
    p.add_argument("--n-boot", type=int, default=10_000,
                   help="Bootstrap yeniden örnekleme sayısı (varsayılan 10000)")
    p.add_argument(
        "--family", choices=list(FAMILIES), default=None,
        help="Bu karşılaştırmanın çokluk ailesi (T2.10 §3).  Çıktıya kaydedilir; "
             "bir karşılaştırma tam olarak bir aileye aittir ve iki kez "
             "düzeltilemez.  Holm YALNIZCA confirmatory aile içinde uygulanır.",
    )
    p.add_argument(
        "--device-comparison", action="store_true", dest="device_comparison",
        help="CİHAZ karşılaştırması modu: aynı checkpoint ve aynı episode "
             "tohumlarıyla farklı donanımı karşılaştırır.  Yalnızca cihaz "
             "önkoşulu gevşetilir; diğer tüm önkoşullar ve ek olarak "
             "checkpoint hash eşitliği zorunlu kalır.  Çıktı cihaz "
             "karşılaştırması olarak etiketlenir.",
    )
    args = p.parse_args(argv)

    r = compare_files(args.file_a, args.file_b, n_boot=args.n_boot,
                      device_comparison=args.device_comparison,
                      family=args.family)
    if args.json:
        print(json.dumps(r, indent=2, ensure_ascii=False))
    else:
        print(format_report(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

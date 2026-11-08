"""
tests/test_schema_integrity.py — T2.6.3 / T2.6.4 / T2.6.6

Yayımlanan eval dosyalarının taşınabilirlik ve provenance bütünlüğü.

Bu testler dosyaların *içeriğini* değil, yeniden üretilebilirlik iddialarını
doğrular:
  • T2.6.3 — hiçbir schema v1 dosyasında mutlak yol ya da ev dizini olmamalı
  • T2.6.4 — legacy seeding'de ep_seed izole replay vaat etmemeli
  • T2.6.6 — kaydedilen commit depo geçmişinde bulunmalı, ağaç temiz olmalı
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from aircombat_bench.eval.runner import path_is_portable

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIRS = [ROOT / "docs" / "evals" / "v0.1", ROOT / "docs" / "evals" / "v1.0"]


def _schema_v1_files():
    """schema_version taşıyan tüm eval JSON'ları."""
    out = []
    for d in EVAL_DIRS:
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if data.get("schema_version"):
                out.append((p, data))
    return out


_FILES = _schema_v1_files()
_IDS = [str(p.relative_to(ROOT)).replace("\\", "/") for p, _ in _FILES]


def test_schema_v1_files_exist():
    assert _FILES, "Doğrulanacak schema v1 dosyası bulunamadı"


# ---------------------------------------------------------------------------
# T2.6.3 — taşınabilir yollar
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path,data", _FILES, ids=_IDS)
def test_checkpoint_path_is_portable(path, data):
    """Mutlak yol veya ev dizini içeren checkpoint.path testi düşürür."""
    p = data.get("checkpoint", {}).get("path", "")
    assert path_is_portable(p), (
        f"{path.name}: checkpoint.path taşınabilir değil: {p!r}. "
        "Depo köküne göreceli olmalı."
    )


@pytest.mark.parametrize("path,data", _FILES, ids=_IDS)
def test_opponent_checkpoint_path_is_portable(path, data):
    opp = data.get("opponent_checkpoint")
    if not isinstance(opp, dict):
        pytest.skip("checkpoint rakip yok")
    assert path_is_portable(opp.get("path", "")), (
        f"{path.name}: opponent_checkpoint.path taşınabilir değil: "
        f"{opp.get('path')!r}"
    )


@pytest.mark.parametrize("path,data", _FILES, ids=_IDS)
def test_no_home_directory_anywhere_in_file(path, data):
    """Dosyanın hiçbir yerinde kullanıcı ev dizini geçmemeli."""
    blob = json.dumps(data, ensure_ascii=False)
    for marker in ("/Users/", "\\Users\\", "/home/", "%USERPROFILE%"):
        assert marker not in blob, (
            f"{path.name}: dosya içinde ev dizini işareti {marker!r} bulundu"
        )


def test_path_is_portable_rejects_absolute():
    """Doğrulayıcının kendisi mutlak yolları reddediyor mu?"""
    assert path_is_portable("checkpoints/a.pt")
    assert not path_is_portable("/home/u/a.pt")
    assert not path_is_portable("C:\\Users\\a\\a.pt")
    assert not path_is_portable("c:/Users/a/a.pt")
    assert not path_is_portable("")


# ---------------------------------------------------------------------------
# T2.6.4 — ep_seed replay garantisi
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path,data", _FILES, ids=_IDS)
def test_legacy_seeding_does_not_promise_isolated_replay(path, data):
    """legacy akışında ep_seed None olmalı; episode'u konum tanımlar.

    `seed + idx` legacy akışında tek bir episode'u izole yeniden üretmez —
    o başlangıç koşuluna ulaşmak önceki tüm episode'ları oynatmayı gerektirir.
    Alanı doldurmak, tutulamayan bir söz vermek olur.
    """
    if data.get("seeding") != "legacy":
        pytest.skip("legacy seeding değil")
    eps = data.get("episodes", [])
    if not eps:
        pytest.skip("episode kaydı yok")
    for r in eps[:50]:
        assert r.get("ep_seed") is None, (
            f"{path.name}: seeding=legacy iken ep_seed doldurulmuş "
            f"({r.get('ep_seed')!r}) — izole replay vaat edilemez"
        )
        assert "stream_position" in r, (
            f"{path.name}: stream_position eksik — legacy akışında episode'u "
            "tanımlayan alan budur"
        )


@pytest.mark.parametrize("path,data", _FILES, ids=_IDS)
def test_stream_position_is_sequential(path, data):
    eps = data.get("episodes", [])
    if not eps or "stream_position" not in eps[0]:
        pytest.skip("stream_position yok")
    pos = [r["stream_position"] for r in eps]
    assert pos == list(range(len(eps))), (
        f"{path.name}: stream_position 0..n-1 sırasında değil"
    )


# ---------------------------------------------------------------------------
# T2.6.6 — provenance
# ---------------------------------------------------------------------------

def _commit_exists(sha: str) -> bool:
    try:
        subprocess.check_output(
            ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
            cwd=str(ROOT), stderr=subprocess.DEVNULL, timeout=10,
        )
        return True
    except Exception:
        return False


@pytest.mark.parametrize("path,data", _FILES, ids=_IDS)
def test_recorded_commit_exists_in_history(path, data):
    """Kaydedilen commit depo geçmişinde bulunmalı."""
    sha = data.get("git_commit")
    if not sha or sha == "unknown":
        pytest.skip("git_commit kaydedilmemiş (eski dosya)")
    assert _commit_exists(sha), (
        f"{path.name}: kaydedilen commit {sha!r} depo geçmişinde yok — "
        "çıktı bu commit'ten yeniden üretilemez"
    )


@pytest.mark.parametrize("path,data", _FILES, ids=_IDS)
def test_canonical_files_not_produced_on_dirty_tree(path, data):
    """Kanonik (v1.0) dosyalar temiz ağaçta üretilmiş olmalı."""
    if "v1.0" not in str(path).replace("\\", "/"):
        pytest.skip("yalnızca kanonik v1.0 dosyaları için")
    assert data.get("git_dirty") is False, (
        f"{path.name}: git_dirty={data.get('git_dirty')!r} — kanonik dosya "
        "commit'lenmiş temiz bir ağaçta üretilmelidir"
    )

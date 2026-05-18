"""
compare_eval.py
===============
mappo_final.pt (Faz-1 baseline) vs mappo_gat_final.pt (GAT) karşılaştırmalı
değerlendirme scripti. Tez "GAT katkısı" bölümü için temel kanıt.

Kullanım:
    python -X utf8 evaluation/compare_eval.py
    python -X utf8 evaluation/compare_eval.py --episodes 100 --seed 42
    python -X utf8 evaluation/compare_eval.py --baseline checkpoints/mappo_final.pt
                                               --gat checkpoints/mappo_gat_final.pt

Çıktılar:
    Konsol  : karşılaştırma tablosu
    JSON    : logs/compare_results.json

Metrikler:
    win_rate         — Blue kazanma oranı
    kill_per_ep      — episode başı ortalama kill
    draw_rate        — beraberlik oranı
    mean_ep_len      — ortalama episode uzunluğu (adım)
    ammo_efficiency  — toplam atış / kill (düşük = verimli)
    wez_intime_ratio — WEZ içinde geçirilen adım oranı

GAT Analizi (sadece GAT modeli, 10 ep):
    fc1_new_norm     — fc1_new ağırlık L2 normu (öğrenme kanıtı)
    msg_l2_near      — mesafeye göre ortalama mesaj L2: yakın (<3000m)
    msg_l2_mid       — orta (3000–8000m)
    msg_l2_far       — uzak (>8000m)
"""

import sys
import json
import argparse
import yaml
import numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

from envs.dogfight_env import DogfightEnv, BLUE, RED
from agents.heuristic_agent import MultiHeuristicPolicy
from utils.normalization import Normalizer
from envs.aircraft_model import (
    STATE_X, STATE_Y, STATE_H, STATE_ALIVE, STATE_AMMO,
)
from envs.geometry_utils import distance_3d, bearing_angle, wrap_to_pi

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------
NEAR_DIST  = 3000.0   # yakın mesafe eşiği (m)
FAR_DIST   = 8000.0   # uzak mesafe eşiği (m)
GAT_ANALYSIS_EPS = 10


# ===========================================================================
# Yardımcı: checkpoint'ten actor yükleme
# ===========================================================================

def _resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_baseline_actor(ckpt_path: str, obs_dim: int, action_dim: int,
                        hidden: int, device: torch.device):
    """MAPPOActor (Faz-1, 50D) yükle."""
    from training.train_mappo import MAPPOActor
    actor = MAPPOActor(obs_dim, action_dim, hidden=hidden).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    ep   = ckpt.get("episode", "?")
    step = ckpt.get("global_step", "?")
    print(f"[Baseline] Yüklendi: {Path(ckpt_path).name}  ep={ep}  step={step}")
    return actor


def load_gat_actor(ckpt_path: str, old_obs_dim: int, new_obs_dim: int,
                   action_dim: int, hidden: int, device: torch.device):
    """GATMAPPOActor (Faz-2, 68D) yükle."""
    from training.train_mappo import GATMAPPOActor
    actor = GATMAPPOActor(old_obs_dim, new_obs_dim, action_dim,
                          hidden=hidden).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    ep   = ckpt.get("episode", "?")
    step = ckpt.get("global_step", "?")
    print(f"[GAT]      Yüklendi: {Path(ckpt_path).name}  ep={ep}  step={step}")
    return actor


def load_gat_comm(ckpt_path: str, node_dim: int, edge_dim: int,
                  n_heads: int, msg_dim: int, device: torch.device):
    """GATComm modülü yükle."""
    from models.gat_comm import GATComm
    comm = GATComm(node_dim=node_dim, edge_dim=edge_dim,
                   n_heads=n_heads, msg_dim=msg_dim).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    comm.load_state_dict(ckpt["gat_comm"])
    comm.eval()
    return comm


# ===========================================================================
# GAT obs uzatma (train_mappo._extend_obs_phase2 kopyası)
# ===========================================================================

def build_gat_edge_feats(env: DogfightEnv, blue_ids: list,
                         wez_range: float) -> np.ndarray:
    N      = len(blue_ids)
    edge   = np.zeros((N, N, 3), dtype=np.float32)
    states = env.get_all_states()
    for i, aid_i in enumerate(blue_ids):
        for j, aid_j in enumerate(blue_ids):
            if i == j:
                continue
            s_i = states.get(aid_i)
            s_j = states.get(aid_j)
            if s_i is None or s_j is None:
                continue
            pos_i = s_i[[STATE_X, STATE_Y, STATE_H]]
            pos_j = s_j[[STATE_X, STATE_Y, STATE_H]]
            dist  = distance_3d(pos_i, pos_j)
            bear  = bearing_angle(pos_i, pos_j)
            ts    = 0.0
            for eid in env.red_ids:
                es = states.get(eid)
                if es is not None and es[STATE_ALIVE] > 0.5:
                    d  = distance_3d(pos_j, es[[STATE_X, STATE_Y, STATE_H]])
                    ts = max(ts, float(np.clip(
                        1.0 - d / (wez_range + 1e-9), 0.0, 1.0
                    )))
            edge[i, j] = [
                float(np.clip(dist / (env.map_size + 1e-9), 0.0, 1.0)),
                float(np.clip(wrap_to_pi(bear) / np.pi, -1.0, 1.0)),
                ts,
            ]
    return edge


def extend_obs_phase2(obs_dict: dict, env: DogfightEnv,
                      gat_comm, base_obs_dim: int,
                      node_dim: int, wez_range: float,
                      device: torch.device) -> dict:
    blue_ids   = [f"blue_{i}" for i in range(env._max_n_per_team)]
    states     = env.get_all_states()
    ego_list   = []
    alive_list = []
    for aid in blue_ids:
        obs = obs_dict.get(aid, np.zeros(base_obs_dim, dtype=np.float32))
        ego_list.append(obs[:node_dim])
        s = states.get(aid)
        alive_list.append(float(s[STATE_ALIVE]) if s is not None else 0.0)

    edge_feats = build_gat_edge_feats(env, blue_ids, wez_range)
    messages   = gat_comm.compute_messages(ego_list, edge_feats, alive_list, device)

    role     = np.array([0.5, 0.5], dtype=np.float32)
    extended = {}
    for i, aid in enumerate(blue_ids):
        base = obs_dict.get(aid, np.zeros(base_obs_dim, dtype=np.float32))
        extended[aid] = np.concatenate([base, role, messages[i]], axis=0)
    return extended, messages, alive_list, edge_feats


# ===========================================================================
# Episode Koşucusu
# ===========================================================================

def run_episodes(env: DogfightEnv, actor, opp_policy,
                 blue_ids: list, n_episodes: int,
                 seed: int, device: torch.device,
                 is_gat: bool = False,
                 gat_comm=None, base_obs_dim: int = 50,
                 node_dim: int = 17, wez_range: float = 8000.0,
                 collect_gat_analysis: bool = False,
                 init_ammo: int = 10) -> dict:
    """
    n_episodes çalıştırır, metrikleri toplar.

    collect_gat_analysis=True ise mesaj L2 norm'ları mesafeye göre biriktirilir.
    """
    from training.train_mappo import MAPPOActor, GATMAPPOActor

    env.seed(seed)

    wins = 0; losses = 0; draws = 0
    kills_list   = []
    win_list     = []   # per-ep: 1=win, 0=other (bootstrap için)
    lengths_list = []
    shots_list   = []
    wez_steps_list = []
    total_steps_list = []

    # GAT analizi için
    msg_l2_near = []
    msg_l2_mid  = []
    msg_l2_far  = []

    for ep in range(n_episodes):
        obs_dict = env.reset()
        opp_policy.reset()

        ep_kills    = 0
        ep_shots    = 0
        ep_wez_steps = 0
        ep_steps    = 0
        done        = {"__all__": False}

        # Başlangıç ammo takibi
        init_states = env.get_all_states()
        ammo_start  = {
            aid: int(init_states[aid][STATE_AMMO])
            for aid in blue_ids
            if init_states.get(aid) is not None
        }

        while not done["__all__"]:
            # GAT obs uzatma
            if is_gat and gat_comm is not None:
                ext_obs, messages, alive_list, edge_feats = extend_obs_phase2(
                    obs_dict, env, gat_comm, base_obs_dim,
                    node_dim, wez_range, device
                )
                # GAT analizi: mesafe + mesaj L2
                if collect_gat_analysis:
                    states = env.get_all_states()
                    for i, aid_i in enumerate(blue_ids):
                        for j, aid_j in enumerate(blue_ids):
                            if i == j:
                                continue
                            si = states.get(aid_i)
                            sj = states.get(aid_j)
                            if (si is None or sj is None or
                                    si[STATE_ALIVE] < 0.5 or
                                    sj[STATE_ALIVE] < 0.5):
                                continue
                            dist = distance_3d(
                                si[[STATE_X, STATE_Y, STATE_H]],
                                sj[[STATE_X, STATE_Y, STATE_H]]
                            )
                            msg_l2 = float(np.linalg.norm(messages[i]))
                            if dist < NEAR_DIST:
                                msg_l2_near.append(msg_l2)
                            elif dist < FAR_DIST:
                                msg_l2_mid.append(msg_l2)
                            else:
                                msg_l2_far.append(msg_l2)
            else:
                ext_obs = obs_dict

            # Actor aksiyonları
            actions = {}
            with torch.no_grad():
                for aid in blue_ids:
                    obs = ext_obs.get(aid)
                    if obs is None:
                        continue
                    obs_t    = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    raw, _   = actor.act(obs_t, deterministic=False)
                    if is_gat:
                        squashed = GATMAPPOActor.squash(raw.squeeze(0))
                    else:
                        squashed = MAPPOActor.squash(raw.squeeze(0))
                    actions[aid] = squashed.cpu().numpy()

            # Heuristic aksiyonları (sadece red ajanlar)
            state_dict      = env.get_all_states()
            all_heuristic   = opp_policy.act(state_dict)
            red_ids         = [aid for aid in all_heuristic if aid not in blue_ids]
            opp_actions     = {k: all_heuristic[k] for k in red_ids}
            action_dict     = {**actions, **opp_actions}

            obs_dict, rew_dict, done, info_dict = env.step(action_dict)
            ep_steps += 1

            # WEZ istatistiği
            for aid in blue_ids:
                info = info_dict.get(aid, {})
                if info.get("r_wez", 0.0) > 0.01:
                    ep_wez_steps += 1
                if info.get("r_kill", 0.0) > 0.5:
                    ep_kills += 1

        # Episode sonucu
        winner = done.get("winner", "draw")
        if winner == BLUE:      wins += 1;   win_list.append(1)
        elif winner == RED:     losses += 1; win_list.append(0)
        else:                   draws += 1;  win_list.append(0)

        # Atış sayısı: başlangıç - kalan ammo
        final_states = env.get_all_states()
        for aid in blue_ids:
            fs = final_states.get(aid)
            if fs is not None:
                remaining = int(fs[STATE_AMMO])
                ep_shots += max(0, ammo_start.get(aid, init_ammo) - remaining)

        kills_list.append(ep_kills)
        lengths_list.append(ep_steps)
        shots_list.append(ep_shots)
        wez_steps_list.append(ep_wez_steps)
        total_steps_list.append(ep_steps * len(blue_ids))

        if (ep + 1) % 10 == 0:
            icon = "W" if winner == BLUE else ("L" if winner == RED else "D")
            print(f"  ep {ep+1:>3}/{n_episodes} [{icon}] "
                  f"kills={ep_kills} shots={ep_shots} "
                  f"wez={ep_wez_steps}/{ep_steps*len(blue_ids)} "
                  f"len={ep_steps}", flush=True)

    # Ammo efficiency: shots / kills (inf → kills=0 durumunda NaN)
    total_shots = sum(shots_list)
    total_kills = sum(kills_list)
    ammo_eff    = total_shots / total_kills if total_kills > 0 else float("nan")

    # WEZ in-time ratio
    total_wez  = sum(wez_steps_list)
    total_ts   = sum(total_steps_list)
    wez_ratio  = total_wez / total_ts if total_ts > 0 else 0.0

    results = {
        "n_episodes":    n_episodes,
        "win_rate":      wins   / n_episodes,
        "loss_rate":     losses / n_episodes,
        "draw_rate":     draws  / n_episodes,
        "kill_per_ep":   float(np.mean(kills_list)),
        "kill_std":      float(np.std(kills_list)),
        "mean_ep_len":   float(np.mean(lengths_list)),
        "ammo_efficiency": round(ammo_eff, 2),
        "wez_intime_ratio": round(wez_ratio, 4),
        "wins": wins, "losses": losses, "draws": draws,
        # Bootstrap için ham diziler
        "_win_list":   win_list,
        "_kills_list": kills_list,
    }

    if collect_gat_analysis:
        results["gat_msg_l2_near"] = round(float(np.mean(msg_l2_near)), 4) if msg_l2_near else float("nan")
        results["gat_msg_l2_mid"]  = round(float(np.mean(msg_l2_mid)),  4) if msg_l2_mid  else float("nan")
        results["gat_msg_l2_far"]  = round(float(np.mean(msg_l2_far)),  4) if msg_l2_far  else float("nan")
        results["gat_msg_l2_near_n"] = len(msg_l2_near)
        results["gat_msg_l2_mid_n"]  = len(msg_l2_mid)
        results["gat_msg_l2_far_n"]  = len(msg_l2_far)

    return results


# ===========================================================================
# Bootstrap Confidence Intervals
# ===========================================================================

def bootstrap_ci(data: list, stat_fn, n_boot: int = 2000,
                 ci: float = 0.95, seed: int = 0) -> tuple:
    """
    Parametrik olmayan bootstrap ile güven aralığı hesapla.

    Parameters
    ----------
    data    : ham episode listesi (0/1 veya sayısal)
    stat_fn : istatistik fonksiyonu (örn. np.mean)
    n_boot  : bootstrap tekrar sayısı
    ci      : güven düzeyi (0.95 = %95)

    Returns
    -------
    (lower, upper) : güven aralığı sınırları
    """
    rng  = np.random.default_rng(seed)
    arr  = np.array(data, dtype=float)
    n    = len(arr)
    boot = np.array([
        stat_fn(arr[rng.integers(0, n, size=n)])
        for _ in range(n_boot)
    ])
    alpha = 1.0 - ci
    lo    = float(np.percentile(boot, alpha / 2 * 100))
    hi    = float(np.percentile(boot, (1 - alpha / 2) * 100))
    return lo, hi


# ===========================================================================
# Karşılaştırma Tablosu
# ===========================================================================

def print_comparison(baseline: dict, gat: dict, gat_analysis: dict,
                     n_boot: int = 2000, baseline_label: str = "Baseline",
                     gat_label: str = "GAT"):
    def delta_str(v_gat, v_base):
        if v_gat is None or v_base is None:
            return ""
        if isinstance(v_gat, float) and isinstance(v_base, float):
            d = v_gat - v_base
            return f"({d:+.3f})"
        return ""

    # Bootstrap CI hesapla
    def ci_str(data, stat_fn, fmt):
        if not data:
            return "N/A"
        lo, hi = bootstrap_ci(data, stat_fn, n_boot=n_boot)
        return f"[{fmt.format(lo)}, {fmt.format(hi)}]"

    b_wins  = baseline.get("_win_list",   [])
    g_wins  = gat.get("_win_list",        [])
    b_kills = baseline.get("_kills_list", [])
    g_kills = gat.get("_kills_list",      [])

    n_ep = baseline['n_episodes']
    print()
    print("=" * 90)
    print(f"  KARŞILAŞTIRMA: {baseline_label}  vs  {gat_label}")
    print(f"  Her biri {n_ep} episode | seed=42 | 2v2 normal spawn | heuristic rakip | bootstrap n={n_boot}")
    print("=" * 90)
    print(f"  {'Metrik':<22} {baseline_label:<22} {'95% CI':<24} {gat_label:<22} {'95% CI':<24} {'Fark'}")
    print("─" * 90)

    # Win rate
    wb = baseline.get("win_rate", 0); wg = gat.get("win_rate", 0)
    print(f"  {'Win Rate':<22} {wb:.1%}{'':17} {ci_str(b_wins, np.mean, '{:.1%}'):<24} "
          f"{wg:.1%}{'':17} {ci_str(g_wins, np.mean, '{:.1%}'):<24} {wg-wb:+.3f}")
    # Kill/ep
    kb = baseline.get("kill_per_ep", 0); kg = gat.get("kill_per_ep", 0)
    print(f"  {'Kill / ep':<22} {kb:.2f}{'':19} {ci_str(b_kills, np.mean, '{:.2f}'):<24} "
          f"{kg:.2f}{'':19} {ci_str(g_kills, np.mean, '{:.2f}'):<24} {kg-kb:+.3f}")

    print("─" * 90)

    # Diğer metrikler (CI yok)
    rows = [
        ("loss_rate",        "Loss Rate",        "{:.1%}"),
        ("draw_rate",        "Draw Rate",        "{:.1%}"),
        ("mean_ep_len",      "Ep Length (adım)", "{:.0f}"),
        ("ammo_efficiency",  "Atış / Kill",      "{:.2f}"),
        ("wez_intime_ratio", "WEZ İçi Süre",     "{:.1%}"),
    ]
    for key, label, fmt in rows:
        vb = baseline.get(key); vg = gat.get(key)
        vb_s = fmt.format(vb) if vb is not None else "N/A"
        vg_s = fmt.format(vg) if vg is not None else "N/A"
        d_s  = delta_str(vg, vb)
        print(f"  {label:<22} {vb_s:<46} {vg_s:<46} {d_s}")

    print("─" * 90)
    print()

    # GAT Analizi
    print("  GAT ATTENTION ANALİZİ")
    print("─" * 72)
    fn = gat_analysis.get("fc1_new_norm", float("nan"))
    print(f"  fc1_new ağırlık L2 normu : {fn:.4f}  "
          f"({'> 0 → öğrenme var ✓' if fn > 0.1 else '≈ 0 → sıfır init durumu'})")
    print()
    print(f"  {'Mesafe Aralığı':<20} {'Ort. Mesaj L2':<18} {'Örnek Sayısı'}")
    print("  " + "─" * 50)
    for label, key_l2, key_n in [
        (f"Yakın (<{NEAR_DIST/1000:.0f}km)",   "gat_msg_l2_near", "gat_msg_l2_near_n"),
        (f"Orta  ({NEAR_DIST/1000:.0f}–{FAR_DIST/1000:.0f}km)", "gat_msg_l2_mid", "gat_msg_l2_mid_n"),
        (f"Uzak  (>{FAR_DIST/1000:.0f}km)",   "gat_msg_l2_far",  "gat_msg_l2_far_n"),
    ]:
        l2 = gat_analysis.get(key_l2, float("nan"))
        n  = gat_analysis.get(key_n, 0)
        l2_str = f"{l2:.4f}" if not (isinstance(l2, float) and np.isnan(l2)) else "N/A"
        print(f"  {label:<20} {l2_str:<18} {n}")

    print("=" * 72)
    print()


# ===========================================================================
# Entry Point
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Baseline vs GAT karşılaştırmalı eval")
    p.add_argument("--baseline",        default="checkpoints/mappo_final.pt")
    p.add_argument("--gat",             default="checkpoints/mappo_gat_final.pt")
    p.add_argument("--baseline-label",  default=None,
                   help="Tablo başlığında gösterilecek baseline ismi")
    p.add_argument("--gat-label",       default="GAT (mappo_gat_final)",
                   help="Tablo başlığında gösterilecek GAT ismi")
    p.add_argument("--config",    default="configs/config.yaml")
    p.add_argument("--episodes",  type=int, default=100)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--device",    default="auto")
    p.add_argument("--output",    default="logs/compare_results.json")
    p.add_argument("--n-bootstrap", type=int, default=2000,
                   help="Bootstrap CI için tekrar sayısı")
    return p.parse_args()


def main():
    args = parse_args()

    config_path = PROJECT_ROOT / args.config
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = _resolve_device() if args.device == "auto" else torch.device(args.device)
    print(f"[Compare] Device: {device}")

    # ── Config parametreleri ───────────────────────────────────────────────
    tr_cfg   = config["training"]
    env_cfg  = config.get("environment", {})
    comm_cfg = config.get("communication", {})
    wpn_cfg  = config.get("weapons", {})

    hidden      = int(tr_cfg.get("hidden_dim", 256))
    action_dim  = 5

    base_obs_dim = 50
    gat_node_dim = int(comm_cfg.get("node_dim", 17))
    gat_msg_dim  = int(comm_cfg.get("msg_dim", 16))
    gat_n_heads  = int(comm_cfg.get("n_heads", 4))
    gat_edge_dim = int(comm_cfg.get("edge_dim", 3))
    role_dim     = int(comm_cfg.get("role_dim", 2))
    gat_obs_dim  = base_obs_dim + role_dim + gat_msg_dim   # 68
    wez_range    = float(wpn_cfg.get("wez_range_max", 8000.0))
    init_ammo    = int(config.get("aircraft", {}).get("initial_ammo", 6))

    # ── Ortam kurulumu (2v2 normal spawn) ─────────────────────────────────
    def make_env():
        env = DogfightEnv(config)
        env.set_curriculum_phase(4)   # Faz-3: 2v2 normal spawn
        return env

    env = make_env()
    blue_ids = list(env.blue_ids)
    all_ids  = list(env.agent_ids)
    team_map = {aid: ("blue" if "blue" in aid else "red") for aid in all_ids}
    opp_policy = MultiHeuristicPolicy(config, all_ids, team_map)

    print(f"[Compare] Ortam: 2v2, blue_ids={blue_ids}")
    print(f"[Compare] GAT obs_dim={gat_obs_dim}, wez_range={wez_range}m, init_ammo={init_ammo}")
    print()

    # ── 1. BASELINE değerlendirme ──────────────────────────────────────────
    baseline_path = PROJECT_ROOT / args.baseline
    if not baseline_path.exists():
        print(f"HATA: baseline checkpoint bulunamadı: {baseline_path}")
        sys.exit(1)

    print(f"[1/3] Baseline değerlendirme ({args.episodes} ep) ...")
    baseline_actor = load_baseline_actor(
        str(baseline_path), base_obs_dim, action_dim, hidden, device
    )
    baseline_results = run_episodes(
        env, baseline_actor, opp_policy, blue_ids,
        n_episodes=args.episodes, seed=args.seed,
        device=device, is_gat=False,
        init_ammo=init_ammo,
    )
    print(f"  Tamamlandı: W={baseline_results['wins']} "
          f"L={baseline_results['losses']} D={baseline_results['draws']}")
    print()

    # ── 2. GAT değerlendirme ───────────────────────────────────────────────
    gat_path = PROJECT_ROOT / args.gat
    if not gat_path.exists():
        print(f"HATA: GAT checkpoint bulunamadı: {gat_path}")
        sys.exit(1)

    print(f"[2/3] GAT değerlendirme ({args.episodes} ep) ...")
    gat_actor = load_gat_actor(
        str(gat_path), base_obs_dim, gat_obs_dim, action_dim, hidden, device
    )
    gat_comm = load_gat_comm(
        str(gat_path), gat_node_dim, gat_edge_dim, gat_n_heads, gat_msg_dim, device
    )
    env2 = make_env()
    opp2 = MultiHeuristicPolicy(config, list(env2.agent_ids), team_map)

    gat_results = run_episodes(
        env2, gat_actor, opp2, blue_ids,
        n_episodes=args.episodes, seed=args.seed,
        device=device, is_gat=True, gat_comm=gat_comm,
        base_obs_dim=base_obs_dim, node_dim=gat_node_dim,
        wez_range=wez_range, init_ammo=init_ammo,
    )
    print(f"  Tamamlandı: W={gat_results['wins']} "
          f"L={gat_results['losses']} D={gat_results['draws']}")
    print()

    # ── 3. GAT Attention Analizi ───────────────────────────────────────────
    print(f"[3/3] GAT attention analizi ({GAT_ANALYSIS_EPS} ep) ...")
    env3 = make_env()
    opp3 = MultiHeuristicPolicy(config, list(env3.agent_ids), team_map)
    gat_analysis_raw = run_episodes(
        env3, gat_actor, opp3, blue_ids,
        n_episodes=GAT_ANALYSIS_EPS, seed=args.seed + 1,
        device=device, is_gat=True, gat_comm=gat_comm,
        base_obs_dim=base_obs_dim, node_dim=gat_node_dim,
        wez_range=wez_range, init_ammo=init_ammo,
        collect_gat_analysis=True,
    )

    # fc1_new ağırlık L2 normu
    fc1_new_norm = float(torch.norm(gat_actor.fc1_new.weight).item())
    gat_analysis = {
        "fc1_new_norm":    round(fc1_new_norm, 4),
        "gat_msg_l2_near": gat_analysis_raw.get("gat_msg_l2_near", float("nan")),
        "gat_msg_l2_mid":  gat_analysis_raw.get("gat_msg_l2_mid",  float("nan")),
        "gat_msg_l2_far":  gat_analysis_raw.get("gat_msg_l2_far",  float("nan")),
        "gat_msg_l2_near_n": gat_analysis_raw.get("gat_msg_l2_near_n", 0),
        "gat_msg_l2_mid_n":  gat_analysis_raw.get("gat_msg_l2_mid_n",  0),
        "gat_msg_l2_far_n":  gat_analysis_raw.get("gat_msg_l2_far_n",  0),
    }

    # ── Karşılaştırma tablosu ──────────────────────────────────────────────
    baseline_label = args.baseline_label or Path(args.baseline).stem
    gat_label      = args.gat_label
    print_comparison(baseline_results, gat_results, gat_analysis,
                     n_boot=args.n_bootstrap,
                     baseline_label=baseline_label,
                     gat_label=gat_label)

    # ── JSON çıktısı ───────────────────────────────────────────────────────
    def _serializable(d: dict) -> dict:
        return {k: v for k, v in d.items() if not k.startswith("_")}

    output = {
        "config": {
            "episodes":   args.episodes,
            "seed":       args.seed,
            "n_bootstrap": args.n_bootstrap,
            "baseline":   str(args.baseline),
            "gat":        str(args.gat),
        },
        "baseline":     _serializable(baseline_results),
        "gat":          _serializable(gat_results),
        "gat_analysis": gat_analysis,
    }
    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"[Compare] Sonuçlar kaydedildi: {out_path}")


if __name__ == "__main__":
    main()

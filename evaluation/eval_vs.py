"""
eval_vs.py
==========
MAPPO_GAT vs FACMAC_OM_R doğrudan kapışma değerlendirmesi.

Kullanım:
    python -X utf8 evaluation/eval_vs.py
    python -X utf8 evaluation/eval_vs.py \
        --mappo  checkpoints/mappo_gat_ep37000.pt \
        --facmac checkpoints/facmac_omr_ep27000.pt \
        --episodes 500 --seed 42

Tasarım:
  - episodes//2 ep: MAPPO=Blue,  FACMAC=Red
  - episodes//2 ep: FACMAC=Blue, MAPPO=Red
  - Toplam: 500 ep, seed=42

OOD notu: Her iki model birbirini hiç görmemiş → generalization testi.
"""

import sys
import json
import argparse
import yaml
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from envs.dogfight_env import DogfightEnv, BLUE, RED
from envs.aircraft_model import STATE_ALIVE, STATE_X, STATE_Y, STATE_H, STATE_PSI, STATE_AMMO
from envs.geometry_utils import distance_3d, bearing_angle, wrap_to_pi, antenna_train_angle
from models.gat_comm import GATComm
from models.facmac_net import FACMACActorOM
from models.om_net import (
    CentralizedOpponentModel, CentralizedRoleAssigner,
    EnemyHistoryBuffer, build_team_state,
)
from training.train_mappo import GATMAPPOActor, MAPPOActor

# ── Sabitler ──────────────────────────────────────────────────────────────────
_BASE       = 50
_GAT_ND     = 17          # GATComm node dim (ego kısmı)
_MAPPO_DIM  = 78          # 50 base + 18 GAT ext + 6 intent (zeros) + 4 role (zeros)
_FACMAC_DIM = 60          # 50 base + 6 intent + 4 role
_WEZ_MIN    = 300.0
_WEZ_MAX    = 8000.0
_WEZ_ANG    = np.radians(30.0)


# ── Wilson CI ─────────────────────────────────────────────────────────────────
def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0
    p   = k / n
    den = 1 + z**2 / n
    ctr = (p + z**2 / (2 * n)) / den
    rng = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return float(max(0.0, ctr - rng)), float(min(1.0, ctr + rng))


# ── Rule-based fire (FACMAC için) ─────────────────────────────────────────────
def _rule_fire(my_ids, opp_ids, obs_dict, states):
    fire = {}
    for aid in my_ids:
        fire[aid] = 0.0
        obs = obs_dict.get(aid, np.zeros(_BASE, np.float32))
        if obs[16] > 1e-4:          # cooldown_norm > 0
            continue
        b = states.get(aid)
        if b is None or b[STATE_ALIVE] < 0.5 or b[STATE_AMMO] < 0.5:
            continue
        b_pos = b[[STATE_X, STATE_Y, STATE_H]]
        for rid in opp_ids:
            r = states.get(rid)
            if r is None or r[STATE_ALIVE] < 0.5:
                continue
            r_pos = r[[STATE_X, STATE_Y, STATE_H]]
            dist  = distance_3d(b_pos, r_pos)
            if dist < _WEZ_MIN or dist > _WEZ_MAX:
                continue
            if abs(antenna_train_angle(b_pos, r_pos, b[STATE_PSI])) <= _WEZ_ANG:
                fire[aid] = 1.0
                break
    return fire


# ── Model yükleme ─────────────────────────────────────────────────────────────
def load_mappo(ckpt_path, config, device):
    hidden     = int(config["training"].get("hidden_dim", 256))
    action_dim = 5
    ckpt       = torch.load(ckpt_path, map_location=device, weights_only=False)
    actor_sd   = ckpt.get("actor", {})

    actor = GATMAPPOActor(
        old_obs_dim=50, ext_dim=18, intent_dim=6, role_dim=4,
        action_dim=action_dim, hidden=hidden,
    ).to(device)
    actor.load_state_dict(actor_sd, strict=False)
    actor.eval()

    gat_comm = GATComm().to(device)
    if "gat_comm" in ckpt:
        gat_comm.load_state_dict(ckpt["gat_comm"])
    gat_comm.eval()

    ep   = ckpt.get("episode", "?")
    step = ckpt.get("total_steps", ckpt.get("total_timesteps", "?"))
    step_str = f"{step:,}" if isinstance(step, int) else str(step)
    print(f"[Eval] MAPPO  : {Path(ckpt_path).name}  ep={ep}  step={step_str}")
    return actor, gat_comm


def load_facmac(ckpt_path, device):
    ckpt     = torch.load(ckpt_path, map_location=device, weights_only=False)
    actor_sd = ckpt.get("actor", {})

    actor = FACMACActorOM(base_obs_dim=50, hidden=256).to(device)
    actor.load_state_dict(actor_sd, strict=False)
    actor.eval()

    cent_om   = CentralizedOpponentModel().to(device)
    cent_role = CentralizedRoleAssigner().to(device)
    if "cent_om"   in ckpt: cent_om.load_state_dict(ckpt["cent_om"])
    if "cent_role" in ckpt: cent_role.load_state_dict(ckpt["cent_role"], strict=False)
    cent_om.eval(); cent_role.eval()

    ep   = ckpt.get("episode", "?")
    step = ckpt.get("total_steps", "?")
    step_str = f"{step:,}" if isinstance(step, int) else str(step)
    print(f"[Eval] FACMAC : {Path(ckpt_path).name}  ep={ep}  step={step_str}")
    return actor, cent_om, cent_role


# ── MAPPO obs uzantısı (50D → 78D) ───────────────────────────────────────────
def _build_mappo_obs(obs_dict, my_ids, opp_ids, gat_comm, env, device):
    states  = env.get_all_states()
    wez_max = float(env.config.get("weapons", {}).get("wez_range_max", _WEZ_MAX))

    ego_list   = []
    alive_list = []
    for aid in my_ids:
        obs = obs_dict.get(aid, np.zeros(_BASE, np.float32))
        ego_list.append(obs[:_GAT_ND])
        s = states.get(aid)
        alive_list.append(float(s[STATE_ALIVE]) if s is not None else 0.0)

    N    = len(my_ids)
    edge = np.zeros((N, N, 3), np.float32)
    for i, ai in enumerate(my_ids):
        for j, aj in enumerate(my_ids):
            if i == j:
                continue
            si = states.get(ai)
            sj = states.get(aj)
            if si is None or sj is None:
                continue
            pi   = si[[STATE_X, STATE_Y, STATE_H]]
            pj   = sj[[STATE_X, STATE_Y, STATE_H]]
            dist = distance_3d(pi, pj)
            bear = bearing_angle(pi, pj)
            ts   = 0.0
            for eid in opp_ids:
                es = states.get(eid)
                if es is not None and es[STATE_ALIVE] > 0.5:
                    d  = distance_3d(pj, es[[STATE_X, STATE_Y, STATE_H]])
                    ts = max(ts, float(np.clip(1.0 - d / (wez_max + 1e-9), 0.0, 1.0)))
            edge[i, j] = [
                float(np.clip(dist / (env.map_size + 1e-9), 0.0, 1.0)),
                float(np.clip(wrap_to_pi(bear) / np.pi, -1.0, 1.0)),
                ts,
            ]

    messages = gat_comm.compute_messages(ego_list, edge, alive_list, device)

    ext_obs = {}
    for i, aid in enumerate(my_ids):
        base  = obs_dict.get(aid, np.zeros(_BASE, np.float32))[:_BASE]
        ext18 = np.concatenate([np.zeros(2, np.float32), messages[i]])  # 2D dummy + 16D GAT
        ext_obs[aid] = np.concatenate([
            base, ext18,
            np.zeros(6, np.float32),   # intent (zeros — ep37000 bu ağırlıklara sahip değil)
            np.zeros(4, np.float32),   # role (zeros)
        ])                             # toplam: 78D
    return ext_obs


# ── FACMAC obs uzantısı (50D → 60D) ──────────────────────────────────────────
def _build_facmac_obs(obs_dict, my_ids, cent_om, cent_role, enemy_hist, states, device):
    base_arr = np.stack([
        obs_dict.get(aid, np.zeros(_BASE, np.float32))[:_BASE]
        for aid in my_ids
    ], axis=0)   # (2, 50)

    hist_960 = enemy_hist.update(base_arr)
    with torch.no_grad():
        hist_t    = torch.from_numpy(hist_960).unsqueeze(0).to(device)
        intent_np = cent_om.intent_flat(hist_t).squeeze(0).cpu().numpy()   # (6,)

    ts_np = build_team_state([states.get(bid) for bid in my_ids])          # (6,)
    with torch.no_grad():
        x_role = torch.from_numpy(
            np.concatenate([intent_np, ts_np]).astype(np.float32)
        ).unsqueeze(0).to(device)
        role_0, role_1 = cent_role.assign(x_role)

    roles = [role_0.squeeze(0).cpu().numpy(), role_1.squeeze(0).cpu().numpy()]  # each (4,)

    ext_obs = {
        aid: np.concatenate([
            obs_dict.get(aid, np.zeros(_BASE, np.float32))[:_BASE],
            intent_np,   # 6D
            roles[i],    # 4D
        ])               # toplam: 60D
        for i, aid in enumerate(my_ids)
    }
    return ext_obs, intent_np, roles


# ── Aksiyon üretimi ───────────────────────────────────────────────────────────
def _mappo_actions(ext_obs, my_ids, actor, device):
    actions = {}
    with torch.no_grad():
        for aid in my_ids:
            o  = ext_obs.get(aid, np.zeros(_MAPPO_DIM, np.float32))
            ot = torch.FloatTensor(o).unsqueeze(0).to(device)
            raw, _ = actor.act(ot, deterministic=True)
            actions[aid] = MAPPOActor.squash(raw.squeeze(0)).cpu().numpy()  # 5D
    return actions


def _facmac_actions(ext_obs, my_ids, opp_ids, actor, obs_dict, states, device):
    ctrl = {}
    with torch.no_grad():
        for aid in my_ids:
            o  = ext_obs.get(aid, np.zeros(_FACMAC_DIM, np.float32))
            ot = torch.FloatTensor(o).unsqueeze(0).to(device)
            c, _ = actor.act(ot, deterministic=True)
            ctrl[aid] = c.squeeze(0).cpu().numpy()   # 4D

    fire = _rule_fire(my_ids, opp_ids, obs_dict, states)
    return {aid: np.append(ctrl[aid], fire[aid]) for aid in my_ids}   # 5D


# ── Episode döngüsü ───────────────────────────────────────────────────────────
def run_half(
    n_episodes, mappo_ids, facmac_ids,
    env, mappo_actor, gat_comm,
    facmac_actor, cent_om, cent_role,
    device, seed, label="",
):
    """
    mappo_ids  : MAPPO'nun kontrol ettiği agent id listesi
    facmac_ids : FACMAC'ın kontrol ettiği agent id listesi
    (opp = diğer takım: MAPPO'nun rakibi = facmac_ids, vice versa)
    """
    env.seed(seed)
    mappo_team = BLUE if any("blue" in a for a in mappo_ids) else RED
    ep_results = []

    for ep in range(n_episodes):
        obs_dict     = env.reset()
        enemy_hist_f = EnemyHistoryBuffer()
        done         = {"__all__": False}

        mappo_kills  = 0
        facmac_kills = 0
        ep_len       = 0

        intent_acc = np.zeros(6, np.float32)
        role_acc   = [np.zeros(4, np.float32), np.zeros(4, np.float32)]
        n_steps    = 0

        while not done["__all__"]:
            states = env.get_all_states()

            # obs uzantıları
            mappo_ext = _build_mappo_obs(
                obs_dict, mappo_ids, facmac_ids, gat_comm, env, device)
            facmac_ext, intent_np, roles = _build_facmac_obs(
                obs_dict, facmac_ids, cent_om, cent_role,
                enemy_hist_f, states, device)

            # rol istatistikleri biriktir
            intent_acc += intent_np
            role_acc[0] += roles[0]
            role_acc[1] += roles[1]
            n_steps += 1

            # aksiyonlar
            mappo_acts  = _mappo_actions(mappo_ext,  mappo_ids,  mappo_actor,  device)
            facmac_acts = _facmac_actions(
                facmac_ext, facmac_ids, mappo_ids,
                facmac_actor, obs_dict, states, device)

            action_dict = {**mappo_acts, **facmac_acts}
            obs_dict, _rew, done, info = env.step(action_dict)

            for aid in mappo_ids:
                if info[aid].get("r_kill", 0.0) > 0.5:
                    mappo_kills += 1
            for aid in facmac_ids:
                if info[aid].get("r_kill", 0.0) > 0.5:
                    facmac_kills += 1
            ep_len += 1

        # sonuç
        winner = done.get("winner", "draw")
        if winner == "draw":
            outcome = "draw"
        elif winner == mappo_team:
            outcome = "mappo_win"
        else:
            outcome = "facmac_win"

        div = max(n_steps, 1)
        ep_results.append({
            "ep":           ep + 1,
            "outcome":      outcome,
            "mappo_kills":  mappo_kills,
            "facmac_kills": facmac_kills,
            "ep_len":       ep_len,
            "intent_mean":  (intent_acc / div).tolist(),
            "role0_mean":   (role_acc[0] / div).tolist(),
            "role1_mean":   (role_acc[1] / div).tolist(),
        })

        if (ep + 1) % 50 == 0:
            done_n = ep + 1
            mw = sum(1 for r in ep_results if r["outcome"] == "mappo_win")
            fw = sum(1 for r in ep_results if r["outcome"] == "facmac_win")
            print(f"  {label} ep {done_n:3d}/{n_episodes} | "
                  f"MAPPO W={mw/done_n:.3f}  FACMAC W={fw/done_n:.3f}", flush=True)

    return ep_results


# ── İstatistik hesaplama ──────────────────────────────────────────────────────
def _compute_stats(ep_results):
    n  = len(ep_results)
    mw = sum(1 for r in ep_results if r["outcome"] == "mappo_win")
    fw = sum(1 for r in ep_results if r["outcome"] == "facmac_win")
    dr = n - mw - fw
    return {
        "n":                n,
        "mappo_wins":       mw,
        "facmac_wins":      fw,
        "draws":            dr,
        "mappo_win_rate":   round(mw / n, 4),
        "facmac_win_rate":  round(fw / n, 4),
        "draw_rate":        round(dr / n, 4),
        "mappo_win_ci":     wilson_ci(mw, n),
        "facmac_win_ci":    wilson_ci(fw, n),
        "mappo_kill_mean":  round(float(np.mean([r["mappo_kills"]  for r in ep_results])), 3),
        "facmac_kill_mean": round(float(np.mean([r["facmac_kills"] for r in ep_results])), 3),
        "mean_ep_len":      round(float(np.mean([r["ep_len"]       for r in ep_results])), 1),
        "intent_mean":      np.mean([r["intent_mean"] for r in ep_results], axis=0).tolist(),
        "role0_mean":       np.mean([r["role0_mean"]  for r in ep_results], axis=0).tolist(),
        "role1_mean":       np.mean([r["role1_mean"]  for r in ep_results], axis=0).tolist(),
    }


# ── Özet yazdır ───────────────────────────────────────────────────────────────
def _print_block(title, s):
    mi, mh = s["mappo_win_ci"]
    fi, fh = s["facmac_win_ci"]
    im     = s["intent_mean"]
    r0     = s["role0_mean"]
    r1     = s["role1_mean"]
    print(f"\n  ── {title} ({s['n']} ep) ──────────────────────────")
    print(f"  MAPPO  win%  : {s['mappo_win_rate']:.1%}  [{mi:.1%}–{mh:.1%}]"
          f"  kill/ep={s['mappo_kill_mean']:.2f}")
    print(f"  FACMAC win%  : {s['facmac_win_rate']:.1%}  [{fi:.1%}–{fh:.1%}]"
          f"  kill/ep={s['facmac_kill_mean']:.2f}")
    print(f"  Draw         : {s['draw_rate']:.1%}  |  mean ep len={s['mean_ep_len']:.0f}")
    print(f"  FACMAC intent: e0[agg={im[0]:.2f} eva={im[1]:.2f} def={im[2]:.2f}] "
          f"e1[agg={im[3]:.2f} eva={im[4]:.2f} def={im[5]:.2f}]")
    print(f"  FACMAC role0 : sniper={r0[0]:.2f} pursuit={r0[1]:.2f} "
          f"def={r0[2]:.2f} sup={r0[3]:.2f}")
    print(f"  FACMAC role1 : sniper={r1[0]:.2f} pursuit={r1[1]:.2f} "
          f"def={r1[2]:.2f} sup={r1[3]:.2f}")


def print_summary(stats_a, stats_b, combined, mappo_ckpt, facmac_ckpt):
    sep = "=" * 62
    print(f"\n{sep}")
    print("  MAPPO_GAT  vs  FACMAC_OM_R  — Doğrudan Kapışma")
    print(f"  MAPPO  : {Path(mappo_ckpt).name}")
    print(f"  FACMAC : {Path(facmac_ckpt).name}")
    print("  OOD    : her iki model birbirini hiç görmemiş")
    print(sep)
    _print_block("Half A — MAPPO=Blue  FACMAC=Red", stats_a)
    _print_block("Half B — FACMAC=Blue MAPPO=Red",  stats_b)
    _print_block("TOPLAM",                           combined)
    print(f"\n{sep}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mappo",    default="checkpoints/mappo_gat_ep37000.pt")
    p.add_argument("--facmac",   default="checkpoints/facmac_omr_ep27000.pt")
    p.add_argument("--config",   default="configs/config.yaml")
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--device",   default="auto")
    p.add_argument("--output",   default="logs/eval_vs_results.json")
    args = p.parse_args()

    with open(ROOT / args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))

    mappo_path  = str(ROOT / args.mappo)
    facmac_path = str(ROOT / args.facmac)

    # ── Modeller ─────────────────────────────────────────────────────────────
    mappo_actor, gat_comm                   = load_mappo(mappo_path,  config, device)
    facmac_actor, cent_om, cent_role        = load_facmac(facmac_path, device)

    n_each = args.episodes // 2
    print(f"[Eval] Device    : {device}")
    print(f"[Eval] Episodes  : {args.episodes}  ({n_each} × 2)  seed={args.seed}")

    # ── Ortam ────────────────────────────────────────────────────────────────
    env      = DogfightEnv(config)
    env.set_curriculum_phase(4)   # 2v2 normal spawn
    blue_ids = env.blue_ids
    red_ids  = env.red_ids

    # ── Half A: MAPPO=Blue, FACMAC=Red ───────────────────────────────────────
    print(f"\n[Half A] MAPPO=Blue  FACMAC=Red  — {n_each} ep")
    half_a = run_half(
        n_each,
        mappo_ids=blue_ids, facmac_ids=red_ids,
        env=env, mappo_actor=mappo_actor, gat_comm=gat_comm,
        facmac_actor=facmac_actor, cent_om=cent_om, cent_role=cent_role,
        device=device, seed=args.seed, label="[A]",
    )

    # ── Half B: FACMAC=Blue, MAPPO=Red ───────────────────────────────────────
    print(f"\n[Half B] FACMAC=Blue MAPPO=Red   — {n_each} ep")
    half_b = run_half(
        n_each,
        mappo_ids=red_ids, facmac_ids=blue_ids,
        env=env, mappo_actor=mappo_actor, gat_comm=gat_comm,
        facmac_actor=facmac_actor, cent_om=cent_om, cent_role=cent_role,
        device=device, seed=args.seed + 1, label="[B]",
    )

    # ── İstatistikler ─────────────────────────────────────────────────────────
    stats_a  = _compute_stats(half_a)
    stats_b  = _compute_stats(half_b)
    combined = _compute_stats(half_a + half_b)

    print_summary(stats_a, stats_b, combined, args.mappo, args.facmac)

    # ── JSON kaydet ───────────────────────────────────────────────────────────
    out = {
        "mappo_checkpoint":  args.mappo,
        "facmac_checkpoint": args.facmac,
        "n_episodes":        args.episodes,
        "seed":              args.seed,
        "device":            str(device),
        "half_a_mappo_blue": stats_a,
        "half_b_facmac_blue": stats_b,
        "combined":          combined,
    }
    out_path = ROOT / args.output
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n[Eval] Sonuçlar kaydedildi: {out_path}")


if __name__ == "__main__":
    main()

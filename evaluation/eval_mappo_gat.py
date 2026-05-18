"""
eval_mappo_gat.py
=================
GAT-MAPPO checkpoint'ini heuristic rakibe karşı değerlendirir.
eval_qmix.py ile aynı metrikler ve koşullar.

Kullanım:
    python -X utf8 evaluation/eval_mappo_gat.py --checkpoint checkpoints/mappo_gat_ep44000.pt
    python -X utf8 evaluation/eval_mappo_gat.py --checkpoint checkpoints/mappo_gat_ep44000.pt --episodes 500 --seed 42
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
from envs.aircraft_model import STATE_ALIVE, STATE_X, STATE_Y, STATE_H
from envs.geometry_utils import distance_3d, bearing_angle, wrap_to_pi
from agents.heuristic_agent import MultiHeuristicPolicy
from utils.normalization import Normalizer
from training.train_mappo import GATMAPPOActor, MAPPOActor
from models.gat_comm import GATComm


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0
    p   = k / n
    den = 1 + z**2 / n
    ctr = (p + z**2 / (2 * n)) / den
    rng = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return float(max(0.0, ctr - rng)), float(min(1.0, ctr + rng))


def _build_gat_edge_feats(env, blue_ids, states, wez_range):
    N    = len(blue_ids)
    edge = np.zeros((N, N, 3), dtype=np.float32)
    for i, ai in enumerate(blue_ids):
        for j, aj in enumerate(blue_ids):
            if i == j: continue
            si = states.get(ai); sj = states.get(aj)
            if si is None or sj is None: continue
            pi = si[[STATE_X, STATE_Y, STATE_H]]
            pj = sj[[STATE_X, STATE_Y, STATE_H]]
            dist = distance_3d(pi, pj)
            bear = bearing_angle(pi, pj)
            ts   = 0.0
            for eid in env.red_ids:
                es = states.get(eid)
                if es is not None and es[STATE_ALIVE] > 0.5:
                    d = distance_3d(pj, es[[STATE_X, STATE_Y, STATE_H]])
                    ts = max(ts, float(np.clip(1.0 - d / (wez_range + 1e-9), 0.0, 1.0)))
            edge[i, j] = [
                float(np.clip(dist / (env.map_size + 1e-9), 0.0, 1.0)),
                float(np.clip(wrap_to_pi(bear) / np.pi, -1.0, 1.0)),
                ts,
            ]
    return edge


def _extend_obs_gat(obs_dict, env, gat_comm, device, wez_range, obs_dim):
    """
    obs_dim=68: [base50 | role2  | gat16]
    obs_dim=76: [base50 | role4  | gat16 | intent6]
    eval'de role=uniform, intent=zeros.
    """
    blue_ids   = sorted([a for a in env.agent_ids if "blue" in a])
    states     = env.get_all_states()
    ego_list   = []; alive_list = []
    for aid in blue_ids:
        obs = obs_dict.get(aid, np.zeros(50, dtype=np.float32))
        ego_list.append(obs[:17])
        s = states.get(aid)
        alive_list.append(float(s[STATE_ALIVE]) if s is not None else 0.0)
    edge_feats = _build_gat_edge_feats(env, blue_ids, states, wez_range)
    messages   = gat_comm.compute_messages(ego_list, edge_feats, alive_list, device)
    extended   = {}
    for i, aid in enumerate(blue_ids):
        base = obs_dict.get(aid, np.zeros(50, dtype=np.float32))
        if obs_dim == 76:
            role   = np.full(4, 0.25, dtype=np.float32)   # uniform role
            intent = np.zeros(6,      dtype=np.float32)    # no OM at eval
            extended[aid] = np.concatenate([base, role, messages[i], intent])
        else:  # 68D
            role = np.array([0.5, 0.5], dtype=np.float32)
            extended[aid] = np.concatenate([base, role, messages[i]])
    return extended


def run_eval(config, checkpoint_path, n_episodes, seed, device):
    env      = DogfightEnv(config)
    env.set_curriculum_phase(4)
    blue_ids = env.blue_ids
    red_ids  = env.red_ids
    all_ids  = blue_ids + red_ids
    team_map = {a: ("blue" if "blue" in a else "red") for a in all_ids}

    # ── Actor yükle (GATMAPPOActor — 68D veya 76D otomatik algılama) ─────────
    ckpt   = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hidden = int(config["training"].get("hidden_dim", 256))
    actor_sd = ckpt["actor"]

    # fc1_new.weight shape'den ext_dim → new_obs_dim belirle
    fc1_new_shape = actor_sd["fc1_new.weight"].shape  # (hidden, ext_dim)
    ext_dim       = fc1_new_shape[1]
    new_obs_dim   = 50 + ext_dim   # 68 veya 76
    has_role      = "role_selector.net.0.weight" in actor_sd
    print(f"[Eval] obs_dim={new_obs_dim}D (ext={ext_dim}D, role_selector={has_role})")

    actor = GATMAPPOActor(old_obs_dim=50, new_obs_dim=new_obs_dim,
                          action_dim=env.action_dim, hidden=hidden,
                          with_role_selector=has_role).to(device)
    actor.load_state_dict(actor_sd)
    actor.eval()

    gat_comm = GATComm(node_dim=17, edge_dim=3, n_heads=4, msg_dim=16).to(device)
    gat_comm.load_state_dict(ckpt["gat_comm"])
    gat_comm.eval()

    wez_range = float(config.get("weapons", {}).get("wez_range_max", 8000.0))
    _obs_dim  = new_obs_dim   # 76D veya 68D — obs construction'da kullanılır

    print(f"[Eval] GAT-MAPPO checkpoint : {checkpoint_path}")
    print(f"[Eval] Saved at             : ep={ckpt.get('episode','?')}, "
          f"step={ckpt.get('global_step', ckpt.get('total_steps','?'))}")
    print(f"[Eval] Device               : {device}")
    print(f"[Eval] Episodes             : {n_episodes} | seed={seed}")

    opp = MultiHeuristicPolicy(config, all_ids, team_map)

    wins = 0; losses = 0; draws = 0
    kills_per_ep = []; lengths = []; double_kills = 0

    env.seed(seed)

    for ep in range(n_episodes):
        obs_dict = env.reset()
        opp.reset()
        done     = {"__all__": False}
        ep_kills = 0; ep_len = 0

        while not done["__all__"]:
            ext_obs = _extend_obs_gat(obs_dict, env, gat_comm, device, wez_range, _obs_dim)
            blue_actions = {}
            with torch.no_grad():
                for aid in blue_ids:
                    o     = ext_obs.get(aid, np.zeros(_obs_dim, dtype=np.float32))
                    obs_t = torch.FloatTensor(o).unsqueeze(0).to(device)
                    raw, _ = actor.act(obs_t, deterministic=False)
                    blue_actions[aid] = MAPPOActor.squash(raw.squeeze(0)).cpu().numpy()

            states      = env.get_all_states()
            all_opp     = opp.act(states)
            red_actions = {r: all_opp[r] for r in red_ids if r in all_opp}

            action_dict          = {**blue_actions, **red_actions}
            obs_dict, rew, done, info = env.step(action_dict)

            for aid in blue_ids:
                if info[aid].get("r_kill", 0.0) > 0.5:
                    ep_kills += 1
            ep_len += 1

        winner = done.get("winner", "draw")
        if   winner == BLUE: wins   += 1
        elif winner == RED:  losses += 1
        else:                draws  += 1

        kills_per_ep.append(ep_kills)
        lengths.append(ep_len)

        final_states = env.get_all_states()
        if all(final_states[r][STATE_ALIVE] < 0.5 for r in red_ids if r in final_states):
            double_kills += 1

        if (ep + 1) % 50 == 0:
            print(f"  ep {ep+1:4d}/{n_episodes} | "
                  f"W={wins/(ep+1):.3f} K/ep={np.mean(kills_per_ep):.2f} "
                  f"D={draws/(ep+1):.3f}", flush=True)

    n              = n_episodes
    win_rate       = wins / n
    ci_lo, ci_hi   = wilson_ci(wins, n)

    return dict(
        checkpoint       = checkpoint_path,
        n_episodes       = n,
        seed             = seed,
        win_rate         = round(win_rate, 4),
        win_ci95_lo      = round(ci_lo, 4),
        win_ci95_hi      = round(ci_hi, 4),
        kill_per_ep      = round(float(np.mean(kills_per_ep)), 4),
        second_kill_rate = round(double_kills / n, 4),
        draw_rate        = round(draws / n, 4),
        loss_rate        = round(losses / n, 4),
        mean_ep_len      = round(float(np.mean(lengths)), 1),
        wins=wins, losses=losses, draws=draws,
    )


def print_summary(r):
    print("\n" + "=" * 57)
    print(f"  GAT-MAPPO EVAL — {r['n_episodes']} episode | seed={r['seed']}")
    print("=" * 57)
    print(f"  Win  rate       : {r['win_rate']:.1%}  "
          f"[{r['win_ci95_lo']:.1%} – {r['win_ci95_hi']:.1%}]  (95% CI)")
    print(f"  Kill / ep       : {r['kill_per_ep']:.3f}")
    print(f"  2nd kill rate   : {r['second_kill_rate']:.1%}")
    print(f"  Draw rate       : {r['draw_rate']:.1%}")
    print(f"  Loss rate       : {r['loss_rate']:.1%}")
    print(f"  Mean ep length  : {r['mean_ep_len']:.0f} steps")
    print(f"  Wins/Draws/Loss : {r['wins']}/{r['draws']}/{r['losses']}")
    print("=" * 57)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/mappo_gat_ep44000.pt")
    p.add_argument("--config",     default="configs/config.yaml")
    p.add_argument("--episodes",   type=int, default=500)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--device",     default="auto")
    p.add_argument("--output",     default="logs/eval_mappo_gat_results.json")
    args = p.parse_args()

    with open(ROOT / args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else torch.device(args.device)

    results = run_eval(config, str(ROOT / args.checkpoint),
                       args.episodes, args.seed, device)
    print_summary(results)

    out = ROOT / args.output
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Eval] Sonuçlar kaydedildi: {out}")


if __name__ == "__main__":
    main()

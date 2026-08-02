"""
eval_facmac.py
==============
FACMAC-TD3 checkpoint'ini heuristic rakibe karşı değerlendirir.

Kullanım:
    python -X utf8 evaluation/eval_facmac.py --checkpoint checkpoints/facmac_ep8400.pt
    python -X utf8 evaluation/eval_facmac.py --checkpoint checkpoints/facmac_ep8400.pt --episodes 500 --seed 42
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
from agents.heuristic_agent import MultiHeuristicPolicy
from models.facmac_net import FACMACActor, FACMACActorOM
from models.om_net import (
    CentralizedOpponentModel, CentralizedRoleAssigner,
    EnemyHistoryBuffer, build_team_state, INTENT_DEFENSIVE,
)
from envs.geometry_utils import antenna_train_angle, distance_3d

_WEZ_RANGE_MIN = 300.0
_WEZ_RANGE_MAX = 8000.0
_WEZ_ANGLE_MAX = np.radians(30.0)


def _rule_based_fire(blue_ids, red_ids, obs_dict, states, obs_dim):
    """WEZ içi + cooldown==0 ise fire=1."""
    fire = {}
    for aid in blue_ids:
        obs = obs_dict.get(aid, np.zeros(obs_dim, dtype=np.float32))
        fire[aid] = 0.0
        if obs[16] > 1e-4:   # cooldown_norm > 0
            continue
        b = states.get(aid)
        if b is None or b[STATE_ALIVE] < 0.5 or b[STATE_AMMO] < 0.5:
            continue
        b_pos = b[[STATE_X, STATE_Y, STATE_H]]
        for rid in red_ids:
            r = states.get(rid)
            if r is None or r[STATE_ALIVE] < 0.5:
                continue
            r_pos = r[[STATE_X, STATE_Y, STATE_H]]
            dist  = distance_3d(b_pos, r_pos)
            if dist < _WEZ_RANGE_MIN or dist > _WEZ_RANGE_MAX:
                continue
            ata = antenna_train_angle(b_pos, r_pos, b[STATE_PSI])
            if abs(ata) <= _WEZ_ANGLE_MAX:
                fire[aid] = 1.0
                break
    return fire


def wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson score %95 güven aralığı → (low, high)."""
    if n == 0:
        return 0.0, 0.0
    p   = k / n
    den = 1 + z**2 / n
    ctr = (p + z**2 / (2 * n)) / den
    rng = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return float(max(0.0, ctr - rng)), float(min(1.0, ctr + rng))


def run_eval(config: dict, checkpoint_path: str, n_episodes: int,
             seed: int, device: torch.device, deterministic: bool = True) -> dict:

    # ── Ortam ────────────────────────────────────────────────────────────────
    env = DogfightEnv(config)
    env.set_curriculum_phase(4)          # 2v2 normal spawn
    blue_ids = env.blue_ids
    red_ids  = env.red_ids
    all_ids  = blue_ids + red_ids
    team_map = {a: ("blue" if "blue" in a else "red") for a in all_ids}

    # ── FACMAC actor ─────────────────────────────────────────────────────────
    fcfg       = config.get("facmac", {})
    base_obs_dim = 50
    hidden     = int(fcfg.get("hidden", 256))

    ckpt       = torch.load(checkpoint_path, map_location=device, weights_only=False)
    actor_sd   = ckpt["actor"]
    use_om     = 'fc1_base.weight' in actor_sd   # yeni FACMACActorOM formatı

    if use_om:
        actor = FACMACActorOM(base_obs_dim=base_obs_dim, hidden=hidden).to(device)
        actor.load_state_dict(actor_sd, strict=False)
        # OM bileşenleri
        cent_om   = CentralizedOpponentModel().to(device)
        cent_role = CentralizedRoleAssigner().to(device)
        if 'cent_om' in ckpt:
            cent_om.load_state_dict(ckpt['cent_om'])
        if 'cent_role' in ckpt:
            cent_role.load_state_dict(ckpt['cent_role'], strict=False)
        cent_om.eval(); cent_role.eval()
        enemy_hist = EnemyHistoryBuffer()
        obs_dim    = 60
    else:
        actor   = FACMACActor(obs_dim=base_obs_dim, hidden=hidden).to(device)
        remap   = {'net.0.weight': 'net.0.weight', 'net.0.bias': 'net.0.bias',
                   'net.2.weight': 'net.2.weight', 'net.2.bias': 'net.2.bias',
                   'ctrl_head.weight': 'ctrl_head.weight', 'ctrl_head.bias': 'ctrl_head.bias'}
        actor.load_state_dict({k: v for k, v in actor_sd.items() if k in remap}, strict=False)
        obs_dim    = base_obs_dim
        cent_om    = None
        enemy_hist = None

    actor.eval()

    print(f"[Eval] FACMAC checkpoint : {checkpoint_path}")
    print(f"[Eval] Saved at          : ep={ckpt.get('episode','?')}, "
          f"step={ckpt.get('total_steps','?'):,}")
    print(f"[Eval] Device            : {device} | obs_dim={obs_dim} | OM={use_om}")
    print(f"[Eval] Episodes          : {n_episodes} | seed={seed} | "
          f"deterministic={deterministic}")

    # ── Heuristic rakip ──────────────────────────────────────────────────────
    opp = MultiHeuristicPolicy(config, all_ids, team_map)

    # ── Sayaçlar ─────────────────────────────────────────────────────────────
    wins = 0; losses = 0; draws = 0
    kills_per_ep = []
    lengths      = []
    double_kills = 0

    # Intent / role takibi (yalnızca OM modeli için)
    intent_steps = []
    role0_steps  = []
    role1_steps  = []

    env.seed(seed)

    for ep in range(n_episodes):
        obs_dict = env.reset()
        opp.reset()
        if enemy_hist is not None:
            enemy_hist.reset()
        done     = {"__all__": False}
        ep_kills = 0
        ep_len   = 0

        while not done["__all__"]:
            # OM obs 확장 (60D) 또는 base obs (50D)
            states = env.get_all_states()
            if use_om and cent_om is not None:
                base_arr = np.stack([
                    obs_dict.get(aid, np.zeros(base_obs_dim, dtype=np.float32))[:base_obs_dim]
                    for aid in blue_ids
                ], axis=0)
                hist_960 = enemy_hist.update(base_arr)
                with torch.no_grad():
                    hist_t    = torch.from_numpy(hist_960).unsqueeze(0).to(device)
                    intent_np = cent_om.intent_flat(hist_t).squeeze(0).cpu().numpy()
                ts_np = build_team_state([states.get(bid) for bid in blue_ids])
                with torch.no_grad():
                    x_role = torch.from_numpy(
                        np.concatenate([intent_np, ts_np]).astype(np.float32)
                    ).unsqueeze(0).to(device)
                    role_0, role_1 = cent_role.assign(x_role)
                roles = [role_0.squeeze(0).cpu().numpy(),
                         role_1.squeeze(0).cpu().numpy()]
                # intent/role takibi
                intent_steps.append(intent_np.copy())
                role0_steps.append(roles[0].copy())
                role1_steps.append(roles[1].copy())
                obs_60d = {
                    aid: np.concatenate([
                        obs_dict.get(aid, np.zeros(base_obs_dim, np.float32))[:base_obs_dim],
                        intent_np, roles[i]
                    ])
                    for i, aid in enumerate(blue_ids)
                }
                obs_for_actor = obs_60d
            else:
                obs_for_actor = obs_dict

            with torch.no_grad():
                ctrl_actions = {}
                for aid in blue_ids:
                    o     = obs_for_actor.get(aid, np.zeros(obs_dim, dtype=np.float32))
                    obs_t = torch.FloatTensor(o[:obs_dim]).unsqueeze(0).to(device)
                    ctrl, _ = actor.act(obs_t, deterministic=deterministic)
                    ctrl_actions[aid] = ctrl.squeeze(0).cpu().numpy()   # 4D

            fire_dict = _rule_based_fire(blue_ids, red_ids, obs_dict, states, base_obs_dim)
            blue_actions = {
                aid: np.append(ctrl_actions[aid], fire_dict[aid])
                for aid in blue_ids
            }

            all_opp     = opp.act(states)
            red_actions = {rid: all_opp[rid] for rid in red_ids if rid in all_opp}

            action_dict              = {**blue_actions, **red_actions}
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
        red_all_dead = all(
            final_states[r][STATE_ALIVE] < 0.5
            for r in red_ids if r in final_states
        )
        if red_all_dead:
            double_kills += 1

        if (ep + 1) % 50 == 0:
            print(f"  ep {ep+1:4d}/{n_episodes} | "
                  f"W={wins/(ep+1):.3f} K/ep={np.mean(kills_per_ep):.2f} "
                  f"D={draws/(ep+1):.3f}", flush=True)

    n = n_episodes
    win_rate     = wins / n
    ci_lo, ci_hi = wilson_ci(wins, n)
    kill_mean    = float(np.mean(kills_per_ep))
    draw_rate    = draws / n
    loss_rate    = losses / n
    dkr          = double_kills / n
    mean_len     = float(np.mean(lengths))

    out = dict(
        checkpoint       = checkpoint_path,
        n_episodes       = n,
        seed             = seed,
        deterministic    = deterministic,
        win_rate         = round(win_rate, 4),
        win_ci95_lo      = round(ci_lo, 4),
        win_ci95_hi      = round(ci_hi, 4),
        kill_per_ep      = round(kill_mean, 4),
        second_kill_rate = round(dkr, 4),
        draw_rate        = round(draw_rate, 4),
        loss_rate        = round(loss_rate, 4),
        mean_ep_len      = round(mean_len, 1),
        wins             = wins,
        losses           = losses,
        draws            = draws,
    )

    # Intent / role istatistikleri (yalnızca OM modellerinde)
    if intent_steps:
        out["intent_mean"] = np.mean(intent_steps, axis=0).round(4).tolist()
        out["role0_mean"]  = np.mean(role0_steps,  axis=0).round(4).tolist()
        out["role1_mean"]  = np.mean(role1_steps,  axis=0).round(4).tolist()

    return out


def print_summary(r: dict):
    print("\n" + "=" * 58)
    print(f"  FACMAC-TD3 EVAL — {r['n_episodes']} episode | seed={r['seed']}")
    print("=" * 58)
    print(f"  Win  rate       : {r['win_rate']:.1%}  "
          f"[{r['win_ci95_lo']:.1%} – {r['win_ci95_hi']:.1%}]  (95% CI)")
    print(f"  Kill / ep       : {r['kill_per_ep']:.3f}")
    print(f"  2nd kill rate   : {r['second_kill_rate']:.1%}")
    print(f"  Draw rate       : {r['draw_rate']:.1%}")
    print(f"  Loss rate       : {r['loss_rate']:.1%}")
    print(f"  Mean ep length  : {r['mean_ep_len']:.0f} steps")
    print(f"  Wins/Draws/Loss : {r['wins']}/{r['draws']}/{r['losses']}")
    if "intent_mean" in r:
        im = r["intent_mean"]
        r0 = r["role0_mean"]
        r1 = r["role1_mean"]
        print(f"  Intent (red_0)  : agg={im[0]:.3f} def={im[1]:.3f} eva={im[2]:.3f}")
        print(f"  Intent (red_1)  : agg={im[3]:.3f} def={im[4]:.3f} eva={im[5]:.3f}")
        print(f"  Role (blue_0)   : sniper={r0[0]:.3f} pursuit={r0[1]:.3f} "
              f"def={r0[2]:.3f} sup={r0[3]:.3f}")
        print(f"  Role (blue_1)   : sniper={r1[0]:.3f} pursuit={r1[1]:.3f} "
              f"def={r1[2]:.3f} sup={r1[3]:.3f}")
    print("=" * 58)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",    default="checkpoints/facmac_ep8400.pt")
    p.add_argument("--config",        default="configs/config.yaml")
    p.add_argument("--episodes",      type=int,  default=500)
    p.add_argument("--seed",          type=int,  default=42)
    p.add_argument("--device",        default="auto")
    p.add_argument("--output",        default="logs/eval_facmac_results.json")
    p.add_argument("--stochastic",    action="store_true",
                   help="Stokastik aksiyon örneklemesi (varsayılan: deterministic)")
    args = p.parse_args()

    with open(ROOT / args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else torch.device(args.device)

    results = run_eval(
        config, str(ROOT / args.checkpoint),
        args.episodes, args.seed, device,
        deterministic=not args.stochastic,
    )
    print_summary(results)

    out = ROOT / args.output
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Eval] Sonuçlar kaydedildi: {out}")


if __name__ == "__main__":
    main()

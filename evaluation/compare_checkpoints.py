"""
compare_checkpoints.py
======================
3 checkpoint karşılaştırması (500 ep, seed=42, heuristic rakip, 2v2):

  Model 1 — mappo_final.pt         : MAPPO baseline (50D, no GAT)
  Model 2 — mappo_gat_ep37000.pt   : GAT iletişim (68D, no OM/Role)
  Model 3 — mappo_gat_ep47000.pt   : GAT + OpponentModel + RoleSelector (76D)

Metrikler:
  win_rate + %95 CI (Wilson), kill/ep, 2nd_kill_rate, draw_rate
  Model 3 için ek: intent dağılımı, role dağılımı
"""

import sys
import numpy as np
import torch
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from envs.dogfight_env import DogfightEnv, BLUE, RED
from envs.aircraft_model import STATE_X, STATE_Y, STATE_H, STATE_ALIVE
from envs.geometry_utils import distance_3d, bearing_angle, wrap_to_pi
from agents.heuristic_agent import MultiHeuristicPolicy

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_EP     = 500
SEED     = 42
GAT_WEZ  = 8000.0

CHECKPOINTS = [
    ("MAPPO baseline",           "checkpoints/mappo_final.pt"),
    ("GAT iletisim",             "checkpoints/mappo_gat_ep37000.pt"),
    ("GAT + OM + RoleSelector",  "checkpoints/mappo_gat_ep47000.pt"),
]


# ---------------------------------------------------------------------------
# %95 Wilson CI
# ---------------------------------------------------------------------------

def wilson_ci(wins, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    denom  = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    margin = (z * np.sqrt(p*(1-p)/n + z**2/(4*n**2))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


# ---------------------------------------------------------------------------
# Model yükleyiciler
# ---------------------------------------------------------------------------

def load_mappo(ckpt_path, config):
    from training.train_mappo import MAPPOActor
    hidden = int(config["training"].get("hidden_dim", 256))
    actor  = MAPPOActor(50, 5, hidden=hidden).to(DEVICE)
    ckpt   = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return {"type": "mappo", "actor": actor}


def load_gat(ckpt_path, config, has_om=True):
    from training.train_mappo import GATMAPPOActor
    from models.gat_comm import GATComm
    comm = config.get("communication", {})
    msg_dim = int(comm.get("msg_dim", 16))

    # obs boyutunu checkpoint'ten otomatik tespit et
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    ext_dim  = ckpt["actor"]["fc1_new.weight"].shape[1]  # 18 veya 26
    obs_dim  = 50 + ext_dim                               # 68 veya 76

    actor = GATMAPPOActor(
        old_obs_dim        = 50,
        new_obs_dim        = obs_dim,
        action_dim         = 5,
        hidden             = int(config["training"].get("hidden_dim", 256)),
        with_role_selector = has_om,
    ).to(DEVICE)
    actor.load_state_dict(ckpt["actor"], strict=False)
    actor.eval()

    gat_comm = GATComm(
        node_dim = int(comm.get("node_dim", 17)),
        edge_dim = int(comm.get("edge_dim", 3)),
        n_heads  = int(comm.get("n_heads", 4)),
        msg_dim  = msg_dim,
    ).to(DEVICE)
    if "gat_comm" in ckpt:
        gat_comm.load_state_dict(ckpt["gat_comm"])
    gat_comm.eval()

    model = {
        "type":     "gat",
        "actor":    actor,
        "gat_comm": gat_comm,
        "obs_dim":  obs_dim,
        "msg_dim":  msg_dim,
    }

    if has_om and "opponent_model" in ckpt:
        from models.opponent_model import OpponentModel
        opp_cfg = config.get("opponent_model", {})
        opp_m = OpponentModel(
            history_steps = int(opp_cfg.get("history_window", 20)),
            hidden1       = int(opp_cfg.get("hidden1", 128)),
            hidden2       = int(opp_cfg.get("hidden2", 64)),
        ).to(DEVICE)
        opp_m.load_state_dict(ckpt["opponent_model"])
        opp_m.eval()
        model["opponent_model"] = opp_m

    return model


# ---------------------------------------------------------------------------
# GAT yardımcıları
# ---------------------------------------------------------------------------

def build_edge_feats(env, blue_ids):
    N      = len(blue_ids)
    edge   = np.zeros((N, N, 3), dtype=np.float32)
    states = env.get_all_states()
    for i, ai in enumerate(blue_ids):
        for j, aj in enumerate(blue_ids):
            if i == j:
                continue
            si = states.get(ai)
            sj = states.get(aj)
            if si is None or sj is None:
                continue
            pi = si[[STATE_X, STATE_Y, STATE_H]]
            pj = sj[[STATE_X, STATE_Y, STATE_H]]
            d  = distance_3d(pi, pj)
            b  = bearing_angle(pi, pj)
            ts = 0.0
            for eid in env.red_ids:
                es = states.get(eid)
                if es is not None and es[STATE_ALIVE] > 0.5:
                    de = distance_3d(pj, es[[STATE_X, STATE_Y, STATE_H]])
                    ts = max(ts, float(np.clip(1.0 - de / (GAT_WEZ + 1e-9), 0, 1)))
            edge[i, j] = [
                float(np.clip(d / (env.map_size + 1e-9), 0, 1)),
                float(np.clip(wrap_to_pi(b) / np.pi, -1, 1)),
                ts,
            ]
    return edge


def extend_68(obs_dict, env, model, blue_ids):
    """GAT-only (68D): base(50) + role(2) + gat_msg(16)."""
    states = env.get_all_states()
    ego_list   = [obs_dict.get(aid, np.zeros(50))[:17] for aid in blue_ids]
    alive_list = [float(states[aid][STATE_ALIVE]) if states.get(aid) is not None else 0.0
                  for aid in blue_ids]
    edge   = build_edge_feats(env, blue_ids)
    msgs   = model["gat_comm"].compute_messages(ego_list, edge, alive_list, DEVICE)
    role   = np.array([0.5, 0.5], dtype=np.float32)
    result = {}
    for i, aid in enumerate(blue_ids):
        base = obs_dict.get(aid, np.zeros(50, dtype=np.float32))
        result[aid] = np.concatenate([base, role, msgs[i]])  # 68D
    return result


def extend_76(obs_dict, env, model, blue_ids, prev_roles):
    """GAT + OM + Role (76D): base(50) + role(4) + gat_msg(16) + intent(6)."""
    states = env.get_all_states()
    ego_list   = [obs_dict.get(aid, np.zeros(50))[:17] for aid in blue_ids]
    alive_list = [float(states[aid][STATE_ALIVE]) if states.get(aid) is not None else 0.0
                  for aid in blue_ids]
    edge = build_edge_feats(env, blue_ids)
    msgs = model["gat_comm"].compute_messages(ego_list, edge, alive_list, DEVICE)

    # OpponentModel: batched GPU forward
    hist_list  = [env.get_enemy_history_flat(aid) for aid in blue_ids]
    with torch.no_grad():
        hist_t   = torch.from_numpy(np.stack(hist_list)).to(DEVICE)
        intents  = model["opponent_model"](hist_t).cpu().numpy()  # (2, 6)

    result  = {}
    roles_out = {}
    for i, aid in enumerate(blue_ids):
        base      = obs_dict.get(aid, np.zeros(50, dtype=np.float32))
        intent    = intents[i]
        resources = base[13:16]
        other     = [a for a in blue_ids if a != aid]
        tm_role   = prev_roles.get(other[0], np.full(4, 0.25, dtype=np.float32)) if other else np.full(4, 0.25, dtype=np.float32)
        with torch.no_grad():
            role_t = model["actor"].role_selector(
                torch.from_numpy(intent).unsqueeze(0).to(DEVICE),
                torch.from_numpy(tm_role).unsqueeze(0).to(DEVICE),
                torch.from_numpy(resources).unsqueeze(0).to(DEVICE),
                hard=True,
            )
        role_np = role_t.squeeze(0).cpu().numpy()
        prev_roles[aid] = role_np.copy()
        roles_out[aid]  = role_np.copy()
        result[aid]     = np.concatenate([base, role_np, msgs[i], intent])  # 76D
    return result, intents, roles_out


# ---------------------------------------------------------------------------
# Tek model eval
# ---------------------------------------------------------------------------

def run_eval(label, ckpt_path, config, n_ep=N_EP, seed=SEED):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"{'='*60}")

    # Model yükle
    if "gat" not in ckpt_path.lower() and "gat" not in label.lower():
        model = load_mappo(ckpt_path, config)
        mode  = "mappo"
    else:
        has_om = "opponent_model" in torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model  = load_gat(ckpt_path, config, has_om=has_om)
        mode   = "gat_om" if has_om else "gat"

    print(f"  Mode       : {mode} | obs_dim={model.get('obs_dim', 50)}")

    env = DogfightEnv(config)
    env.set_curriculum_phase(4)   # Faz-3: 2v2 normal spawn
    env.seed(seed)

    team_map   = {aid: ("blue" if "blue" in aid else "red") for aid in env.agent_ids}
    opp_policy = MultiHeuristicPolicy(config, env.agent_ids, team_map)

    from training.train_mappo import MAPPOActor

    wins = 0; losses = 0; draws = 0
    kills_total   = 0
    second_kills  = 0          # ep başına 2. kill tamamlanan ep sayısı
    intent_acc    = []         # (6,) per step
    role_acc      = []         # (4,) per step

    for ep in range(n_ep):
        obs_dict = env.reset()
        opp_policy.reset()
        prev_roles = {aid: np.full(4, 0.25, dtype=np.float32) for aid in env.blue_ids}
        done       = {"__all__": False}
        ep_kills   = 0

        while not done["__all__"]:
            # Obs uzat (gerekirse)
            if mode == "gat":
                obs_ext = extend_68(obs_dict, env, model, env.blue_ids)
            elif mode == "gat_om":
                obs_ext, ep_intents, ep_roles = extend_76(
                    obs_dict, env, model, env.blue_ids, prev_roles
                )
                intent_acc.append(np.mean(ep_intents, axis=0))
                role_acc.append(np.mean(list(ep_roles.values()), axis=0))
            else:
                obs_ext = obs_dict

            # MAPPO aksiyonlar
            actions = {}
            with torch.no_grad():
                for aid in env.blue_ids:
                    if aid not in obs_ext:
                        continue
                    obs_t  = torch.FloatTensor(obs_ext[aid]).unsqueeze(0).to(DEVICE)
                    raw, _ = model["actor"].act(obs_t, deterministic=False)
                    squash = MAPPOActor.squash(raw.squeeze(0))
                    actions[aid] = squash.cpu().numpy()

            # Heuristic aksiyonlar (yalnızca red ajanlar)
            all_heuristic = opp_policy.act(env.get_all_states())
            opp_actions   = {k: v for k, v in all_heuristic.items()
                             if k not in set(env.blue_ids)}
            action_dict   = {**actions, **opp_actions}

            obs_dict, rew_dict, done, info_dict = env.step(action_dict)

            for aid in env.blue_ids:
                if info_dict.get(aid, {}).get("r_kill", 0.0) > 0.5:
                    ep_kills += 1

        winner = done.get("winner", "draw")
        if winner == BLUE:
            wins    += 1
        elif winner == RED:
            losses  += 1
        else:
            draws   += 1

        kills_total  += ep_kills
        if ep_kills >= 2:
            second_kills += 1

        if (ep + 1) % 50 == 0:
            print(f"  [{ep+1:>3}/{n_ep}] W={wins/(ep+1):.2f} "
                  f"kill/ep={kills_total/(ep+1):.2f} "
                  f"2nd={second_kills/(ep+1):.2f}")

    # Metrikler
    win_rate    = wins   / n_ep
    draw_rate   = draws  / n_ep
    loss_rate   = losses / n_ep
    kill_per_ep = kills_total / n_ep
    second_rate = second_kills / n_ep
    ci_lo, ci_hi = wilson_ci(wins, n_ep)

    result = {
        "label":       label,
        "ckpt":        ckpt_path,
        "mode":        mode,
        "win_rate":    win_rate,
        "ci_lo":       ci_lo,
        "ci_hi":       ci_hi,
        "kill_per_ep": kill_per_ep,
        "2nd_kill_rate": second_rate,
        "draw_rate":   draw_rate,
        "loss_rate":   loss_rate,
        "wins": wins, "losses": losses, "draws": draws,
    }

    if intent_acc:
        ia = np.mean(intent_acc, axis=0)
        result["intent"] = {
            "e0_agg": float(ia[0]), "e0_def": float(ia[1]), "e0_eva": float(ia[2]),
            "e1_agg": float(ia[3]), "e1_def": float(ia[4]), "e1_eva": float(ia[5]),
        }
    if role_acc:
        ra = np.mean(role_acc, axis=0)
        result["role"] = {
            "sniper":    float(ra[0]),
            "pursuit":   float(ra[1]),
            "defensive": float(ra[2]),
            "support":   float(ra[3]),
        }

    return result


# ---------------------------------------------------------------------------
# Tablo yazdır
# ---------------------------------------------------------------------------

def print_table(results):
    print("\n" + "="*75)
    print("KARSILASTIRMA SONUCLARI — 500 ep | seed=42 | heuristic rakip | 2v2")
    print("="*75)

    hdr = f"{'Model':<28} {'Win%':>7} {'95% CI':>14} {'Kill/ep':>8} {'2nd%':>7} {'Draw%':>7}"
    print(hdr)
    print("-"*75)
    for r in results:
        lbl  = r["label"][:27]
        win  = r["win_rate"]
        lo   = r["ci_lo"]
        hi   = r["ci_hi"]
        k    = r["kill_per_ep"]
        sec  = r["2nd_kill_rate"]
        draw = r["draw_rate"]
        print(f"{lbl:<28} {win:>6.1%}  [{lo:.2f}-{hi:.2f}]  {k:>7.2f}  {sec:>6.1%}  {draw:>6.1%}")

    print("="*75)

    # Model 3 intent ve role
    for r in results:
        if "intent" in r:
            print(f"\n{r['label']} — Intent dagilimi (OpponentModel):")
            it = r["intent"]
            print(f"  Dusman-0: agg={it['e0_agg']:.2f}  def={it['e0_def']:.2f}  eva={it['e0_eva']:.2f}")
            print(f"  Dusman-1: agg={it['e1_agg']:.2f}  def={it['e1_def']:.2f}  eva={it['e1_eva']:.2f}")
        if "role" in r:
            ro = r["role"]
            print(f"\n{r['label']} — Role dagilimi (RoleSelector):")
            print(f"  sniper={ro['sniper']:.2f}  pursuit={ro['pursuit']:.2f}  "
                  f"defensive={ro['defensive']:.2f}  support={ro['support']:.2f}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config_path = PROJECT_ROOT / "configs/config.yaml"
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    results = []
    for label, ckpt in CHECKPOINTS:
        full_path = str(PROJECT_ROOT / ckpt)
        r = run_eval(label, full_path, config)
        results.append(r)

    print_table(results)


if __name__ == "__main__":
    main()

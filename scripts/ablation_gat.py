"""
ablation_gat.py
===============
GAT mesajlarinin etkisini olcer.

Kosul 1 - Normal : obs[50:68] = [role(2) + GAT_msg(16)]  (standart)
Kosul 2 - Ablation: obs[50:68] = zeros                   (GAT devre disi)

Her kosulda N_EP episode kosturulur; kill/ep ve win_rate karsilastirilir.
Fark yoksa actor GAT mesajlarini ignore ediyor demektir.

Kullanim:
    python -X utf8 scripts/ablation_gat.py
    python -X utf8 scripts/ablation_gat.py --checkpoint checkpoints/mappo_ep3000.pt --episodes 10
"""

import sys, argparse, yaml, numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from envs.dogfight_env import DogfightEnv, BLUE, RED
from envs.aircraft_model import STATE_ALIVE
from agents.heuristic_agent import MultiHeuristicPolicy
from utils.normalization import Normalizer
from training.train_mappo import GATMAPPOActor, MAPPOActor
from models.gat_comm import GATComm
from scripts.analyze_gat_attention import build_gat_inputs   # yardimci

N_EP_DEFAULT = 5


def load_model(ckpt_path, device):
    with open(PROJECT_ROOT / "configs/config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["communication"]["enable_comms"] = True
    comm   = config["communication"]
    hidden = int(config["training"].get("hidden_dim", 256))

    gat = GATComm(
        node_dim=int(comm.get("node_dim", 17)),
        edge_dim=int(comm.get("edge_dim",  3)),
        n_heads =int(comm.get("n_heads",   4)),
        msg_dim =int(comm.get("msg_dim",  16)),
    ).to(device)

    actor = GATMAPPOActor(
        old_obs_dim=50, new_obs_dim=68,
        action_dim=5, hidden=int(config["training"].get("hidden_dim", 256))
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    actor.load_state_dict(ckpt["actor"])
    gat.load_state_dict(ckpt["gat_comm"])
    actor.eval(); gat.eval()
    return gat, actor, config


def run_episodes(env, actor, gat, opp_policy, norm,
                 gat_node_dim, map_size, wez_range, device,
                 n_ep, zero_gat=False, seed_offset=0):
    """
    n_ep episode calistir.
    zero_gat=True ise GAT mesajlari sifirlanir (ablation koşulu).
    """
    kills_list, wins = [], 0
    MSG_DIM = 16; ROLE_DIM = 2

    for ep in range(n_ep):
        env.seed(seed_offset + ep)
        obs_dict = env.reset()
        opp_policy.reset()
        done = {"__all__": False}
        ep_kills = 0
        prev_alive = {a: 1.0 for a in env.agent_ids}

        while not done["__all__"]:
            state_dict = env.get_all_states()

            # GAT mesajlari
            with torch.no_grad():
                node_t, edge_t, alive_t = build_gat_inputs(
                    state_dict, env.blue_ids, env.red_ids,
                    gat_node_dim, map_size, wez_range, norm, device
                )
                if zero_gat:
                    msgs_np = np.zeros((len(env.blue_ids), MSG_DIM), dtype=np.float32)
                else:
                    msgs_t  = gat.forward(node_t, edge_t, alive_t)
                    msgs_np = msgs_t.squeeze(0).cpu().numpy()

            # Aksiyonlar
            actions = {}
            for i, aid in enumerate(env.blue_ids):
                role   = np.array([1.0, 0.0] if i == 0 else [0.0, 1.0],
                                   dtype=np.float32)
                obs_68 = np.concatenate([obs_dict[aid], role, msgs_np[i]])
                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs_68).unsqueeze(0).to(device)
                    raw, _ = actor.act(obs_t, deterministic=False)
                    actions[aid] = MAPPOActor.squash(raw.squeeze(0)).cpu().numpy()

            opp_actions = opp_policy.act(state_dict)
            obs_dict, _, done, _ = env.step({**actions, **opp_actions})

            ns = env.get_all_states()
            for aid in env.agent_ids:
                if prev_alive[aid] > 0.5 and ns[aid][STATE_ALIVE] < 0.5:
                    ep_kills += 1
                prev_alive[aid] = ns[aid][STATE_ALIVE]

        if done.get("winner") == BLUE:
            wins += 1
        kills_list.append(ep_kills)

    return {
        "kill_per_ep": float(np.mean(kills_list)),
        "win_rate":    wins / n_ep,
        "kills_list":  kills_list,
        "wins":        wins,
        "n_ep":        n_ep,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/mappo_ep3000.pt")
    p.add_argument("--episodes",   type=int, default=N_EP_DEFAULT)
    p.add_argument("--seed",       type=int, default=42)
    args = p.parse_args()

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_name = Path(args.checkpoint).stem
    print(f"[Ablation] Checkpoint : {args.checkpoint}")
    print(f"[Ablation] Episodes   : {args.episodes} x 2 kosul")
    print(f"[Ablation] Device     : {device}\n")

    gat, actor, config = load_model(PROJECT_ROOT / args.checkpoint, device)

    config["communication"]["enable_comms"] = True
    env = DogfightEnv(config)
    env.set_curriculum_phase(4)

    norm         = Normalizer(config)
    gat_node_dim = int(config["communication"].get("node_dim", 17))
    wez_range    = float(config.get("weapons", {}).get("wez_range_max", 8000.0))
    map_size     = float(config.get("env", {}).get("map_size", 50000.0))

    team_map   = {a: ("blue" if "blue" in a else "red") for a in env.agent_ids}
    opp_policy = MultiHeuristicPolicy(config, env.agent_ids, team_map)

    # ── Kosul 1: Normal (GAT aktif) ──────────────────────────────────────
    print("Kosul 1 — Normal (GAT aktif)...")
    res_normal = run_episodes(
        env, actor, gat, opp_policy, norm,
        gat_node_dim, map_size, wez_range, device,
        args.episodes, zero_gat=False, seed_offset=args.seed
    )
    print(f"  kill/ep={res_normal['kill_per_ep']:.3f}  "
          f"win_rate={res_normal['win_rate']:.2f}  "
          f"kills={res_normal['kills_list']}")

    # ── Kosul 2: Ablation (GAT=zeros) ────────────────────────────────────
    print("\nKosul 2 — Ablation (GAT mesajlari=zeros)...")
    res_ablation = run_episodes(
        env, actor, gat, opp_policy, norm,
        gat_node_dim, map_size, wez_range, device,
        args.episodes, zero_gat=True, seed_offset=args.seed
    )
    print(f"  kill/ep={res_ablation['kill_per_ep']:.3f}  "
          f"win_rate={res_ablation['win_rate']:.2f}  "
          f"kills={res_ablation['kills_list']}")

    # ── Karsilastirma ────────────────────────────────────────────────────
    d_kill = res_normal["kill_per_ep"] - res_ablation["kill_per_ep"]
    d_win  = res_normal["win_rate"]    - res_ablation["win_rate"]

    print(f"\n{'='*55}")
    print(f"GAT ABLATION SONUCU  —  {ckpt_name}")
    print(f"{'='*55}")
    print(f"{'':20s}  {'Normal':>10s}  {'Ablation':>10s}  {'Fark':>8s}")
    print(f"{'kill/ep':20s}  {res_normal['kill_per_ep']:>10.3f}  "
          f"{res_ablation['kill_per_ep']:>10.3f}  {d_kill:>+8.3f}")
    print(f"{'win_rate':20s}  {res_normal['win_rate']:>10.3f}  "
          f"{res_ablation['win_rate']:>10.3f}  {d_win:>+8.3f}")
    print(f"{'='*55}")

    # Yorum
    kill_sig = abs(d_kill) >= 0.2
    win_sig  = abs(d_win)  >= 0.1
    if kill_sig or win_sig:
        direction = "olumlu" if d_kill >= 0 else "olumsuz"
        print(f"\n=> GAT ETKI GOSTERDI ({direction})")
        print(f"   Actor mesajlari kullanıyor — koordinasyon sinyal uretiyor.")
    else:
        print(f"\n=> GAT ETKISIZ (fark kucuk)")
        print(f"   Actor mesajlari ignore ediyor ya da ep sayisi cok az.")
        print(f"   Daha fazla episode ile tekrar deneyin (--episodes 20).")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()

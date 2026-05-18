"""
analyze_gat_attention.py
========================
GAT mesaj vektorlerini analiz eder:
- Blue_0'in aldigi 16D mesajin mesafeye gore farklilasip farklilasmadigini test eder
- Mesafe < 3000m vs > 8000m mesaj ortalamalari karsilastirilir (L2 norm)
- Attention agirlik analizi (2 ajan: her zaman 1.0 oldugunu da raporlar)

Kullanim:
    python -X utf8 scripts/analyze_gat_attention.py
    python -X utf8 scripts/analyze_gat_attention.py --checkpoint checkpoints/mappo_gat_ep4000.pt
"""

import sys
import argparse
import numpy as np
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

from envs.dogfight_env import DogfightEnv, BLUE, RED
from envs.aircraft_model import STATE_X, STATE_Y, STATE_H, STATE_PSI, STATE_ALIVE
from agents.heuristic_agent import MultiHeuristicPolicy
from utils.normalization import Normalizer
from training.train_mappo import GATMAPPOActor, MAPPOActor
from models.gat_comm import GATComm

N_EPISODES  = 10
# Düşman mesafesine göre bant sınırları (blue_0'ın en yakın red ajana mesafesi)
DIST_NEAR   = 3000.0   # m  — düşman yakında (WEZ içi/yakını)
DIST_FAR    = 8000.0   # m  — düşman uzakta (WEZ dışı)


def load_phase2_model(ckpt_path, device):
    with open(PROJECT_ROOT / "configs/config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["communication"]["enable_comms"] = True

    comm   = config["communication"]
    hidden = int(config["training"].get("hidden_dim", 256))

    gat = GATComm(
        node_dim = int(comm.get("node_dim", 17)),
        edge_dim = int(comm.get("edge_dim",  3)),
        n_heads  = int(comm.get("n_heads",   4)),
        msg_dim  = int(comm.get("msg_dim",  16)),
    ).to(device)

    actor = GATMAPPOActor(
        old_obs_dim=50, new_obs_dim=68,
        action_dim=5, hidden=hidden
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    actor.load_state_dict(ckpt["actor"])
    gat.load_state_dict(ckpt["gat_comm"])
    actor.eval(); gat.eval()
    return gat, actor, config


def build_gat_inputs(states, blue_ids, red_ids, gat_node_dim,
                     map_size, wez_range, norm, device):
    """
    state_dict -> (node_t, edge_t, alive_t) tensors.
    Egitimle birebir ayni edge feature hesabini kullanir:
      distance_norm = dist(blue_i, blue_j) / map_size
      bearing_norm  = bearing(i->j) / pi
      threat_j      = max red threat to blue_j  (based on wez_range)
    """
    N = len(blue_ids)
    ego_list, alive_list = [], []
    for aid in blue_ids:
        s = states[aid]
        ego_17 = norm.ego_obs(s, cooldown_norm=0.0)
        ego_list.append(ego_17[:gat_node_dim])
        alive_list.append(float(s[STATE_ALIVE]))

    edge = np.zeros((N, N, 3), dtype=np.float32)
    for i, ai in enumerate(blue_ids):
        for j, aj in enumerate(blue_ids):
            if i == j:
                continue
            si, sj = states[ai], states[aj]
            dx_ij = sj[STATE_X] - si[STATE_X]
            dy_ij = sj[STATE_Y] - si[STATE_Y]
            dh_ij = sj[STATE_H] - si[STATE_H]
            dist_ij   = float(np.sqrt(dx_ij**2 + dy_ij**2 + dh_ij**2) + 1e-6)
            dist_norm = float(np.clip(dist_ij / (map_size + 1e-9), 0, 1))
            bearing   = np.arctan2(dy_ij, dx_ij)
            psi_i     = si[STATE_PSI]
            rel_bear  = float(np.arctan2(np.sin(bearing - psi_i),
                                          np.cos(bearing - psi_i)) / np.pi)
            # threat_j: j'nin en yakin kirmiziya mesafesi
            ts = 0.0
            pos_j = np.array([sj[STATE_X], sj[STATE_Y], sj[STATE_H]])
            for rid in red_ids:
                sr = states.get(rid)
                if sr is not None and sr[STATE_ALIVE] > 0.5:
                    dr = float(np.sqrt(np.sum((pos_j - np.array(
                        [sr[STATE_X], sr[STATE_Y], sr[STATE_H]]))**2)))
                    ts = max(ts, float(np.clip(1.0 - dr / (wez_range + 1e-9), 0, 1)))
            edge[i, j] = [dist_norm, rel_bear, ts]

    node_t  = torch.tensor(np.stack(ego_list), dtype=torch.float32,
                            device=device).unsqueeze(0)
    edge_t  = torch.tensor(edge, dtype=torch.float32, device=device).unsqueeze(0)
    alive_t = torch.tensor(alive_list, dtype=torch.float32, device=device).unsqueeze(0)
    return node_t, edge_t, alive_t


def nearest_red_dist(states, blue_id, red_ids):
    """blue_id'nin en yakin kirmizi ajana mesafesi (m)."""
    s0 = states[blue_id]
    if s0[STATE_ALIVE] < 0.5:
        return float('inf')
    pos0 = np.array([s0[STATE_X], s0[STATE_Y], s0[STATE_H]])
    best = float('inf')
    for rid in red_ids:
        sr = states.get(rid)
        if sr is not None and sr[STATE_ALIVE] > 0.5:
            pr = np.array([sr[STATE_X], sr[STATE_Y], sr[STATE_H]])
            d  = float(np.linalg.norm(pos0 - pr))
            if d < best:
                best = d
    return best


def run_collection(gat, actor, env, opp_policy, norm,
                   gat_node_dim, map_size, wez_range, device, n_episodes):
    """
    n_episodes boyunca koş, her adımda blue_0 mesajını ve mesafeyi kaydet.

    Mesafe = blue_0'ın en yakın kırmızı ajana olan mesafesi (dusman mesafesi).
    Bu, GAT'ın düşman tehdidine duyarlılığını test eder.

    Returns:
        msgs_near : list[np.ndarray(16,)] — dusman_dist < DIST_NEAR
        msgs_far  : list[np.ndarray(16,)] — dusman_dist > DIST_FAR
        msgs_all  : list[np.ndarray(16,)] — tüm adımlar
        dists_all : list[float]            — dusman mesafesi (m)
        wins, losses, draws
    """
    msgs_near, msgs_far, msgs_all, dists_all = [], [], [], []
    wins = losses = draws = 0

    for ep in range(n_episodes):
        obs_dict = env.reset()
        opp_policy.reset()
        done = {"__all__": False}

        while not done["__all__"]:
            state_dict = env.get_all_states()
            s0 = state_dict["blue_0"]
            s1 = state_dict["blue_1"]

            # Her iki mavi ajan da hayattaysa analiz et
            if s0[STATE_ALIVE] > 0.5 and s1[STATE_ALIVE] > 0.5:
                with torch.no_grad():
                    node_t, edge_t, alive_t = build_gat_inputs(
                        state_dict, env.blue_ids, env.red_ids,
                        gat_node_dim, map_size, wez_range, norm, device
                    )
                    msgs_t  = gat.forward(node_t, edge_t, alive_t)
                    msgs_np = msgs_t.squeeze(0).cpu().numpy()  # (N, 16)

                msg_b0 = msgs_np[0].copy()  # blue_0'ın aldığı mesaj

                # Dusmanin mesafesi: blue_0 -> en yakin kirmizi
                dist = nearest_red_dist(state_dict, "blue_0", env.red_ids)

                msgs_all.append(msg_b0)
                dists_all.append(dist)
                if dist < DIST_NEAR:
                    msgs_near.append(msg_b0)
                elif dist > DIST_FAR:
                    msgs_far.append(msg_b0)

            # Aksiyonlar (mesaj vektoru obs uzatmasi icin)
            with torch.no_grad():
                node_t2, edge_t2, alive_t2 = build_gat_inputs(
                    state_dict, env.blue_ids, env.red_ids,
                    gat_node_dim, map_size, wez_range, norm, device
                )
                msgs_t2  = gat.forward(node_t2, edge_t2, alive_t2)
                msgs_np2 = msgs_t2.squeeze(0).cpu().numpy()

            actions = {}
            for i, aid in enumerate(env.blue_ids):
                role   = np.array([1.0, 0.0] if i == 0 else [0.0, 1.0],
                                   dtype=np.float32)
                obs_68 = np.concatenate([obs_dict[aid], role, msgs_np2[i]])
                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs_68).unsqueeze(0).to(device)
                    raw, _ = actor.act(obs_t, deterministic=False)
                    squashed = MAPPOActor.squash(raw.squeeze(0))
                    actions[aid] = squashed.cpu().numpy()

            opp_actions = opp_policy.act(state_dict)
            action_dict = {**actions, **opp_actions}
            obs_dict, _, done, _ = env.step(action_dict)

        winner = done.get("winner", "draw")
        if winner == BLUE:   wins   += 1
        elif winner == RED:  losses += 1
        else:                draws  += 1

    return msgs_near, msgs_far, msgs_all, dists_all, wins, losses, draws


def print_report(msgs_near, msgs_far, msgs_all, dists_all,
                 wins, losses, draws, n_ep, ckpt_name):
    print(f"\n{'='*65}")
    print(f"GAT MESAJ ANALIZI  —  {ckpt_name}")
    print(f"{'='*65}")

    dists = np.array(dists_all)
    n_all = len(msgs_all)

    print(f"\n[0] EPISODE SONUCLARI  ({n_ep} episode)")
    print(f"    Win={wins}  Loss={losses}  Draw={draws}  | "
          f"win_rate={wins/n_ep:.2f}")

    print(f"\n[1] DUSMAN MESAFE DAGILIMI  ({n_all} adim, blue_0 -> en yakin red)")
    print(f"    Ortalama : {dists.mean()/1000:.2f} km  |  "
          f"std : {dists.std()/1000:.2f} km")
    print(f"    Min/Max  : {dists.min()/1000:.2f} / {dists.max()/1000:.2f} km")
    n_near = len(msgs_near)
    n_far  = len(msgs_far)
    n_mid  = n_all - n_near - n_far
    print(f"    < {DIST_NEAR/1000:.0f} km (yakin) : {n_near:5d} adim  "
          f"({100*n_near/max(n_all,1):.1f}%)")
    print(f"    mid                : {n_mid:5d} adim  "
          f"({100*n_mid/max(n_all,1):.1f}%)")
    print(f"    > {DIST_FAR/1000:.0f} km (uzak)  : {n_far:5d} adim  "
          f"({100*n_far/max(n_all,1):.1f}%)")

    if n_near == 0 or n_far == 0:
        print("\n  !! Yetersiz veri: yakın veya uzak bant boş — analiz yapılamıyor")
        print(f"{'='*65}\n")
        return

    arr_near = np.stack(msgs_near)   # (n_near, 16)
    arr_far  = np.stack(msgs_far)    # (n_far,  16)
    arr_all  = np.stack(msgs_all)    # (n_all,  16)

    mean_near = arr_near.mean(axis=0)  # (16,)
    mean_far  = arr_far.mean(axis=0)   # (16,)

    l2_diff = float(np.linalg.norm(mean_near - mean_far))
    cos_sim = float(
        np.dot(mean_near, mean_far) /
        (np.linalg.norm(mean_near) * np.linalg.norm(mean_far) + 1e-9)
    )

    print(f"\n[2] MESAJ ORTALAMA VEKTORLERI (16D)")
    print(f"    Yakin (<{DIST_NEAR/1000:.0f}km) ||mean|| = "
          f"{np.linalg.norm(mean_near):.4f}")
    print(f"    Uzak  (>{DIST_FAR/1000:.0f}km) ||mean|| = "
          f"{np.linalg.norm(mean_far):.4f}")

    print(f"\n[3] MESAJ FARKLILIGI")
    print(f"    L2(mean_near, mean_far)   = {l2_diff:.4f}")
    print(f"    cosine_sim(near, far)     = {cos_sim:.4f}")

    # Genel std (mesajin ne kadar degistigini goster)
    msg_std_per_dim = arr_all.std(axis=0)
    print(f"\n[4] MESAJ VARYANSI (tum adimlar)")
    print(f"    std_mean (16 dim ort.)    = {msg_std_per_dim.mean():.4f}")
    print(f"    std_max  (en degerli dim) = {msg_std_per_dim.max():.4f}  "
          f"(dim {int(msg_std_per_dim.argmax())})")
    print(f"    std_min  (en sabit dim)   = {msg_std_per_dim.min():.4f}")

    # Boyut bazli fark
    dim_diff = np.abs(mean_near - mean_far)
    top3     = dim_diff.argsort()[::-1][:3]
    print(f"\n[5] EN COK FARK EDEN BOYUTLAR (yakin vs uzak)")
    for rank, d in enumerate(top3, 1):
        print(f"    #{rank}  dim[{d:2d}]: "
              f"near={mean_near[d]:+.4f}  far={mean_far[d]:+.4f}  "
              f"diff={dim_diff[d]:.4f}")

    print(f"\n[6] YORUM")
    if l2_diff > 0.5:
        print(f"    ✓ GAT ANLAM TASIYOR — L2={l2_diff:.3f} > 0.5")
        print(f"      Mesafe degisince mesaj icerig degisiyor.")
        print(f"      Koordinasyon icin potansiyel bilgi akisi mevcut.")
    elif l2_diff > 0.2:
        print(f"    ~ ZAYIF SINYAL — L2={l2_diff:.3f} (0.2–0.5)")
        print(f"      Mesafe bir miktar etkiliyor, ancak gürültülü.")
    else:
        print(f"    !! GAT ANLAMSIZ — L2={l2_diff:.3f} < 0.2")
        print(f"      Mesaj vektoru mesafeden bagimsiz (sabit offset gibi).")
        if msg_std_per_dim.mean() < 0.1:
            print(f"      Cok dusuk varyans: GAT hemen hemen sabit mesaj gonderiyor.")

    if cos_sim > 0.95:
        print(f"    !! Cosine benzerlik cok yuksek ({cos_sim:.3f}): "
              f"mesajlar yonsel olarak ayni (sadece buyukluk farki)")

    print(f"{'='*65}\n")

    return {
        "l2_near_far": l2_diff,
        "cos_sim":     cos_sim,
        "msg_std_mean": float(msg_std_per_dim.mean()),
        "n_near":       n_near,
        "n_far":        n_far,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  default="checkpoints/mappo_ep3000.pt")
    p.add_argument("--episodes",    type=int, default=N_EPISODES)
    p.add_argument("--seed",        type=int, default=7)
    p.add_argument("--spawn-dist",  type=float, default=6000.0,
                   help="Analiz icin spawn min mesafe (m); genis aralik icin 6000)")
    args = p.parse_args()

    ckpt_path = PROJECT_ROOT / args.checkpoint
    ckpt_name = Path(args.checkpoint).stem

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[GAT-Analiz] Device    : {device}")
    print(f"[GAT-Analiz] Checkpoint: {args.checkpoint}")

    gat, actor, config = load_phase2_model(ckpt_path, device)

    # Analiz icin genis spawn araligini zorla (egitim config'inden bagimsiz)
    config["communication"]["enable_comms"] = True
    config["curriculum_v2"]["phase3_spawn_dist"]     = args.spawn_dist
    config["curriculum_v2"]["phase3_spawn_dist_max"] = 12000.0
    print(f"[GAT-Analiz] Spawn aralik: {args.spawn_dist/1000:.0f}–12 km "
          f"(analiz icin override)")

    env = DogfightEnv(config)
    env.set_curriculum_phase(4)
    env.seed(args.seed)

    norm         = Normalizer(config)
    gat_node_dim = int(config["communication"].get("node_dim", 17))
    wez_range    = float(config.get("weapons", {}).get("wez_range_max", 8000.0))
    map_size     = float(config.get("env", {}).get("map_size", 50000.0))

    team_map   = {aid: ("blue" if "blue" in aid else "red")
                  for aid in env.agent_ids}
    opp_policy = MultiHeuristicPolicy(config, env.agent_ids, team_map)

    print(f"[GAT-Analiz] Mesafe analizi: blue_0 -> en yakin RED ajani")
    print(f"             Yakin bant: <{DIST_NEAR/1000:.0f}km  |  "
          f"Uzak bant: >{DIST_FAR/1000:.0f}km")
    print(f"\n[GAT-Analiz] {args.episodes} episode kosuluyor...\n")

    msgs_near, msgs_far, msgs_all, dists_all, wins, losses, draws = run_collection(
        gat, actor, env, opp_policy, norm,
        gat_node_dim, map_size, wez_range, device, args.episodes
    )

    result = print_report(
        msgs_near, msgs_far, msgs_all, dists_all,
        wins, losses, draws, args.episodes, ckpt_name
    )

    # Sonuclari dosyaya kaydet
    out_path = PROJECT_ROOT / "logs/gat_msg_analysis.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"checkpoint={args.checkpoint}\n")
        f.write(f"n_episodes={args.episodes}\n")
        f.write(f"wins={wins} losses={losses} draws={draws}\n")
        if result:
            for k, v in result.items():
                f.write(f"{k}={v}\n")
    print(f"[GAT-Analiz] Sonuclar kaydedildi: {out_path}")


if __name__ == "__main__":
    main()

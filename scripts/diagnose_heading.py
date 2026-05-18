"""
diagnose_heading.py
===================
Blue ajanının düşmana yönelim analizi.

Her 50 adımda:
  - Blue-Red mesafesi
  - Bearing: Red'in Blue'ya göre açısı (ENU, 0=Kuzey, CW)
  - Blue heading (PSI, derece)
  - Açı farkı: |bearing - heading| (0 = tam karşıda, 180 = tam arkada)
"""

import sys
import os
import numpy as np
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import torch
from pathlib import Path
from envs.dogfight_env import DogfightEnv, BLUE, RED
from envs.aircraft_model import STATE_X, STATE_Y, STATE_H, STATE_PSI, STATE_ALIVE
from envs.geometry_utils import distance_3d
from agents.heuristic_agent import MultiHeuristicPolicy
from training.train_mappo import MAPPOActor
from utils.normalization import Normalizer

CHECKPOINT = "checkpoints/mappo_ep2299.pt"
N_EPISODES = 5
LOG_EVERY   = 50

def rad2deg(r):
    return float(np.degrees(r))

def bearing_deg(ax, ay, bx, by):
    """A'dan B'ye coğrafi bearing (derece, 0=Kuzey, CW)."""
    dx = bx - ax
    dy = by - ay
    angle = np.degrees(np.arctan2(dx, dy))  # ENU: arctan2(east, north)
    return float(angle % 360)

def angle_diff(a, b):
    """İki açı arasındaki mutlak fark, [0, 180] aralığında."""
    diff = abs(a - b) % 360
    return float(min(diff, 360 - diff))

def psi_to_heading(psi_rad):
    """PSI (radyan, math convention: 0=Doğu, CCW) → coğrafi heading (0=Kuzey, CW)."""
    deg = np.degrees(psi_rad)
    heading = (90 - deg) % 360
    return float(heading)

def main():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    # Env — normal spawn (faz 3)
    env = DogfightEnv(config)
    env.set_curriculum_phase(3)

    # Actor yükle
    device = "cuda" if torch.cuda.is_available() else "cpu"
    obs_dim    = 50
    action_dim = 5
    actor = MAPPOActor(obs_dim, action_dim).to(device)

    ckpt = torch.load(CHECKPOINT, map_location=device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    print(f"Checkpoint yüklendi: {CHECKPOINT} (ep={ckpt.get('episode',0)})")

    normalizer = Normalizer(config)
    team_map   = {aid: ("blue" if "blue" in aid else "red") for aid in env.agent_ids}
    heuristic  = MultiHeuristicPolicy(config, env.agent_ids, team_map)

    print(f"\n{'='*80}")
    print(f"{'Ep':>3} {'Step':>5} | {'Dist(m)':>9} | {'Bearing':>8} | {'Heading':>8} | {'DeltaAngle':>10} | Yorum")
    print(f"{'='*80}")

    for ep in range(1, N_EPISODES + 1):
        obs_dict = env.reset()
        step = 0
        ep_deltas = []

        while True:
            # Blue aksiyon
            actions = {}
            states = env.get_all_states()

            for aid in env.blue_ids:
                s = states[aid]
                if s[STATE_ALIVE] < 0.5:
                    continue
                obs_t = torch.FloatTensor(obs_dict[aid]).unsqueeze(0).to(device)
                with torch.no_grad():
                    action, _ = actor.act(obs_t, deterministic=True)
                actions[aid] = action.squeeze(0).cpu().numpy()

            # Red heuristic
            red_actions = heuristic.act(states)
            actions.update(red_actions)

            # Her LOG_EVERY adımda log
            if step % LOG_EVERY == 0:
                b_state = states.get("blue_0")
                r_state = states.get("red_0")

                if b_state is not None and r_state is not None:
                    if b_state[STATE_ALIVE] > 0.5 and r_state[STATE_ALIVE] > 0.5:
                        bx, by = b_state[STATE_X], b_state[STATE_Y]
                        rx, ry = r_state[STATE_X], r_state[STATE_Y]
                        bh     = b_state[STATE_H]
                        rh     = r_state[STATE_H]

                        dist    = distance_3d(
                            np.array([bx, by, bh]),
                            np.array([rx, ry, rh])
                        )
                        bearing = bearing_deg(bx, by, rx, ry)
                        heading = psi_to_heading(b_state[STATE_PSI])
                        delta   = angle_diff(bearing, heading)
                        ep_deltas.append(delta)

                        yorum = "HEDEFLIYOR" if delta < 30 else ("YAKIN" if delta < 60 else ("UZAK" if delta < 120 else "TERS"))
                        print(f"{ep:>3} {step:>5} | {dist:>9.0f} | {bearing:>8.1f} | {heading:>8.1f} | {delta:>10.1f} | {yorum}")

            obs_dict, _, done_dict, _ = env.step(actions)
            step += 1

            if done_dict.get("__all__", False):
                winner = done_dict.get("winner", "draw")
                mean_delta = np.mean(ep_deltas) if ep_deltas else float('nan')
                print(f"  --> Ep {ep} bitti: {winner}, {step} adım | Ortalama delta: {mean_delta:.1f}°")
                print()
                break

    print(f"{'='*80}")
    print("Analiz tamamlandi.")
    print()
    print("Yorumlama:")
    print("  DeltaAngle < 30  : Ajan düşmana doğru bakiyor (hedefleme var)")
    print("  DeltaAngle 30-60 : Yaklaşik yönelim")
    print("  DeltaAngle 60-120: Yanlış yön")
    print("  DeltaAngle > 120 : Düşmandan kaçiyor / ters yön")

if __name__ == "__main__":
    main()

"""200 episode WEZ fire rate testi — fix sonrası."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml, numpy as np, torch
from envs.dogfight_env import DogfightEnv
from agents.heuristic_agent import MultiHeuristicPolicy
from training.train_mappo import MAPPOActor
from envs.aircraft_model import ACTION_FIRE, STATE_ALIVE

with open('configs/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

env       = DogfightEnv(config)
team_map  = {aid: ("blue" if "blue" in aid else "red") for aid in env.agent_ids}
heuristic = MultiHeuristicPolicy(config, env.red_ids, team_map)

torch.manual_seed(42)
actor = MAPPOActor(obs_dim=50, action_dim=5, hidden=256)
actor.eval()

N_EP = 200
total_steps = wez_steps = wez_fire_cmd = wez_fired = wez_cooldown_blocked = wins = 0

for ep in range(N_EP):
    obs_dict  = env.reset()
    heuristic.reset()
    done      = False

    while not done:
        states      = env.get_all_states()
        action_dict = {}

        with torch.no_grad():
            for bid in env.blue_ids:
                obs_t = torch.FloatTensor(obs_dict[bid]).unsqueeze(0)
                raw, _ = actor.act(obs_t, deterministic=False)
                action_dict[bid] = MAPPOActor.squash(raw.squeeze(0)).cpu().numpy()

        action_dict.update(heuristic.act(states))

        for bid in env.blue_ids:
            b_state = states[bid]
            wep     = env._weapons[bid]
            for rid in env.red_ids:
                rst = states[rid]
                if rst[STATE_ALIVE] < 0.5:
                    continue
                wez = wep.compute_wez(b_state, rst)
                if wez["in_wez"]:
                    wez_steps += 1
                    if float(action_dict[bid][ACTION_FIRE]) >= 0.5:
                        wez_fire_cmd += 1
                        if wep.can_fire:
                            wez_fired += 1
                        else:
                            wez_cooldown_blocked += 1
                break

        obs_dict, _, done_dict, _ = env.step(action_dict)
        done = done_dict["__all__"]
        total_steps += 1

    if done_dict.get("winner") == "blue":
        wins += 1

print(f"\n=== 200 Episode WEZ Fire Test (obs_dim=50, cooldown=0.5s) ===")
print(f"Toplam adım       : {total_steps:,}")
print(f"WEZ içi adım      : {wez_steps} ({100*wez_steps/max(total_steps,1):.1f}%)")
print(f"WEZ + fire_cmd    : {wez_fire_cmd} ({100*wez_fire_cmd/max(wez_steps,1):.1f}% of WEZ)")
print(f"WEZ + ateş OK     : {wez_fired}   ({100*wez_fired/max(wez_steps,1):.1f}% of WEZ)")
print(f"WEZ + cooldown blk: {wez_cooldown_blocked}")
print(f"Win rate          : {wins}/{N_EP} = {100*wins/N_EP:.1f}%")

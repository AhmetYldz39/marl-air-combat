"""
diagnose_fire.py
================
En iyi checkpoint'i yükleyip 10 episode çalıştır.
Her adımda:
  - Blue'nun düşmana mesafesi
  - WEZ içinde mi
  - Fire aksiyon değeri (raw, 0-1)
  - Gerçekten ateş edildi mi (fire_cmd >= 0.5 AND fired)
  - Ateş tetiklenmediyse neden (fail_reason)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import yaml
from pathlib import Path
from collections import defaultdict

from envs.dogfight_env   import DogfightEnv
from envs.aircraft_model import ACTION_FIRE
from utils.normalization  import Normalizer
from training.train_mappo import MAPPOActor

# ── Config ────────────────────────────────────────────────────────────────────
PROJ     = Path(__file__).parent.parent
CFG_PATH = PROJ / "configs" / "config.yaml"
with open(CFG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_EP     = 10
MAX_ROWS = 3000   # toplam satır sınırı (uzun episode'lar için)

# ── Checkpoint seç (pool'dan en iyi) ─────────────────────────────────────────
ckpt_dir = PROJ / "checkpoints"
candidates = sorted(ckpt_dir.glob("pool_actor_ep*.pt"),
                    key=lambda p: int(p.stem.replace("pool_actor_ep", "")))
if not candidates:
    candidates = sorted(ckpt_dir.glob("mappo_ep*.pt"),
                        key=lambda p: int(p.stem.replace("mappo_ep", "")))

# ep400 varsa onu seç, yoksa en yüksek ep
target = next((p for p in candidates if "ep400" in p.name), candidates[-1])
print(f"Kullanılan checkpoint: {target.name}\n")

# ── Actor yükle ───────────────────────────────────────────────────────────────
ckpt  = torch.load(str(target), map_location=DEVICE, weights_only=False)
actor = MAPPOActor(obs_dim=49, action_dim=5, hidden=256)
actor.load_state_dict(ckpt["actor"])
actor.to(DEVICE)
actor.eval()

# ── Env kur ───────────────────────────────────────────────────────────────────
env        = DogfightEnv(config)
normalizer = Normalizer(config)
blue_ids   = env.blue_ids
red_ids    = env.red_ids

# ── Heuristic (Red için) ──────────────────────────────────────────────────────
from agents.heuristic_agent import MultiHeuristicPolicy
team_map  = {aid: ("blue" if "blue" in aid else "red") for aid in env.agent_ids}
heuristic = MultiHeuristicPolicy(config, red_ids, team_map)

# ── Veri toplama ──────────────────────────────────────────────────────────────
rows = []

for ep in range(N_EP):
    obs_dict = env.reset()
    heuristic.reset()
    done     = False
    step     = 0

    while not done:
        step += 1
        states     = env.get_all_states()
        action_dict = {}

        # Blue: neural policy
        with torch.no_grad():
            for bid in blue_ids:
                obs   = obs_dict[bid]
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
                raw, _ = actor.act(obs_t, deterministic=False)
                sq     = MAPPOActor.squash(raw.squeeze(0)).cpu().numpy()
                action_dict[bid] = sq

        # Red: heuristic
        action_dict.update(heuristic.act(states))

        # WEZ + fire analizi (env.step'ten ÖNCE — process_fire simülasyonu)
        for bid in blue_ids:
            b_state = states[bid]

            # En yakın hayatta düşmanı bul
            closest_dist = float("inf")
            closest_st   = None
            for rid in red_ids:
                rst  = states[rid]
                from envs.aircraft_model import STATE_ALIVE
                if rst[STATE_ALIVE] < 0.5:
                    continue
                diff = rst[:3] - b_state[:3]
                d    = float(np.linalg.norm(diff))
                if d < closest_dist:
                    closest_dist = d
                    closest_st   = rst

            wep = env._weapons[bid]

            # WEZ hesapla
            if closest_st is not None:
                wez_info = wep.compute_wez(b_state, closest_st)
            else:
                wez_info = {"in_wez": False, "dist": float("inf")}

            fire_val = float(action_dict[bid][ACTION_FIRE])
            fire_cmd = fire_val >= 0.5

            # process_fire simüle et (state'i değiştirmez, sadece okur)
            if closest_st is not None:
                fr = wep.process_fire(b_state, closest_st, action_dict[bid], env.dt)
                fired  = fr.get("fired", False)
                reason = fr.get("fail_reason", "")
            else:
                fired  = False
                reason = "no_target"

            rows.append({
                "ep":       ep + 1,
                "step":     step,
                "agent":    bid,
                "dist_m":   round(closest_dist, 1),
                "in_wez":   wez_info.get("in_wez", False),
                "fire_val": round(fire_val, 3),
                "fire_cmd": fire_cmd,
                "fired":    fired,
                "reason":   reason,
            })

        # Env adımı
        obs_dict, _, done_dict, _ = env.step(action_dict)
        done = done_dict["__all__"]

        if len(rows) >= MAX_ROWS:
            done = True

    print(f"Ep {ep+1} tamamlandı ({step} adım)")

# ── Tablo yazdır ──────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print(f"{'Ep':>3} {'Step':>5} {'Agent':>8} {'Dist(m)':>8} {'InWEZ':>6} {'FireVal':>8} {'FireCmd':>8} {'Fired':>6} {'Reason'}")
print("-" * 90)

wez_fire_rows   = []   # WEZ içinde + fire_cmd=True
wez_nofire_rows = []   # WEZ içinde + fire_cmd=False
total_wez_steps = 0

for r in rows:
    in_wez = r["in_wez"]
    if in_wez:
        total_wez_steps += 1
        if r["fire_cmd"]:
            wez_fire_rows.append(r)
        else:
            wez_nofire_rows.append(r)

    # Sadece WEZ içindeki adımları veya fire_cmd=True adımları göster
    if in_wez or r["fire_cmd"]:
        print(f"{r['ep']:>3} {r['step']:>5} {r['agent']:>8} "
              f"{r['dist_m']:>8.0f} {str(r['in_wez']):>6} "
              f"{r['fire_val']:>8.3f} {str(r['fire_cmd']):>8} "
              f"{str(r['fired']):>6}  {r['reason'] or ''}")

print("=" * 90)

# ── Özet istatistikler ────────────────────────────────────────────────────────
print(f"\n{'=' * 50}")
print("ÖZET")
print(f"{'=' * 50}")
print(f"Toplam adım         : {len(rows)}")
print(f"WEZ içi adım        : {total_wez_steps} ({100*total_wez_steps/max(len(rows),1):.1f}%)")
print(f"WEZ içi + ateş cmd  : {len(wez_fire_rows)}")
print(f"WEZ içi + ateş yok  : {len(wez_nofire_rows)}")

if wez_fire_rows:
    fired_ok = sum(1 for r in wez_fire_rows if r["fired"])
    print(f"WEZ+ateş → gerçek   : {fired_ok} / {len(wez_fire_rows)}")
    reasons = defaultdict(int)
    for r in wez_fire_rows:
        if not r["fired"]:
            reasons[r["reason"]] += 1
    if reasons:
        print("Ateş tetiklenmeme nedenleri (WEZ içi, cmd=True):")
        for k, v in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v}")

# fire_val dağılımı (WEZ içinde)
if total_wez_steps > 0:
    wez_vals = [r["fire_val"] for r in rows if r["in_wez"]]
    print(f"\nWEZ içi fire_val istatistikleri:")
    print(f"  mean : {np.mean(wez_vals):.3f}")
    print(f"  std  : {np.std(wez_vals):.3f}")
    print(f"  >0.5 : {sum(1 for v in wez_vals if v > 0.5)} / {len(wez_vals)}")
    print(f"  >0.9 : {sum(1 for v in wez_vals if v > 0.9)} / {len(wez_vals)}")

"""
eval_structural_hypothesis.py
==============================
Yapısal hipotez testi: rol sinyali sıfırlandığında FACMAC ve MAPPO ajan
davranışı birbirinden ayrışmaya devam ediyor mu?

Hipotez: FACMAC'ta rol, VAR olan farklılaşmayı etiketler (değil yaratır).
MAPPO'da ise zaten farklılaşma yok.

Tahmin:
  - FACMAC zeroed → davranış asimetrisini korur (rol olmadan da ayrışır)
  - MAPPO zeroed  → davranış simetrik kalır (rol olmadan da simetrik)

Kullanım:
    python -X utf8 evaluation/eval_structural_hypothesis.py facmac
    python -X utf8 evaluation/eval_structural_hypothesis.py mappo
    python -X utf8 evaluation/eval_structural_hypothesis.py facmac --episodes 200 --seed 42

Ölçülen metrikler (her ajan için ayrı):
  - ep başı kill sayısı
  - ortalama angajman mesafesi (en yakın canlı düşmana, her adım)
  - WEZ'de geçen adım sayısı (±30°, 300-8000m)
  - fire sayısı

Fark koşulları:
  - normal : cent_role.assign() ile normal rol ataması
  - zeroed : roles[i] = [0,0,0,0]  (fc1_role girişi sıfır)
"""

import sys
import argparse
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
import torch.nn as nn

from envs.dogfight_env import DogfightEnv, BLUE, RED
from envs.aircraft_model import (
    STATE_ALIVE, STATE_X, STATE_Y, STATE_H, STATE_PSI, STATE_AMMO,
)
from envs.geometry_utils import distance_3d, bearing_angle, wrap_to_pi, antenna_train_angle
from agents.heuristic_agent import MultiHeuristicPolicy
from models.om_net import (
    CentralizedOpponentModel, CentralizedRoleAssigner,
    EnemyHistoryBuffer, build_team_state,
)
import yaml

_WEZ_RANGE_MIN   = 300.0
_WEZ_RANGE_MAX   = 8000.0
_WEZ_ANGLE_MAX   = np.radians(30.0)
_BASE_OBS_DIM    = 50


# ---------------------------------------------------------------------------
# Yardımcı: WEZ içi mi?
# ---------------------------------------------------------------------------
def _in_wez(agent_state, enemy_states: list) -> bool:
    if agent_state[STATE_ALIVE] < 0.5:
        return False
    a_pos = agent_state[[STATE_X, STATE_Y, STATE_H]]
    for es in enemy_states:
        if es is None or es[STATE_ALIVE] < 0.5:
            continue
        e_pos = es[[STATE_X, STATE_Y, STATE_H]]
        d = distance_3d(a_pos, e_pos)
        if _WEZ_RANGE_MIN <= d <= _WEZ_RANGE_MAX:
            ata = antenna_train_angle(a_pos, e_pos, agent_state[STATE_PSI])
            if abs(ata) <= _WEZ_ANGLE_MAX:
                return True
    return False


def _nearest_enemy_dist(agent_state, enemy_states: list) -> float:
    if agent_state[STATE_ALIVE] < 0.5:
        return float("nan")
    a_pos  = agent_state[[STATE_X, STATE_Y, STATE_H]]
    dists  = []
    for es in enemy_states:
        if es is None or es[STATE_ALIVE] < 0.5:
            continue
        dists.append(distance_3d(a_pos, es[[STATE_X, STATE_Y, STATE_H]]))
    return float(np.min(dists)) if dists else float("nan")


# ---------------------------------------------------------------------------
# Legacy MAPPO sequential role assigner (ep44000 formatı)
# ---------------------------------------------------------------------------
class _LegacySequentialRoleAssigner(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp0 = nn.Sequential(nn.Linear(12, 64), nn.ReLU(), nn.Linear(64, 4))
        self.mlp1 = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 4))

    @torch.no_grad()
    def assign(self, x):
        logit0 = self.mlp0(x)
        role0  = F.one_hot(logit0.argmax(dim=-1), 4).float()
        logit1 = self.mlp1(torch.cat([x, role0], dim=-1))
        role1  = F.one_hot(logit1.argmax(dim=-1), 4).float()
        return role0, role1


# ---------------------------------------------------------------------------
# Ana eval döngüsü
# ---------------------------------------------------------------------------
def run_condition(
    model_type: str,
    ckpt_path: str,
    config: dict,
    n_episodes: int,
    seed: int,
    zero_role: bool,
    device: torch.device,
) -> dict:
    """
    Tek koşul için eval döngüsü çalıştır.

    zero_role=True  → roles[i] = [0,0,0,0]
    zero_role=False → cent_role.assign(x_role) ile normal atama
    """
    label = "zeroed" if zero_role else "normal"
    print(f"\n[Eval] Koşul: {label} | model: {model_type} | ep: {n_episodes} | seed: {seed}")

    # Ortam
    env = DogfightEnv(config)
    env.set_curriculum_phase(4)
    blue_ids = env.blue_ids
    red_ids  = env.red_ids
    all_ids  = blue_ids + red_ids
    team_map = {a: ("blue" if "blue" in a else "red") for a in all_ids}

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # ── Model yükle ────────────────────────────────────────────────────────
    if model_type == "facmac":
        from models.facmac_net import FACMACActorOM
        fcfg    = config.get("facmac", {})
        hidden  = int(fcfg.get("hidden", 256))
        actor   = FACMACActorOM(base_obs_dim=_BASE_OBS_DIM, hidden=hidden).to(device)
        actor.load_state_dict(ckpt["actor"], strict=False)
        obs_dim_eval = 60   # 50 + 6 intent + 4 role
        use_rule_fire = True

        cent_om   = CentralizedOpponentModel().to(device)
        cent_role = CentralizedRoleAssigner().to(device)
        if "cent_om"   in ckpt: cent_om.load_state_dict(ckpt["cent_om"])
        if "cent_role" in ckpt: cent_role.load_state_dict(ckpt["cent_role"], strict=False)
        cent_om.eval(); cent_role.eval()
        enemy_hist = EnemyHistoryBuffer()
        gat_comm   = None

    elif model_type == "mappo":
        from training.train_mappo import GATMAPPOActor, MAPPOActor
        from models.gat_comm import GATComm
        hidden     = int(config["training"].get("hidden_dim", 256))
        action_dim = env.action_dim

        actor_sd = ckpt.get("actor", {})
        if "fc1_intent.weight" not in actor_sd:
            raise ValueError(
                "Bu script yalnızca GAT+OM MAPPO checkpoint'lerini (78D) destekler. "
                "fc1_intent.weight bulunamadı."
            )

        obs_dim_eval = 78
        actor = GATMAPPOActor(
            old_obs_dim=50, ext_dim=18, intent_dim=6, role_dim=4,
            action_dim=action_dim, hidden=hidden,
        ).to(device)
        actor.load_state_dict(actor_sd, strict=False)
        use_rule_fire = False

        comm_cfg = config.get("communication", {})
        gat_node_dim  = int(comm_cfg.get("node_dim", 17))
        gat_wez_range = float(config.get("weapons", {}).get("wez_range_max", 8000.0))

        gat_comm = GATComm().to(device)
        if "gat_comm" in ckpt:
            gat_comm.load_state_dict(ckpt["gat_comm"])
        gat_comm.eval()

        cent_om   = CentralizedOpponentModel().to(device)
        role_sd   = ckpt.get("cent_role", {})
        cent_role = (_LegacySequentialRoleAssigner() if "mlp0.0.weight" in role_sd
                     else CentralizedRoleAssigner()).to(device)
        cent_role.load_state_dict(role_sd, strict=False)
        if "cent_om" in ckpt: cent_om.load_state_dict(ckpt["cent_om"])
        cent_om.eval(); cent_role.eval()
        enemy_hist = EnemyHistoryBuffer()
    else:
        raise ValueError(f"Bilinmeyen model_type: {model_type}")

    actor.eval()
    opp = MultiHeuristicPolicy(config, all_ids, team_map)
    env.seed(seed)

    # ── Sayaçlar ──────────────────────────────────────────────────────────
    # Per-agent: kills, eng_dists (list of floats), wez_steps, fires
    n_agents = len(blue_ids)
    ag_kills  = {aid: [] for aid in blue_ids}  # kills per episode
    ag_eng    = {aid: [] for aid in blue_ids}  # mean eng dist per step (per episode)
    ag_wez    = {aid: [] for aid in blue_ids}  # wez_steps per episode
    ag_fire   = {aid: [] for aid in blue_ids}  # fire_count per episode

    role0_buf = []
    role1_buf = []
    wins = losses = draws = 0

    for ep in range(n_episodes):
        obs_dict = env.reset()
        opp.reset()
        if enemy_hist is not None:
            enemy_hist.reset()

        ep_kills  = {aid: 0   for aid in blue_ids}
        ep_eng    = {aid: []  for aid in blue_ids}
        ep_wez    = {aid: 0   for aid in blue_ids}
        ep_fire   = {aid: 0   for aid in blue_ids}
        done      = {"__all__": False}

        while not done["__all__"]:
            states       = env.get_all_states()
            red_states   = [states.get(r) for r in red_ids]

            # ── Build obs ─────────────────────────────────────────────────
            if model_type == "facmac":
                base_arr = np.stack([
                    obs_dict.get(aid, np.zeros(_BASE_OBS_DIM, np.float32))[:_BASE_OBS_DIM]
                    for aid in blue_ids
                ], axis=0)
                hist_960 = enemy_hist.update(base_arr)
                with torch.no_grad():
                    hist_t    = torch.from_numpy(hist_960).unsqueeze(0).to(device)
                    intent_np = cent_om.intent_flat(hist_t).squeeze(0).cpu().numpy()
                ts_np = build_team_state([states.get(bid) for bid in blue_ids])
                if zero_role:
                    roles = [np.zeros(4, np.float32), np.zeros(4, np.float32)]
                else:
                    with torch.no_grad():
                        x_role = torch.from_numpy(
                            np.concatenate([intent_np, ts_np]).astype(np.float32)
                        ).unsqueeze(0).to(device)
                        r0, r1 = cent_role.assign(x_role)
                    roles = [r0.squeeze(0).cpu().numpy(), r1.squeeze(0).cpu().numpy()]
                obs_for_actor = {
                    aid: np.concatenate([
                        obs_dict.get(aid, np.zeros(_BASE_OBS_DIM, np.float32))[:_BASE_OBS_DIM],
                        intent_np, roles[i]
                    ])
                    for i, aid in enumerate(blue_ids)
                }
                if ep == 0:
                    role0_buf.append(roles[0].copy())
                    role1_buf.append(roles[1].copy())

            elif model_type == "mappo":
                # GAT mesajları
                ego_list   = []
                alive_list = []
                for aid in blue_ids:
                    o_base = obs_dict.get(aid, np.zeros(_BASE_OBS_DIM, np.float32))
                    ego_list.append(o_base[:gat_node_dim])
                    s = states.get(aid)
                    alive_list.append(float(s[STATE_ALIVE]) if s is not None else 0.0)
                edge_feats = _build_gat_edge_feats(
                    blue_ids, red_ids, states, gat_wez_range, env.map_size
                )
                messages = gat_comm.compute_messages(ego_list, edge_feats, alive_list, device)

                if cent_om is not None:
                    base_arr = np.stack([
                        obs_dict.get(aid, np.zeros(_BASE_OBS_DIM, np.float32))[:_BASE_OBS_DIM]
                        for aid in blue_ids
                    ], axis=0)
                    hist_960 = enemy_hist.update(base_arr)
                    with torch.no_grad():
                        hist_t    = torch.from_numpy(hist_960).unsqueeze(0).to(device)
                        intent_np = cent_om.intent_flat(hist_t).squeeze(0).cpu().numpy()
                    ts_np = build_team_state([states.get(bid) for bid in blue_ids])
                    if zero_role:
                        roles = [np.zeros(4, np.float32)] * n_agents
                    else:
                        with torch.no_grad():
                            x_role = torch.from_numpy(
                                np.concatenate([intent_np, ts_np]).astype(np.float32)
                            ).unsqueeze(0).to(device)
                            r0, r1 = cent_role.assign(x_role)
                        roles = [r0.squeeze(0).cpu().numpy(), r1.squeeze(0).cpu().numpy()]
                    if ep == 0:
                        role0_buf.append(roles[0].copy())
                        role1_buf.append(roles[1].copy())
                else:
                    intent_np = np.zeros(6, np.float32)
                    roles     = [np.zeros(4, np.float32)] * n_agents

                obs_for_actor = {}
                for i, aid in enumerate(blue_ids):
                    base = obs_dict.get(aid, np.zeros(_BASE_OBS_DIM, np.float32))[:_BASE_OBS_DIM]
                    ext18 = np.concatenate([np.zeros(2, np.float32), messages[i]])
                    obs_for_actor[aid] = np.concatenate([base, ext18, intent_np, roles[i]])

            # ── Actor adımı ───────────────────────────────────────────────
            with torch.no_grad():
                if model_type == "facmac":
                    ctrl_actions = {}
                    for aid in blue_ids:
                        o   = obs_for_actor[aid][:obs_dim_eval]
                        t   = torch.FloatTensor(o).unsqueeze(0).to(device)
                        c, _ = actor.act(t, deterministic=True)
                        ctrl_actions[aid] = c.squeeze(0).cpu().numpy()
                    fire_dict = _rule_fire(blue_ids, red_ids, obs_dict, states)
                    blue_actions = {
                        aid: np.append(ctrl_actions[aid], fire_dict[aid]) for aid in blue_ids
                    }
                else:
                    from training.train_mappo import MAPPOActor
                    blue_actions = {}
                    for aid in blue_ids:
                        o   = obs_for_actor[aid][:obs_dim_eval]
                        t   = torch.FloatTensor(o).unsqueeze(0).to(device)
                        raw, _ = actor.act(t, deterministic=True)
                        sq = MAPPOActor.squash(raw.squeeze(0))
                        blue_actions[aid] = sq.cpu().numpy()

            all_opp     = opp.act(states)
            red_actions = {rid: all_opp[rid] for rid in red_ids if rid in all_opp}
            action_dict = {**blue_actions, **red_actions}
            obs_dict, rew_dict, done, info_dict = env.step(action_dict)

            # ── Per-agent metrikleri topla ────────────────────────────────
            for aid in blue_ids:
                s = states.get(aid)
                if s is None or s[STATE_ALIVE] < 0.5:
                    continue
                if info_dict[aid].get("r_kill", 0.0) > 0.5:
                    ep_kills[aid] += 1
                d = _nearest_enemy_dist(s, red_states)
                if not np.isnan(d):
                    ep_eng[aid].append(d)
                if _in_wez(s, red_states):
                    ep_wez[aid] += 1
                if model_type == "facmac":
                    ep_fire[aid] += int(fire_dict.get(aid, 0) > 0.5)
                else:
                    ep_fire[aid] += int(float(blue_actions[aid][4]) > 0.5)

        winner = done.get("winner", "draw")
        if   winner == BLUE: wins   += 1
        elif winner == RED:  losses += 1
        else:                draws  += 1

        for aid in blue_ids:
            ag_kills[aid].append(ep_kills[aid])
            ag_eng[aid].append(float(np.mean(ep_eng[aid])) if ep_eng[aid] else 0.0)
            ag_wez[aid].append(ep_wez[aid])
            ag_fire[aid].append(ep_fire[aid])

        if (ep + 1) % 50 == 0:
            print(f"  ep {ep+1:4d}/{n_episodes} | W={wins/(ep+1):.3f}", flush=True)

    # ── Sonuçları derle ───────────────────────────────────────────────────
    result = {
        "condition":  label,
        "model_type": model_type,
        "n_episodes": n_episodes,
        "win_rate":   round(wins / n_episodes, 4),
        "draw_rate":  round(draws / n_episodes, 4),
        "agents":     {},
    }
    all_kill_means = []
    all_eng_means  = []
    all_wez_means  = []

    for aid in blue_ids:
        km = float(np.mean(ag_kills[aid]))
        em = float(np.mean(ag_eng[aid]))
        wm = float(np.mean(ag_wez[aid]))
        fm = float(np.mean(ag_fire[aid]))
        result["agents"][aid] = {
            "kill_mean":  round(km, 4),
            "kill_std":   round(float(np.std(ag_kills[aid])), 4),
            "eng_mean":   round(em, 1),
            "wez_mean":   round(wm, 2),
            "fire_mean":  round(fm, 2),
        }
        all_kill_means.append(km)
        all_eng_means.append(em)
        all_wez_means.append(wm)

    # Fark metrikleri
    result["kill_imbalance"] = round(
        float(abs(all_kill_means[0] - all_kill_means[1])), 4
    ) if len(all_kill_means) == 2 else None
    result["eng_imbalance"] = round(
        float(abs(all_eng_means[0] - all_eng_means[1])), 1
    ) if len(all_eng_means) == 2 else None
    result["wez_imbalance"] = round(
        float(abs(all_wez_means[0] - all_wez_means[1])), 2
    ) if len(all_wez_means) == 2 else None

    # kill korelasyonu (negatif → ayrışma, pozitif → simetri)
    k0 = ag_kills[blue_ids[0]]
    k1 = ag_kills[blue_ids[1]]
    if len(k0) > 1 and np.std(k0) > 1e-9 and np.std(k1) > 1e-9:
        result["kill_corr"] = round(float(np.corrcoef(k0, k1)[0, 1]), 4)
    else:
        result["kill_corr"] = None

    # Rol dağılımı özeti (ep=0 adımlarından)
    if role0_buf:
        result["role0_mean"] = np.mean(role0_buf, axis=0).round(4).tolist()
    if role1_buf and len(blue_ids) > 1:
        result["role1_mean"] = np.mean(role1_buf, axis=0).round(4).tolist()

    return result


# ---------------------------------------------------------------------------
# GAT kenar özellikleri
# ---------------------------------------------------------------------------
def _build_gat_edge_feats(blue_ids, red_ids, states, wez_range, map_size):
    from envs.geometry_utils import distance_3d, bearing_angle, wrap_to_pi
    N    = len(blue_ids)
    edge = np.zeros((N, N, 3), np.float32)
    for i, ai in enumerate(blue_ids):
        for j, aj in enumerate(blue_ids):
            if i == j: continue
            si = states.get(ai); sj = states.get(aj)
            if si is None or sj is None: continue
            pi = si[[STATE_X, STATE_Y, STATE_H]]
            pj = sj[[STATE_X, STATE_Y, STATE_H]]
            d  = distance_3d(pi, pj)
            b  = bearing_angle(pi, pj)
            ts = 0.0
            for eid in red_ids:
                es = states.get(eid)
                if es is not None and es[STATE_ALIVE] > 0.5:
                    de = distance_3d(pj, es[[STATE_X, STATE_Y, STATE_H]])
                    ts = max(ts, float(np.clip(1.0 - de / (wez_range + 1e-9), 0.0, 1.0)))
            edge[i, j] = [
                float(np.clip(d / (map_size + 1e-9), 0.0, 1.0)),
                float(np.clip(wrap_to_pi(b) / np.pi, -1.0, 1.0)),
                ts,
            ]
    return edge


# ---------------------------------------------------------------------------
# Rule-based fire (FACMAC için)
# ---------------------------------------------------------------------------
def _rule_fire(blue_ids, red_ids, obs_dict, states):
    fire = {}
    for aid in blue_ids:
        obs = obs_dict.get(aid, np.zeros(_BASE_OBS_DIM, np.float32))
        fire[aid] = 0.0
        if obs[16] > 1e-4: continue
        b = states.get(aid)
        if b is None or b[STATE_ALIVE] < 0.5 or b[STATE_AMMO] < 0.5: continue
        b_pos = b[[STATE_X, STATE_Y, STATE_H]]
        for rid in red_ids:
            r = states.get(rid)
            if r is None or r[STATE_ALIVE] < 0.5: continue
            r_pos = r[[STATE_X, STATE_Y, STATE_H]]
            d = distance_3d(b_pos, r_pos)
            if _WEZ_RANGE_MIN <= d <= _WEZ_RANGE_MAX:
                ata = antenna_train_angle(b_pos, r_pos, b[STATE_PSI])
                if abs(ata) <= _WEZ_ANGLE_MAX:
                    fire[aid] = 1.0
                    break
    return fire


# ---------------------------------------------------------------------------
# Sonuç yazdır
# ---------------------------------------------------------------------------
def print_comparison(r_normal: dict, r_zeroed: dict):
    mt = r_normal["model_type"]
    n  = r_normal["n_episodes"]
    print(f"\n{'='*62}")
    print(f"  YAPISAL HİPOTEZ TESTİ  — {mt.upper()}  ({n} ep)")
    print(f"{'='*62}")
    header = f"  {'Metrik':<28} {'NORMAL':>10} {'ZEROED':>10}"
    print(header)
    print(f"  {'-'*58}")
    print(f"  {'Win rate':<28} {r_normal['win_rate']:>10.3f} {r_zeroed['win_rate']:>10.3f}")
    print()

    for cond, r in [("NORMAL", r_normal), ("ZEROED", r_zeroed)]:
        print(f"  [{cond}]")
        for aid, ag in r["agents"].items():
            print(f"    {aid}: kill={ag['kill_mean']:.3f}±{ag['kill_std']:.2f}  "
                  f"eng={ag['eng_mean']:.0f}m  wez={ag['wez_mean']:.1f}st  "
                  f"fire={ag['fire_mean']:.1f}")
        print(f"    kill_imbalance = {r['kill_imbalance']}  "
              f"  eng_imbalance = {r['eng_imbalance']}m  "
              f"  wez_imbalance = {r['wez_imbalance']}")
        print(f"    kill_corr      = {r['kill_corr']}")
        if "role0_mean" in r:
            rn = ["snp","prs","def","sup"]
            r0s = " ".join(f"{rn[k]}={v:.3f}" for k, v in enumerate(r["role0_mean"]))
            r1s = " ".join(f"{rn[k]}={v:.3f}" for k, v in enumerate(r.get("role1_mean", [0]*4)))
            print(f"    role0_mean: {r0s}")
            print(f"    role1_mean: {r1s}")
        print()

    # Yorumlama
    print(f"  {'─'*58}")
    print("  YORUM:")
    ki_n = r_normal.get("kill_imbalance") or 0.0
    ki_z = r_zeroed.get("kill_imbalance") or 0.0
    threshold = 0.10   # 0.10 kill/ep fark → anlamlı asimetri

    if ki_z >= threshold and ki_n >= threshold:
        print("  FACMAC hipotezi DESTEKLENİYOR: rol sıfırlandığında asimetri koruyor.")
        print("  → Rol, var olan farklılaşmayı etiketliyor (değil yaratıyor).")
    elif ki_z < threshold and ki_n >= threshold:
        print("  Rol DESTEKLENİYOR: normal modda asimetri var, zeroed'de kayboluyor.")
        print("  → Rol, davranışı aktif olarak farklılaştırıyor.")
    elif ki_z < threshold and ki_n < threshold:
        print("  Her iki koşulda da asimetri yok → ajan davranışı simetrik.")
        print("  → Rol etkisiz; MAPPO için beklenen sonuç.")
    else:
        print("  Belirsiz: zeroed'de asimetri var ama normal'de yok (beklenmedik).")
    print(f"{'='*62}\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_type", choices=["facmac", "mappo"])
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--config",     default="configs/config.yaml")
    p.add_argument("--episodes",   type=int, default=200)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--output",     default=None)
    args = p.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.checkpoint is None:
        if args.model_type == "facmac":
            ckpt_path = "checkpoints/facmac_omr_ep27000.pt"
        else:
            ckpt_path = "checkpoints/mappo_gat_ep44000.pt"
    else:
        ckpt_path = args.checkpoint

    print(f"[Eval] Checkpoint : {ckpt_path}")
    print(f"[Eval] Device     : {device}")
    print(f"[Eval] Episodes   : {args.episodes} per condition | seed={args.seed}")

    r_normal = run_condition(
        args.model_type, ckpt_path, config,
        args.episodes, args.seed, zero_role=False, device=device,
    )
    r_zeroed = run_condition(
        args.model_type, ckpt_path, config,
        args.episodes, args.seed, zero_role=True, device=device,
    )

    print_comparison(r_normal, r_zeroed)

    if args.output:
        out = {"normal": r_normal, "zeroed": r_zeroed}
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"[Eval] Kaydedildi: {args.output}")


if __name__ == "__main__":
    main()

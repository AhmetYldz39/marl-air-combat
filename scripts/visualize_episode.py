"""
visualize_episode.py
====================
Egitilmis MAPPO politikasi ile 2v2 dogfight episode'lari calistirir,
3D veya top-down 2D animasyon olarak MP4 kaydeder.

Kullanim:
    python -X utf8 scripts/visualize_episode.py
    python -X utf8 scripts/visualize_episode.py --checkpoint checkpoints/mappo_gat_final.pt --phase2
    python -X utf8 scripts/visualize_episode.py --view topdown --mode win
    python -X utf8 scripts/visualize_episode.py --all-outcomes --view topdown
    python -X utf8 scripts/visualize_episode.py --view chase --output logs/chase.mp4

Kamera modlari (--view):
    3d         — varsayilan, yavasca donen 3D bakis
    topdown    — tepeden 2D taktik harita (en net dinamik gosterimi)
    side       — 3D yan profil (irtifa degisimini gormek icin)
    chase      — blue_0'i takip eden 3D kamera (arkadan bakar)
    cinematic  — 360 derece donen 3D film cekimi

Cikti:
    MP4 video (varsayilan: logs/dogfight_{view}_{mode}.mp4)
    --all-outcomes ile: logs/dogfight_{view}_win.mp4, _loss.mp4, _draw.mp4
"""

import sys
import argparse
import numpy as np
import yaml
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import imageio.v2 as imageio

from envs.dogfight_env import DogfightEnv, BLUE, RED
from envs.aircraft_model import (
    STATE_X, STATE_Y, STATE_H, STATE_PSI,
    STATE_HP, STATE_ALIVE, STATE_V, STATE_AMMO,
)
from envs.geometry_utils import distance_3d, bearing_angle, wrap_to_pi
from agents.heuristic_agent import MultiHeuristicPolicy
from utils.normalization import Normalizer
from training.train_mappo import MAPPOActor

# QMIX — opsiyonel (--qmix flag ile etkinleşir)
try:
    from models.qmix_net import AgentQNetwork, ActionMapper as QMIXActionMapper
    _QMIX_AVAILABLE = True
except ImportError:
    _QMIX_AVAILABLE = False

# Başlık etiketi — main() tarafından set edilir
ALGO_LABEL = "MAPPO"

# ---------------------------------------------------------------------------
# Renk ve stil sabitleri
# ---------------------------------------------------------------------------
BG_COLOR   = "#0d1117"
GRID_COLOR = "#1e2530"
TEXT_COLOR = "#e0e0e0"
DIM_COLOR  = "#555555"

BLUE_ALIVE  = "#3af0ff"
BLUE_DEAD   = "#1a5060"
BLUE_TRAIL  = "#1a8090"
RED_ALIVE   = "#ff4444"
RED_DEAD    = "#601414"
RED_TRAIL   = "#902020"

FIRE_COLOR  = "#ffaa00"
KILL_COLOR  = "#ff6600"

TRAIL_LEN        = 80
WEZ_RANGE        = 8000.0
WEZ_HALF_ANG     = np.deg2rad(30.0)
FIRE_FLASH_FRAMES = 8
KILL_FLASH_FRAMES = 15

FIG_W, FIG_H = 16, 9
DPI          = 80


# ===========================================================================
# GAT Obs Extension (compare_eval.py ile aynı mantık)
# ===========================================================================

def load_gat_comm(ckpt_path: str, node_dim: int, edge_dim: int,
                  n_heads: int, msg_dim: int, device: torch.device):
    from models.gat_comm import GATComm
    comm = GATComm(node_dim=node_dim, edge_dim=edge_dim,
                   n_heads=n_heads, msg_dim=msg_dim).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    comm.load_state_dict(ckpt["gat_comm"])
    comm.eval()
    return comm


def _build_gat_edge_feats(env: DogfightEnv, blue_ids: list,
                           wez_range: float) -> np.ndarray:
    N      = len(blue_ids)
    edge   = np.zeros((N, N, 3), dtype=np.float32)
    states = env.get_all_states()
    for i, aid_i in enumerate(blue_ids):
        for j, aid_j in enumerate(blue_ids):
            if i == j:
                continue
            s_i = states.get(aid_i)
            s_j = states.get(aid_j)
            if s_i is None or s_j is None:
                continue
            pos_i = s_i[[STATE_X, STATE_Y, STATE_H]]
            pos_j = s_j[[STATE_X, STATE_Y, STATE_H]]
            dist  = distance_3d(pos_i, pos_j)
            bear  = bearing_angle(pos_i, pos_j)
            ts    = 0.0
            for eid in env.red_ids:
                es = states.get(eid)
                if es is not None and es[STATE_ALIVE] > 0.5:
                    d  = distance_3d(pos_j, es[[STATE_X, STATE_Y, STATE_H]])
                    ts = max(ts, float(np.clip(1.0 - d / (wez_range + 1e-9), 0.0, 1.0)))
            edge[i, j] = [
                float(np.clip(dist / (env.map_size + 1e-9), 0.0, 1.0)),
                float(np.clip(wrap_to_pi(bear) / np.pi, -1.0, 1.0)),
                ts,
            ]
    return edge


def _extend_obs_gat(obs_dict: dict, env: DogfightEnv,
                    gat_comm, base_obs_dim: int,
                    node_dim: int, wez_range: float,
                    device: torch.device) -> dict:
    """50D obs'u 68D'ye uzat: [base50 | role2 | msg16]."""
    blue_ids   = sorted([a for a in env.agent_ids if "blue" in a])
    states     = env.get_all_states()
    ego_list   = []
    alive_list = []
    for aid in blue_ids:
        obs = obs_dict.get(aid, np.zeros(base_obs_dim, dtype=np.float32))
        ego_list.append(obs[:node_dim])
        s = states.get(aid)
        alive_list.append(float(s[STATE_ALIVE]) if s is not None else 0.0)
    edge_feats = _build_gat_edge_feats(env, blue_ids, wez_range)
    messages   = gat_comm.compute_messages(ego_list, edge_feats, alive_list, device)
    role       = np.array([0.5, 0.5], dtype=np.float32)
    extended   = {}
    for i, aid in enumerate(blue_ids):
        base = obs_dict.get(aid, np.zeros(base_obs_dim, dtype=np.float32))
        extended[aid] = np.concatenate([base, role, messages[i]], axis=0)
    return extended


# ===========================================================================
# Episode Toplama
# ===========================================================================

class QMIXPolicy:
    """AgentQNetwork'ü MAPPOActor.act() arayüzüyle sarmalar."""

    def __init__(self, agent_net, action_mapper, device):
        self.agent_net     = agent_net
        self.action_mapper = action_mapper
        self.device        = device

    def act(self, obs_t: torch.Tensor, deterministic: bool = True):
        with torch.no_grad():
            q_vals = self.agent_net(obs_t)          # (1, 162)
        idx    = int(q_vals.argmax(dim=-1).item())
        cont   = self.action_mapper(idx)            # (5,) float32
        return torch.FloatTensor(cont).unsqueeze(0).to(self.device), None


def load_qmix_policy(checkpoint_path: str, device: torch.device,
                     obs_dim: int = 50, n_actions: int = 162,
                     agent_hidden: int = 128) -> "QMIXPolicy":
    assert _QMIX_AVAILABLE, "models/qmix_net.py bulunamadı"
    ckpt       = torch.load(checkpoint_path, map_location=device, weights_only=False)
    agent_net  = AgentQNetwork(obs_dim, n_actions, agent_hidden).to(device)
    agent_net.load_state_dict(ckpt["agent_net"])
    agent_net.eval()
    mapper = QMIXActionMapper()
    print(f"[Viz] QMIXPolicy yüklendi — ep={ckpt.get('episode','?')}, "
          f"step={ckpt.get('total_steps','?')}")
    return QMIXPolicy(agent_net, mapper, device)


def load_actor(checkpoint_path: str, device: torch.device,
               obs_dim: int, action_dim: int, hidden: int,
               phase2: bool = False):
    """Actor (ve gerekirse GATComm) yukle. (actor, gat_comm_or_None) dondurur."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if phase2:
        try:
            from training.train_mappo import GATMAPPOActor
            actor = GATMAPPOActor(old_obs_dim=50, new_obs_dim=68,
                                   action_dim=action_dim, hidden=hidden).to(device)
            actor.load_state_dict(ckpt["actor"])
            actor.eval()
            print(f"[Viz] GATMAPPOActor yuklendi: {checkpoint_path}")
            # GATComm de ayni checkpoint'ten yukle
            gat_comm = load_gat_comm(checkpoint_path,
                                     node_dim=17, edge_dim=3,
                                     n_heads=4, msg_dim=16, device=device)
            print(f"[Viz] GATComm yuklendi.")
            return actor, gat_comm
        except Exception as e:
            print(f"[Viz] GATMAPPOActor yuklenemedi ({e}), MAPPOActor deneniyor...")

    actor = MAPPOActor(obs_dim, action_dim, hidden=hidden).to(device)
    if "actor" in ckpt:
        actor.load_state_dict(ckpt["actor"])
        print(f"[Viz] MAPPOActor yuklendi — ep={ckpt.get('episode','?')}, "
              f"step={ckpt.get('global_step','?')}")
    else:
        actor.load_state_dict(ckpt)
        print("[Viz] MAPPOActor yuklendi (raw state_dict)")
    actor.eval()
    return actor, None


def get_blue_actions(obs_dict: dict, blue_ids: list, actor,
                     device: torch.device, deterministic: bool = True,
                     gat_comm=None, env=None, wez_range: float = 8000.0) -> dict:
    # phase2 ise obs'u 68D'ye uzat
    if gat_comm is not None and env is not None:
        obs_dict = _extend_obs_gat(obs_dict, env, gat_comm,
                                   base_obs_dim=50, node_dim=17,
                                   wez_range=wez_range, device=device)
    actions = {}
    with torch.no_grad():
        for aid in blue_ids:
            obs_t    = torch.FloatTensor(obs_dict[aid]).unsqueeze(0).to(device)
            raw, _   = actor.act(obs_t, deterministic=deterministic)
            squashed = MAPPOActor.squash(raw.squeeze(0))
            actions[aid] = squashed.cpu().numpy()
    return actions


def _nearest_alive_pos(state_dict: dict, ids: list):
    alive = [state_dict[a] for a in ids if state_dict[a][STATE_ALIVE] > 0.5]
    if not alive:
        return None
    s = alive[0]
    return np.array([s[STATE_X], s[STATE_Y], s[STATE_H]])


def collect_episode(env: DogfightEnv, actor, opp_policy: MultiHeuristicPolicy,
                    device: torch.device, deterministic: bool = True,
                    seed: int = 0, gat_comm=None,
                    wez_range: float = 8000.0) -> dict:
    env.seed(seed)
    obs_dict = env.reset()
    opp_policy.reset()

    traj        = {aid: [] for aid in env.agent_ids}
    fire_events = []
    kill_events = []
    prev_alive  = {aid: 1.0 for aid in env.agent_ids}

    done = {"__all__": False}

    while not done["__all__"]:
        state_dict = env.get_all_states()

        for aid in env.agent_ids:
            traj[aid].append(state_dict[aid].copy())

        blue_actions = get_blue_actions(obs_dict, env.blue_ids, actor,
                                        device, deterministic,
                                        gat_comm=gat_comm, env=env,
                                        wez_range=wez_range)
        all_opp_acts = opp_policy.act(state_dict)
        red_actions  = {rid: all_opp_acts[rid]
                        for rid in env.red_ids if rid in all_opp_acts}

        step = len(traj[env.agent_ids[0]]) - 1
        for aid in env.blue_ids:
            if blue_actions[aid][4] > 0.5:
                opp_pos = _nearest_alive_pos(state_dict, env.red_ids)
                fire_events.append((step, aid, state_dict[aid][STATE_PSI], opp_pos))
        for aid in env.red_ids:
            if red_actions[aid][4] > 0.5:
                opp_pos = _nearest_alive_pos(state_dict, env.blue_ids)
                fire_events.append((step, aid, state_dict[aid][STATE_PSI], opp_pos))

        action_dict = {**blue_actions, **red_actions}
        obs_dict, _, done, _ = env.step(action_dict)

        new_state = env.get_all_states()
        for aid in env.agent_ids:
            na = new_state[aid][STATE_ALIVE]
            if prev_alive[aid] > 0.5 and na < 0.5:
                kill_events.append((step + 1, aid))
            prev_alive[aid] = na

    state_dict = env.get_all_states()
    for aid in env.agent_ids:
        traj[aid].append(state_dict[aid].copy())

    winner  = done.get("winner", "draw")
    n_steps = len(traj[env.agent_ids[0]])
    print(f"[Viz] Episode: {n_steps} adim | kazanan={winner} | "
          f"kills={len(kill_events)} | fires={len(fire_events)}")
    return dict(traj=traj, fire_events=fire_events,
                kill_events=kill_events, winner=winner, n_steps=n_steps)


def find_good_episode(env, actor, opp_policy, device,
                      n_tries: int = 15, seed_offset: int = 0,
                      mode: str = "win", gat_comm=None,
                      wez_range: float = 8000.0,
                      min_steps: int = 0) -> dict:
    """
    Belirli outcome tipini arar. min_steps ile kısa episodları filtreler.
    mode: 'win' | 'loss' | 'draw'
    Eşit outcome'lar arasında her zaman daha uzun (daha fazla adım) olanı tercih eder.
    """
    target_map = {"win": BLUE, "loss": RED, "draw": "draw"}
    target     = target_map.get(mode, BLUE)

    candidates = []   # (ep, n_steps) — hedef outcome'a uyanlar
    best_fallback = None

    for i in range(n_tries):
        ep = collect_episode(env, actor, opp_policy, device,
                             deterministic=False, seed=seed_offset + i,
                             gat_comm=gat_comm, wez_range=wez_range)
        if ep["winner"] == target:
            if ep["n_steps"] >= min_steps:
                candidates.append(ep)
            else:
                print(f"[Viz] seed={seed_offset+i} hedef '{mode}' ama çok kısa "
                      f"({ep['n_steps']}<{min_steps}), atlanıyor.")

        # Fallback: hiç uygun bulunamazsa en uzun döner
        if best_fallback is None or ep["n_steps"] > best_fallback["n_steps"]:
            best_fallback = ep

    if candidates:
        # En uzun olanı seç
        best = max(candidates, key=lambda e: e["n_steps"])
        print(f"[Viz] '{mode}' için {len(candidates)} aday — "
              f"en uzun: steps={best['n_steps']}, kills={len(best['kill_events'])}")
        return best

    print(f"[Viz] {n_tries} denemede '{mode}' (min_steps={min_steps}) bulunamadı — "
          f"fallback kullanılıyor (winner={best_fallback['winner']}, "
          f"steps={best_fallback['n_steps']})")
    return best_fallback


# ===========================================================================
# Kamera & Geometri Yardimcilari
# ===========================================================================

def heading_vec(psi: float):
    """ENU: psi=0 → Kuzey. East=sin(psi), North=cos(psi)."""
    return np.array([np.sin(psi), np.cos(psi), 0.0])


def wez_cone_lines_3d(pos: np.ndarray, psi: float,
                      n_lines: int = 8, range_m: float = 4000.0):
    """3D WEZ koni icin cizgi listesi uret."""
    x0, y0, h0 = pos
    lines = []
    for i in range(n_lines + 1):
        ang = -WEZ_HALF_ANG + i * (2 * WEZ_HALF_ANG / n_lines)
        dx  = np.sin(psi + ang) * range_m
        dy  = np.cos(psi + ang) * range_m
        lines.append(([x0, x0 + dx], [y0, y0 + dy], [h0, h0]))
    n_arc = 16
    arc_x, arc_y, arc_h = [], [], []
    for i in range(n_arc + 1):
        ang = -WEZ_HALF_ANG + i * (2 * WEZ_HALF_ANG / n_arc)
        arc_x.append(x0 + np.sin(psi + ang) * range_m)
        arc_y.append(y0 + np.cos(psi + ang) * range_m)
        arc_h.append(h0)
    lines.append((arc_x, arc_y, arc_h))
    return lines


def get_3d_camera(view: str, frame_idx: int, total_frames: int,
                  data: dict, step: int) -> tuple:
    """(elev, azim) dondur."""
    progress = frame_idx / max(total_frames - 1, 1)
    if view == "3d":
        return 28, -50 + progress * 40
    elif view == "side":
        return 8, -90
    elif view == "cinematic":
        return 22, -180 + progress * 360
    elif view == "chase":
        blue_ids = sorted([a for a in data["traj"] if "blue" in a])
        if blue_ids:
            states = data["traj"][blue_ids[0]]
            s_idx  = min(step, len(states) - 1)
            if states[s_idx][STATE_ALIVE] > 0.5:
                psi  = states[s_idx][STATE_PSI]
                # Kamera arkasindan baksin: -90 - degrees(psi)
                azim = -90 - np.degrees(psi)
                return 15, float(azim)
        return 28, -50 + progress * 40
    return 28, -50 + progress * 40


# ===========================================================================
# Durum Paneli (sagda, tum viewlarda ortaktir)
# ===========================================================================

def draw_info_panel(ax_info, data: dict, step: int):
    """Sag bilgi panelini cizer — hem 2D hem 3D modlarinda kullanilir."""
    ax_info.cla()
    ax_info.set_facecolor(BG_COLOR)
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)
    ax_info.axis("off")

    traj        = data["traj"]
    kill_events = data["kill_events"]
    n_steps     = data["n_steps"]
    winner      = data["winner"]
    agent_ids   = list(traj.keys())

    t_elapsed = step * 0.05
    ax_info.text(0.5, 0.97, f"T = {t_elapsed:5.1f} s",
                 color=TEXT_COLOR, fontsize=11, ha="center", va="top",
                 fontweight="bold")
    ax_info.text(0.5, 0.90, f"Step {step} / {n_steps - 1}",
                 color=DIM_COLOR, fontsize=8, ha="center", va="top")

    ax_info.text(0.5, 0.82, f"BLUE  ({ALGO_LABEL})",
                 color=BLUE_ALIVE, fontsize=9, ha="center", va="top",
                 fontweight="bold")
    ax_info.text(0.5, 0.44, "RED  (Heuristic)",
                 color=RED_ALIVE, fontsize=9, ha="center", va="top",
                 fontweight="bold")

    def draw_agent_status(aids, y_start, col_live, col_dead):
        for i, aid in enumerate(aids):
            states = traj[aid]
            if step < len(states):
                s      = states[step]
                alive  = s[STATE_ALIVE] > 0.5
                hp     = s[STATE_HP]
                ammo   = int(s[STATE_AMMO])
                col    = col_live if alive else col_dead
                name   = aid.replace("_", " ").title()
                status = "ALIVE" if alive else "DEAD"
                y      = y_start - i * 0.16

                ax_info.text(0.1, y, name,
                             color=col, fontsize=8, va="top", fontweight="bold")
                ax_info.text(0.9, y, status,
                             color=col, fontsize=7, va="top", ha="right")
                bar_y = y - 0.05
                bar_w = 0.8
                ax_info.add_patch(plt.Rectangle(
                    (0.1, bar_y - 0.015), bar_w, 0.025,
                    facecolor=GRID_COLOR, transform=ax_info.transAxes, clip_on=False))
                ax_info.add_patch(plt.Rectangle(
                    (0.1, bar_y - 0.015), bar_w * max(0, hp), 0.025,
                    facecolor=col, alpha=0.85,
                    transform=ax_info.transAxes, clip_on=False))
                ax_info.text(0.5, bar_y + 0.003, f"HP {hp:.0%}",
                             color=TEXT_COLOR, fontsize=6.5,
                             ha="center", va="bottom", alpha=0.8)
                ax_info.text(0.9, bar_y - 0.02, f"Ammo:{ammo}",
                             color=DIM_COLOR, fontsize=6, ha="right", va="top")

    draw_agent_status([a for a in agent_ids if "blue" in a], 0.78, BLUE_ALIVE, BLUE_DEAD)
    draw_agent_status([a for a in agent_ids if "red"  in a], 0.40, RED_ALIVE,  RED_DEAD)

    blue_kills = sum(1 for (s_, aid) in kill_events if "red"  in aid and s_ <= step)
    red_kills  = sum(1 for (s_, aid) in kill_events if "blue" in aid and s_ <= step)
    ax_info.text(0.5, 0.12,
                 f"Kills   Blue {blue_kills}  :  {red_kills}  Red",
                 color=TEXT_COLOR, fontsize=9, ha="center", va="top",
                 fontweight="bold")

    prog = step / max(n_steps - 1, 1)
    ax_info.add_patch(plt.Rectangle(
        (0.05, 0.04), 0.90, 0.018,
        facecolor=GRID_COLOR, transform=ax_info.transAxes, clip_on=False))
    ax_info.add_patch(plt.Rectangle(
        (0.05, 0.04), 0.90 * prog, 0.018,
        facecolor=BLUE_TRAIL, alpha=0.8,
        transform=ax_info.transAxes, clip_on=False))

    # Sonuc bilgisi
    if step >= n_steps - 2:
        result_colors = {BLUE: BLUE_ALIVE, RED: RED_ALIVE, "draw": "#aaaaaa"}
        result_texts  = {BLUE: "BLUE WINS", RED: "RED WINS", "draw": "DRAW"}
        rc = result_colors.get(winner, "#aaaaaa")
        rt = result_texts.get(winner, "DRAW")
        ax_info.text(0.5, 0.27, rt,
                     color=rc, fontsize=14, fontweight="bold",
                     ha="center", va="center",
                     bbox=dict(facecolor=BG_COLOR, edgecolor=rc,
                               boxstyle="round,pad=0.3", alpha=0.9))


# ===========================================================================
# 3D Frame Render
# ===========================================================================

def render_frame_3d(fig: plt.Figure, ax3d, ax_info,
                    data: dict, frame_idx: int,
                    total_frames: int, step_skip: int, view: str = "3d"):
    ax3d.cla()

    step        = frame_idx * step_skip
    traj        = data["traj"]
    fire_events = data["fire_events"]
    kill_events = data["kill_events"]
    winner      = data["winner"]
    n_steps     = data["n_steps"]
    agent_ids   = list(traj.keys())

    all_x = np.concatenate([np.array([s[STATE_X] for s in traj[a]]) for a in agent_ids])
    all_y = np.concatenate([np.array([s[STATE_Y] for s in traj[a]]) for a in agent_ids])
    all_h = np.concatenate([np.array([s[STATE_H] for s in traj[a]]) for a in agent_ids])
    pad   = 4000.0
    cx    = (all_x.max() + all_x.min()) / 2
    cy    = (all_y.max() + all_y.min()) / 2
    rng   = max(all_x.max() - all_x.min(), all_y.max() - all_y.min()) / 2 + pad
    xmin, xmax = cx - rng, cx + rng
    ymin, ymax = cy - rng, cy + rng
    hmin  = max(0, all_h.min() - 500)
    hmax  = all_h.max() + 1000

    ax3d.set_facecolor(BG_COLOR)
    for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor(GRID_COLOR)
    ax3d.grid(True, color=GRID_COLOR, linewidth=0.4, alpha=0.6)

    elev, azim = get_3d_camera(view, frame_idx, total_frames, data, step)
    ax3d.view_init(elev=elev, azim=azim)

    ax3d.set_xlim(xmin, xmax)
    ax3d.set_ylim(ymin, ymax)
    ax3d.set_zlim(hmin, hmax)
    ax3d.set_xlabel("East (m)",  color=DIM_COLOR, fontsize=7, labelpad=4)
    ax3d.set_ylabel("North (m)", color=DIM_COLOR, fontsize=7, labelpad=4)
    ax3d.set_zlabel("Alt (m)",   color=DIM_COLOR, fontsize=7, labelpad=4)
    ax3d.tick_params(colors=DIM_COLOR, labelsize=6)
    for spine in ax3d.spines.values():
        spine.set_color(GRID_COLOR)

    # Zemin izgarasi
    for gxi in np.linspace(xmin, xmax, 6):
        ax3d.plot([gxi, gxi], [ymin, ymax], [hmin, hmin],
                  color=GRID_COLOR, lw=0.4, alpha=0.5)
    for gyi in np.linspace(ymin, ymax, 6):
        ax3d.plot([xmin, xmax], [gyi, gyi], [hmin, hmin],
                  color=GRID_COLOR, lw=0.4, alpha=0.5)

    # Ajanlar
    for aid in agent_ids:
        is_blue  = "blue" in aid
        col_live = BLUE_ALIVE if is_blue else RED_ALIVE
        col_dead = BLUE_DEAD  if is_blue else RED_DEAD
        col_trl  = BLUE_TRAIL if is_blue else RED_TRAIL
        label    = aid.replace("_", " ").title()

        states_so_far = traj[aid][:step + 1]
        if not states_so_far:
            continue
        xs = np.array([s[STATE_X] for s in states_so_far])
        ys = np.array([s[STATE_Y] for s in states_so_far])
        hs = np.array([s[STATE_H] for s in states_so_far])
        cur_alive = states_so_far[-1][STATE_ALIVE] > 0.5

        trail_start = max(0, len(xs) - TRAIL_LEN)
        n_seg = len(xs) - trail_start
        for j in range(n_seg - 1):
            alpha = 0.1 + 0.7 * (j / max(n_seg - 1, 1))
            si = trail_start + j
            ax3d.plot(xs[si:si+2], ys[si:si+2], hs[si:si+2],
                      color=col_trl, lw=1.2, alpha=alpha)

        cur_x, cur_y, cur_h = xs[-1], ys[-1], hs[-1]
        col_cur = col_live if cur_alive else col_dead
        ax3d.scatter([cur_x], [cur_y], [cur_h],
                     color=col_cur, s=80 if cur_alive else 30,
                     zorder=10,
                     edgecolors="white" if cur_alive else col_dead,
                     linewidths=0.5 if cur_alive else 0)

        if cur_alive:
            psi   = states_so_far[-1][STATE_PSI]
            arrow = 1400.0
            hv    = heading_vec(psi)
            ax3d.quiver(cur_x, cur_y, cur_h,
                        hv[0] * arrow, hv[1] * arrow, 0,
                        color=col_cur, lw=2.0, arrow_length_ratio=0.35, alpha=0.95)
            ax3d.plot([cur_x, cur_x], [cur_y, cur_y], [hmin, cur_h],
                      color=col_cur, lw=0.5, alpha=0.3, linestyle=":")
            ax3d.text(cur_x, cur_y, cur_h + 400,
                      label, color=col_live, fontsize=7,
                      ha="center", va="bottom", fontweight="bold")

    # Ates efektleri
    last_fire: dict = {}
    for ev in fire_events:
        f_step = ev[0]; f_aid = ev[1]
        if f_step <= step:
            if f_aid not in last_fire or ev[0] > last_fire[f_aid][0]:
                last_fire[f_aid] = ev

    for f_aid, (f_step, _, f_psi, opp_pos) in last_fire.items():
        age = step - f_step
        if 0 <= age < FIRE_FLASH_FRAMES:
            alpha_f = max(0.0, 1.0 - age / FIRE_FLASH_FRAMES)
            states  = traj[f_aid]
            if f_step >= len(states):
                continue
            s = states[f_step]
            if s[STATE_ALIVE] < 0.5:
                continue
            pos          = np.array([s[STATE_X], s[STATE_Y], s[STATE_H]])
            hv           = heading_vec(f_psi)
            is_blue_agent = "blue" in f_aid
            laser_color  = "#00ffff" if is_blue_agent else "#ff2222"

            ax3d.scatter([pos[0]], [pos[1]], [pos[2]],
                         color=laser_color, s=500 * alpha_f,
                         alpha=min(1.0, alpha_f * 1.2), zorder=25,
                         edgecolors="#ffffff", linewidths=1.2)
            end = pos + hv * WEZ_RANGE
            ax3d.plot([pos[0], end[0]], [pos[1], end[1]], [pos[2], end[2]],
                      color=laser_color, lw=5.0, alpha=min(1.0, alpha_f),
                      zorder=24, solid_capstyle="round")
            ax3d.plot([pos[0], end[0]], [pos[1], end[1]], [pos[2], end[2]],
                      color="#ffffff", lw=9.0, alpha=alpha_f * 0.25,
                      zorder=23, solid_capstyle="round")

            if age < 2:
                for xs_c, ys_c, hs_c in wez_cone_lines_3d(pos, f_psi, range_m=WEZ_RANGE):
                    ax3d.plot(xs_c, ys_c, hs_c,
                              color=laser_color, lw=1.5, alpha=alpha_f * 0.6, zorder=22)
            if opp_pos is not None and age < 5:
                ax3d.scatter([opp_pos[0]], [opp_pos[1]], [opp_pos[2]],
                             color="#ffffff", s=400 * alpha_f,
                             alpha=alpha_f, zorder=26,
                             edgecolors=laser_color, linewidths=1.5)

    # Patlama efektleri
    for (k_step, k_aid) in kill_events:
        age = step - k_step
        if 0 <= age < KILL_FLASH_FRAMES:
            states = traj[k_aid]
            s_idx  = min(k_step, len(states) - 1)
            s      = states[s_idx]
            exp_r  = 500 + age * 400
            alpha_k = max(0.0, 0.9 * (1 - age / KILL_FLASH_FRAMES))
            u  = np.linspace(0, 2 * np.pi, 20)
            px = s[STATE_X] + exp_r * np.cos(u)
            py = s[STATE_Y] + exp_r * np.sin(u)
            ph = np.full_like(px, s[STATE_H])
            ax3d.plot(px, py, ph, color=KILL_COLOR, lw=1.5, alpha=alpha_k)
            ax3d.scatter([s[STATE_X]], [s[STATE_Y]], [s[STATE_H]],
                         color=KILL_COLOR, s=300 * (1 - age / KILL_FLASH_FRAMES),
                         alpha=alpha_k, zorder=30)

    # Baslik
    title_y = 0.97
    fig.text(0.38, title_y, "2v2 MARL Air Combat",
             color=TEXT_COLOR, fontsize=14, fontweight="bold",
             ha="center", va="top", transform=fig.transFigure)
    fig.text(0.38, title_y - 0.045,
             f"{ALGO_LABEL}  |  Master's Thesis",
             color=DIM_COLOR, fontsize=8, ha="center", va="top",
             transform=fig.transFigure)

    # Kamera modu etiketi
    cam_labels = {"3d": "3D Orbit", "side": "Side View",
                  "chase": "Chase Cam", "cinematic": "Cinematic"}
    cam_lbl = cam_labels.get(view, view.upper())
    fig.text(0.02, 0.97, f"CAM: {cam_lbl}",
             color=DIM_COLOR, fontsize=7, ha="left", va="top",
             transform=fig.transFigure)

    # Son kare sonuc
    if step >= n_steps - 2:
        result_colors = {BLUE: BLUE_ALIVE, RED: RED_ALIVE, "draw": "#aaaaaa"}
        result_texts  = {BLUE: "BLUE WINS", RED: "RED WINS", "draw": "DRAW"}
        rc = result_colors.get(winner, "#aaaaaa")
        rt = result_texts.get(winner, "DRAW")
        ax3d.text2D(0.5, 0.55, rt,
                    transform=ax3d.transAxes,
                    color=rc, fontsize=26, fontweight="bold",
                    ha="center", va="center", alpha=0.85,
                    bbox=dict(facecolor=BG_COLOR, edgecolor=rc,
                              boxstyle="round,pad=0.4", alpha=0.75))

    draw_info_panel(ax_info, data, step)


# ===========================================================================
# TOP-DOWN 2D Frame Render
# ===========================================================================

def _psi_to_mpl_deg(psi: float) -> float:
    """ENU psi (Kuzeyden CW) → matplotlib dereceye cevir (Dogudan CCW)."""
    return 90.0 - np.degrees(psi)


def render_frame_topdown(fig: plt.Figure, ax2d, ax_info,
                         data: dict, frame_idx: int,
                         total_frames: int, step_skip: int):
    ax2d.cla()

    step        = frame_idx * step_skip
    traj        = data["traj"]
    fire_events = data["fire_events"]
    kill_events = data["kill_events"]
    winner      = data["winner"]
    n_steps     = data["n_steps"]
    agent_ids   = list(traj.keys())

    # Aks sinirlari
    all_x = np.concatenate([np.array([s[STATE_X] for s in traj[a]]) for a in agent_ids])
    all_y = np.concatenate([np.array([s[STATE_Y] for s in traj[a]]) for a in agent_ids])
    pad   = 5000.0
    cx    = (all_x.max() + all_x.min()) / 2
    cy    = (all_y.max() + all_y.min()) / 2
    half  = max(all_x.max() - all_x.min(), all_y.max() - all_y.min()) / 2 + pad
    xmin, xmax = cx - half, cx + half
    ymin, ymax = cy - half, cy + half

    ax2d.set_facecolor(BG_COLOR)
    ax2d.set_xlim(xmin, xmax)
    ax2d.set_ylim(ymin, ymax)
    ax2d.set_aspect("equal")
    ax2d.tick_params(colors=DIM_COLOR, labelsize=6)
    ax2d.set_xlabel("East (m)",  color=DIM_COLOR, fontsize=7)
    ax2d.set_ylabel("North (m)", color=DIM_COLOR, fontsize=7)
    for spine in ax2d.spines.values():
        spine.set_color(GRID_COLOR)

    # Izgara
    for gxi in np.linspace(xmin, xmax, 7):
        ax2d.axvline(gxi, color=GRID_COLOR, lw=0.4, alpha=0.5)
    for gyi in np.linspace(ymin, ymax, 7):
        ax2d.axhline(gyi, color=GRID_COLOR, lw=0.4, alpha=0.5)

    # Kuzey oku (sol ust kosede)
    arrow_x = xmin + (xmax - xmin) * 0.06
    arrow_y = ymin + (ymax - ymin) * 0.88
    arrow_len = half * 0.06
    ax2d.annotate("", xy=(arrow_x, arrow_y + arrow_len),
                  xytext=(arrow_x, arrow_y),
                  arrowprops=dict(arrowstyle="->", color=TEXT_COLOR, lw=1.5))
    ax2d.text(arrow_x, arrow_y + arrow_len * 1.25, "N",
              color=TEXT_COLOR, fontsize=8, ha="center", va="bottom", fontweight="bold")

    # Range halkasi (10km)
    ring_r = 10000.0
    ring_theta = np.linspace(0, 2 * np.pi, 120)
    ax2d.plot(cx + ring_r * np.cos(ring_theta),
              cy + ring_r * np.sin(ring_theta),
              color=GRID_COLOR, lw=0.6, alpha=0.4, linestyle="--")
    ax2d.text(cx + ring_r * 0.72, cy + ring_r * 0.72, "10km",
              color=DIM_COLOR, fontsize=6, alpha=0.6)

    # ── WEZ konileri (hayatta olan her ajan icin) ──────────────────────────
    for aid in agent_ids:
        states_so_far = traj[aid][:step + 1]
        if not states_so_far:
            continue
        s = states_so_far[-1]
        if s[STATE_ALIVE] < 0.5:
            continue
        cx_a, cy_a = s[STATE_X], s[STATE_Y]
        psi        = s[STATE_PSI]
        is_blue    = "blue" in aid
        wez_color  = "#00ccff" if is_blue else "#ff4444"
        mpl_deg    = _psi_to_mpl_deg(psi)
        wedge      = mpatches.Wedge(
            (cx_a, cy_a), WEZ_RANGE,
            mpl_deg - 30.0, mpl_deg + 30.0,
            facecolor=wez_color, alpha=0.06, edgecolor=wez_color,
            linewidth=0.8, linestyle="--",
        )
        ax2d.add_patch(wedge)

    # ── Ajanlar: trail + marker + heading oku ─────────────────────────────
    for aid in agent_ids:
        is_blue  = "blue" in aid
        col_live = BLUE_ALIVE if is_blue else RED_ALIVE
        col_dead = BLUE_DEAD  if is_blue else RED_DEAD
        col_trl  = BLUE_TRAIL if is_blue else RED_TRAIL

        states_so_far = traj[aid][:step + 1]
        if not states_so_far:
            continue
        xs = np.array([s[STATE_X] for s in states_so_far])
        ys = np.array([s[STATE_Y] for s in states_so_far])
        cur_alive = states_so_far[-1][STATE_ALIVE] > 0.5

        # Trail (gradient alpha)
        trail_start = max(0, len(xs) - TRAIL_LEN)
        n_seg = len(xs) - trail_start
        for j in range(n_seg - 1):
            alpha = 0.08 + 0.65 * (j / max(n_seg - 1, 1))
            si = trail_start + j
            ax2d.plot(xs[si:si+2], ys[si:si+2],
                      color=col_trl, lw=1.5, alpha=alpha)

        cur_x, cur_y = xs[-1], ys[-1]
        col_cur = col_live if cur_alive else col_dead

        if cur_alive:
            # Ucak sembolu: ikizkenar ucgen, heading yonune bakiyor
            psi    = states_so_far[-1][STATE_PSI]
            s_cur  = states_so_far[-1]
            alt    = s_cur[STATE_H]

            # Ucgen boyutu
            sz     = half * 0.022
            # Ucgen kosenoktalar (yerel koordinat, +Y ileri)
            tri_local = np.array([[0, sz * 1.4],
                                   [-sz * 0.6, -sz * 0.8],
                                   [sz * 0.6, -sz * 0.8]])
            # ENU psi = Kuzeyden CW; matplotlib: rotasyon matrisi
            c_p, s_p = np.cos(psi), np.sin(psi)
            # Kuzey = +Y ekseninde, psi Kuzeyden CW → rotasyon matrisi:
            # x' = x*cos(psi) + y*sin(psi)
            # y' = -x*sin(psi) + y*cos(psi)   (ENU → ekranda North=up)
            rot      = np.array([[c_p, s_p], [-s_p, c_p]])
            tri_world = (rot @ tri_local.T).T + np.array([cur_x, cur_y])
            tri_patch = mpatches.Polygon(
                tri_world, closed=True,
                facecolor=col_live, edgecolor="white",
                linewidth=0.8, alpha=0.95, zorder=10
            )
            ax2d.add_patch(tri_patch)

            # Irtifa etiketi
            ax2d.text(cur_x, cur_y + sz * 2.2,
                      f"{aid.split('_')[0].title()} {aid.split('_')[1]}\n{alt/1000:.1f}km",
                      color=col_live, fontsize=6.5, ha="center", va="bottom",
                      fontweight="bold",
                      bbox=dict(facecolor=BG_COLOR, alpha=0.5, pad=1.0,
                                edgecolor="none"))
        else:
            # Olen ajan: X isareti
            s_kl = sz = half * 0.018
            ax2d.plot([cur_x - s_kl, cur_x + s_kl],
                      [cur_y - s_kl, cur_y + s_kl],
                      color=col_dead, lw=2.0, alpha=0.7)
            ax2d.plot([cur_x + s_kl, cur_x - s_kl],
                      [cur_y - s_kl, cur_y + s_kl],
                      color=col_dead, lw=2.0, alpha=0.7)

    # ── Ates lazer efektleri ───────────────────────────────────────────────
    last_fire: dict = {}
    for ev in fire_events:
        f_step = ev[0]; f_aid = ev[1]
        if f_step <= step:
            if f_aid not in last_fire or ev[0] > last_fire[f_aid][0]:
                last_fire[f_aid] = ev

    for f_aid, (f_step, _, f_psi, opp_pos) in last_fire.items():
        age = step - f_step
        if 0 <= age < FIRE_FLASH_FRAMES:
            alpha_f = max(0.0, 1.0 - age / FIRE_FLASH_FRAMES)
            states  = traj[f_aid]
            if f_step >= len(states):
                continue
            s = states[f_step]
            if s[STATE_ALIVE] < 0.5:
                continue
            pos_x, pos_y = s[STATE_X], s[STATE_Y]
            hv           = heading_vec(f_psi)
            is_blue_agent = "blue" in f_aid
            laser_color  = "#00ffff" if is_blue_agent else "#ff2222"

            end_x = pos_x + hv[0] * WEZ_RANGE
            end_y = pos_y + hv[1] * WEZ_RANGE
            ax2d.plot([pos_x, end_x], [pos_y, end_y],
                      color=laser_color, lw=3.5, alpha=min(1.0, alpha_f),
                      zorder=20, solid_capstyle="round")
            ax2d.plot([pos_x, end_x], [pos_y, end_y],
                      color="#ffffff", lw=7.0, alpha=alpha_f * 0.2,
                      zorder=19, solid_capstyle="round")
            ax2d.scatter([pos_x], [pos_y], color=laser_color,
                         s=300 * alpha_f, alpha=alpha_f, zorder=21,
                         edgecolors="#ffffff", linewidths=0.8)

            # WEZ koni (sadece ilk 2 frame)
            if age < 2:
                mpl_deg = _psi_to_mpl_deg(f_psi)
                wez_w   = mpatches.Wedge(
                    (pos_x, pos_y), WEZ_RANGE,
                    mpl_deg - 30.0, mpl_deg + 30.0,
                    facecolor=laser_color, alpha=0.15 * alpha_f,
                    edgecolor=laser_color, linewidth=1.2,
                )
                ax2d.add_patch(wez_w)

            if opp_pos is not None and age < 5:
                ax2d.scatter([opp_pos[0]], [opp_pos[1]], color="#ffffff",
                             s=250 * alpha_f, alpha=alpha_f, zorder=22,
                             edgecolors=laser_color, linewidths=1.2)

    # ── Patlama efektleri ──────────────────────────────────────────────────
    for (k_step, k_aid) in kill_events:
        age = step - k_step
        if 0 <= age < KILL_FLASH_FRAMES:
            states = traj[k_aid]
            s_idx  = min(k_step, len(states) - 1)
            s      = states[s_idx]
            alpha_k = max(0.0, 0.9 * (1 - age / KILL_FLASH_FRAMES))
            r_exp   = half * 0.02 + age * half * 0.012
            circle  = mpatches.Circle(
                (s[STATE_X], s[STATE_Y]), r_exp,
                facecolor=KILL_COLOR, alpha=alpha_k * 0.45,
                edgecolor=KILL_COLOR, linewidth=1.5, zorder=15
            )
            ax2d.add_patch(circle)
            # Ikincidalga halkasi
            circle2 = mpatches.Circle(
                (s[STATE_X], s[STATE_Y]), r_exp * 0.5,
                facecolor="#ffdd44", alpha=alpha_k * 0.6,
                edgecolor="none", zorder=16
            )
            ax2d.add_patch(circle2)
            ax2d.text(s[STATE_X], s[STATE_Y] + r_exp * 1.4, "💥" if age < 4 else "",
                      fontsize=12, ha="center", va="bottom", zorder=17)

    # ── Baslik ve etiketler ────────────────────────────────────────────────
    fig.text(0.38, 0.97, "2v2 MARL Air Combat — Top-Down View",
             color=TEXT_COLOR, fontsize=14, fontweight="bold",
             ha="center", va="top", transform=fig.transFigure)
    fig.text(0.38, 0.925,
             f"{ALGO_LABEL}  |  Master's Thesis",
             color=DIM_COLOR, fontsize=8, ha="center", va="top",
             transform=fig.transFigure)
    fig.text(0.02, 0.97, "CAM: Top-Down",
             color=DIM_COLOR, fontsize=7, ha="left", va="top",
             transform=fig.transFigure)

    # Adim ve sure
    t_elapsed = step * 0.05
    ax2d.text(0.98, 0.98, f"T={t_elapsed:.1f}s  Step={step}",
              transform=ax2d.transAxes, color=TEXT_COLOR,
              fontsize=8, ha="right", va="top", fontweight="bold",
              bbox=dict(facecolor=BG_COLOR, alpha=0.7, pad=2.0, edgecolor=GRID_COLOR))

    # Efsane
    legend_handles = [
        mpatches.Patch(color=BLUE_ALIVE, label=f"Blue ({ALGO_LABEL})"),
        mpatches.Patch(color=RED_ALIVE,  label="Red (Heuristic)"),
        mpatches.Patch(color="#00ccff", alpha=0.3, label="Blue WEZ"),
        mpatches.Patch(color="#ff4444", alpha=0.3, label="Red WEZ"),
    ]
    ax2d.legend(handles=legend_handles, loc="lower left",
                fontsize=6.5, facecolor=BG_COLOR, edgecolor=GRID_COLOR,
                labelcolor=TEXT_COLOR, framealpha=0.85)

    # Son kare sonuc
    if step >= n_steps - 2:
        result_colors = {BLUE: BLUE_ALIVE, RED: RED_ALIVE, "draw": "#aaaaaa"}
        result_texts  = {BLUE: "BLUE WINS", RED: "RED WINS", "draw": "DRAW"}
        rc = result_colors.get(winner, "#aaaaaa")
        rt = result_texts.get(winner, "DRAW")
        ax2d.text(0.5, 0.5, rt,
                  transform=ax2d.transAxes,
                  color=rc, fontsize=30, fontweight="bold",
                  ha="center", va="center", alpha=0.85, zorder=99,
                  bbox=dict(facecolor=BG_COLOR, edgecolor=rc,
                            boxstyle="round,pad=0.5", alpha=0.8))

    draw_info_panel(ax_info, data, step)


# ===========================================================================
# MP4 Kayit
# ===========================================================================

def render_video(data: dict, output_path: str,
                 fps: int = 20, step_skip: int = 2,
                 view: str = "3d"):
    n_steps      = data["n_steps"]
    total_frames = (n_steps - 1) // step_skip + 1

    print(f"[Viz] View: {view} | {total_frames} frame | "
          f"{n_steps} adim, skip={step_skip}, {fps} fps")
    print(f"[Viz] Tahmini video suresi: {total_frames / fps:.0f} sn")

    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    fig.patch.set_facecolor(BG_COLOR)

    gs = gridspec.GridSpec(1, 2, figure=fig,
                           width_ratios=[4.2, 1],
                           left=0.02, right=0.98,
                           top=0.91, bottom=0.05,
                           wspace=0.02)

    is_topdown = (view == "topdown")
    if is_topdown:
        ax_main = fig.add_subplot(gs[0, 0])        # 2D
    else:
        ax_main = fig.add_subplot(gs[0, 0], projection="3d")  # 3D
    ax_info = fig.add_subplot(gs[0, 1])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    writer_kwargs = {
        "fps": fps,
        "codec": "libx264",
        "output_params": ["-crf", "18", "-pix_fmt", "yuv420p"],
    }

    with imageio.get_writer(output_path, **writer_kwargs) as writer:
        for fi in range(total_frames):
            if is_topdown:
                render_frame_topdown(fig, ax_main, ax_info, data,
                                     fi, total_frames, step_skip)
            else:
                render_frame_3d(fig, ax_main, ax_info, data,
                                fi, total_frames, step_skip, view)

            fig.canvas.draw()
            w, h  = fig.canvas.get_width_height()
            frame = np.frombuffer(fig.canvas.buffer_rgba(),
                                  dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
            writer.append_data(frame)

            if fi % 50 == 0 or fi == total_frames - 1:
                pct = (fi + 1) / total_frames * 100
                print(f"  {fi+1:>4}/{total_frames}  ({pct:.0f}%)", flush=True)

    plt.close(fig)
    size_mb = Path(output_path).stat().st_size / 1e6
    print(f"[Viz] Kaydedildi: {output_path}  ({size_mb:.1f} MB)")


# ===========================================================================
# Main
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Dogfight Visualizer — multi-view")
    p.add_argument("--checkpoint", default="checkpoints/mappo_gat_final.pt",
                   help="Checkpoint yolu")
    p.add_argument("--config",  default="configs/config.yaml")
    p.add_argument("--output",  default=None,
                   help="Cikti MP4 yolu (varsayilan: logs/dogfight_{view}_{mode}.mp4)")
    p.add_argument("--fps",     type=int, default=20)
    p.add_argument("--skip",    type=int, default=2,
                   help="Her kac adimda bir frame (varsayilan:2)")
    p.add_argument("--tries",   type=int, default=20,
                   help="Hedef outcome icin deneme sayisi")
    p.add_argument("--seed",    type=int, default=0)
    p.add_argument("--mode",    default="win",
                   choices=["win", "draw", "loss"],
                   help="Tek mod: win / draw / loss")
    p.add_argument("--view",    default="topdown",
                   choices=["3d", "topdown", "side", "chase", "cinematic"],
                   help="Kamera modu (varsayilan: topdown)")
    p.add_argument("--all-outcomes", action="store_true",
                   help="win + loss + draw icin ayri videolar olustur")
    p.add_argument("--count", type=int, default=1,
                   help="Her outcome tipi icin kac video (varsayilan: 1)")
    p.add_argument("--min-steps", type=int, default=0,
                   help="Bu adımdan kısa episodları hedef olarak kabul etme")
    p.add_argument("--phase2",  action="store_true",
                   help="GATMAPPOActor modunu kullan")
    p.add_argument("--qmix",    action="store_true",
                   help="QMIX AgentQNetwork ile visualize et")
    p.add_argument("--device",  default="auto")
    return p.parse_args()


def main():
    args = parse_args()

    cfg_path = PROJECT_ROOT / args.config
    with open(cfg_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    env = DogfightEnv(config)
    env.set_curriculum_phase(4)

    global ALGO_LABEL

    norm       = Normalizer(config)
    obs_dim    = norm.obs_dim(n_teammates=1, n_enemies=2)   # 50D
    action_dim = env.action_dim
    hidden     = int(config["training"].get("hidden_dim", 256))

    if args.qmix:
        qcfg       = config.get("qmix", {})
        actor      = load_qmix_policy(
            args.checkpoint, device,
            obs_dim       = int(qcfg.get("obs_dim",      50)),
            n_actions     = int(qcfg.get("n_actions",   162)),
            agent_hidden  = int(qcfg.get("agent_hidden", 128)),
        )
        gat_comm   = None
        ALGO_LABEL = "QMIX"
    else:
        actor, gat_comm = load_actor(args.checkpoint, device, obs_dim, action_dim,
                                      hidden, phase2=args.phase2)
        actor.eval()
        ALGO_LABEL = "GAT-MAPPO" if args.phase2 else "MAPPO"

    wez_range  = float(config.get("weapons", {}).get("wez_range_max", 8000.0))
    team_map   = {aid: ("blue" if "blue" in aid else "red")
                  for aid in env.agent_ids}
    opp_policy = MultiHeuristicPolicy(config, env.agent_ids, team_map)

    # Cikti yolu helper
    def out_path(mode: str, idx: int) -> str:
        if args.output and not args.all_outcomes and args.count == 1:
            return args.output
        suffix = f"_{idx+1}" if args.count > 1 else ""
        return f"logs/dogfight_{args.view}_{mode}{suffix}.mp4"

    modes = ["win", "loss", "draw"] if args.all_outcomes else [args.mode]

    for mode in modes:
        for i in range(args.count):
            print(f"\n[Viz] === Mod: {mode.upper()} {i+1}/{args.count} | View: {args.view} ===")
            data = find_good_episode(env, actor, opp_policy, device,
                                     n_tries=args.tries,
                                     seed_offset=args.seed + modes.index(mode) * 1000 + i * 100,
                                     mode=mode,
                                     gat_comm=gat_comm,
                                     wez_range=wez_range,
                                     min_steps=args.min_steps)
            render_video(data, out_path(mode, i),
                         fps=args.fps, step_skip=args.skip, view=args.view)

    print("\n[Viz] Tamamlandi.")


if __name__ == "__main__":
    main()

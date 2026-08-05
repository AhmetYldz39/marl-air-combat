"""
visualize_episode_print.py
==========================
MAPPO baseline'dan iki episode alir ve yazdir icin uygun statik figur uretir.
Sol panel: sifir-kill berabere episode. Sag panel: iki-kill kazanma episode.

Bu script, visualize_episode.py'daki video render'ini DEGISTIRMEZ.
Episode toplama fonksiyonlari ortak kullanilir; render katmani tamamen ayridir.

Cikti:
    figures/fig3_trajectory.png  (300 DPI, 6.5 in x 3.0 in)
    figures/fig3_trajectory.pdf  (varsa pdflatex icin)

Kullanim:
    python -X utf8 scripts/visualize_episode_print.py
    python -X utf8 scripts/visualize_episode_print.py --checkpoint checkpoints/mappo_final.pt
    python -X utf8 scripts/visualize_episode_print.py --tries 30 --output figures/fig3.png
"""

import sys
import argparse
import numpy as np
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from envs.dogfight_env import DogfightEnv
from envs.aircraft_model import (
    STATE_X, STATE_Y, STATE_H, STATE_PSI, STATE_HP, STATE_ALIVE, STATE_AMMO,
)
from agents.heuristic_agent import MultiHeuristicPolicy
from utils.normalization import Normalizer

# Episode collection and policy loading re-used from the video renderer
from scripts.visualize_episode import (
    collect_episode,
    find_good_episode,
    load_actor,
    heading_vec,
)

# ---------------------------------------------------------------------------
# Print style constants (white background, grayscale)
# ---------------------------------------------------------------------------
PRINT_DPI      = 300
PRINT_W_IN     = 6.5   # figure* width in IEEE two-column layout
PRINT_H_IN     = 3.0   # chosen to keep aspect ratio comfortable

# Trajectory line styles
BLUE_COLOR     = "black"
BLUE_LS        = "solid"
BLUE_LW        = 1.4
BLUE_MARKER    = "o"
BLUE_MSIZE     = 5

RED_COLOR      = "black"
RED_LS         = "dashed"
RED_LW         = 1.1
RED_MARKER     = "s"
RED_MSIZE      = 4

DEAD_ALPHA     = 0.35
WEZ_RANGE      = 8000.0
WEZ_HALF_ANG   = np.deg2rad(30.0)

MIN_FONT_PT    = 11   # declared sizes ≥11 pt: single-column (3.25 in) has no scaling


# ---------------------------------------------------------------------------
# Print-quality top-down renderer for one episode (static, full trajectory)
# ---------------------------------------------------------------------------

def _psi_to_mpl_deg(psi: float) -> float:
    """ENU psi (Kuzeyden CW) → matplotlib derece (Dogudan CCW)."""
    return 90.0 - np.degrees(psi)


def _kill_count_for(data: dict, agent_id: str) -> int:
    """Bu agentin kac rakip oldurdugunu say (kill_events'te karsi taraf olarak gecen)."""
    is_blue = "blue" in agent_id
    opp_prefix = "red" if is_blue else "blue"
    return sum(1 for (_, k_aid) in data["kill_events"] if opp_prefix in k_aid)


def render_print_panel(ax: plt.Axes, data: dict, title: str,
                       show_ylabel: bool = True) -> None:
    """
    Tek episode panelini yuksek kaliteli statik bicimde cizer.

    Kurallar:
    - Beyaz zemin, koyu cizgiler
    - Blue solid, Red dashed
    - En fazla BIR WEZ koni (en fazla kill yapan veya hayatta kalan Blue)
    - Yan panel yok
    - Font >= MIN_FONT_PT
    """
    traj        = data["traj"]
    kill_events = data["kill_events"]
    winner      = data["winner"]
    agent_ids   = list(traj.keys())
    blue_ids    = sorted([a for a in agent_ids if "blue" in a])
    red_ids     = sorted([a for a in agent_ids if "red" in a])

    # Tam yörünge (tüm adımlar)
    all_x = np.concatenate([np.array([s[STATE_X] for s in traj[a]]) for a in agent_ids])
    all_y = np.concatenate([np.array([s[STATE_Y] for s in traj[a]]) for a in agent_ids])
    pad   = 4000.0
    cx    = (all_x.max() + all_x.min()) / 2
    cy    = (all_y.max() + all_y.min()) / 2
    half  = max(all_x.max() - all_x.min(), all_y.max() - all_y.min()) / 2 + pad

    xmin, xmax = cx - half, cx + half
    ymin, ymax = cy - half, cy + half

    ax.set_facecolor("white")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")

    # Izgara (cok acik gri)
    ax.grid(True, color="#cccccc", linewidth=0.4, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    # Eksen etiketleri (km cinsinden)
    def km_fmt(val, pos):
        return f"{val/1000:.0f}"

    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(km_fmt))
    ax.yaxis.set_major_formatter(FuncFormatter(km_fmt))
    ax.set_xlabel("East (km)", fontsize=MIN_FONT_PT)
    if show_ylabel:
        ax.set_ylabel("North (km)", fontsize=MIN_FONT_PT)
    else:
        ax.set_ylabel("")
    ax.tick_params(labelsize=MIN_FONT_PT - 1)

    # 10 km referans halkasi
    ring_r = 10000.0
    if ring_r < half * 1.5:
        ring_t = np.linspace(0, 2 * np.pi, 120)
        ax.plot(cx + ring_r * np.cos(ring_t), cy + ring_r * np.sin(ring_t),
                color="#cccccc", lw=0.5, linestyle=":", zorder=0)
        ax.text(cx + ring_r * 0.72, cy + ring_r * 0.72, "10 km",
                fontsize=MIN_FONT_PT - 2, color="#aaaaaa", ha="left", va="bottom")

    # Kuzey oku
    arrow_x = xmin + (xmax - xmin) * 0.06
    arrow_y = ymin + (ymax - ymin) * 0.87
    arrow_len = half * 0.07
    ax.annotate("", xy=(arrow_x, arrow_y + arrow_len),
                xytext=(arrow_x, arrow_y),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2))
    ax.text(arrow_x, arrow_y + arrow_len * 1.3, "N",
            fontsize=MIN_FONT_PT - 1, ha="center", va="bottom", fontweight="bold")

    # Başlangıç noktaları (küçük içi boş sembol)
    for aid in agent_ids:
        s0 = traj[aid][0]
        is_blue = "blue" in aid
        mk = BLUE_MARKER if is_blue else RED_MARKER
        ax.plot(s0[STATE_X], s0[STATE_Y],
                marker=mk, markersize=BLUE_MSIZE if is_blue else RED_MSIZE,
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=0.8,
                linestyle="none", zorder=5)

    # Yörüngeler (tam rota çizgisi)
    for aid in agent_ids:
        is_blue = "blue" in aid
        ls = BLUE_LS if is_blue else RED_LS
        lw = BLUE_LW if is_blue else RED_LW
        xs = np.array([s[STATE_X] for s in traj[aid]])
        ys = np.array([s[STATE_Y] for s in traj[aid]])
        ax.plot(xs, ys, color="black", linestyle=ls, linewidth=lw,
                alpha=0.85, zorder=3)

    # Son konum işaretçileri
    for aid in agent_ids:
        s_final = traj[aid][-1]
        alive   = s_final[STATE_ALIVE] > 0.5
        is_blue = "blue" in aid
        mk = BLUE_MARKER if is_blue else RED_MARKER
        ms = BLUE_MSIZE  if is_blue else RED_MSIZE
        alpha = 1.0 if alive else DEAD_ALPHA
        fc = "black" if alive else "white"
        ax.plot(s_final[STATE_X], s_final[STATE_Y],
                marker=mk, markersize=ms + 1,
                markerfacecolor=fc,
                markeredgecolor="black",
                markeredgewidth=1.0,
                linestyle="none", alpha=alpha, zorder=7)
        # Başlık etiketi: "B0", "B1", "R0", "R1"
        prefix = "B" if is_blue else "R"
        idx = aid.split("_")[-1]
        label_x = s_final[STATE_X]
        label_y = s_final[STATE_Y] + half * 0.04
        ax.text(label_x, label_y, f"{prefix}{idx}",
                fontsize=MIN_FONT_PT - 1, ha="center", va="bottom",
                color="black", alpha=alpha,
                bbox=dict(facecolor="white", edgecolor="none",
                          alpha=0.7, pad=0.5))

    # Heading oku (son konumda, yalnizca hayatta olanlar)
    for aid in agent_ids:
        s_final = traj[aid][-1]
        if s_final[STATE_ALIVE] < 0.5:
            continue
        psi = s_final[STATE_PSI]
        hv = heading_vec(psi)
        arrow_scale = half * 0.07
        ax.annotate("",
                    xy=(s_final[STATE_X] + hv[0] * arrow_scale,
                        s_final[STATE_Y] + hv[1] * arrow_scale),
                    xytext=(s_final[STATE_X], s_final[STATE_Y]),
                    arrowprops=dict(arrowstyle="-|>", color="black",
                                   lw=1.0, mutation_scale=8),
                    zorder=8)

    # Kill event isaretleri (X sembolu)
    for (k_step, k_aid) in kill_events:
        s_k = traj[k_aid][min(k_step, len(traj[k_aid]) - 1)]
        marker_sz = half * 0.025
        for sign_x, sign_y in [(1, 1), (-1, 1)]:
            ax.plot([s_k[STATE_X] - sign_x * marker_sz,
                     s_k[STATE_X] + sign_x * marker_sz],
                    [s_k[STATE_Y] - sign_y * marker_sz,
                     s_k[STATE_Y] + sign_y * marker_sz],
                    color="black", lw=1.5, alpha=0.7, zorder=9)

    # Tek WEZ koni: en fazla kill yapan hayatta Blue, yoksa ilk Blue
    wez_agent = None
    best_kills = -1
    for bid in blue_ids:
        s_final = traj[bid][-1]
        if s_final[STATE_ALIVE] < 0.5:
            continue
        n_kills = sum(1 for (_, k) in kill_events if "red" in k)
        if n_kills > best_kills:
            best_kills = n_kills
            wez_agent = bid
    if wez_agent is None and blue_ids:
        wez_agent = blue_ids[0]

    if wez_agent:
        s_final = traj[wez_agent][-1]
        if s_final[STATE_ALIVE] > 0.5:
            psi = s_final[STATE_PSI]
            mpl_deg = _psi_to_mpl_deg(psi)
            wedge = mpatches.Wedge(
                (s_final[STATE_X], s_final[STATE_Y]),
                WEZ_RANGE,
                mpl_deg - 30.0, mpl_deg + 30.0,
                facecolor="#dddddd", alpha=0.5,
                edgecolor="black", linewidth=0.8,
                linestyle="--", zorder=2,
            )
            ax.add_patch(wedge)

    # Panel basligı
    ax.set_title(title, fontsize=MIN_FONT_PT + 1, fontweight="bold", pad=4)

    # Sonuc etiketi (sag ust kose)
    outcome_labels = {"blue": "Win", "red": "Loss", "draw": "Draw (timeout)"}
    outcome_text = outcome_labels.get(winner, "Draw")
    n_kills_blue = sum(1 for (_, k) in kill_events if "red" in k)
    result_str = f"{outcome_text}  |  {n_kills_blue} kill{'s' if n_kills_blue != 1 else ''}"
    ax.text(0.98, 0.98, result_str,
            transform=ax.transAxes,
            fontsize=MIN_FONT_PT, ha="right", va="top",
            bbox=dict(facecolor="white", edgecolor="#aaaaaa",
                      boxstyle="round,pad=0.3", linewidth=0.6))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/mappo_final.pt",
                   help="MAPPO baseline checkpoint (default: checkpoints/mappo_final.pt)")
    p.add_argument("--tries", type=int, default=25,
                   help="Max tries per outcome type (default: 25)")
    p.add_argument("--min-steps", type=int, default=200,
                   help="Min episode length to consider (default: 200)")
    p.add_argument("--seed-offset", type=int, default=0,
                   help="Seed offset for episode search (default: 0)")
    p.add_argument("--output", default="figures/fig3_trajectory.png",
                   help="Output path (default: figures/fig3_trajectory.png)")
    p.add_argument("--pdf", action="store_true",
                   help="Also save a PDF alongside the PNG")
    p.add_argument("--config", default="configs/config.yaml",
                   help="Config file (default: configs/config.yaml)")
    return p.parse_args()


def main():
    args = parse_args()

    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Print] Device: {device}")

    ckpt_path = args.checkpoint
    if not Path(ckpt_path).exists():
        print(f"[Print] ERROR: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    # MAPPO baseline is always non-phase2 (50D obs)
    actor, gat_comm = load_actor(ckpt_path, device,
                                 obs_dim=50, action_dim=5, hidden=256,
                                 phase2=False)
    print(f"[Print] Checkpoint yüklendi: {ckpt_path}")

    env = DogfightEnv(cfg)
    env.set_curriculum_phase(4)   # 2v2
    team_map   = {aid: ("blue" if "blue" in aid else "red") for aid in env.agent_ids}
    opp_policy = MultiHeuristicPolicy(cfg, env.agent_ids, team_map)

    print(f"[Print] Sifir-kill berabere episode aranıyor (max {args.tries} deneme)...")
    ep_draw = find_good_episode(
        env, actor, opp_policy, device,
        n_tries=args.tries,
        seed_offset=args.seed_offset,
        mode="draw",
        gat_comm=None,
        min_steps=args.min_steps,
    )
    # Sıfır-kill koşulunu kontrol et; eğer kill varsa tekrar ara
    if len(ep_draw["kill_events"]) > 0:
        print(f"[Print] Berabere episodda kill var ({len(ep_draw['kill_events'])}); "
              f"sıfır-kill episode aranıyor...")
        for seed in range(args.seed_offset + args.tries,
                          args.seed_offset + args.tries * 3):
            ep_cand = collect_episode(env, actor, opp_policy, device,
                                      deterministic=False, seed=seed, gat_comm=None)
            if ep_cand["winner"] == "draw" and len(ep_cand["kill_events"]) == 0 \
                    and ep_cand["n_steps"] >= args.min_steps:
                ep_draw = ep_cand
                print(f"[Print] Sıfır-kill berabere bulundu: seed={seed}, "
                      f"steps={ep_draw['n_steps']}")
                break

    print(f"[Print] İki-kill kazanma episode aranıyor (max {args.tries} deneme)...")
    ep_win = None
    for seed in range(args.seed_offset, args.seed_offset + args.tries * 2):
        ep_cand = collect_episode(env, actor, opp_policy, device,
                                  deterministic=False, seed=seed, gat_comm=None)
        n_kills_blue = sum(1 for (_, k) in ep_cand["kill_events"] if "red" in k)
        if ep_cand["winner"] == "blue" and n_kills_blue >= 2 \
                and ep_cand["n_steps"] >= args.min_steps:
            ep_win = ep_cand
            print(f"[Print] İki-kill kazanma bulundu: seed={seed}, "
                  f"steps={ep_win['n_steps']}, kills={n_kills_blue}")
            break

    if ep_win is None:
        print("[Print] UYARI: İki-kill win bulunamadı; en iyi win kullanılıyor.")
        ep_win = find_good_episode(
            env, actor, opp_policy, device,
            n_tries=args.tries * 2,
            seed_offset=args.seed_offset,
            mode="win",
            gat_comm=None,
            min_steps=args.min_steps,
        )

    # Çıktı dizini oluştur
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Şekil oluştur
    fig, (ax_left, ax_right) = plt.subplots(1, 2,
                                             figsize=(PRINT_W_IN, PRINT_H_IN),
                                             dpi=PRINT_DPI)
    fig.patch.set_facecolor("white")
    plt.subplots_adjust(left=0.10, right=0.97, top=0.84, bottom=0.15,
                        wspace=0.30)

    n_draw_kills = sum(1 for (_, k) in ep_draw["kill_events"] if "red" in k)
    n_win_kills  = sum(1 for (_, k) in ep_win["kill_events"] if "red" in k)

    render_print_panel(ax_left,  ep_draw,
                       title="(a) No engagement established",
                       show_ylabel=True)
    render_print_panel(ax_right, ep_win,
                       title="(b) Two-kill win",
                       show_ylabel=False)

    # Ortak legend
    leg_elements = [
        mlines.Line2D([], [], color="black", linestyle="solid",
                      linewidth=BLUE_LW, label="Blue (MAPPO)"),
        mlines.Line2D([], [], color="black", linestyle="dashed",
                      linewidth=RED_LW,  label="Red (heuristic)"),
        mlines.Line2D([], [], marker="o", color="black", linestyle="none",
                      markersize=4, label="Start (open) / End (filled)"),
        mpatches.Patch(facecolor="#dddddd", edgecolor="black",
                       linewidth=0.6, linestyle="--",
                       label="WEZ cone (Blue$_0$, final step)"),
    ]
    fig.legend(handles=leg_elements, loc="lower center",
               ncol=4, fontsize=MIN_FONT_PT - 1,
               frameon=True, edgecolor="#aaaaaa",
               bbox_to_anchor=(0.535, -0.02))

    # Suptitle (figur basliginin altinda)
    n_draw_steps = ep_draw["n_steps"]
    n_win_steps  = ep_win["n_steps"]
    fig.text(0.535, 0.98,
             f"MAPPO baseline — illustrative single episodes, not evidence. "
             f"(a) {n_draw_steps} steps, 0 kills; "
             f"(b) {n_win_steps} steps, {n_win_kills} kills.",
             ha="center", va="top",
             fontsize=MIN_FONT_PT - 1, color="#555555",
             transform=fig.transFigure)

    fig.savefig(out_path, dpi=PRINT_DPI, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"[Print] Kaydedildi: {out_path}")

    if args.pdf:
        pdf_path = out_path.with_suffix(".pdf")
        fig.savefig(pdf_path, dpi=PRINT_DPI, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"[Print] PDF kaydedildi: {pdf_path}")

    plt.close(fig)

    # LaTeX figure* kodu yaz (kullanim kolayligi icin)
    latex_path = out_path.with_suffix(".tex")
    n_draw_kills_total = len(ep_draw["kill_events"])
    n_win_kills_total  = len(ep_win["kill_events"])
    latex_snippet = f"""%% --- Trajectory figure (Task 3) ---
%% Yerlesim: Section 8.2 sonrasinda, yer varsa.
%% \\input icin: bu dosyayi docs/conference_paper/F3/fig3_trajectory.tex olarak kopyala.
%%
\\begin{{figure*}}[t]
\\centering
\\includegraphics[width=\\linewidth]{{{out_path.name}}}
\\caption{{%
  Top-down trajectory plots from two illustrative episodes of the MAPPO baseline
  against the heuristic opponent. These are \\emph{{single hand-picked episodes
  and do not constitute evidence about policy quality}}.
  (a)~Zero-kill draw: both Blue agents fail to close within weapon engagement zone
  (WEZ), illustrating the engagement-initiation failure identified in the
  outcome decomposition.
  (b)~Two-kill win: Blue engages and defeats both Red agents within
  {n_win_steps} steps.
  Solid lines: Blue (MAPPO); dashed lines: Red (heuristic).
  Gray wedge: WEZ cone (Blue$_0$, final step only).
  All trajectories are top-down (altitude not shown).%
}}
\\label{{fig:trajectory}}
\\end{{figure*}}
"""
    latex_path.write_text(latex_snippet, encoding="utf-8")
    print(f"[Print] LaTeX snippet kaydedildi: {latex_path}")


if __name__ == "__main__":
    main()

"""
thesis_figures.py
=================
Tez figürlerini üretir → figures/ klasörüne kaydeder (300 dpi, PNG).

Figure 1 : three_marl_paradigms.png
Figure 2 : activation_functions.png
Figure 3 : gat_architecture.png

Kullanım:
    python -X utf8 scripts/thesis_figures.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

# ── Genel stil ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":    "serif",
    "font.serif":     ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size":      11,
    "figure.dpi":     150,           # ekran önizleme; kayıt 300 dpi
    "text.usetex":    False,
})

ROOT    = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

SAVEKW = dict(dpi=300, bbox_inches="tight", facecolor="white")

# ── Renk paleti ───────────────────────────────────────────────────────────────
C_BLUE   = "#2B6CB0"   # ajan / ana kutu
C_LBLUE  = "#BEE3F8"   # açık mavi dolgu
C_ENV    = "#4A5568"   # environment kutu
C_LENV   = "#E2E8F0"   # açık gri dolgu
C_CRIT   = "#C05621"   # centralized critic / turuncu
C_LCRIT  = "#FEEBC8"
C_GREEN  = "#276749"
C_LGREEN = "#C6F6D5"
C_RED    = "#C53030"
C_LRED   = "#FED7D7"
C_ARROW  = "#2D3748"
C_GRAY   = "#718096"


# ─────────────────────────────────────────────────────────────────────────────
# Ortak çizim yardımcıları
# ─────────────────────────────────────────────────────────────────────────────

def box(ax, cx, cy, w, h, text, fc, ec, tc="white",
        fs=9.5, bold=False, alpha=1.0, lw=1.4):
    r = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.03", linewidth=lw,
        facecolor=fc, edgecolor=ec, alpha=alpha, zorder=3,
    )
    ax.add_patch(r)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal",
            multialignment="center", zorder=4)


def arrow(ax, x0, y0, x1, y1, color=C_ARROW, lw=1.3,
          label="", ls="->", rad=0.0, lfs=8):
    cs = f"arc3,rad={rad}" if rad else "arc3,rad=0"
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle=ls, color=color, lw=lw,
                    connectionstyle=cs,
                ), zorder=2)
    if label:
        mx = (x0 + x1) / 2 + 0.015
        my = (y0 + y1) / 2
        ax.text(mx, my, label, fontsize=lfs, color=C_GRAY,
                ha="left", va="center", style="italic", zorder=5)


def dashed_line(ax, y, xmin=0.05, xmax=0.95, color=C_GRAY, lw=1.0):
    ax.axhline(y, xmin=xmin, xmax=xmax,
               color=color, lw=lw, linestyle="--", zorder=1)


# =============================================================================
# Figure 1 — Three MARL Paradigms
# =============================================================================

def make_figure1():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    fig.patch.set_facecolor("white")

    # ── Panel 1: Fully Decentralized ──────────────────────────────────────────
    ax = axes[0]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ajan kutuları
    box(ax, 0.25, 0.72, 0.30, 0.13, "Agent 1", fc=C_BLUE,  ec=C_BLUE,  fs=10, bold=True)
    box(ax, 0.75, 0.72, 0.30, 0.13, "Agent 2", fc=C_BLUE,  ec=C_BLUE,  fs=10, bold=True)
    # ayrı env kutuları
    box(ax, 0.25, 0.38, 0.30, 0.13, "Env 1",   fc=C_LENV,  ec=C_ENV,   fs=10, tc=C_ENV)
    box(ax, 0.75, 0.38, 0.30, 0.13, "Env 2",   fc=C_LENV,  ec=C_ENV,   fs=10, tc=C_ENV)

    # action okları (aşağı) — sağ tarafa hizalı
    arrow(ax, 0.28, 0.655, 0.28, 0.445)
    arrow(ax, 0.78, 0.655, 0.78, 0.445)
    # obs okları (yukarı) — sol tarafa hizalı
    arrow(ax, 0.22, 0.445, 0.22, 0.655)
    arrow(ax, 0.72, 0.445, 0.72, 0.655)
    # etiketler — çakışmayan konumlar
    ax.text(0.315, 0.550, "action", fontsize=8, color=C_GRAY, va="center", ha="left")
    ax.text(0.815, 0.550, "action", fontsize=8, color=C_GRAY, va="center", ha="left")
    ax.text(0.185, 0.550, "obs",    fontsize=8, color=C_GRAY, va="center", ha="right")
    ax.text(0.685, 0.550, "obs",    fontsize=8, color=C_GRAY, va="center", ha="right")

    # uyarı
    ax.text(0.50, 0.18, "✗  Non-stationary environment",
            ha="center", fontsize=10, color=C_RED, fontweight="bold",
            fontfamily="DejaVu Sans")
    ax.text(0.50, 0.08, "Agents ignore teammates → unstable learning",
            ha="center", fontsize=8.5, color=C_GRAY, style="italic")

    ax.set_title("Fully Decentralized\n(Independent Learners)",
                 fontsize=11.5, fontweight="bold", pad=10)

    # ── Panel 2: Fully Centralized ────────────────────────────────────────────
    ax = axes[1]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    box(ax, 0.50, 0.72, 0.48, 0.15,
        "Central Controller", fc=C_CRIT, ec=C_CRIT, fs=10, bold=True)
    box(ax, 0.50, 0.38, 0.48, 0.15,
        "Environment",        fc=C_LENV, ec=C_ENV,  fs=10, tc=C_ENV)

    # çift yönlü oklar — etiketler sağa/sola ayrılmış
    arrow(ax, 0.54, 0.645, 0.54, 0.455)
    arrow(ax, 0.46, 0.455, 0.46, 0.645)
    ax.text(0.60, 0.555, "joint action", fontsize=8.5, color=C_GRAY, va="center", ha="left")
    ax.text(0.40, 0.545, "global state", fontsize=8.5, color=C_GRAY, va="center", ha="right")

    # Uyarı kutusu
    r = FancyBboxPatch((0.08, 0.14), 0.84, 0.12,
                       boxstyle="round,pad=0.02", lw=1.2,
                       facecolor=C_LRED, edgecolor=C_RED)
    ax.add_patch(r)
    ax.text(0.50, 0.20,
            r"$|\mathcal{A}|^n$ joint action space",
            ha="center", fontsize=10.5, color=C_RED, fontweight="bold")
    ax.text(0.50, 0.08, "Exponential complexity — impractical for n > 2",
            ha="center", fontsize=8.5, color=C_GRAY, style="italic")

    ax.set_title("Fully Centralized\n(Joint Action Space)",
                 fontsize=11.5, fontweight="bold", pad=10)

    # ── Panel 3: CTDE ─────────────────────────────────────────────────────────
    ax = axes[2]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # Kesik çizgi — training / execution ayırıcı
    dashed_line(ax, 0.52, xmin=0.02, xmax=0.98)
    # Faz etiketleri — sol tarafa, Actor 1 hizasında
    ax.text(0.04, 0.575, "Training\nPhase",   ha="left", fontsize=8,
            color=C_GRAY, style="italic", va="center")
    ax.text(0.04, 0.465, "Execution\nPhase", ha="left", fontsize=8,
            color=C_GRAY, style="italic", va="center")

    # Centralized Critic (training) — biraz sağa kaydır, sol etiketlere yer aç
    box(ax, 0.57, 0.74, 0.44, 0.13,
        "Centralized Critic\n(Global State)", fc=C_CRIT, ec=C_CRIT, fs=9.5, bold=True)

    # Global state input ok (sol kenardan critic'e)
    arrow(ax, 0.05, 0.74, 0.35, 0.74, label="")
    ax.text(0.04, 0.80, "Global\nState", ha="left", fontsize=8, color=C_GRAY)

    # Decentralized actors (execution)
    box(ax, 0.28, 0.31, 0.33, 0.13,
        "Actor 1\n(Local Obs)", fc=C_BLUE, ec=C_BLUE, fs=9.5, bold=True)
    box(ax, 0.78, 0.31, 0.33, 0.13,
        "Actor 2\n(Local Obs)", fc=C_BLUE, ec=C_BLUE, fs=9.5, bold=True)

    # Local obs okları (aşağıdan yukarı)
    arrow(ax, 0.28, 0.245, 0.28, 0.10)
    arrow(ax, 0.78, 0.245, 0.78, 0.10)
    ax.text(0.28, 0.055, "Local Obs", ha="center", fontsize=8.5, color=C_GRAY)
    ax.text(0.78, 0.055, "Local Obs", ha="center", fontsize=8.5, color=C_GRAY)

    # Gradient: Critic → Actors (kesik, turuncu)
    arrow(ax, 0.40, 0.68, 0.25, 0.375,
          color="#C05621", lw=1.1, rad=0.20)
    arrow(ax, 0.74, 0.68, 0.85, 0.375,
          color="#C05621", lw=1.1, rad=-0.15)
    # "value gradient" etiketi — sol gradient ok'un soluna
    ax.text(0.04, 0.630, "value gradient\n(train only)",
            ha="left", fontsize=7.5, color="#C05621", style="italic")

    ax.set_title("Centralized Training,\nDecentralized Execution (CTDE)",
                 fontsize=11.5, fontweight="bold", pad=10)

    plt.tight_layout(pad=1.8)
    out = FIG_DIR / "three_marl_paradigms.png"
    fig.savefig(out, **SAVEKW)
    plt.close(fig)
    print(f"[OK] {out}")


# =============================================================================
# Figure 2 — Activation Functions
# =============================================================================

def make_figure2():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x        = np.linspace(-4, 4, 1000)
    relu     = np.maximum(0, x)
    tanh_v   = np.tanh(x)
    sigmoid  = 1.0 / (1.0 + np.exp(-x))
    clamp    = np.clip(x, -1.0, 1.0)

    lw = 2.1
    ax.plot(x, relu,    color="#2196F3", lw=lw,
            label=r"ReLU: $\max(0,\, x)$")
    ax.plot(x, tanh_v,  color="#FF9800", lw=lw,
            label=r"$\tanh(x)$")
    ax.plot(x, sigmoid, color="#4CAF50", lw=lw,
            label=r"Sigmoid: $\sigma(x) = \frac{1}{1+e^{-x}}$")
    ax.plot(x, clamp,   color="#E53E3E", lw=lw, linestyle="--",
            label=r"Clamp$(-1,\,1)$  [this work]")

    # Referans çizgileri
    ax.axhline(0,  color="black", lw=0.8, alpha=0.45)
    ax.axvline(0,  color="black", lw=0.8, alpha=0.45)
    ax.axhline( 1, color=C_GRAY,  lw=0.6, linestyle=":", alpha=0.5)
    ax.axhline(-1, color=C_GRAY,  lw=0.6, linestyle=":", alpha=0.5)

    ax.set_xlabel(r"$x$", fontsize=13)
    ax.set_ylabel(r"$f(x)$", fontsize=13)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-1.6, 3.6)
    ax.legend(fontsize=10.5, loc="upper left", framealpha=0.96,
              edgecolor=C_GRAY)
    ax.grid(True, alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Satürasyon bölgesi notu (tanh için)
    ax.annotate("saturation",
                xy=(3.2, np.tanh(3.2)), xytext=(2.2, 1.45),
                fontsize=8.5, color="#FF9800",
                arrowprops=dict(arrowstyle="->", color="#FF9800", lw=1.0),
                style="italic")
    # Clamp kırılma noktaları
    for xv in (-1, 1):
        ax.scatter([xv], [xv], color="#E53E3E", s=40, zorder=5)

    plt.tight_layout()
    out = FIG_DIR / "activation_functions.png"
    fig.savefig(out, **SAVEKW)
    plt.close(fig)
    print(f"[OK] {out}")


# =============================================================================
# Figure 3 — GATComm Architecture
# =============================================================================

def make_figure3():
    """
    GATComm (gat_comm.py) gerçek mimarisine göre:
        1. Query  = W_q(x_i)               17→16 (4H×4D)
        2. Key    = W_k(concat(x_j,e_ij))  20→16
           Value  = W_v(concat(x_j,e_ij))  20→16
        3. attn   = softmax(Q·K^T / √D)  + dead-agent masking
        4. msg_i  = Σ_j attn × V         → concat 4 heads → 16D
        5. out    = LayerNorm(W_out(16→16))
    """
    fig = plt.figure(figsize=(15, 6.5))
    fig.patch.set_facecolor("white")

    # Sol 3/4: akış diyagramı
    ax = fig.add_axes([0.01, 0.04, 0.70, 0.92])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    BW, BH = 0.115, 0.115    # standart kutu w, h

    # ── Renk / stil tanımları ────────────────────────────────────────────────
    KW_IN  = dict(fc=C_LBLUE,   ec=C_BLUE,  tc=C_BLUE,  lw=1.3)
    KW_PRJ = dict(fc=C_LCRIT,   ec=C_CRIT,  tc=C_CRIT,  lw=1.3)
    KW_ATT = dict(fc=C_LRED,    ec=C_RED,   tc=C_RED,   lw=1.3)
    KW_OUT = dict(fc=C_LGREEN,  ec=C_GREEN, tc=C_GREEN, lw=1.3)

    def bx(cx, cy, text, kw, fs=8.8, w=BW, h=BH):
        box(ax, cx, cy, w, h, text, **kw, fs=fs)

    def ar(x0, y0, x1, y1, lbl="", rad=0.0, color=C_ARROW, lw=1.25):
        arrow(ax, x0, y0, x1, y1, label=lbl, rad=rad, color=color, lw=lw)

    # ── Girdi sütunu (x ≈ 0.09) ──────────────────────────────────────────────
    X_IN = 0.09
    Y_NI = 0.74    # node_i
    Y_NJ = 0.50    # node_j (komşu)
    Y_EJ = 0.26    # edge_ij

    bx(X_IN, Y_NI, "Node $\\mathbf{x}_i$\n(17D)", KW_IN, fs=9)
    bx(X_IN, Y_NJ, "Node $\\mathbf{x}_j$\n(17D)", KW_IN, fs=9)
    bx(X_IN, Y_EJ, "Edge $\\mathbf{e}_{ij}$\n(3D)",  KW_IN, fs=9)

    ax.text(X_IN, 0.935, "Inputs", ha="center", fontsize=9,
            color=C_BLUE, fontweight="bold")

    # ── Concat bloğu (x ≈ 0.27) ──────────────────────────────────────────────
    X_CAT = 0.27
    Y_CAT = 0.38   # concat(x_j, e_ij) = 20D

    bx(X_CAT, Y_CAT, "Concat\n$[\\mathbf{x}_j\\,\\|\\,\\mathbf{e}_{ij}]$\n(20D)",
       KW_PRJ, fs=8.5)

    # node_j → concat
    ar(X_IN + BW/2, Y_NJ, X_CAT - BW/2, Y_CAT + 0.02, rad=0.0)
    # edge_ij → concat
    ar(X_IN + BW/2, Y_EJ, X_CAT - BW/2, Y_CAT - 0.02, rad=0.0)

    # ── Projeksiyon sütunu (x ≈ 0.445) ───────────────────────────────────────
    X_PRJ = 0.445
    Y_Q   = 0.74
    Y_KV  = 0.38

    bx(X_PRJ, Y_Q,  "Query $W_q$\n$17\\!\\to\\!16$\n$(4H\\!\\times\\!4D)$",
       KW_PRJ, fs=8.5)
    bx(X_PRJ, Y_KV, "Key $W_k$  /  Value $W_v$\n$20\\!\\to\\!16$\n$(4H\\!\\times\\!4D)$",
       KW_PRJ, fs=8.0, w=BW*1.35, h=BH)

    # node_i → Query
    ar(X_IN  + BW/2, Y_NI,       X_PRJ - BW/2, Y_Q)
    # concat → Key/Value
    ar(X_CAT + BW/2, Y_CAT,      X_PRJ - BW*1.35/2, Y_KV)

    ax.text(X_PRJ, 0.935, "Projections", ha="center", fontsize=9,
            color=C_CRIT, fontweight="bold")

    # ── Attention (x ≈ 0.625) ─────────────────────────────────────────────────
    X_ATT = 0.625
    Y_ATT = 0.56

    bx(X_ATT, Y_ATT,
       "Attention Score\n"
       "$\\alpha_{ij}^h = \\mathrm{softmax}$\n"
       "$(Q_i \\cdot K_{ij}^T / \\sqrt{D})$",
       KW_ATT, fs=8.3)

    ar(X_PRJ + BW/2,       Y_Q,   X_ATT - BW/2, Y_ATT + 0.03, rad=-0.12)
    ar(X_PRJ + BW*1.35/2,  Y_KV,  X_ATT - BW/2, Y_ATT - 0.03, rad=0.08)

    # Masking not
    ax.text(X_ATT, Y_ATT - BH/2 - 0.07,
            "+ dead-agent mask", ha="center", fontsize=7.5,
            color=C_GRAY, style="italic")

    ax.text(X_ATT, 0.935, "Attention", ha="center", fontsize=9,
            color=C_RED, fontweight="bold")

    # ── Ağırlıklı toplam + concat (x ≈ 0.785) ─────────────────────────────────
    X_AGG = 0.785
    Y_AGG = 0.56

    bx(X_AGG, Y_AGG,
       "Weighted Sum\n$\\sum_j \\alpha_{ij}^h \\mathbf{v}_{ij}$\nConcat $H$ heads\n$\\to$ 16D",
       KW_ATT, fs=8.3, h=BH * 1.2)

    ar(X_ATT + BW/2, Y_ATT, X_AGG - BW/2, Y_AGG)

    # Value'ya ok (dashed: K/V → aggregation)
    ar(X_PRJ + BW*1.35/2, Y_KV, X_AGG - BW/2, Y_AGG,
       rad=0.22, color="#999999", lw=1.0)
    ax.text(X_PRJ + 0.09, Y_KV - 0.12, "value path",
            ha="center", fontsize=7.2, color=C_GRAY, style="italic")

    # ── Çıkış (x ≈ 0.945) ────────────────────────────────────────────────────
    X_OUT = 0.945
    Y_OUT = 0.56

    bx(X_OUT, Y_OUT,
       "LayerNorm\n$W_{out}$\n$\\mathbf{m}_i$\n(16D)",
       KW_OUT, fs=8.8, h=BH * 1.3)

    ar(X_AGG + BW/2, Y_AGG, X_OUT - BW/2, Y_OUT)

    ax.text(X_OUT, 0.935, "Output", ha="center", fontsize=9,
            color=C_GREEN, fontweight="bold")

    # ── Alt açıklama ─────────────────────────────────────────────────────────
    ax.text(0.50, 0.035,
            "$H=4$ attention heads $\\times$ 4D/head $=$ 16D message  "
            "|  node$_i$: ego obs (17D)  "
            "|  edge: [dist$_{\\mathrm{norm}}$, bearing$_{\\mathrm{norm}}$, threat]",
            ha="center", fontsize=8.5, color=C_GRAY, style="italic")

    # ── Başlık ────────────────────────────────────────────────────────────────
    ax.text(0.50, 0.985,
            "GATComm — Multi-Head Graph Attention Message Computation",
            ha="center", va="top", fontsize=11.5, fontweight="bold")

    # ── Renk açıklaması ──────────────────────────────────────────────────────
    legend_elems = [
        mpatches.Patch(fc=C_LBLUE,  ec=C_BLUE,  label="Input features"),
        mpatches.Patch(fc=C_LCRIT,  ec=C_CRIT,  label="Linear projection"),
        mpatches.Patch(fc=C_LRED,   ec=C_RED,   label="Attention"),
        mpatches.Patch(fc=C_LGREEN, ec=C_GREEN, label="Output"),
    ]
    ax.legend(handles=legend_elems, loc="lower left", fontsize=8.5,
              framealpha=0.95, ncol=2)

    # ── Sağ panel: örnek graf ─────────────────────────────────────────────────
    ax2 = fig.add_axes([0.72, 0.08, 0.27, 0.84])
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.axis("off")

    # Arka plan kutusu
    ax2.add_patch(FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.96,
        boxstyle="round,pad=0.02", lw=1.0,
        facecolor="#F7FAFF", edgecolor="#CBD5E0"))

    ax2.text(0.50, 0.94, "Example: 2 Blue Agents",
             ha="center", fontsize=9.5, fontweight="bold", color="#1A365D")

    # Blue düğümleri
    ax2.scatter([0.28], [0.72], s=1600, color=C_BLUE,  zorder=5)
    ax2.scatter([0.72], [0.28], s=1600, color=C_BLUE,  zorder=5)
    ax2.text(0.28, 0.72, "$B_0$", ha="center", va="center",
             fontsize=10, color="white", fontweight="bold", zorder=6)
    ax2.text(0.72, 0.28, "$B_1$", ha="center", va="center",
             fontsize=10, color="white", fontweight="bold", zorder=6)

    ax2.text(0.15, 0.84, "$\\mathbf{x}_0$\n(17D)", ha="center",
             fontsize=8.5, color=C_BLUE)
    ax2.text(0.85, 0.16, "$\\mathbf{x}_1$\n(17D)", ha="center",
             fontsize=8.5, color=C_BLUE)

    # Çift yönlü kenar
    ax2.annotate("", xy=(0.63, 0.37), xytext=(0.37, 0.63),
                 arrowprops=dict(arrowstyle="<->", color=C_ARROW, lw=1.8))

    # Kenar özelliği etiketi
    ax2.text(0.50, 0.555, "$\\mathbf{e}_{01}$",
             ha="center", fontsize=9, color="#444", fontweight="bold")
    ax2.text(0.50, 0.49,
             "[dist$_{\\mathrm{norm}}$,\nbearing, threat]",
             ha="center", fontsize=7.8, color=C_GRAY)

    # Red ajan (tehdit kaynağı)
    ax2.scatter([0.80], [0.78], s=700, color="#C53030", marker="*", zorder=5)
    ax2.text(0.80, 0.88, "Red\n(threat)", ha="center",
             fontsize=8, color="#C53030")
    ax2.annotate("", xy=(0.73, 0.72), xytext=(0.79, 0.79),
                 arrowprops=dict(arrowstyle="->", color="#C53030",
                                 lw=1.1, linestyle="dashed"))

    # Mesaj gösterimi
    ax2.add_patch(FancyBboxPatch(
        (0.08, 0.06), 0.84, 0.15,
        boxstyle="round,pad=0.02", lw=1.0,
        facecolor=C_LGREEN, edgecolor=C_GREEN))
    ax2.text(0.50, 0.135, "$\\mathbf{m}_0 \\leftarrow$ GATComm($B_0, B_1$)\n(16D message)",
             ha="center", fontsize=8.0, color=C_GREEN)

    out = FIG_DIR / "gat_architecture.png"
    fig.savefig(out, **SAVEKW)
    plt.close(fig)
    print(f"[OK] {out}")


# =============================================================================
# Figure 4 — MDP Agent-Environment Interaction Loop
# =============================================================================

def make_mdp_figure():
    """
    Circular MDP loop: Agent ↔ Environment, timeline strip, annotations.
    2400×1400 px @ 200 DPI  →  figsize=(12, 7)
    Sans-serif font, white background.
    """
    # ── Renk paleti ──────────────────────────────────────────────────────────
    CA   = "#2C5F8A"      # agent dark blue
    CA_L = "#D6E4F2"      # agent light fill
    CE   = "#C0634A"      # env dark (text)
    CE_L = "#FDECD8"      # env light fill
    CE_B = "#E8896A"      # env border coral
    CN   = "#6B7280"      # neutral gray
    CN_L = "#F3F4F6"      # neutral light
    ARR  = "#374151"      # arrow / main text
    ARR2 = "#9B4D3E"      # reward arrow

    with plt.rc_context({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"]}):
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor("white")
        fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        ax.set_facecolor("white")
        ax.set_xlim(-0.25, 12.25)   # padding → content tam görünür
        ax.set_ylim(-0.30,  7.30)
        ax.axis("off")

        # ── Kutu yardımcısı (data koordinatları) ─────────────────────────────
        def dbox(cx, cy, w, h, title, sub1="", sub2="",
                 fc="#D6E4F2", ec="#2C5F8A", tc="#2C5F8A", lw=2.2):
            ax.add_patch(FancyBboxPatch(
                (cx - w/2, cy - h/2), w, h,
                boxstyle="round,pad=0.15", linewidth=lw,
                facecolor=fc, edgecolor=ec, zorder=3))
            dy = 0.35 if (sub1 or sub2) else 0.0
            ax.text(cx, cy + dy, title,
                    ha="center", va="center", fontsize=19, fontweight="bold",
                    color=tc, zorder=4)
            if sub1:
                ax.text(cx, cy - 0.18, sub1,
                        ha="center", va="center", fontsize=13, color=tc, zorder=4)
            if sub2:
                ax.text(cx, cy - 0.65, sub2,
                        ha="center", va="center", fontsize=11,
                        color=tc, style="italic", zorder=4)

        def arr(x0, y0, x1, y1, rad=0.0, color=ARR, lw=2.0, ms=20):
            ax.annotate("",
                xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="-|>", color=color, lw=lw,
                    mutation_scale=ms,
                    connectionstyle=f"arc3,rad={rad}",
                ), zorder=2)

        # ── Ana kutular ───────────────────────────────────────────────────────
        BW, BH = 2.7, 1.85
        AX, AY = 2.9,  4.50   # Agent center
        EX, EY = 9.1,  4.50   # Environment center

        dbox(AX, AY, BW, BH,
             "Agent",
             r"$\pi(a \mid o,\, \theta)$",
             r"Value  $V(s)$",
             fc=CA_L, ec=CA, tc=CA)

        dbox(EX, EY, BW, BH,
             "Environment",
             r"$f(s,\,a) \;\rightarrow\; s'$",
             fc=CE_L, ec=CE_B, tc=CE)

        # ── Üst ok: Agent → Environment (yukarı kıvrım) ──────────────────────
        #   ok uçları: sağ kenarda biraz yukarı
        arr(AX + BW/2, AY + 0.42,
            EX - BW/2, EY + 0.42,
            rad=0.40, color=ARR)

        ax.text(6.0, 5.95,
                r"Action   $a_t$",
                ha="center", va="bottom", fontsize=14,
                color=ARR, fontweight="bold")

        # ── Alt oklar: Environment → Agent (aşağı kıvrım) ───────────────────
        #   gözlem: biraz yukarı başlangıç noktası
        arr(EX - BW/2, EY - 0.28,
            AX + BW/2, AY - 0.28,
            rad=0.32, color=ARR)
        #   ödül: daha aşağı başlangıç noktası + daha derin kıvrım
        arr(EX - BW/2, EY - 0.58,
            AX + BW/2, AY - 0.58,
            rad=0.50, color=ARR2)

        ax.text(6.0, 3.05,
                r"Observation   $o_{t+1}$",
                ha="center", va="top", fontsize=14,
                color=ARR, fontweight="bold")
        ax.text(6.0, 2.52,
                r"Reward   $r_{t+1}$",
                ha="center", va="top", fontsize=14,
                color=ARR2, fontweight="bold")

        # ── Başlık ───────────────────────────────────────────────────────────
        ax.text(6.0, 6.78,
                "Markov Decision Process — Agent–Environment Interaction",
                ha="center", va="top", fontsize=16.5,
                fontweight="bold", color="#1F2937")

        # ── Timeline şeridi ──────────────────────────────────────────────────
        TY  = 1.62           # çizgi y
        TX0 = 1.20           # başlangıç
        TX1 = 11.20          # bitiş

        # Arka plan şeridi
        ax.add_patch(FancyBboxPatch(
            (TX0 - 0.2, TY - 0.55), TX1 - TX0 + 0.4, 1.10,
            boxstyle="round,pad=0.05", lw=1.0,
            facecolor=CN_L, edgecolor="#D1D5DB", zorder=1))

        # Yatay çizgi
        ax.plot([TX0, TX1], [TY, TY], color=CN, lw=1.8, zorder=2)
        # Yön oku
        arr(TX1 - 0.1, TY, TX1 + 0.35, TY,
            rad=0.0, color=CN, lw=1.5, ms=14)

        # Tick'ler ve etiketler
        ticks = [
            (2.20,  "$t$",   r"$(s_t,\; a_t)$"),
            (4.85,  "$t{+}1$", r"$(r_{t+1},\; s_{t+1},\; a_{t+1})$"),
            (7.50,  "$t{+}2$", r"$(r_{t+2},\; s_{t+2},\; a_{t+2})$"),
            (10.15, "$t{+}3$", r"$(r_{t+3},\; \cdots)$"),
        ]
        for tx, tlbl, tup in ticks:
            ax.plot([tx, tx], [TY - 0.14, TY + 0.14],
                    color=CN, lw=1.8, zorder=3)
            ax.text(tx, TY + 0.28, tlbl,
                    ha="center", va="bottom", fontsize=13,
                    color=ARR, fontweight="bold")
            ax.text(tx, TY - 0.28, tup,
                    ha="center", va="top", fontsize=9,
                    color=CN, style="italic")

        ax.text(TX0 - 0.55, TY, "Time",
                ha="center", va="center", fontsize=10,
                color=CN, fontweight="bold", style="italic",
                rotation=90)

        # ── Alt açıklamalar ───────────────────────────────────────────────────
        ax.text(0.55, 0.38,
                r"Markov property:   $P(s_{t+1} \mid s_t,\; a_t)$",
                ha="left", va="center", fontsize=10.5,
                color=CN, style="italic")
        ax.text(11.45, 0.38,
                r"Objective:   maximize   $\mathbb{E}\!\left[\,\sum_t \gamma^t r_t\right]$",
                ha="right", va="center", fontsize=10.5,
                color=CN, style="italic")

        out = FIG_DIR / "mdp_interaction_loop.png"
        fig.savefig(out, dpi=200, facecolor="white")
        plt.close(fig)
        print(f"[OK] {out}")


# =============================================================================
# Figure 5 — Actor-Critic Architecture
# =============================================================================

def make_actor_critic_figure():
    """
    Actor-Critic Architecture — MAPPO Policy Network.
    2600×1600 px @ 200 DPI (figsize 13×8 in, no bbox crop).
    """
    # ── Renk sabit ────────────────────────────────────────────────────────────
    C_ACTOR   = "#2B6CB0"   # koyu mavi — actor tower
    C_CRITIC  = "#C05621"   # mercan — critic tower
    C_LACTOR  = "#BEE3F8"   # açık mavi
    C_LCRIT   = "#FEEBC8"   # açık turuncu
    C_INPUT   = "#E2E8F0"   # açık gri — input kutu
    C_INBDR   = "#4A5568"   # koyu gri kenarlık
    C_OUT_A   = "#276749"   # yeşil — actor çıktı
    C_LOUT_A  = "#C6F6D5"   # açık yeşil
    C_OUT_C   = "#B7791F"   # amber — critic çıktı
    C_LOUT_C  = "#FEFCE8"   # açık sarı
    C_LGRAD_A = "#DBEAFE"   # gradient kutu arka plan — actor
    C_LGRAD_C = "#FFEDD5"   # gradient kutu arka plan — critic
    ARR       = "#2D3748"
    GR        = "#718096"

    W_IN, H_IN, DPI = 13.0, 8.0, 200

    fig, ax = plt.subplots(figsize=(W_IN, H_IN))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    ax.set_facecolor("white")
    ax.set_xlim(0.0, 13.0)
    ax.set_ylim(0.0,  8.0)
    ax.axis("off")

    # ── Yerel yardımcılar ─────────────────────────────────────────────────────
    def B(cx, cy, w, h, text, fc, ec, tc="white",
          fs=10.5, bold=True, lw=1.6, dashed=False):
        ls = "--" if dashed else "-"
        r = FancyBboxPatch(
            (cx - w/2, cy - h/2), w, h,
            boxstyle="round,pad=0.04", linewidth=lw,
            facecolor=fc, edgecolor=ec, linestyle=ls, zorder=3)
        ax.add_patch(r)
        ax.text(cx, cy, text, ha="center", va="center",
                fontsize=fs, color=tc,
                fontweight="bold" if bold else "normal",
                multialignment="center", zorder=4)

    def AR(x0, y0, x1, y1, color=ARR, lw=1.5, dashed=False):
        ls = "--" if dashed else "-"
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(
                        arrowstyle="->", color=color, lw=lw,
                        linestyle=ls,
                        connectionstyle="arc3,rad=0"), zorder=2)

    # ── Layout ────────────────────────────────────────────────────────────────
    CX_L = 3.40    # actor tower x
    CX_R = 9.60    # critic tower x
    CX_M = 6.50    # orta
    BW   = 3.00    # kutu genişliği
    BH   = 0.62    # kutu yüksekliği

    Y_TITLE  = 7.68
    Y_TH_LBL = 7.15   # tower başlığı
    Y_INPUT  = 6.62
    Y_SPLIT  = 6.28   # split noktası
    Y_ENC    = 5.55   # MLP Encoder
    Y_HEAD   = 4.30   # Action / Value Head
    Y_OUT    = 3.10   # output kutu
    Y_OUTLAB = 2.65   # label altı
    Y_SEP    = 2.28   # separator çizgi
    Y_GRAD   = 1.50   # gradient kutu
    GBH      = 0.70   # gradient kutu yüksekliği
    GBW      = 3.60   # gradient kutu genişliği
    Y_NOTE   = 0.16   # alt not

    # ── Başlık ────────────────────────────────────────────────────────────────
    ax.text(CX_M, Y_TITLE,
            "Actor-Critic Architecture — MAPPO Policy Network",
            ha="center", va="center", fontsize=14.5,
            fontweight="bold", color="#1F2937")

    # ── Tower başlıkları ──────────────────────────────────────────────────────
    ax.text(CX_L, Y_TH_LBL,
            r"Actor   $\pi(a \mid o,\;\theta_\pi)$",
            ha="center", va="center", fontsize=12.5,
            fontweight="bold", color=C_ACTOR)
    ax.text(CX_R, Y_TH_LBL,
            r"Critic   $V(s,\;\theta_V)$",
            ha="center", va="center", fontsize=12.5,
            fontweight="bold", color=C_CRITIC)

    # ── Ortak giriş kutusu ────────────────────────────────────────────────────
    B(CX_M, Y_INPUT, 3.20, BH,
      r"Observation   $o_t$",
      fc=C_INPUT, ec=C_INBDR, tc=C_INBDR, fs=12.0, bold=True, lw=1.8)

    # Aşağı ok → split noktasına
    ax.plot([CX_M, CX_M], [Y_INPUT - BH/2, Y_SPLIT],
            color=ARR, lw=1.5, zorder=2)

    # Sol dal → Actor MLP
    ax.annotate("", xy=(CX_L, Y_ENC + BH/2 + 0.06),
                xytext=(CX_M, Y_SPLIT),
                arrowprops=dict(arrowstyle="->", color=ARR, lw=1.5,
                                connectionstyle="arc3,rad=0"), zorder=2)

    # Sağ dal → Critic MLP
    ax.annotate("", xy=(CX_R, Y_ENC + BH/2 + 0.06),
                xytext=(CX_M, Y_SPLIT),
                arrowprops=dict(arrowstyle="->", color=ARR, lw=1.5,
                                connectionstyle="arc3,rad=0"), zorder=2)

    # ── Actor Tower ───────────────────────────────────────────────────────────
    B(CX_L, Y_ENC, BW, BH,
      "MLP Encoder\n[256, 256]",
      fc=C_ACTOR, ec=C_ACTOR, fs=10.5, bold=True)

    AR(CX_L, Y_ENC - BH/2, CX_L, Y_HEAD + BH/2 + 0.05)

    B(CX_L, Y_HEAD, BW, BH,
      "Action Head",
      fc=C_ACTOR, ec=C_ACTOR, fs=10.5, bold=True)

    AR(CX_L, Y_HEAD - BH/2, CX_L, Y_OUT + BH/2 + 0.05)

    # Actor çıktı (yeşil, kesik kenarlık)
    B(CX_L, Y_OUT, BW + 0.30, BH,
      r"$\Delta\psi,\;\Delta\gamma,\;\Delta n,\;"
      r"\mathrm{fire}\!\in\!\{0,1\}$",
      fc=C_LOUT_A, ec=C_OUT_A, tc=C_OUT_A,
      fs=9.5, bold=False, lw=2.0, dashed=True)

    ax.text(CX_L, Y_OUTLAB,
            "Continuous + Discrete Actions",
            ha="center", va="top", fontsize=9.0,
            color=C_OUT_A, style="italic")

    # ── Critic Tower ──────────────────────────────────────────────────────────
    B(CX_R, Y_ENC, BW, BH,
      "MLP Encoder\n[256, 256]",
      fc=C_CRITIC, ec=C_CRITIC, fs=10.5, bold=True)

    AR(CX_R, Y_ENC - BH/2, CX_R, Y_HEAD + BH/2 + 0.05)

    B(CX_R, Y_HEAD, BW, BH,
      "Value Head",
      fc=C_CRITIC, ec=C_CRITIC, fs=10.5, bold=True)

    AR(CX_R, Y_HEAD - BH/2, CX_R, Y_OUT + BH/2 + 0.05)

    # Critic çıktı (amber, kesik kenarlık)
    B(CX_R, Y_OUT, BW, BH,
      r"$V(s) \in \mathbb{R}$",
      fc=C_LOUT_C, ec=C_OUT_C, tc=C_OUT_C,
      fs=11.0, bold=False, lw=2.0, dashed=True)

    ax.text(CX_R, Y_OUTLAB,
            "Baseline for Variance Reduction",
            ha="center", va="top", fontsize=9.0,
            color=C_OUT_C, style="italic")

    # ── Shared Backbone annotation ─────────────────────────────────────────────
    Y_SB    = (Y_ENC + Y_HEAD) / 2          # iki kutu arasında
    x_left  = CX_L + BW/2 + 0.20
    x_right = CX_R - BW/2 - 0.20
    ax.plot([x_left, x_right], [Y_SB, Y_SB],
            color=GR, lw=1.0, linestyle="--", zorder=1, alpha=0.65)
    ax.text(CX_M, Y_SB + 0.09,
            "Shared Backbone (optional)",
            ha="center", va="bottom", fontsize=8.5,
            color=GR, style="italic")

    # ── Separator + "Gradient Flow" etiket ────────────────────────────────────
    ax.plot([0.55, 12.45], [Y_SEP, Y_SEP],
            color="#CBD5E0", lw=1.1, linestyle="--", zorder=1)
    ax.text(CX_M, Y_SEP - 0.06,
            "Gradient Flow",
            ha="center", va="top", fontsize=8.5,
            color=GR, style="italic", fontweight="bold")

    # ── Policy Gradient kutu ──────────────────────────────────────────────────
    B(CX_L, Y_GRAD, GBW, GBH,
      "Policy Gradient\n"
      r"$\nabla_\theta J = \mathbb{E}[\nabla \log \pi \cdot A(s,a)]$",
      fc=C_LGRAD_A, ec=C_ACTOR, tc=C_ACTOR,
      fs=8.5, bold=False, lw=1.5)

    # ── Value Loss kutu ───────────────────────────────────────────────────────
    B(CX_R, Y_GRAD, GBW, GBH,
      "Value Loss\n"
      r"$\mathcal{L}_V = (V(s) - V_{\mathrm{tar}})^2$",
      fc=C_LGRAD_C, ec=C_CRITIC, tc=C_CRITIC,
      fs=8.5, bold=False, lw=1.5)

    # Backprop okları (kesik, yukarı)
    AR(CX_L, Y_GRAD + GBH/2 + 0.06,
       CX_L, Y_OUT  - BH/2  - 0.06,
       color=C_ACTOR, lw=1.3, dashed=True)
    AR(CX_R, Y_GRAD + GBH/2 + 0.06,
       CX_R, Y_OUT  - BH/2  - 0.06,
       color=C_CRITIC, lw=1.3, dashed=True)

    # "backprop" etiketler
    bp_y = (Y_GRAD + GBH/2 + Y_OUT - BH/2) / 2
    ax.text(CX_L - 0.20, bp_y,
            "backprop", ha="right", va="center",
            fontsize=8.0, color=C_ACTOR, style="italic")
    ax.text(CX_R + 0.20, bp_y,
            "backprop", ha="left", va="center",
            fontsize=8.0, color=C_CRITIC, style="italic")

    # ── Alt not ───────────────────────────────────────────────────────────────
    ax.text(CX_M, Y_NOTE,
            r"Advantage   $A(s,a) = r + \gamma V(s') - V(s)$"
            r"   decouples policy improvement from value estimation",
            ha="center", va="bottom", fontsize=9.0,
            color=GR, style="italic")

    # ── Kaydet ────────────────────────────────────────────────────────────────
    out = FIG_DIR / "actor_critic_architecture.png"
    fig.savefig(out, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"[OK] {out}")


# =============================================================================
# Figure 6 — ENU Coordinate Frame (3D) + Euler angle arcs
# =============================================================================

def make_enu_figure():
    """
    ENU Coordinate Frame — iki panelli düzen.
    SOL (60%): 3D ENU eksenleri, temiz, uçaksız.
    SAĞ (40%): 2D Euler açı diyagramı (yan görünüm uçak + 3 yay).
    2400×1200 px @ 200 DPI (figsize 12×6 in, bbox_inches yok).
    """
    from mpl_toolkits.mplot3d import Axes3D           # noqa: F401
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches

    # ── Renkler ───────────────────────────────────────────────────────────────
    C_E     = "#E8896A"    # East  — mercan
    C_N     = "#2C5F8A"    # North — koyu mavi
    C_U     = "#4CAF50"    # Up    — yeşil
    C_PHI   = "#D32F2F"    # φ roll  — kırmızı
    C_THETA = "#2E7D32"    # θ pitch — koyu yeşil
    C_PSI   = "#1565C0"    # ψ yaw   — koyu mavi
    GR      = "#718096"
    DARK    = "#1F2937"

    W_IN, H_IN, DPI = 12.0, 6.0, 200    # → 2400×1200 px

    with plt.rc_context({"font.family": "sans-serif",
                         "font.sans-serif": ["DejaVu Sans", "Arial", "sans-serif"],
                         "font.size": 10}):

        fig = plt.figure(figsize=(W_IN, H_IN))
        fig.patch.set_facecolor("white")
        # Üst %8 başlık, alt %14 alt not için boşluk bırak
        fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.15,
                            wspace=0.04)

        gs = gridspec.GridSpec(1, 2, width_ratios=[5.5, 4.5], figure=fig)

        # ══════════════════════════════════════════════════════════════════════
        # SOL PANEL — 3D ENU eksenleri
        # ══════════════════════════════════════════════════════════════════════
        ax3 = fig.add_subplot(gs[0], projection="3d")
        ax3.set_facecolor("white")

        L  = 0.72    # eksen uzunluğu
        AR = 0.11    # ok başı oranı

        # ENU eksen okları (quiver)
        ax3.quiver(0, 0, 0, L, 0, 0, color=C_E, lw=3.2, arrow_length_ratio=AR)
        ax3.quiver(0, 0, 0, 0, L, 0, color=C_N, lw=3.2, arrow_length_ratio=AR)
        ax3.quiver(0, 0, 0, 0, 0, L, color=C_U, lw=3.2, arrow_length_ratio=AR)

        # Eksen etiketleri — ok ucundan 0.14 offset
        ax3.text(L + 0.14, 0.00, 0.00, "E  (East)",
                 color=C_E, fontsize=13, fontweight="bold",
                 ha="center", va="center")
        ax3.text(0.00, L + 0.14, 0.00, "N  (North)",
                 color=C_N, fontsize=13, fontweight="bold",
                 ha="center", va="center")
        ax3.text(0.00, 0.00, L + 0.14, "U  (Up)",
                 color=C_U, fontsize=13, fontweight="bold",
                 ha="center", va="center")

        # Orijin noktası + etiketi
        ax3.scatter([0], [0], [0], color=DARK, s=65, zorder=10)
        ax3.text(0.05, -0.09, -0.05, "Origin",
                 color=DARK, fontsize=9, ha="left", va="top")

        # XY zemin düzlemi ızgara (hafif gri kesikli)
        g_vals = np.linspace(0, 0.65, 5)
        for gv in g_vals:
            ax3.plot([gv, gv], [0.0, 0.65], [0, 0],
                     color="#D0D0D0", lw=0.7, linestyle="--", alpha=0.55)
            ax3.plot([0.0, 0.65], [gv, gv], [0, 0],
                     color="#D0D0D0", lw=0.7, linestyle="--", alpha=0.55)

        ax3.set_xlim(0, 1.0)
        ax3.set_ylim(0, 1.0)
        ax3.set_zlim(0, 1.0)
        ax3.set_axis_off()
        ax3.view_init(elev=22, azim=218)
        ax3.dist = 9.0

        # Açıklama kutusu — 3D eksen koordinatlarıyla (text2D)
        annot = ("ENU Convention\n"
                 "• Right-handed inertial frame\n"
                 "• x → East,  y → North,  z → Up\n"
                 "• Unity-compatible\n"
                 r"• RK4  $\Delta t = 0.05\,$s  (20 Hz)")
        ax3.text2D(0.97, 0.97, annot,
                   transform=ax3.transAxes,
                   fontsize=8.5, va="top", ha="right",
                   color=DARK,
                   bbox=dict(boxstyle="round,pad=0.5",
                              fc="#F7FAFC", ec="#CBD5E0", lw=1.2))

        # ══════════════════════════════════════════════════════════════════════
        # SAĞ PANEL — 2D Euler açı diyagramı
        # ══════════════════════════════════════════════════════════════════════
        ax2 = fig.add_subplot(gs[1])
        ax2.set_facecolor("#F8F8F8")
        ax2.set_xlim(0.0, 11.0)
        ax2.set_ylim(0.0, 10.0)
        ax2.axis("off")
        ax2.set_title("Euler Angle Convention  (ZYX sequence)",
                      fontsize=10.5, fontweight="bold", color=DARK, pad=5)

        # ── Uçak gövdesi (yan görünüm, burun sağda) ───────────────────────────
        CX, CY = 5.0, 5.60   # gövde merkezi

        # Gövde (konik beşgen, burun sağda)
        fuse_x = [2.40, 7.45, 7.85, 7.45, 2.40]
        fuse_y = [CY-0.22, CY-0.14, CY, CY+0.14, CY+0.22]
        ax2.fill(fuse_x, fuse_y, color="#4A5568", alpha=0.82, zorder=3)
        ax2.plot(fuse_x + [fuse_x[0]], fuse_y + [fuse_y[0]],
                 color=DARK, lw=1.0, zorder=4)

        # Ana kanat (yukarı sweep)
        wing_x = [4.50, 3.80, 5.80, 4.50]
        wing_y = [CY+0.22, CY+2.00, CY+0.22, CY+0.22]
        ax2.fill(wing_x, wing_y, color="#4A5568", alpha=0.78, zorder=3)
        ax2.plot(wing_x + [wing_x[0]], wing_y + [wing_y[0]],
                 color=DARK, lw=0.9, zorder=4)

        # Yatay stabilizer (kuyruk)
        htail_x = [2.40, 2.10, 3.15, 2.40]
        htail_y = [CY+0.22, CY+1.00, CY+0.22, CY+0.22]
        ax2.fill(htail_x, htail_y, color="#718096", alpha=0.70, zorder=3)
        ax2.plot(htail_x + [htail_x[0]], htail_y + [htail_y[0]],
                 color=DARK, lw=0.7, zorder=4)

        # Gövde eksenleri (x_b, z_b) — kısa referans okları
        bax_ox, bax_oy = 5.80, CY
        ba_len = 0.72
        ax2.annotate("", xy=(bax_ox + ba_len, bax_oy),
                     xytext=(bax_ox, bax_oy),
                     arrowprops=dict(arrowstyle="-|>", color=C_E,
                                     lw=1.8, mutation_scale=11), zorder=6)
        ax2.text(bax_ox + ba_len + 0.14, bax_oy + 0.12,
                 r"$x_b$", color=C_E, fontsize=10.5,
                 fontweight="bold", zorder=7)
        ax2.annotate("", xy=(bax_ox, bax_oy - ba_len),
                     xytext=(bax_ox, bax_oy),
                     arrowprops=dict(arrowstyle="-|>", color="#9B4D3E",
                                     lw=1.8, mutation_scale=11), zorder=6)
        ax2.text(bax_ox + 0.12, bax_oy - ba_len - 0.20,
                 r"$z_b$", color="#9B4D3E", fontsize=10.5,
                 fontweight="bold", zorder=7)

        # ── φ (roll) — SOL: gövde ekseni etrafında dönme yayı ────────────────
        #   Uçak gövdesinin solunda, yanal eksen etrafında oluşan yay
        phi_cx, phi_cy = 1.35, CY
        phi_r = 1.00
        phi_arc = mpatches.Arc((phi_cx, phi_cy), 2*phi_r, 2*phi_r,
                                angle=0, theta1=30, theta2=150,
                                color=C_PHI, lw=2.8, zorder=5)
        ax2.add_patch(phi_arc)
        # Ok başı (150° ucunda)
        t_e = np.radians(150)
        phi_tip = (phi_cx + phi_r*np.cos(t_e), phi_cy + phi_r*np.sin(t_e))
        phi_tan = (-np.sin(t_e), np.cos(t_e))
        ax2.annotate("", xy=(phi_tip[0] + phi_tan[0]*0.18,
                              phi_tip[1] + phi_tan[1]*0.18),
                     xytext=phi_tip,
                     arrowprops=dict(arrowstyle="-|>", color=C_PHI,
                                     lw=1.5, mutation_scale=10), zorder=6)
        # Referans yatay kesik çizgi
        ax2.plot([phi_cx, phi_cx + phi_r], [phi_cy, phi_cy],
                 color=C_PHI, lw=1.0, linestyle="--", alpha=0.45, zorder=2)
        ax2.text(phi_cx, phi_cy + phi_r + 0.42,
                 r"$\phi$  roll", color=C_PHI, fontsize=12,
                 fontweight="bold", ha="center", va="bottom", zorder=7)

        # ── θ (pitch) — SAĞ: burun yukarı eğim yayı ──────────────────────────
        #   Burun ucundan başlayan, yataydan θ derecelik açıyı gösteren yay
        th_cx, th_cy = 7.85, CY
        th_r = 1.00
        # Referans yatay kesik çizgi (kanat düzlemi)
        ax2.plot([th_cx, th_cx + th_r], [th_cy, th_cy],
                 color=C_THETA, lw=1.0, linestyle="--", alpha=0.45, zorder=2)
        theta_arc = mpatches.Arc((th_cx, th_cy), 2*th_r, 2*th_r,
                                  angle=0, theta1=0, theta2=62,
                                  color=C_THETA, lw=2.8, zorder=5)
        ax2.add_patch(theta_arc)
        # Ok başı (62° ucunda)
        t_e = np.radians(62)
        th_tip = (th_cx + th_r*np.cos(t_e), th_cy + th_r*np.sin(t_e))
        th_tan = (-np.sin(t_e), np.cos(t_e))
        ax2.annotate("", xy=(th_tip[0] + th_tan[0]*0.18,
                              th_tip[1] + th_tan[1]*0.18),
                     xytext=th_tip,
                     arrowprops=dict(arrowstyle="-|>", color=C_THETA,
                                     lw=1.5, mutation_scale=10), zorder=6)
        ax2.text(th_cx + th_r + 0.28, th_cy + th_r*0.42,
                 r"$\theta$  pitch", color=C_THETA, fontsize=12,
                 fontweight="bold", ha="left", va="center", zorder=7)

        # ── ψ (yaw) — ALT: yatay düzlemde dönme yayı ─────────────────────────
        #   Uçağın altında, dikey eksen etrafındaki sapma açısını gösterir
        psi_cx, psi_cy = 5.0, 2.60
        psi_r = 1.30
        # Kuzey referans oku (yukarı kesik)
        ax2.annotate("", xy=(psi_cx, psi_cy + psi_r + 0.20),
                     xytext=(psi_cx, psi_cy),
                     arrowprops=dict(arrowstyle="-|>", color=GR,
                                     lw=1.0, mutation_scale=8,
                                     linestyle="dashed"), zorder=3)
        ax2.text(psi_cx + 0.12, psi_cy + psi_r + 0.25,
                 "N", color=GR, fontsize=9.5, ha="left", va="bottom")
        psi_arc = mpatches.Arc((psi_cx, psi_cy), 2*psi_r, 2*psi_r,
                                angle=0, theta1=20, theta2=160,
                                color=C_PSI, lw=2.8, zorder=5)
        ax2.add_patch(psi_arc)
        # Ok başı (160° ucunda)
        t_e = np.radians(160)
        psi_tip = (psi_cx + psi_r*np.cos(t_e), psi_cy + psi_r*np.sin(t_e))
        psi_tan = (-np.sin(t_e), np.cos(t_e))
        ax2.annotate("", xy=(psi_tip[0] + psi_tan[0]*0.18,
                              psi_tip[1] + psi_tan[1]*0.18),
                     xytext=psi_tip,
                     arrowprops=dict(arrowstyle="-|>", color=C_PSI,
                                     lw=1.5, mutation_scale=10), zorder=6)
        ax2.text(psi_cx, psi_cy - psi_r - 0.48,
                 r"$\psi$  yaw / heading", color=C_PSI, fontsize=12,
                 fontweight="bold", ha="center", va="top", zorder=7)

        # ── Tam genişlik başlık ───────────────────────────────────────────────
        fig.text(0.50, 0.97,
                 "East-North-Up (ENU) Inertial Reference Frame",
                 ha="center", va="top", fontsize=14,
                 fontweight="bold", color=DARK)

        # ── Alt notlar ────────────────────────────────────────────────────────
        fig.text(
            0.50, 0.09,
            r"State vector:  $\mathbf{x}_t = "
            r"[x,\; y,\; h,\; V,\; \alpha,\; \beta,\; \gamma,\; "
            r"\phi,\; \theta,\; \psi,\; p,\; q,\; r,\; "
            r"m_f,\; m_a,\; HP,\; r_d,\; a]^\top \in \mathbb{R}^{18}$",
            ha="center", va="bottom", fontsize=10,
            style="italic", color=GR)
        fig.text(
            0.50, 0.03,
            r"Body$\!\to\!$Inertial rotation:  "
            r"$R(\phi,\;\theta,\;\psi)$ via ZYX Euler sequence",
            ha="center", va="bottom", fontsize=10,
            style="italic", color=GR)

        # ── Kaydet ────────────────────────────────────────────────────────────
        out = FIG_DIR / "enu_coordinate_frame.png"
        fig.savefig(out, dpi=DPI, facecolor="white")
        plt.close(fig)

    print(f"[OK] {out}")


# =============================================================================
# Figure 7 — Observation & Action Space Mapping
# =============================================================================

def make_action_state_figure():
    """
    Observation and Action Space Definition — two-column mapping diagram.
    2800×1600 px @ 200 DPI (figsize 14×8 in, no bbox crop).
    """
    # ── Renkler ───────────────────────────────────────────────────────────────
    C_L   = "#2C5F8A"   # koyu mavi — obs sütunu
    C_LL  = "#DBEAFE"   # açık mavi dolgu
    C_R   = "#E8896A"   # mercan — action sütunu
    C_LR  = "#FDE8D8"   # açık mercan dolgu
    C_G4F = "#F3F4F6"   # padding grubu dolgu
    C_G4E = "#9CA3AF"   # padding grubu kenarlık
    GR    = "#718096"
    DARK  = "#1F2937"

    W_IN, H_IN, DPI = 14.0, 8.0, 200

    fig, ax = plt.subplots(figsize=(W_IN, H_IN))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    ax.set_facecolor("white")
    ax.set_xlim(0.0, 14.0)
    ax.set_ylim(0.0,  8.0)
    ax.axis("off")

    # ── Geometri sabitler ─────────────────────────────────────────────────────
    BH   = 0.22   # kutu yüksekliği
    SLOT = 0.28   # dikey adım (box + gap)

    # Sol sütun
    SBX  = 0.22   # sembol kutu sol kenar x
    SBW  = 1.85   # sembol kutu genişliği
    SBC  = SBX + SBW / 2          # sembol kutu merkez x
    DSX  = SBX + SBW + 0.32       # açıklama text başlangıç x

    # Sağ sütun
    RSX  = 8.55   # sağ sembol kutu sol kenar x
    RSBW = 2.05   # sağ sembol kutu genişliği
    RSBC = RSX + RSBW / 2         # sağ merkez x
    RDSX = RSX + RSBW + 0.30      # sağ açıklama başlangıç x

    # Merkez
    X_SEP = 7.00
    ABW, ABH = 2.20, 1.48
    ABY = 4.15    # ajan kutu merkez y

    # Sütun arka plan panelleri (çok açık)
    ax.add_patch(FancyBboxPatch(
        (0.10, 1.08), 5.15, 6.15,
        boxstyle="round,pad=0.04", lw=0.6,
        facecolor=C_LL, edgecolor=C_L, alpha=0.18, zorder=0))
    ax.add_patch(FancyBboxPatch(
        (8.45, 1.95), 5.40, 5.28,
        boxstyle="round,pad=0.04", lw=0.6,
        facecolor=C_LR, edgecolor=C_R, alpha=0.20, zorder=0))

    # ── Yerel yardımcılar ─────────────────────────────────────────────────────
    def obs_row(cy, sym, desc, dashed=False):
        fc = C_G4F if dashed else C_LL
        ec = C_G4E if dashed else C_L
        tc = GR    if dashed else C_L
        ls = "--"  if dashed else "-"
        ax.add_patch(FancyBboxPatch(
            (SBX, cy - BH/2), SBW, BH,
            boxstyle="round,pad=0.025", lw=1.2,
            facecolor=fc, edgecolor=ec, linestyle=ls, zorder=3))
        ax.text(SBC, cy, sym, ha="center", va="center",
                fontsize=8.0, fontweight="bold", color=tc, zorder=4)
        ax.annotate("", xy=(DSX - 0.14, cy), xytext=(SBX + SBW + 0.04, cy),
                    arrowprops=dict(arrowstyle="->", color=ec, lw=1.0,
                                    connectionstyle="arc3,rad=0"), zorder=2)
        ax.text(DSX, cy, desc,
                ha="left", va="center", fontsize=7.8, color=DARK, zorder=4)

    def act_row(cy, sym, desc):
        ax.add_patch(FancyBboxPatch(
            (RSX, cy - BH/2), RSBW, BH,
            boxstyle="round,pad=0.025", lw=1.2,
            facecolor=C_LR, edgecolor=C_R, zorder=3))
        ax.text(RSBC, cy, sym, ha="center", va="center",
                fontsize=8.2, fontweight="bold", color=C_R, zorder=4)
        ax.annotate("", xy=(RDSX - 0.12, cy), xytext=(RSX + RSBW + 0.04, cy),
                    arrowprops=dict(arrowstyle="->", color=C_R, lw=1.0,
                                    connectionstyle="arc3,rad=0"), zorder=2)
        ax.text(RDSX, cy, desc,
                ha="left", va="center", fontsize=7.8, color=DARK, zorder=4)

    def grp_hdr(cy, text, color):
        ax.add_patch(FancyBboxPatch(
            (SBX, cy - 0.13), 5.15, 0.26,
            boxstyle="round,pad=0.01", lw=0.7,
            facecolor="white", edgecolor=color, alpha=0.90, zorder=2))
        ax.text(SBX + 0.10, cy, text,
                ha="left", va="center", fontsize=7.6,
                fontweight="bold", color=color, zorder=4)

    # ── Başlık ────────────────────────────────────────────────────────────────
    ax.text(7.0, 7.72, "Observation and Action Space Definition",
            ha="center", va="center", fontsize=14.0,
            fontweight="bold", color=DARK)

    # ── Sütun başlıkları ──────────────────────────────────────────────────────
    ax.text(2.85, 7.38,
            r"Observation Space   $\mathbf{o}_i \in \mathbb{R}^{50}$",
            ha="center", va="center", fontsize=11.5,
            fontweight="bold", color=C_L)
    ax.text(11.15, 7.38,
            r"Action Space   $\mathbf{a}_i \in \mathbb{R}^{4} \times \{0,1\}$",
            ha="center", va="center", fontsize=11.5,
            fontweight="bold", color=C_R)

    # ── Merkez kesik çizgi ────────────────────────────────────────────────────
    ax.plot([X_SEP, X_SEP], [7.18, 1.05],
            color="#CBD5E0", lw=1.3, linestyle="--", zorder=1)

    # ── Ajan kutusu ───────────────────────────────────────────────────────────
    ax.add_patch(FancyBboxPatch(
        (X_SEP - ABW/2, ABY - ABH/2), ABW, ABH,
        boxstyle="round,pad=0.08", lw=1.8,
        facecolor="#F7FAFC", edgecolor="#4A5568", zorder=4))
    ax.text(X_SEP, ABY + 0.30,
            r"Agent   $\pi_i(a_i \mid o_i)$",
            ha="center", va="center", fontsize=10.5,
            fontweight="bold", color=DARK, zorder=5)
    ax.text(X_SEP, ABY - 0.25,
            "MLP  [256, 256]",
            ha="center", va="center", fontsize=9.5,
            color=GR, zorder=5)

    # Konsept okları (obs → ajan → action)
    LEFT_EDGE  = SBX + SBW + 0.06 + 0.30  # obs kutu sağ kenarı üzeri
    RIGHT_EDGE = RSX - 0.08               # action kutu sol kenarı üzeri

    ax.annotate("", xy=(X_SEP - ABW/2 - 0.06, ABY),
                xytext=(5.40, ABY),
                arrowprops=dict(arrowstyle="->", color=C_L, lw=1.8,
                                connectionstyle="arc3,rad=0"), zorder=3)
    ax.text((5.40 + X_SEP - ABW/2)/2, ABY + 0.14,
            r"$\mathbf{o}_i$", ha="center", va="bottom",
            fontsize=10, color=C_L, fontweight="bold")

    ax.annotate("", xy=(RIGHT_EDGE, ABY),
                xytext=(X_SEP + ABW/2 + 0.06, ABY),
                arrowprops=dict(arrowstyle="->", color=C_R, lw=1.8,
                                connectionstyle="arc3,rad=0"), zorder=3)
    ax.text((X_SEP + ABW/2 + RIGHT_EDGE)/2, ABY + 0.14,
            r"$\mathbf{a}_i$", ha="center", va="bottom",
            fontsize=10, color=C_R, fontweight="bold")

    # ── Sol sütun items (cursor tabanlı yerleşim) ─────────────────────────────
    cursor = 7.07

    # Grup 1: OWN STATE — 18D
    grp_hdr(cursor - 0.13, "OWN STATE — 18D", C_L)
    cursor -= 0.30

    for sym, desc in [
        ("x, y, h",       "Position in ENU  [m]"),
        ("V",             "True airspeed  [m/s]"),
        ("α, β",          "AoA, sideslip  [rad]"),
        ("γ",             "Flight path angle  [rad]"),
        ("φ, θ, ψ",       "Roll, pitch, yaw  [rad]"),
        ("p, q, r",       "Body angular rates  [rad/s]"),
        ("m_f,  m_a",     "Fuel,  ammunition"),
        ("HP,  r_d,  a",  "Hit pts,  radar lock,  alive"),
    ]:
        obs_row(cursor - BH/2, sym, desc)
        cursor -= SLOT

    # Grup 2: RELATIVE TO ENEMY — 6D
    cursor -= 0.10
    grp_hdr(cursor - 0.13, "RELATIVE TO ENEMY — 6D", C_L)
    cursor -= 0.30

    for sym, desc in [
        ("d_norm",         "Normalised distance"),
        ("bearing_norm",   "Normalised bearing"),
        ("alt_diff_norm",  "Normalised alt. diff."),
        ("ATA_norm",       "Antenna train angle"),
        ("in_wez",         r"WEZ flag  $\in\{0,1\}$"),
        ("enemy_V_norm",   "Enemy speed  (norm.)"),
    ]:
        obs_row(cursor - BH/2, sym, desc)
        cursor -= SLOT

    # Grup 3: TEAMMATE — 6D
    cursor -= 0.10
    grp_hdr(cursor - 0.13, "TEAMMATE — 6D", C_L)
    cursor -= 0.30
    obs_row(cursor - BH/2, "teammate  (6D)", "Relative kinematics")
    cursor -= SLOT

    # Grup 4: PADDING / EXTRAS — 20D
    cursor -= 0.10
    ax.add_patch(FancyBboxPatch(
        (SBX, cursor - 0.13 - 0.13), 5.15, 0.26,
        boxstyle="round,pad=0.01", lw=0.7, linestyle="--",
        facecolor="white", edgecolor=C_G4E, alpha=0.90, zorder=2))
    ax.text(SBX + 0.10, cursor - 0.13,
            "PADDING / EXTRAS — 20D",
            ha="left", va="center", fontsize=7.6,
            fontweight="bold", color=C_G4E, zorder=4)
    cursor -= 0.30
    obs_row(cursor - BH/2, "Additional features",
            "Normalised  →  50D total", dashed=True)
    cursor -= SLOT

    # Dim sayacı
    ax.text(SBX, cursor - 0.05,
            "=  50 dims total",
            ha="left", va="top", fontsize=8.5,
            fontweight="bold", color=C_L)

    # ── Sağ sütun items ───────────────────────────────────────────────────────
    y_acts = np.linspace(6.50, 2.10, 5)
    for (sym, desc), cy in zip([
        (r"$\delta_a \in [-1,\,1]$",  "Aileron command"),
        (r"$\delta_e \in [-1,\,1]$",  "Elevator command"),
        (r"$\delta_r \in [-1,\,1]$",  "Rudder command"),
        (r"$\delta_t \in [0,\,1]$",   "Throttle  (sigmoid)"),
        (r"$\delta_f \in \{0,1\}$",   "Fire  (Bernoulli head)"),
    ], y_acts):
        act_row(cy, sym, desc)

    # ── Alt açıklama kutuları ─────────────────────────────────────────────────
    BT_Y = 0.52
    BT_H = 0.72
    BT_W = 4.18

    for cx, fc, ec, text in [
        (2.35, "#DBEAFE", C_L,
         r"Clamp$(−1,1)$ on $\delta_a,\,\delta_e,\,\delta_r$" + "\n"
         "avoids tanh saturation"),
        (7.00, "#FFEDD5", C_R,
         r"Bernoulli head:  $P(\mathrm{fire}) = \sigma(\mathrm{logit})$" + "\n"
         "bias₀ = −0.85"),
        (11.65, "#E2E8F0", GR,
         r"$\delta_t$: sigmoid $\to$ [0, 1]" + "\n"
         "QMIX: discretises to 162 joint actions"),
    ]:
        ax.add_patch(FancyBboxPatch(
            (cx - BT_W/2, BT_Y - BT_H/2), BT_W, BT_H,
            boxstyle="round,pad=0.06", lw=1.2,
            facecolor=fc, edgecolor=ec, zorder=3))
        ax.text(cx, BT_Y, text,
                ha="center", va="center", fontsize=8.0,
                color=DARK, multialignment="center", zorder=4)

    # ── Kaydet ────────────────────────────────────────────────────────────────
    out = FIG_DIR / "action_state_mapping.png"
    fig.savefig(out, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"[OK] {out}")


# =============================================================================
# Figure 8 — WEZ (Weapon Engagement Zone) Geometry
# =============================================================================

def make_wez_figure():
    """
    WEZ Geometry — yukarıdan bakış 2D taktik görünüm.
    2200×1800 px @ 200 DPI (figsize 11×9 in).
    xlim=11, ylim=9 → her iki eksende doğal 200 px/birim (1:1 piksel).
    """
    # ── Renkler ───────────────────────────────────────────────────────────────
    C_B  = "#2C5F8A"   # Blue ajan
    C_LB = "#63B3ED"   # WEZ koni dolgu
    C_R  = "#C53030"   # Red ajan
    GR   = "#718096"
    DARK = "#1F2937"
    BG   = "#F5F5F5"

    W_IN, H_IN, DPI = 11.0, 9.0, 200

    fig, ax = plt.subplots(figsize=(W_IN, H_IN))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    ax.set_facecolor(BG)
    ax.set_xlim(0.0, 11.0)
    ax.set_ylim(0.0,  9.0)
    ax.axis("off")

    # ── Geometri ──────────────────────────────────────────────────────────────
    BX, BY   = 2.5, 4.0          # Blue konumu
    RX, RY   = 6.8, 6.3          # Red konumu
    PSI_B    = 50.0               # Blue başlık [°]
    PSI_BR   = np.radians(PSI_B)
    RED_PSI  = 240.0              # Red başlık [°]
    RED_PSIR = np.radians(RED_PSI)
    HALF     = 25.0               # WEZ yarı açısı [°]
    R_MAX    = 5.0
    R_MIN    = 0.55

    hx, hy   = np.cos(PSI_BR), np.sin(PSI_BR)
    LOS_deg  = np.degrees(np.arctan2(RY - BY, RX - BX))   # ≈ 28.2°
    R2B_deg  = np.degrees(np.arctan2(BY - RY, BX - RX))   # ≈ 208.2°

    # ── WEZ koni dolgu ────────────────────────────────────────────────────────
    t_fill = np.linspace(np.radians(PSI_B - HALF),
                          np.radians(PSI_B + HALF), 120)
    ax.fill(np.concatenate([[BX], BX + R_MAX * np.cos(t_fill), [BX]]),
            np.concatenate([[BY], BY + R_MAX * np.sin(t_fill), [BY]]),
            color=C_LB, alpha=0.18, zorder=1)

    # Koni kenar çizgileri (katı)
    for sgn in (-1, 1):
        ang = np.radians(PSI_B + sgn * HALF)
        ax.plot([BX, BX + R_MAX * np.cos(ang)],
                [BY, BY + R_MAX * np.sin(ang)],
                color=C_B, lw=1.6, zorder=2)

    # R_max yay (kesik)
    t_rm = np.linspace(np.radians(PSI_B - HALF), np.radians(PSI_B + HALF), 120)
    ax.plot(BX + R_MAX * np.cos(t_rm), BY + R_MAX * np.sin(t_rm),
            color=C_B, lw=1.6, linestyle="--", dashes=(5, 3), zorder=2)

    # R_min iç yay (katı, küçük)
    t_rmin = np.linspace(np.radians(PSI_B - HALF), np.radians(PSI_B + HALF), 40)
    ax.plot(BX + R_MIN * np.cos(t_rmin), BY + R_MIN * np.sin(t_rmin),
            color=C_B, lw=1.3, zorder=2)

    # ── Blue üçgen ────────────────────────────────────────────────────────────
    S  = 0.24
    bp = np.array([[1.2, 0], [-0.7, 0.7], [-0.7, -0.7]]) * S
    Rm = np.array([[np.cos(PSI_BR), -np.sin(PSI_BR)],
                   [np.sin(PSI_BR),  np.cos(PSI_BR)]])
    bw = (Rm @ bp.T).T + np.array([BX, BY])
    ax.add_patch(plt.Polygon(bw, closed=True,
                              facecolor=C_B, edgecolor="white",
                              lw=1.5, zorder=6))

    # Burun noktası (hız oku başlangıcı)
    nx = BX + bp[0, 0] * np.cos(PSI_BR) - bp[0, 1] * np.sin(PSI_BR)
    ny = BY + bp[0, 0] * np.sin(PSI_BR) + bp[0, 1] * np.cos(PSI_BR)

    # ── Hız vektörü (Blue) ────────────────────────────────────────────────────
    VL = 1.10
    ax.annotate("", xy=(nx + hx * VL, ny + hy * VL), xytext=(nx, ny),
                arrowprops=dict(arrowstyle="->", color=C_B, lw=2.2,
                                connectionstyle="arc3,rad=0"), zorder=5)
    ax.text(nx + hx * (VL + 0.24), ny + hy * (VL + 0.24),
            r"$V_B,\;\psi_B$",
            ha="center", va="center", fontsize=10.5,
            color=C_B, fontweight="bold")

    # ── Red yıldız ────────────────────────────────────────────────────────────
    ax.plot(RX, RY, '*', color=C_R, markersize=32, zorder=6,
            markeredgecolor="white", markeredgewidth=0.9)

    # Red hız oku
    RVL = 0.88
    ax.annotate("",
                xy=(RX + RVL * np.cos(RED_PSIR), RY + RVL * np.sin(RED_PSIR)),
                xytext=(RX, RY),
                arrowprops=dict(arrowstyle="->", color=C_R, lw=1.8,
                                connectionstyle="arc3,rad=0"), zorder=5)

    # ── LOS kesik çizgi Blue→Red ──────────────────────────────────────────────
    ax.plot([BX, RX], [BY, RY],
            color=GR, lw=1.6, linestyle="--", dashes=(5, 3), zorder=3)
    mx, my   = (BX + RX) / 2, (BY + RY) / 2
    perp_ang = np.radians(LOS_deg + 90)
    ax.text(mx + 0.36 * np.cos(perp_ang), my + 0.36 * np.sin(perp_ang),
            r"$d_{ij}$",
            ha="center", va="center", fontsize=12.5,
            fontweight="bold", color=GR, style="italic")

    # ── ATA yay (Blue'da, LOS→başlık) ────────────────────────────────────────
    ATA_r   = 1.08
    t_ata   = np.linspace(np.radians(LOS_deg), np.radians(PSI_B), 40)
    ax.plot(BX + ATA_r * np.cos(t_ata), BY + ATA_r * np.sin(t_ata),
            color=C_B, lw=1.8, zorder=4)
    # Yay uçlarına küçük tik işaretler
    for t_end in (t_ata[0], t_ata[-1]):
        ex, ey   = BX + ATA_r * np.cos(t_end), BY + ATA_r * np.sin(t_end)
        perp_t   = t_end + np.pi / 2
        ax.plot([ex - 0.06 * np.cos(perp_t), ex + 0.06 * np.cos(perp_t)],
                [ey - 0.06 * np.sin(perp_t), ey + 0.06 * np.sin(perp_t)],
                color=C_B, lw=1.2, zorder=4)
    # ATA etiketi
    mid_ata = np.radians((LOS_deg + PSI_B) / 2)
    ax.text(BX + (ATA_r + 0.28) * np.cos(mid_ata),
            BY + (ATA_r + 0.28) * np.sin(mid_ata),
            r"$\mathrm{ATA}_{ij}$",
            ha="left", va="center", fontsize=10.5,
            color=C_B, fontweight="bold")

    # ── 2θ_WEZ açı yayı ───────────────────────────────────────────────────────
    WAR   = 1.65
    t_wez = np.linspace(np.radians(PSI_B - HALF), np.radians(PSI_B + HALF), 80)
    ax.plot(BX + WAR * np.cos(t_wez), BY + WAR * np.sin(t_wez),
            color=C_B, lw=1.2, linestyle="--", alpha=0.65, zorder=4)
    wez_lbl_ang = np.radians(PSI_B + HALF + 7)
    ax.text(BX + (WAR + 0.30) * np.cos(wez_lbl_ang),
            BY + (WAR + 0.30) * np.sin(wez_lbl_ang),
            r"$2\theta_{\mathrm{WEZ}}$",
            ha="left", va="center", fontsize=9.5, color=C_B)

    # ── WEZ etiketi (koni içi) ────────────────────────────────────────────────
    wlr = R_MAX * 0.51
    ax.text(BX + wlr * hx, BY + wlr * hy, "WEZ",
            ha="center", va="center", fontsize=13.5,
            fontweight="bold", color=C_B, alpha=0.58)

    # ── R_max etiketi (alt koni kenarı altında) ───────────────────────────────
    rm_ang = np.radians(PSI_B - HALF - 7)
    ax.text(BX + (R_MAX + 0.22) * np.cos(rm_ang),
            BY + (R_MAX + 0.22) * np.sin(rm_ang),
            r"$R_{\max}$",
            ha="center", va="top", fontsize=10,
            fontweight="bold", color=C_B)

    # ── R_min etiketi (iç yay üstünde) ───────────────────────────────────────
    rmin_ang = np.radians(PSI_B + HALF + 10)
    ax.text(BX + (R_MIN + 0.52) * np.cos(rmin_ang),
            BY + (R_MIN + 0.52) * np.sin(rmin_ang),
            r"$R_{\min}$",
            ha="left", va="center", fontsize=9.5, color=C_B)

    # ── Aspect Angle yay (Red'de) ─────────────────────────────────────────────
    AA_r  = 0.66
    t_aa  = np.linspace(np.radians(R2B_deg), np.radians(RED_PSI), 40)
    ax.plot(RX + AA_r * np.cos(t_aa), RY + AA_r * np.sin(t_aa),
            color=C_R, lw=1.5, zorder=4)
    # Yay uçlarına tik
    for t_end in (t_aa[0], t_aa[-1]):
        ex, ey   = RX + AA_r * np.cos(t_end), RY + AA_r * np.sin(t_end)
        perp_t   = t_end + np.pi / 2
        ax.plot([ex - 0.05 * np.cos(perp_t), ex + 0.05 * np.cos(perp_t)],
                [ey - 0.05 * np.sin(perp_t), ey + 0.05 * np.sin(perp_t)],
                color=C_R, lw=1.2, zorder=4)
    # AA etiketi
    mid_aa  = np.radians((R2B_deg + RED_PSI) / 2)
    aa_dist = AA_r + 0.46
    ax.text(RX + aa_dist * np.cos(mid_aa),
            RY + aa_dist * np.sin(mid_aa),
            "AA\n(Aspect Angle)",
            ha="center", va="center", fontsize=8.8,
            color=C_R, fontweight="bold", multialignment="center")

    # ── Ajan etiketleri ───────────────────────────────────────────────────────
    ax.text(BX - 0.12, BY + 0.60, "Blue  (shooter)",
            ha="center", va="bottom", fontsize=11.5,
            color=C_B, fontweight="bold")
    ax.text(RX, RY + 0.60, "Red  (target)",
            ha="center", va="bottom", fontsize=11.5,
            color=C_R, fontweight="bold")

    # ── WEZ koşul kutusu (sağ alt, sarı) ─────────────────────────────────────
    wez_cond = (
        r"WEZ Active  $\Leftrightarrow$:" + "\n"
        r"$R_{\min} \leq d_{ij} \leq R_{\max}$" + "\n"
        r"$\mathrm{ATA}_{ij} \leq \theta_{\mathrm{WEZ}}$" + "\n"
        r"$\Delta h \leq h_{\max}$" + "\n"
        r"cooldown $= 0$   (0.5 s / 10 steps)"
    )
    ax.text(10.75, 0.42, wez_cond,
            ha="right", va="bottom", fontsize=9.2,
            color=DARK, multialignment="left", linespacing=1.60,
            bbox=dict(boxstyle="round,pad=0.55",
                      fc="#FFFDE7", ec="#F9A825", lw=1.4), zorder=7)

    # ── Ödül kutusu (sol alt, yeşil) ──────────────────────────────────────────
    rwd_txt = (
        "Reward components:\n"
        r"$r_{\mathrm{wez}} = \cos(\mathrm{ATA})"
        r"\cdot \mathbf{1}[d \!\leq\! R_{\max}]$   $w\!=\!5.0$" + "\n"
        r"$r_{\mathrm{kill}} = +25.0$   (enemy HP $\to$ 0)" + "\n"
        r"$r_{\mathrm{ammo}} = +0.5\,/\,-0.5$   (fire in/out WEZ)"
    )
    ax.text(0.25, 0.42, rwd_txt,
            ha="left", va="bottom", fontsize=9.2,
            color=DARK, multialignment="left", linespacing=1.60,
            bbox=dict(boxstyle="round,pad=0.55",
                      fc="#F1F8E9", ec="#558B2F", lw=1.4), zorder=7)

    # ── Kuzey göstergesi (sol üst) ────────────────────────────────────────────
    ax.annotate("", xy=(0.58, 8.55), xytext=(0.58, 7.80),
                arrowprops=dict(arrowstyle="->", color=DARK, lw=1.6,
                                connectionstyle="arc3,rad=0"), zorder=5)
    ax.text(0.58, 8.62, "N", ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=DARK)

    # ── Başlık ────────────────────────────────────────────────────────────────
    ax.text(5.5, 8.72, "Weapon Engagement Zone (WEZ) Geometry",
            ha="center", va="center", fontsize=14.5,
            fontweight="bold", color=DARK)

    # ── Alt not ───────────────────────────────────────────────────────────────
    ax.text(5.5, 0.16,
            r"ATA: angle between Blue velocity vector and Blue$\to$Red"
            r" line-of-sight",
            ha="center", va="bottom", fontsize=9.5,
            style="italic", color=GR)

    # ── Kaydet ────────────────────────────────────────────────────────────────
    out = FIG_DIR / "wez_geometry.png"
    fig.savefig(out, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"[OK] {out}")


# =============================================================================
# Figure 9 — Credit Assignment Problem
# =============================================================================

def make_credit_assignment_figure():
    """
    Credit Assignment Problem in Cooperative MARL.
    LEFT: Lazy Agent Problem.  RIGHT: MAPPO Centralized Critic Solution.
    2400×1400 px @ 200 DPI (figsize 12×7 in).
    """
    C_GOLD  = "#B7860B"    # altın kenarlık
    C_LGOLD = "#FFF8DC"    # açık altın dolgu
    CA      = "#2C5F8A"    # ajan koyu mavi
    C_LA    = "#DBEAFE"    # açık mavi dolgu
    CE      = "#E8896A"    # mercan — critic
    C_LE    = "#FDE8D8"    # açık mercan
    C_GRN   = "#276749"    # yeşil — advantage
    C_LGRN  = "#C6F6D5"    # açık yeşil
    C_RD    = "#C53030"    # kırmızı
    C_LRD   = "#FED7D7"    # açık kırmızı
    GR      = "#718096"
    DARK    = "#1F2937"

    W_IN, H_IN, DPI = 12.0, 7.0, 200   # → 2400×1400 px

    with plt.rc_context({"font.family": "sans-serif",
                         "font.sans-serif": ["DejaVu Sans", "Arial", "sans-serif"],
                         "font.size": 10}):
        fig, ax = plt.subplots(figsize=(W_IN, H_IN))
        fig.patch.set_facecolor("white")
        fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        ax.set_facecolor("white")
        ax.set_xlim(0.0, 12.0)
        ax.set_ylim(0.0,  7.0)
        ax.axis("off")

        def B(cx, cy, w, h, text, fc, ec, tc="white",
              fs=9.5, bold=True, lw=1.6, dashed=False):
            ls = "--" if dashed else "-"
            ax.add_patch(FancyBboxPatch(
                (cx - w/2, cy - h/2), w, h,
                boxstyle="round,pad=0.06", linewidth=lw,
                facecolor=fc, edgecolor=ec, linestyle=ls, zorder=3))
            ax.text(cx, cy, text, ha="center", va="center",
                    fontsize=fs, color=tc,
                    fontweight="bold" if bold else "normal",
                    multialignment="center", zorder=4)

        def AR(x0, y0, x1, y1, color=DARK, lw=1.5, ms=14):
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>", color=color,
                                        lw=lw, mutation_scale=ms,
                                        connectionstyle="arc3,rad=0"), zorder=5)

        def line(x0, y0, x1, y1, color=DARK, lw=1.5):
            ax.plot([x0, x1], [y0, y1], color=color, lw=lw, zorder=2)

        # ── Başlık ────────────────────────────────────────────────────────────
        ax.text(6.0, 6.78,
                "Credit Assignment Problem in Cooperative MARL",
                ha="center", va="top", fontsize=14.5,
                fontweight="bold", color=DARK)

        # ── Merkez kesik çizgi + "vs" etiketi ────────────────────────────────
        ax.plot([6.0, 6.0], [0.72, 6.55],
                color="#CBD5E0", lw=1.4, linestyle="--", zorder=1)
        ax.text(6.0, 3.62, "vs",
                ha="center", va="center", fontsize=13,
                fontweight="bold", color=GR,
                bbox=dict(boxstyle="round,pad=0.35", fc="white",
                          ec="#CBD5E0", lw=1.2))

        # ══════════════════════════════════════════════════════════════════════
        # SOL PANEL — The Problem: Lazy Agent
        # ══════════════════════════════════════════════════════════════════════
        ax.text(2.90, 6.46, "The Problem — Lazy Agent",
                ha="center", va="top", fontsize=11.5,
                fontweight="bold", color=C_RD)

        # r_team paylaşılan ödül kutusu (altın)
        B(2.90, 5.70, 3.30, 0.60,
          "Team Reward     $r_{\\mathrm{team}}$ = +25.0  (kill)",
          fc=C_LGOLD, ec=C_GOLD, tc="#7D5600", fs=10.0, lw=1.8)

        # Aşağı dal: r_team → split noktası → B0 + B1
        SPLIT_Y = 5.04
        line(2.90, 5.40, 2.90, SPLIT_Y)            # dikey gövde
        line(1.55, SPLIT_Y, 4.25, SPLIT_Y)          # yatay dal
        AR(1.55, SPLIT_Y, 1.55, 4.75)               # → B0
        AR(4.25, SPLIT_Y, 4.25, 4.75)               # → B1

        # Ajan kutuları
        B(1.55, 4.40, 2.30, 0.60,
          "Agent  B0\nSecured kill  ✓",
          fc=C_LA, ec=CA, tc=CA, fs=9.5)
        B(4.25, 4.40, 2.30, 0.60,
          "Agent  B1\nPassive orbit  ✗",
          fc=C_LA, ec=CA, tc=CA, fs=9.5)

        # "same reward" etiketi
        ax.text(2.90, 5.02,
                r"$\longleftarrow$  same reward  $\longrightarrow$",
                ha="center", va="top", fontsize=9.5,
                color=C_RD, fontweight="bold")

        # Lazy Agent Problem kutusu (B1 altında)
        AR(4.25, 4.10, 4.25, 3.87, color=C_RD)
        B(4.25, 3.57, 2.30, 0.52,
          "Lazy Agent Problem",
          fc=C_LRD, ec=C_RD, tc=C_RD, fs=9.5, lw=1.8)
        ax.text(4.25, 3.27,
                "\"B1 learns:  orbit = free reward\"",
                ha="center", va="top", fontsize=8.5,
                style="italic", color=GR)

        # ══════════════════════════════════════════════════════════════════════
        # SAĞ PANEL — MAPPO Solution: Centralized Critic
        # ══════════════════════════════════════════════════════════════════════
        ax.text(9.10, 6.46, "MAPPO Solution — Centralized Critic",
                ha="center", va="top", fontsize=11.5,
                fontweight="bold", color=C_GRN)

        # Global state kutusu (koyu mavi)
        B(9.10, 5.70, 3.50, 0.60,
          r"$\mathbf{s}_{\mathrm{global}} = "
          r"[o_{B0} \;\|\; o_{B1}] \in \mathbb{R}^{100}$",
          fc=C_LA, ec=CA, tc=CA, fs=10.0, lw=1.8)

        # → Centralized Critic (mercan)
        AR(9.10, 5.40, 9.10, 5.07)
        B(9.10, 4.75, 3.10, 0.58,
          r"Centralized Critic   $V(\mathbf{s}_{\mathrm{global}})$",
          fc=C_LE, ec=CE, tc="#7D3A1A", fs=9.5)

        # → Advantage hesaplama (yeşil)
        AR(9.10, 4.46, 9.10, 4.13)
        B(9.10, 3.80, 3.50, 0.58,
          r"$\hat{A}_t = G_t - V(\mathbf{s}_{\mathrm{global},t})$",
          fc=C_LGRN, ec=C_GRN, tc=C_GRN, fs=10.0)

        # İki dala ayrıl → B0 high Â, B1 low Â
        SPLIT2_Y = 3.24
        line(9.10, 3.51, 9.10, SPLIT2_Y)
        line(7.55, SPLIT2_Y, 10.65, SPLIT2_Y)
        AR(7.55, SPLIT2_Y, 7.55, 2.99, color=C_GRN)
        AR(10.65, SPLIT2_Y, 10.65, 2.99, color=C_RD)

        B(7.55, 2.67, 2.40, 0.58,
          "High  $\\hat{A}$  →  B0\ngradient  $\\uparrow$",
          fc=C_LGRN, ec=C_GRN, tc=C_GRN, fs=9.0)
        B(10.65, 2.67, 2.40, 0.58,
          "Low  $\\hat{A}$  →  B1\ngradient  $\\downarrow$",
          fc=C_LRD, ec=C_RD, tc=C_RD, fs=9.0)

        ax.text(9.10, 2.22,
                "Implicit credit via advantage baseline",
                ha="center", va="top", fontsize=8.5,
                style="italic", color=GR)

        # ── Alt not kutusu (tam genişlik) ─────────────────────────────────────
        ax.add_patch(FancyBboxPatch(
            (0.35, 0.10), 11.30, 0.72,
            boxstyle="round,pad=0.05", lw=1.0,
            facecolor="#F7F7F7", edgecolor="#CBD5E0", zorder=1))
        ax.text(6.0, 0.58,
                r"QMIX explicit decomposition:  "
                r"$Q_{\mathrm{tot}}(\mathbf{o},\mathbf{a}) = "
                r"f\!\left(Q_1(o^1,a^1),\;Q_2(o^2,a^2)\right)$"
                r"     monotonicity:  "
                r"$\partial Q_{\mathrm{tot}}/\partial Q_i \geq 0$"
                r"  guarantees IGM condition",
                ha="center", va="top", fontsize=9.0, color=DARK, zorder=4)

        out = FIG_DIR / "credit_assignment_problem.png"
        fig.savefig(out, dpi=DPI, facecolor="white")
        plt.close(fig)
    print(f"[OK] {out}")


# =============================================================================
# Figure 10 — Curriculum Progression
# =============================================================================

def make_curriculum_figure():
    """
    Four-Phase Curriculum Learning Pipeline.
    Horizontal timeline + spawn distance number line.
    2600×1400 px @ 200 DPI (figsize 13×7 in).
    """
    # Faz renk skalası: açık → koyu mavi
    P_FC = ["#DBEAFE", "#93C5FD", "#2563EB", "#1E3A8A"]   # dolgu
    P_EC = ["#3B82F6", "#2563EB", "#1E3A8A", "#0F172A"]   # kenarlık
    P_TC = [CA_DARK := "#1E3A8A", "#1E3A8A", "white", "white"]

    GR   = "#718096"
    DARK = "#1F2937"
    CA   = "#2C5F8A"

    W_IN, H_IN, DPI = 13.0, 7.0, 200   # → 2600×1400 px

    phases = [
        {
            "cx": 1.55, "num": "Phase 1",
            "sub": "WEZ-Close  1v1",
            "body": ("d ∈ [500, 2000] m\n"
                     "Opponent: Heuristic\n"
                     "Transition:\n"
                     "kill/ep ≥ 0.30\n"
                     "win_rate ≥ 0.55"),
        },
        {
            "cx": 4.35, "num": "Phase 2",
            "sub": "Progressive Range  1v1",
            "body": ("d: 4000 → 16 000 m\n"
                     "+1000 m / 100 ep\n"
                     "Pullback: kill/ep < 0.10\n"
                     "Transition:\n"
                     "kill/ep ≥ 0.20  (300 ep)"),
        },
        {
            "cx": 7.15, "num": "Phase 3",
            "sub": "Normal Spawn  1v1",
            "body": ("d ∈ [6000, 12 000] m\n"
                     "Heading align: ±45°\n"
                     "Opponent: Heuristic+Pool\n"
                     "Transition:\n"
                     "kill/ep ≥ 0.15  (500 ep)"),
        },
        {
            "cx": 9.95, "num": "Phase 4",
            "sub": "Full  2v2",
            "body": ("d ∈ [6000, 12 000] m\n"
                     "Self-play pool\n"
                     "N_pool = 1200 ep\n"
                     "\n"
                     "Budget exhausted"),
        },
    ]

    with plt.rc_context({"font.family": "sans-serif",
                         "font.sans-serif": ["DejaVu Sans", "Arial", "sans-serif"],
                         "font.size": 10}):

        fig, ax = plt.subplots(figsize=(W_IN, H_IN))
        fig.patch.set_facecolor("white")
        fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        ax.set_facecolor("white")
        ax.set_xlim(0.0, 13.0)
        ax.set_ylim(0.0,  7.0)
        ax.axis("off")

        def AR(x0, y0, x1, y1, color=DARK, lw=1.5, ms=13):
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>", color=color,
                                        lw=lw, mutation_scale=ms,
                                        connectionstyle="arc3,rad=0"), zorder=5)

        # ── Başlık ────────────────────────────────────────────────────────────
        ax.text(6.50, 6.80, "Four-Phase Curriculum Learning Pipeline",
                ha="center", va="top", fontsize=14.5,
                fontweight="bold", color=DARK)

        # ── Faz kutuları ──────────────────────────────────────────────────────
        PW   = 2.30    # kutu genişliği
        PH   = 2.80    # kutu toplam yüksekliği
        CY   = 4.65    # kutu merkez y
        HDR  = 0.52    # başlık şeridi yüksekliği

        BOX_BOT = CY - PH/2      # = 3.25
        BOX_TOP = CY + PH/2      # = 6.05
        HDR_BOT = BOX_TOP - HDR  # = 5.53

        for i, ph in enumerate(phases):
            cx = ph["cx"]
            fc, ec, tc = P_FC[i], P_EC[i], P_TC[i]

            # Arka plan kutu
            ax.add_patch(FancyBboxPatch(
                (cx - PW/2, BOX_BOT), PW, PH,
                boxstyle="round,pad=0.07", lw=1.8,
                facecolor=fc, edgecolor=ec, zorder=2))

            # Başlık şeridi (daha koyu, üstte)
            ax.add_patch(FancyBboxPatch(
                (cx - PW/2, HDR_BOT), PW, HDR,
                boxstyle="round,pad=0.04", lw=0,
                facecolor=ec, edgecolor=ec, zorder=3))
            ax.text(cx, HDR_BOT + HDR/2, ph["num"],
                    ha="center", va="center", fontsize=11,
                    fontweight="bold", color="white", zorder=4)

            # Alt başlık
            ax.text(cx, HDR_BOT - 0.19,
                    ph["sub"],
                    ha="center", va="top", fontsize=8.5,
                    fontweight="bold", color=tc, zorder=4)

            # Gövde metni
            body_tc = "#1F2937" if i < 2 else "white"
            ax.text(cx, (BOX_BOT + HDR_BOT - 0.36) / 2,
                    ph["body"],
                    ha="center", va="center",
                    fontsize=7.8, color=body_tc,
                    linespacing=1.55, multialignment="center", zorder=4)

        # Faz arası geçiş okları
        for i in range(3):
            x0 = phases[i]["cx"] + PW/2 + 0.06
            x1 = phases[i+1]["cx"] - PW/2 - 0.06
            xm = (x0 + x1) / 2
            AR(x0, CY, x1, CY, color=P_EC[i+1])
            ax.text(xm, CY + 0.22,
                    "threshold\nmet",
                    ha="center", va="bottom", fontsize=7.0,
                    color=GR, style="italic", multialignment="center")

        # ── Spawn mesafe sayı doğrusu ──────────────────────────────────────────
        NL_Y   = 2.10     # sayı doğrusu merkez y
        NL_X0  = 0.80     # 0 m
        NL_X1  = 10.70    # 16 000 m (Phase 4 sağ kenarı)
        SCALE  = (NL_X1 - NL_X0) / 16000.0   # unit/m

        ax.plot([NL_X0, NL_X1], [NL_Y, NL_Y],
                color="#9CA3AF", lw=1.4, zorder=1)
        AR(NL_X1 + 0.02, NL_Y, NL_X1 + 0.25, NL_Y,
           color="#9CA3AF", lw=1.2, ms=10)

        # km işaret ve etiketi
        for m, lbl in [(0, "0"), (2000, "2k"), (4000, "4k"),
                       (6000, "6k"), (8000, "8k"), (12000, "12k"), (16000, "16k")]:
            xp = NL_X0 + m * SCALE
            ax.plot([xp, xp], [NL_Y - 0.10, NL_Y + 0.10],
                    color="#9CA3AF", lw=1.0)
            ax.text(xp, NL_Y - 0.18, f"{lbl}m",
                    ha="center", va="top", fontsize=7.6, color=GR)

        ax.text(NL_X0 - 0.12, NL_Y,
                "Spawn\ndist.",
                ha="right", va="center", fontsize=8.0,
                fontweight="bold", color=GR)

        # Faz aralığı barları (her faz ayrı y-offset)
        ranges = [(500, 2000), (4000, 16000), (6000, 12000), (6000, 12000)]
        y_offs = [0.38, 0.20, 0.02, -0.16]
        ph_lbl = ["Ph 1", "Ph 2", "Ph 3", "Ph 4"]

        for i, ((m0, m1), yo) in enumerate(zip(ranges, y_offs)):
            x0b = NL_X0 + m0 * SCALE
            x1b = NL_X0 + m1 * SCALE
            yb  = NL_Y + yo
            ax.add_patch(FancyBboxPatch(
                (x0b, yb - 0.08), x1b - x0b, 0.15,
                boxstyle="round,pad=0.01", lw=1.0,
                facecolor=P_FC[i], edgecolor=P_EC[i],
                alpha=0.88, zorder=3))
            ax.text(x1b + 0.12, yb, ph_lbl[i],
                    ha="left", va="center", fontsize=7.6,
                    color=P_EC[i], fontweight="bold")

        # ── Sağ ek not kutusu ─────────────────────────────────────────────────
        NB_X, NB_Y = 11.15, 3.20
        ax.add_patch(FancyBboxPatch(
            (NB_X, NB_Y), 1.72, 2.85,
            boxstyle="round,pad=0.10", lw=1.2,
            facecolor="#FEF3C7", edgecolor="#D4A017", zorder=3))
        ax.text(NB_X + 0.86, NB_Y + 2.72,
                "Policy Collapse\nWithout Curriculum:",
                ha="center", va="top", fontsize=8.0,
                fontweight="bold", color="#7D5600",
                multialignment="center", zorder=4)
        ax.text(NB_X + 0.86, NB_Y + 1.90,
                "agents → 'flee'\nkill/ep ≈ 0.00\nat direct 2v2",
                ha="center", va="top", fontsize=7.6,
                color="#7D5600", multialignment="center",
                linespacing=1.5, zorder=4)

        # ── Alt not ────────────────────────────────────────────────────────────
        ax.text(6.5, 0.30,
                r"Heading alignment at spawn:  "
                r"$\psi_i^{(0)} = \angle(p_{\mathrm{enemy}} - p_i) "
                r"+ \mathcal{U}(-\pi/4,\;+\pi/4)$",
                ha="center", va="bottom", fontsize=9.5,
                style="italic", color=GR)

        out = FIG_DIR / "curriculum_progression.png"
        fig.savefig(out, dpi=DPI, facecolor="white")
        plt.close(fig)
    print(f"[OK] {out}")


# =============================================================================
# Figure 11 — Heading Alignment
# =============================================================================

def make_heading_alignment_figure():
    """
    Spawn Heading Alignment — Training Efficiency.
    LEFT: random init (~90° deviation).  RIGHT: aligned init (≤45°).
    2200×1400 px @ 200 DPI (figsize 11×7 in).
    """
    CA    = "#2C5F8A"    # Blue ajan
    C_LA  = "#DBEAFE"    # açık mavi
    C_RD  = "#C53030"    # Red / kötü
    C_LRD = "#FED7D7"
    C_GRN = "#276749"    # yeşil / iyi
    C_LGR = "#C6F6D5"
    GR    = "#718096"
    DARK  = "#1F2937"

    W_IN, H_IN, DPI = 11.0, 7.0, 200   # → 2200×1400 px

    with plt.rc_context({"font.family": "sans-serif",
                         "font.sans-serif": ["DejaVu Sans", "Arial", "sans-serif"],
                         "font.size": 10}):

        fig, ax = plt.subplots(figsize=(W_IN, H_IN))
        fig.patch.set_facecolor("white")
        fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        ax.set_facecolor("white")
        ax.set_xlim(0.0, 11.0)
        ax.set_ylim(0.0,  7.0)
        ax.axis("off")

        def AR(x0, y0, x1, y1, color=DARK, lw=1.6, ms=13):
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>", color=color,
                                        lw=lw, mutation_scale=ms,
                                        connectionstyle="arc3,rad=0"), zorder=5)

        def aircraft_tri(cx, cy, mpl_angle_deg, size=0.26, color=CA):
            """Üçgen uçak: burun mpl_angle_deg yönünde (0°=sağ, 90°=yukarı)."""
            a  = np.radians(mpl_angle_deg)
            fv = np.array([np.cos(a), np.sin(a)])
            lv = np.array([-fv[1],  fv[0]])
            nose = np.array([cx, cy]) + fv * size * 1.40
            rl   = np.array([cx, cy]) - fv * size * 0.55 + lv * size
            rr   = np.array([cx, cy]) - fv * size * 0.55 - lv * size
            ax.add_patch(mpatches.Polygon(
                [nose, rl, rr], closed=True,
                facecolor=color, edgecolor="white",
                linewidth=1.3, zorder=6))

        # ── Başlık ────────────────────────────────────────────────────────────
        ax.text(5.5, 6.82,
                "Spawn Heading Alignment — Training Efficiency",
                ha="center", va="top", fontsize=14.0,
                fontweight="bold", color=DARK)

        # ── Merkez bölücü ─────────────────────────────────────────────────────
        ax.plot([5.5, 5.5], [0.95, 6.60],
                color="#CBD5E0", lw=1.4, linestyle="--", zorder=1)

        # ══════════════════════════════════════════════════════════════════════
        # SOL PANEL — Hizasız (random)
        # ══════════════════════════════════════════════════════════════════════
        ax.add_patch(FancyBboxPatch(
            (0.15, 1.00), 5.15, 5.60,
            boxstyle="round,pad=0.05", lw=0.8,
            facecolor="#F5F5F5", edgecolor="#E0E0E0", zorder=0))
        ax.text(2.72, 6.44,
                "Without Heading Alignment",
                ha="center", va="top", fontsize=11,
                fontweight="bold", color=C_RD)

        # Blue: sol-orta, kuzeybatıya (135°) yönlü → Red kuzeydoğuda
        BX_L, BY_L = 1.90, 3.20
        RX_L, RY_L = 3.70, 5.30
        LOS_L = np.degrees(np.arctan2(RY_L - BY_L, RX_L - BX_L))  # ≈ 48°

        # Heading: 135° (kuzeybatı) → LOS ile ~87° sapma
        HDG_L = 135.0
        dev_L = abs(HDG_L - LOS_L)
        if dev_L > 180:
            dev_L = 360 - dev_L

        # Uçak ikon (kuzeybatı)
        aircraft_tri(BX_L, BY_L, mpl_angle_deg=HDG_L, size=0.26, color=CA)
        ax.text(BX_L - 0.30, BY_L - 0.18, "Blue",
                ha="right", va="top", fontsize=8.5,
                color=CA, fontweight="bold")

        # Red yıldız
        ax.scatter([RX_L], [RY_L], s=300, color=C_RD, marker="*", zorder=6)
        ax.text(RX_L + 0.14, RY_L + 0.08, "Red",
                ha="left", va="bottom", fontsize=8.5,
                color=C_RD, fontweight="bold")

        # LOS kesik çizgi
        ax.plot([BX_L, RX_L], [BY_L, RY_L],
                color=GR, lw=1.2, linestyle="--", zorder=2)
        ax.text((BX_L+RX_L)/2 + 0.15, (BY_L+RY_L)/2 - 0.05,
                "d = 10 000 m", ha="left", va="center",
                fontsize=8.0, color=GR, style="italic")

        # Heading oku (kuzeybatı yönünde)
        h_rad_L = np.radians(HDG_L)
        HL = 1.20
        AR(BX_L, BY_L,
           BX_L + HL * np.cos(h_rad_L),
           BY_L + HL * np.sin(h_rad_L),
           color=CA, lw=2.2, ms=13)

        # LOS yön oku (ince, referans)
        los_r_L = np.radians(LOS_L)
        AR(BX_L, BY_L,
           BX_L + 0.65 * np.cos(los_r_L),
           BY_L + 0.65 * np.sin(los_r_L),
           color=GR, lw=1.0, ms=9)

        # Sapma yayı
        arc_r_L = 0.65
        t1_L = min(LOS_L, HDG_L)
        t2_L = max(LOS_L, HDG_L)
        ax.add_patch(mpatches.Arc(
            (BX_L, BY_L), 2*arc_r_L, 2*arc_r_L,
            angle=0, theta1=t1_L, theta2=t2_L,
            color=C_RD, lw=2.0, zorder=3))
        mid_L = np.radians((t1_L + t2_L) / 2)
        ax.text(BX_L + (arc_r_L + 0.12) * np.cos(mid_L),
                BY_L + (arc_r_L + 0.12) * np.sin(mid_L),
                f"≈ {dev_L:.0f}°\ndeviation",
                ha="center", va="center", fontsize=8.5,
                color=C_RD, fontweight="bold")

        ax.text(2.72, 1.52, "Mean deviation ≈ 90°",
                ha="center", va="center", fontsize=10.5,
                fontweight="bold", color=C_RD)
        ax.text(2.72, 1.20,
                "Agent must first learn to turn toward enemy",
                ha="center", va="center", fontsize=8.5,
                style="italic", color=GR)

        # ══════════════════════════════════════════════════════════════════════
        # SAĞ PANEL — Hizalı
        # ══════════════════════════════════════════════════════════════════════
        ax.add_patch(FancyBboxPatch(
            (5.70, 1.00), 5.15, 5.60,
            boxstyle="round,pad=0.05", lw=0.8,
            facecolor="#F1F8E9", edgecolor="#C3E6CB", zorder=0))
        ax.text(8.27, 6.44,
                r"With Alignment   "
                r"$\psi_i^{(0)} = \angle(p_e - p_i) + \mathcal{U}(\pm\pi/4)$",
                ha="center", va="top", fontsize=9.8,
                fontweight="bold", color=C_GRN)

        # Blue: aynı göreceli konum, kuzeydo yönlü (LOS±20°)
        BX_R, BY_R = 7.45, 3.20
        RX_R, RY_R = 9.25, 5.30
        LOS_R = np.degrees(np.arctan2(RY_R - BY_R, RX_R - BX_R))  # ≈ 48°
        OFFSET_R = 20.0     # LOS'tan sapma (≤ 45°)
        HDG_R = LOS_R + OFFSET_R

        aircraft_tri(BX_R, BY_R, mpl_angle_deg=HDG_R, size=0.26, color=CA)
        ax.text(BX_R - 0.30, BY_R - 0.18, "Blue",
                ha="right", va="top", fontsize=8.5,
                color=CA, fontweight="bold")

        # Red yıldız
        ax.scatter([RX_R], [RY_R], s=300, color=C_RD, marker="*", zorder=6)
        ax.text(RX_R + 0.14, RY_R + 0.08, "Red",
                ha="left", va="bottom", fontsize=8.5,
                color=C_RD, fontweight="bold")

        # ±45° koni (çok açık mavi dolgu)
        cone_r = 1.85
        cone_angles = np.linspace(np.radians(LOS_R - 45), np.radians(LOS_R + 45), 40)
        cx_cone = [BX_R] + list(BX_R + cone_r * np.cos(cone_angles)) + [BX_R]
        cy_cone = [BY_R] + list(BY_R + cone_r * np.sin(cone_angles)) + [BY_R]
        ax.fill(cx_cone, cy_cone, color=C_LA, alpha=0.42, zorder=1)

        # LOS kesik çizgi
        ax.plot([BX_R, RX_R], [BY_R, RY_R],
                color=GR, lw=1.2, linestyle="--", zorder=2)

        # Heading oku (LOS + 20°)
        h_rad_R = np.radians(HDG_R)
        HR = 1.20
        AR(BX_R, BY_R,
           BX_R + HR * np.cos(h_rad_R),
           BY_R + HR * np.sin(h_rad_R),
           color=CA, lw=2.2, ms=13)

        # Sapma yayı (küçük, yeşil)
        arc_r_R = 0.65
        ax.add_patch(mpatches.Arc(
            (BX_R, BY_R), 2*arc_r_R, 2*arc_r_R,
            angle=0,
            theta1=LOS_R, theta2=HDG_R,
            color=C_GRN, lw=2.0, zorder=3))
        mid_R = np.radians((LOS_R + HDG_R) / 2)
        ax.text(BX_R + (arc_r_R + 0.20) * np.cos(mid_R),
                BY_R + (arc_r_R + 0.20) * np.sin(mid_R),
                f"≤ 45°\ndeviation",
                ha="left", va="center", fontsize=8.5,
                color=C_GRN, fontweight="bold")

        # ±45° koni açıklama etiketi
        ax.text(BX_R + 0.10, BY_R + cone_r + 0.08,
                "±45° heading cone",
                ha="center", va="bottom", fontsize=7.8,
                color=CA, style="italic")

        ax.text(8.27, 1.52, "Mean deviation ≈ 21.6°",
                ha="center", va="center", fontsize=10.5,
                fontweight="bold", color=C_GRN)
        ax.text(8.27, 1.20,
                "Agent begins pursuing immediately",
                ha="center", va="center", fontsize=8.5,
                style="italic", color=GR)

        # ── Alt metrik kutuları ───────────────────────────────────────────────
        ax.add_patch(FancyBboxPatch(
            (0.18, 0.10), 5.10, 0.72,
            boxstyle="round,pad=0.05", lw=1.0,
            facecolor=C_LRD, edgecolor=C_RD, zorder=2))
        ax.text(2.73, 0.46,
                "Random init:  ~90° deviation  →  slow initial learning",
                ha="center", va="center", fontsize=9.0,
                color="#7B1C1C", fontweight="bold")

        ax.add_patch(FancyBboxPatch(
            (5.72, 0.10), 5.10, 0.72,
            boxstyle="round,pad=0.05", lw=1.0,
            facecolor=C_LGR, edgecolor=C_GRN, zorder=2))
        ax.text(8.27, 0.46,
                "Aligned init:  21.6° deviation  →  kill signal within first episodes",
                ha="center", va="center", fontsize=9.0,
                color="#14532D", fontweight="bold")

        out = FIG_DIR / "heading_alignment.png"
        fig.savefig(out, dpi=DPI, facecolor="white")
        plt.close(fig)
    print(f"[OK] {out}")


# =============================================================================
# Figure 12 — Opponent Modeling Pipeline
# =============================================================================

def make_opponent_modeling_figure():
    """
    Opponent Modeling Pipeline — Intent Classification.
    2600×1400 px @ 200 DPI (figsize 13×7 in).
    """
    CA    = "#2C5F8A"; C_LA   = "#DBEAFE"
    CE    = "#E8896A"; C_LE   = "#FDE8D8"
    C_GRN = "#276749"; C_LGRN = "#C6F6D5"
    C_RD  = "#C53030"; C_LRD  = "#FED7D7"
    C_MED = "#2563EB"; C_LMED = "#BFDBFE"
    GR    = "#718096"; DARK   = "#1F2937"
    W_IN, H_IN, DPI = 13.0, 7.0, 200

    with plt.rc_context({"font.family": "sans-serif",
                         "font.sans-serif": ["DejaVu Sans", "Arial", "sans-serif"],
                         "font.size": 10}):
        fig, ax = plt.subplots(figsize=(W_IN, H_IN))
        fig.patch.set_facecolor("white")
        fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        ax.set_facecolor("white")
        ax.set_xlim(0.0, 13.0); ax.set_ylim(0.0, 7.0)
        ax.axis("off")

        def B(cx, cy, w, h, text, fc, ec, tc="white", fs=9.0, bold=True, lw=1.5):
            ax.add_patch(FancyBboxPatch(
                (cx - w/2, cy - h/2), w, h,
                boxstyle="round,pad=0.05", lw=lw,
                facecolor=fc, edgecolor=ec, zorder=3))
            ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
                    color=tc, fontweight="bold" if bold else "normal",
                    multialignment="center", zorder=4)

        def AR(x0, y0, x1, y1, color=DARK, lw=1.4, ms=12):
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>", color=color,
                                        lw=lw, mutation_scale=ms,
                                        connectionstyle="arc3,rad=0"), zorder=5)

        # ── Başlık ────────────────────────────────────────────────────────────
        ax.text(6.5, 6.84, "Opponent Modeling Pipeline — Intent Classification",
                ha="center", va="top", fontsize=14.0,
                fontweight="bold", color=DARK)

        # Adım başlıkları
        for cx, lbl, col in [(1.40, "① INPUT", CA),
                              (3.90, "② NETWORK", C_MED),
                              (6.45, "③ OUTPUT", CE)]:
            ax.text(cx, 6.50, lbl, ha="center", va="top",
                    fontsize=9.5, fontweight="bold", color=col)

        # ADIM 1 — Enemy history
        B(1.40, 5.65, 2.20, 0.90,
          "Enemy History Buffer\n20 steps × 48D = 960D",
          fc=C_LA, ec=CA, tc=CA, fs=9.5)
        AR(2.51, 5.65, 3.13, 5.65)

        # ADIM 2 — 3 FC katmanı (yığılmış)
        FC_CX = 3.90
        for cy_fc, txt_fc in [(5.93, "FC  960 → 256   ReLU"),
                               (5.55, "FC  256 → 128   ReLU"),
                               (5.17, "FC  128 → 6")]:
            B(FC_CX, cy_fc, 1.84, 0.33, txt_fc,
              fc=C_LMED, ec=C_MED, tc=DARK, fs=8.5, lw=1.2)
        for y_s, y_e in [(5.765, 5.720), (5.385, 5.340)]:
            ax.annotate("", xy=(FC_CX, y_e), xytext=(FC_CX, y_s),
                        arrowprops=dict(arrowstyle="-|>", color=C_MED,
                                        lw=1.0, mutation_scale=8,
                                        connectionstyle="arc3,rad=0"), zorder=2)
        AR(4.82, 5.55, 5.47, 5.65)

        # ADIM 3 — İki Softmax çıktısı
        for cx_o, lbl_o in [(5.95, "Softmax₃\nEnemy 0\nintent"),
                             (7.05, "Softmax₃\nEnemy 1\nintent")]:
            B(cx_o, 5.65, 1.00, 0.84, lbl_o,
              fc=C_LE, ec=CE, tc="#7D3A1A", fs=8.5)

        # Intent label kutuları
        INTENT_Y = 4.40
        for cx_i, txt_i, fc_i, ec_i, tc_i in [
            (4.30, "AGGRESSIVE\nα_AA ≤ 45°\n~34%",   C_LRD,    C_RD,  C_RD),
            (6.10, "DEFENSIVE\n45° – 120°\n~38%",    "#F3F4F6", GR,    DARK),
            (7.90, "EVASIVE\nα_AA ≥ 120°\n~28%",     C_LGRN,   C_GRN, C_GRN),
        ]:
            B(cx_i, INTENT_Y, 1.60, 0.72, txt_i,
              fc=fc_i, ec=ec_i, tc=tc_i, fs=8.5)

        # Output → intent okları
        for x_s, x_d in [(5.95, 4.30), (6.45, 6.10), (7.05, 7.90)]:
            AR(x_s, 5.23, x_d, INTENT_Y + 0.36, color=GR, lw=1.0, ms=9)

        # Eğitim notu (sarı arka plan)
        ax.add_patch(FancyBboxPatch((0.25, 0.42), 8.52, 3.48,
                                    boxstyle="round,pad=0.06", lw=1.0,
                                    facecolor="#FFFDE7", edgecolor="#F9A825",
                                    zorder=1))
        ax.text(4.51, 3.72, "Training Note",
                ha="center", va="top", fontsize=10.0,
                fontweight="bold", color="#7D5600", zorder=4)
        ax.text(4.51, 3.36,
                "Loss: cross-entropy vs aspect-angle labels\n"
                "Separate Adam optimizer — decoupled from PPO / FACMAC updates\n"
                "✗   Earlier: closing-velocity thresholds → 100% Defensive  (all wrong)\n"
                "✓   Fix: aspect-angle formulation → balanced label distribution",
                ha="center", va="top", fontsize=8.8, color=DARK,
                linespacing=1.62, multialignment="center", zorder=4)

        # Dikey bölücü
        ax.plot([8.95, 8.95], [0.45, 6.60],
                color="#CBD5E0", lw=1.0, linestyle="--", alpha=0.70, zorder=1)

        # SAĞ PANEL — Gözlem boyutu büyümesi
        ax.text(11.05, 6.58, "Observation\nDimension Growth",
                ha="center", va="top", fontsize=10.5,
                fontweight="bold", color=DARK, multialignment="center")

        obs_rows = [
            ("Base obs",       "50D", C_LA,   CA,    CA),
            ("+ GAT message",  "16D", C_LMED, C_MED, C_MED),
            ("+ Intent (OM)",  " 6D", C_LE,   CE,    "#7D3A1A"),
            ("+ Role",         " 4D", C_LGRN, C_GRN, C_GRN),
        ]
        BX0, BX1 = 9.20, 12.88
        BW = BX1 - BX0
        bar_y = 5.80; bh = 0.52; bg = 0.14

        for name, dim_s, fc, ec, tc in obs_rows:
            ax.add_patch(FancyBboxPatch((BX0, bar_y - bh/2), BW, bh,
                                        boxstyle="round,pad=0.02", lw=1.3,
                                        facecolor=fc, edgecolor=ec, zorder=3))
            ax.text(BX0 + 0.12, bar_y, name,
                    ha="left", va="center", fontsize=8.8,
                    color=tc, fontweight="bold", zorder=4)
            ax.text(BX1 - 0.12, bar_y, dim_s,
                    ha="right", va="center", fontsize=9.0,
                    color=tc, fontweight="bold", zorder=4)
            bar_y -= (bh + bg)

        # Toplam satırı
        ax.add_patch(FancyBboxPatch((BX0, bar_y - bh/2), BW, bh,
                                    boxstyle="round,pad=0.02", lw=1.8,
                                    facecolor="#1E3A8A", edgecolor="#0F172A",
                                    zorder=3))
        ax.text(BX0 + 0.12, bar_y, "= Extended obs",
                ha="left", va="center", fontsize=9.0,
                color="white", fontweight="bold", zorder=4)
        ax.text(BX1 - 0.12, bar_y, "76D",
                ha="right", va="center", fontsize=9.0,
                color="white", fontweight="bold", zorder=4)

        ax.text(11.05, bar_y - 0.52,
                "(FACMAC: 60D\nbase + intent + role,\nno GAT)",
                ha="center", va="top", fontsize=7.8,
                color=GR, style="italic", multialignment="center")

        out = FIG_DIR / "opponent_modeling_pipeline.png"
        fig.savefig(out, dpi=DPI, facecolor="white")
        plt.close(fig)
    print(f"[OK] {out}")


# =============================================================================
# Figure 13 — Role Assigner Architecture
# =============================================================================

def make_role_assigner_figure():
    """
    Dynamic Role Assigner — CentralizedRoleAssigner Architecture.
    2400×1600 px @ 200 DPI (figsize 12×8 in).
    """
    CA    = "#2C5F8A"; C_LA   = "#DBEAFE"
    CE    = "#E8896A"; C_LE   = "#FDE8D8"
    C_GRN = "#276749"; C_LGRN = "#C6F6D5"
    C_RD  = "#C53030"; C_LRD  = "#FED7D7"
    C_GOLD = "#D4A017"; C_LGOLD = "#FEF3C7"
    GR    = "#718096"; DARK   = "#1F2937"
    W_IN, H_IN, DPI = 12.0, 8.0, 200

    with plt.rc_context({"font.family": "sans-serif",
                         "font.sans-serif": ["DejaVu Sans", "Arial", "sans-serif"],
                         "font.size": 10}):
        fig, ax = plt.subplots(figsize=(W_IN, H_IN))
        fig.patch.set_facecolor("white")
        fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        ax.set_facecolor("white")
        ax.set_xlim(0.0, 12.0); ax.set_ylim(0.0, 8.0)
        ax.axis("off")

        def B(cx, cy, w, h, text, fc, ec, tc="white", fs=9.0, bold=True, lw=1.5):
            ax.add_patch(FancyBboxPatch(
                (cx - w/2, cy - h/2), w, h,
                boxstyle="round,pad=0.05", lw=lw,
                facecolor=fc, edgecolor=ec, zorder=3))
            ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
                    color=tc, fontweight="bold" if bold else "normal",
                    multialignment="center", zorder=4)

        def AR(x0, y0, x1, y1, color=DARK, lw=1.4, ms=12, rad=0.0):
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>", color=color,
                                        lw=lw, mutation_scale=ms,
                                        connectionstyle=f"arc3,rad={rad}"),
                        zorder=5)

        CX = 4.80   # merkez akış x

        # ── Başlık ────────────────────────────────────────────────────────────
        ax.text(6.0, 7.90,
                "Dynamic Role Assigner — CentralizedRoleAssigner Architecture",
                ha="center", va="top", fontsize=13.5,
                fontweight="bold", color=DARK)

        # GİRDİ SATIRI: Intent + Team state
        B(2.90, 7.28, 2.60, 0.72,
          "Intent vector   6D\n(2 × Softmax₃)",
          fc=C_LE, ec=CE, tc="#7D3A1A", fs=9.0)
        B(6.70, 7.28, 2.60, 0.72,
          "Team state   6D\n(HP, WEZ flags, pos)",
          fc=C_LA, ec=CA, tc=CA, fs=9.0)

        # Her iki kutudan concat'a dal
        for bx in [2.90, 6.70]:
            ax.plot([bx, bx], [6.92, 6.74], color=DARK, lw=1.5, zorder=2)
        ax.plot([2.90, CX], [6.74, 6.74], color=DARK, lw=1.5, zorder=2)
        ax.plot([6.70, CX], [6.74, 6.74], color=DARK, lw=1.5, zorder=2)
        ax.text(CX, 6.64, "concat → 12D",
                ha="center", va="top", fontsize=8.5,
                color=GR, style="italic")
        AR(CX, 6.60, CX, 6.44)

        # FC katmanları
        B(CX, 6.24, 2.80, 0.38,
          "FC   12 → 64    ReLU",
          fc=C_LA, ec=CA, tc=CA, fs=9.0)
        AR(CX, 6.05, CX, 5.87)
        B(CX, 5.67, 2.80, 0.38,
          "FC   64 → 12    (12 valid role pairs)",
          fc=C_LA, ec=CA, tc=CA, fs=9.0)
        AR(CX, 5.48, CX, 5.22)

        # Gumbel-Softmax (amber)
        B(CX, 4.75, 3.30, 0.92,
          "Gumbel-Softmax\n"
          "τ: 1.0 → 0.3   (annealed over 5M steps)\n"
          "Training: soft sample  |  Eval: argmax",
          fc=C_LGOLD, ec=C_GOLD, tc="#7D5600", fs=9.0)
        AR(CX, 4.29, CX, 4.00)

        # ÇIKTI SATIRI: B0 + B1 roller
        for cx_o, lbl_o in [(2.90, "B0 Role:\nSniper | Pursuit\nDefensive | Support"),
                             (6.70, "B1 Role:\nSniper | Pursuit\nDefensive | Support")]:
            B(cx_o, 3.68, 2.60, 0.88, lbl_o,
              fc=C_LA, ec=CA, tc=CA, fs=9.0)

        ax.text(CX, 3.20,
                "Same-role pairs excluded → 12 valid combinations of 16",
                ha="center", va="top", fontsize=8.5,
                style="italic", color=GR)

        # Eğitim kutusu (alt, açık mavi)
        ax.add_patch(FancyBboxPatch((0.25, 0.30), 7.50, 2.60,
                                    boxstyle="round,pad=0.06", lw=1.0,
                                    facecolor="#EFF6FF", edgecolor=CA,
                                    zorder=1))
        ax.text(4.00, 2.73, "REINFORCE Update",
                ha="center", va="top", fontsize=10.0,
                fontweight="bold", color=CA, zorder=4)
        ax.text(4.00, 2.38,
                r"$\nabla_\psi L = -\mathbb{E}\!\left[\left(\sum_t \gamma^t r_t\right) "
                r"\cdot \nabla_\psi \log P_\psi(\mathrm{pair}_0)\right]$"
                "\n"
                "Role fixed for episode duration\n"
                r"Diversity entropy:  $L_{\mathrm{role}} = L_{\mathrm{REINFORCE}} "
                r"- \beta\,H[P_\psi(\mathrm{pair})]$,   $\beta = 0.01$"
                "\n"
                "✗  On-policy (MAPPO): gradient disconnected at numpy buffer → failed\n"
                "✓  Off-policy (FACMAC): training structure compatible",
                ha="center", va="top", fontsize=8.5, color=DARK,
                linespacing=1.58, multialignment="center", zorder=4)

        # SAĞ PANEL — Role Taxonomy
        TX0, TX1 = 7.95, 11.75
        TW = TX1 - TX0
        ax.add_patch(FancyBboxPatch((TX0 - 0.05, 3.45), TW + 0.10, 4.30,
                                    boxstyle="round,pad=0.06", lw=1.0,
                                    facecolor="#F8FAFC", edgecolor="#CBD5E0",
                                    zorder=1))
        ax.text((TX0 + TX1) / 2, 7.62, "Role Definitions",
                ha="center", va="top", fontsize=10.5,
                fontweight="bold", color=DARK, zorder=4)

        # Tablo başlıkları
        for x_col, hdr in [(TX0 + 0.55, "Role"),
                            (TX0 + 1.85, "Behaviour"),
                            (TX0 + 3.45, "Trigger")]:
            ax.text(x_col, 7.28, hdr, ha="center", va="top",
                    fontsize=8.5, fontweight="bold", color=CA, zorder=4)
        ax.plot([TX0, TX1], [7.12, 7.12], color="#CBD5E0", lw=0.8, zorder=2)

        role_rows = [
            ("Sniper",     "Optimal WEZ pos; patience",  "Enemy evasive"),
            ("Pursuit",    "Close range rapidly",         "Enemy fleeing"),
            ("Defensive",  "Preserve HP; avoid contact",  "Enemy aggressive"),
            ("Support",    "Enable teammate engagement",  "Teammate in WEZ"),
        ]
        row_colors = [C_LGRN, C_LA, C_LRD, "#FEF3C7"]
        row_edges  = [C_GRN, CA, C_RD, C_GOLD]
        ry = 6.75
        for (role, behav, trig), rfc, rec in zip(role_rows, row_colors, row_edges):
            ax.add_patch(FancyBboxPatch((TX0, ry - 0.28), TW, 0.52,
                                        boxstyle="round,pad=0.02", lw=0.8,
                                        facecolor=rfc, edgecolor=rec,
                                        alpha=0.7, zorder=2))
            ax.text(TX0 + 0.55, ry, role,
                    ha="center", va="center", fontsize=8.5,
                    fontweight="bold", color=DARK, zorder=4)
            ax.text(TX0 + 1.85, ry, behav,
                    ha="center", va="center", fontsize=8.0,
                    color=DARK, zorder=4)
            ax.text(TX0 + 3.45, ry, trig,
                    ha="center", va="center", fontsize=8.0,
                    color=DARK, style="italic", zorder=4)
            ry -= 0.82

        out = FIG_DIR / "role_assigner_architecture.png"
        fig.savefig(out, dpi=DPI, facecolor="white")
        plt.close(fig)
    print(f"[OK] {out}")


# =============================================================================
# Figure 14 — QMIX Architecture
# =============================================================================

def make_qmix_figure():
    """
    QMIX — Monotonic Value Function Factorization.
    2600×1600 px @ 200 DPI (figsize 13×8 in).
    """
    CA    = "#2C5F8A"; C_LA   = "#DBEAFE"
    CE    = "#E8896A"; C_LE   = "#FDE8D8"
    C_GRN = "#276749"; C_LGRN = "#C6F6D5"
    C_MED = "#2563EB"; C_LMED = "#BFDBFE"
    GR    = "#718096"; DARK   = "#1F2937"
    W_IN, H_IN, DPI = 13.0, 8.0, 200

    with plt.rc_context({"font.family": "sans-serif",
                         "font.sans-serif": ["DejaVu Sans", "Arial", "sans-serif"],
                         "font.size": 10}):
        fig, ax = plt.subplots(figsize=(W_IN, H_IN))
        fig.patch.set_facecolor("white")
        fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        ax.set_facecolor("white")
        ax.set_xlim(0.0, 13.0); ax.set_ylim(0.0, 8.0)
        ax.axis("off")

        def B(cx, cy, w, h, text, fc, ec, tc="white", fs=9.0, bold=True, lw=1.5):
            ax.add_patch(FancyBboxPatch(
                (cx - w/2, cy - h/2), w, h,
                boxstyle="round,pad=0.05", lw=lw,
                facecolor=fc, edgecolor=ec, zorder=3))
            ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
                    color=tc, fontweight="bold" if bold else "normal",
                    multialignment="center", zorder=4)

        def AR(x0, y0, x1, y1, color=DARK, lw=1.4, ms=12, rad=0.0):
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>", color=color,
                                        lw=lw, mutation_scale=ms,
                                        connectionstyle=f"arc3,rad={rad}"),
                        zorder=5)

        # ── Başlık ────────────────────────────────────────────────────────────
        ax.text(6.5, 7.88, "QMIX — Monotonic Value Function Factorization",
                ha="center", va="top", fontsize=14.0,
                fontweight="bold", color=DARK)

        # ── Tier başlık şeritleri ─────────────────────────────────────────────
        tier_lbls = [
            (7.10, "Individual Agent Networks", CA),
            (5.00, "Mixing Network", CE),
            (4.10, "IGM Condition", C_GRN),
        ]
        for ty, tlbl, tc in [(7.08, "Tier 1 — Individual Agent Networks", CA),
                              (5.20, "Tier 2 — Mixing Network", CE),
                              (2.68, "Tier 3 — IGM Condition", C_GRN)]:
            ax.text(5.50, ty, tlbl, ha="center", va="top",
                    fontsize=9.5, fontweight="bold", color=tc)

        # ══════════════════════════════════════════════════════════════════════
        # TIER 1 — İki ajan ağı
        # ══════════════════════════════════════════════════════════════════════
        for cx_ag, lbl_ag, q_lbl in [
            (2.60, "Agent B0", "$Q_1(o^1, a^1)$"),
            (8.00, "Agent B1", "$Q_2(o^2, a^2)$"),
        ]:
            # Input kutusu
            B(cx_ag, 6.80, 2.60, 0.52,
              f"$o^i \\in \\mathbb{{R}}^{{50}}$  ‖  $a^i_{{\\mathrm{{onehot}}}} \\in \\mathbb{{R}}^{{162}}$",
              fc=C_LA, ec=CA, tc=CA, fs=8.5)
            AR(cx_ag, 6.54, cx_ag, 6.36)
            # FC katmanları
            for cy_fc, fc_txt in [(6.17, "FC 128   ReLU"),
                                   (5.82, "FC 128   ReLU")]:
                B(cx_ag, cy_fc, 2.40, 0.32, fc_txt,
                  fc="#1E3A8A", ec="#0F172A", tc="white", fs=8.5)
            ax.annotate("", xy=(cx_ag, 5.98), xytext=(cx_ag, 6.01),
                        arrowprops=dict(arrowstyle="-|>", color="#0F172A",
                                        lw=1.0, mutation_scale=8,
                                        connectionstyle="arc3,rad=0"), zorder=2)
            AR(cx_ag, 5.66, cx_ag, 5.49)
            # Q çıktısı
            B(cx_ag, 5.31, 2.00, 0.34, q_lbl + r"  $\in \mathbb{R}$",
              fc=C_LE, ec=CE, tc="#7D3A1A", fs=9.0)

        # Parametre paylaşım oku (çift yönlü)
        ax.annotate("", xy=(6.75, 6.17), xytext=(3.85, 6.17),
                    arrowprops=dict(arrowstyle="<->", color=C_MED,
                                    lw=1.8, mutation_scale=13,
                                    connectionstyle="arc3,rad=0"), zorder=5)
        ax.text(5.30, 6.26, "Parameter Sharing   $\\theta_1 = \\theta_2$",
                ha="center", va="bottom", fontsize=9.0,
                color=C_MED, fontweight="bold")

        # ══════════════════════════════════════════════════════════════════════
        # TIER 2 — Mixing Network
        # ══════════════════════════════════════════════════════════════════════
        # Arka plan kutu
        ax.add_patch(FancyBboxPatch((0.30, 3.55), 10.10, 1.70,
                                    boxstyle="round,pad=0.04", lw=1.2,
                                    facecolor="#FFF5F5", edgecolor=CE,
                                    alpha=0.6, zorder=1))

        # Q1, Q2'den aşağı oklar
        AR(2.60, 5.14, 2.60, 4.92, color=CE)
        AR(8.00, 5.14, 8.00, 4.92, color=CE)
        ax.text(2.60, 4.85, "$Q_1$", ha="center", va="top", fontsize=9.0,
                color=CE, fontweight="bold")
        ax.text(8.00, 4.85, "$Q_2$", ha="center", va="top", fontsize=9.0,
                color=CE, fontweight="bold")

        # Hypernetwork kutusu
        B(8.80, 4.42, 2.20, 0.80,
          "Hypernetwork\n"
          r"$\mathbf{s}_{\mathrm{global}} \in \mathbb{R}^{100}$"
          "\n"
          r"$W_{\mathrm{mix}} = |W_{\mathrm{hyper}}(\mathbf{s})| \geq 0$",
          fc=C_LE, ec=CE, tc="#7D3A1A", fs=8.5)

        # ELU/Mixing kutusu
        B(4.80, 4.42, 3.40, 0.80,
          "ELU activation\n"
          r"hidden dim $d_{\mathrm{mix}} = 32$"
          "\n"
          r"$\rightarrow\; Q_{\mathrm{tot}} \in \mathbb{R}$",
          fc=C_LE, ec=CE, tc="#7D3A1A", fs=8.8)

        # Q1, Q2 → mixing, Hyper → mixing okları
        for x_q in [2.60, 8.00]:
            AR(x_q, 4.72, 4.80 + (1.70 if x_q > 5 else -1.70), 4.42, color=CE)
        AR(7.90, 4.42, 6.52, 4.42, color=CE)

        # Monotonicity notu
        AR(4.80, 4.02, 4.80, 3.78, color=C_GRN)
        ax.text(4.80, 3.73,
                r"$\partial Q_{\mathrm{tot}}/\partial Q_i \geq 0$   (monotonicity enforced)",
                ha="center", va="top", fontsize=8.8,
                fontweight="bold", color=C_GRN)

        # ══════════════════════════════════════════════════════════════════════
        # TIER 3 — IGM koşul kutusu
        # ══════════════════════════════════════════════════════════════════════
        ax.add_patch(FancyBboxPatch((0.30, 1.95), 10.10, 1.42,
                                    boxstyle="round,pad=0.06", lw=1.5,
                                    facecolor=C_LGRN, edgecolor=C_GRN,
                                    zorder=2))
        ax.text(5.35, 3.28, "IGM Condition",
                ha="center", va="top", fontsize=10.0,
                fontweight="bold", color=C_GRN, zorder=4)
        ax.text(5.35, 2.95,
                r"$\arg\max_{\mathbf{a}}\; Q_{\mathrm{tot}}(\mathbf{o}, \mathbf{a})"
                r"\;=\;\left(\arg\max_{a^1} Q_1(o^1, a^1),\;"
                r"\arg\max_{a^2} Q_2(o^2, a^2)\right)$",
                ha="center", va="top", fontsize=10.5,
                color=DARK, zorder=4)
        ax.text(5.35, 2.42,
                "Decentralized greedy execution  =  globally optimal joint action",
                ha="center", va="top", fontsize=9.0,
                style="italic", color=C_GRN, fontweight="bold", zorder=4)

        # SAĞ MARGIN — Eylem uzayı kutusu
        ax.add_patch(FancyBboxPatch((10.55, 1.90), 2.35, 5.90,
                                    boxstyle="round,pad=0.08", lw=1.0,
                                    facecolor="#F8FAFC", edgecolor="#CBD5E0",
                                    zorder=2))
        ax.text(11.72, 7.68, "Discrete\nAction Space",
                ha="center", va="top", fontsize=9.5,
                fontweight="bold", color=DARK,
                multialignment="center", zorder=4)
        ax.text(11.72, 7.10,
                "$\\delta_a \\in \\{-1,0,+1\\}$  → 3\n"
                "$\\delta_e \\in \\{-1,0,+1\\}$  → 3\n"
                "$\\delta_r \\in \\{-1,0,+1\\}$  → 3\n"
                "$\\delta_t \\in \\{0,0.5,1\\}$  → 3\n"
                "$\\delta_f \\in \\{0,1\\}$       → 2\n"
                "\n"
                "Total:\n"
                "$3^3 \\times 3 \\times 2 = 162$",
                ha="center", va="top", fontsize=8.5,
                color=DARK, linespacing=1.55,
                multialignment="left", zorder=4)

        # Alt not
        ax.text(5.35, 1.70,
                r"TD loss with target network $\theta^-$ — hard copy every 200 episodes     "
                r"$\varepsilon$-greedy: $\varepsilon$ annealed 1.0 → 0.05 over 500 000 steps",
                ha="center", va="top", fontsize=8.8,
                style="italic", color=GR)

        out = FIG_DIR / "qmix_architecture.png"
        fig.savefig(out, dpi=DPI, facecolor="white")
        plt.close(fig)
    print(f"[OK] {out}")


# =============================================================================
# Figure 15 — TD3 Enhancements
# =============================================================================

def make_td3_figure():
    """
    TD3 Enhancements for FACMAC Training Stability.
    2400×1500 px @ 200 DPI (figsize 12×7.5 in).
    LEFT: stylised divergence chart.  RIGHT: 4 enhancement boxes.
    """
    CA    = "#2C5F8A"; C_LA   = "#DBEAFE"
    CE    = "#E8896A"; C_LE   = "#FDE8D8"
    C_GRN = "#276749"; C_LGRN = "#C6F6D5"
    C_RD  = "#C53030"; C_LRD  = "#FED7D7"
    C_MED = "#2563EB"; C_LMED = "#BFDBFE"
    GR    = "#718096"; DARK   = "#1F2937"
    W_IN, H_IN, DPI = 12.0, 7.5, 200

    with plt.rc_context({"font.family": "sans-serif",
                         "font.sans-serif": ["DejaVu Sans", "Arial", "sans-serif"],
                         "font.size": 10}):

        fig = plt.figure(figsize=(W_IN, H_IN))
        fig.patch.set_facecolor("white")
        fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)

        # Ana eksen (tam canvas) — açıklama kutuları için
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.set_facecolor("white"); ax.set_xlim(0, 12); ax.set_ylim(0, 7.5)
        ax.axis("off")

        # Sol panel — stilize diverjans grafiği
        ax_c = fig.add_axes([0.04, 0.16, 0.28, 0.73])
        ax_c.set_facecolor("#FAFAFA")
        ax_c.spines[["top", "right"]].set_visible(False)
        ax_c.set_yscale("log")
        ax_c.set_xlim(0, 3000); ax_c.set_ylim(1e3, 1e12)
        ax_c.set_xlabel("Episode", fontsize=9)
        ax_c.set_ylabel("Critic Loss  (log scale)", fontsize=9)
        ax_c.tick_params(labelsize=7.5)
        ax_c.set_title("Without TD3 — Divergence",
                        fontsize=9.5, fontweight="bold", color=C_RD, pad=4)

        ep  = [0, 300, 700, 1200, 1800, 2400, 2600]
        lss = [1e4, 3e4, 2e5, 5e6, 3e8, 8e10, 6.6e10]
        ax_c.plot(ep, lss, color=C_RD, lw=2.5, zorder=3)
        ax_c.fill_between(ep, lss, 1e8,
                          where=[l > 1e8 for l in lss],
                          color=C_LRD, alpha=0.55, label="Divergence zone",
                          zorder=2)
        ax_c.axhline(1e8, color=C_RD, lw=0.9, linestyle="--", alpha=0.5)
        ax_c.text(2800, 6.6e10, "$6.6\\!\\times\\!10^{10}$\n~ep 2600",
                  ha="right", va="top", fontsize=8.0,
                  color=C_RD, fontweight="bold")
        ax_c.text(1500, 3e3, "Training\nfailed",
                  ha="center", va="bottom", fontsize=9.0,
                  color=C_RD, fontweight="bold",
                  bbox=dict(boxstyle="round,pad=0.2", fc=C_LRD, ec=C_RD, lw=0.8))
        ax_c.legend(fontsize=7.5, loc="upper left", framealpha=0.9)

        # ── Ana eksen: başlık + sağ kutular + alt not ─────────────────────────
        def B(cx, cy, w, h, text, fc, ec, tc="white", fs=9.0, bold=True, lw=1.5):
            ax.add_patch(FancyBboxPatch(
                (cx - w/2, cy - h/2), w, h,
                boxstyle="round,pad=0.05", lw=lw,
                facecolor=fc, edgecolor=ec, zorder=3))
            ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
                    color=tc, fontweight="bold" if bold else "normal",
                    multialignment="center", zorder=4)

        ax.text(6.0, 7.40, "TD3 Enhancements for FACMAC Training Stability",
                ha="center", va="top", fontsize=14.0,
                fontweight="bold", color=DARK)

        ax.text(8.15, 7.10, "TD3 Enhancements",
                ha="center", va="top", fontsize=11.5,
                fontweight="bold", color=CA)

        # 4 kutu: eşit yükseklik, hizalanmış
        BOX_W = 6.90; BOX_H = 1.22; BOX_CX = 8.15
        BOX_YS = [6.35, 4.98, 3.61, 2.24]
        box_specs = [
            ("① Twin Critics",
             "Two independent $Q_{\\varphi 1}$, $Q_{\\varphi 2}$\n"
             r"Target:  $y = r + \gamma \cdot \min(Q_{\mathrm{tot},1}(s',\tilde{a}'),\;"
             r"Q_{\mathrm{tot},2}(s',\tilde{a}'))$"
             "\n→ Pessimistic targets suppress Q-overestimation",
             CA, C_LA, CA),
            ("② Delayed Actor Updates",
             "Actor updated every $d_{\\mathrm{policy}} = 4$ critic steps\n"
             "→ Stable value estimates before policy update",
             C_MED, C_LMED, C_MED),
            ("③ Target Policy Smoothing",
             r"$\tilde{a}' = \mathrm{clamp}(\pi_{\theta^-}(o') + \varepsilon,\; -1,\; 1)$"
             "\n"
             r"$\varepsilon \sim \mathrm{clip}(\mathcal{N}(0,\;0.2),\;-0.5,\;0.5)$"
             "\n→ Smoother Q-surface; prevents exploitation of peaks",
             CE, C_LE, "#7D3A1A"),
            ("④ Polyak Averaging + Q-Clamping",
             r"$\theta^- \leftarrow \tau\theta + (1-\tau)\theta^-$,   $\tau = 0.005$"
             "\n"
             r"$y \leftarrow \mathrm{clamp}(y,\;-2000,\;+2000)$"
             "\nBound = 3× max theoretical $Q_{\\mathrm{tot}} \\approx 709$",
             C_GRN, C_LGRN, C_GRN),
        ]

        for (title, body, ec, fc, tc), cy in zip(box_specs, BOX_YS):
            ax.add_patch(FancyBboxPatch(
                (BOX_CX - BOX_W/2, cy - BOX_H/2), BOX_W, BOX_H,
                boxstyle="round,pad=0.06", lw=1.6,
                facecolor=fc, edgecolor=ec, zorder=3))
            ax.text(BOX_CX - BOX_W/2 + 0.18, cy + BOX_H/2 - 0.16,
                    title, ha="left", va="top", fontsize=9.5,
                    fontweight="bold", color=tc, zorder=4)
            ax.text(BOX_CX, cy - 0.08, body,
                    ha="center", va="center", fontsize=8.5,
                    color=DARK, linespacing=1.55,
                    multialignment="center", zorder=4)

        # Alt sonuç kutusu
        ax.add_patch(FancyBboxPatch((3.60, 0.12), 8.25, 0.80,
                                    boxstyle="round,pad=0.05", lw=1.2,
                                    facecolor=C_LGRN, edgecolor=C_GRN,
                                    zorder=2))
        ax.text(7.72, 0.82,
                "Result: critic loss stabilized → FACMAC training converges\n"
                "Pure FACMAC win rate: 82.0%  [78.4–85.1%]     "
                "FACMAC + OM + REINFORCE: 84.2%  [80.7–87.1%]",
                ha="center", va="top", fontsize=9.0,
                color="#14532D", fontweight="bold",
                multialignment="center", zorder=4)

        out = FIG_DIR / "td3_enhancements.png"
        fig.savefig(out, dpi=DPI, facecolor="white")
        plt.close(fig)
    print(f"[OK] {out}")


# =============================================================================
# Figure 16 — FACMAC Pipeline
# =============================================================================

def make_facmac_pipeline_figure():
    """
    FACMAC + OM + REINFORCE — Complete Training Pipeline.
    2800×1600 px @ 200 DPI (figsize 14×8 in).
    """
    CA    = "#2C5F8A"; C_LA   = "#DBEAFE"
    CE    = "#E8896A"; C_LE   = "#FDE8D8"
    C_GRN = "#276749"; C_LGRN = "#C6F6D5"
    C_RD  = "#C53030"; C_LRD  = "#FED7D7"
    C_MED = "#2563EB"; C_LMED = "#BFDBFE"
    C_TEAL = "#0E7490"; C_LTEAL = "#CFFAFE"
    GR    = "#718096"; DARK   = "#1F2937"
    W_IN, H_IN, DPI = 14.0, 8.0, 200

    with plt.rc_context({"font.family": "sans-serif",
                         "font.sans-serif": ["DejaVu Sans", "Arial", "sans-serif"],
                         "font.size": 10}):
        fig, ax = plt.subplots(figsize=(W_IN, H_IN))
        fig.patch.set_facecolor("white")
        fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        ax.set_facecolor("white")
        ax.set_xlim(0.0, 14.0); ax.set_ylim(0.0, 8.0)
        ax.axis("off")

        def B(cx, cy, w, h, text, fc, ec, tc="white", fs=8.5, bold=True, lw=1.5):
            ax.add_patch(FancyBboxPatch(
                (cx - w/2, cy - h/2), w, h,
                boxstyle="round,pad=0.05", lw=lw,
                facecolor=fc, edgecolor=ec, zorder=3))
            ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
                    color=tc, fontweight="bold" if bold else "normal",
                    multialignment="center", zorder=4)

        def AR(x0, y0, x1, y1, color=DARK, lw=1.4, ms=12, rad=0.0, ls="-"):
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>", color=color,
                                        lw=lw, mutation_scale=ms,
                                        linestyle=ls,
                                        connectionstyle=f"arc3,rad={rad}"),
                        zorder=5)

        def col_hdr(cx, text, color):
            ax.text(cx, 7.84, text, ha="center", va="top",
                    fontsize=9.5, fontweight="bold", color=color)
            ax.plot([cx - 1.15, cx + 1.15], [7.72, 7.72],
                    color=color, lw=1.0, alpha=0.5)

        # ── Başlık ────────────────────────────────────────────────────────────
        ax.text(7.0, 7.98,
                "FACMAC + OM + REINFORCE — Complete Training Pipeline",
                ha="center", va="top", fontsize=14.0,
                fontweight="bold", color=DARK)

        # ── Sütun merkezleri ──────────────────────────────────────────────────
        CX = [1.30, 3.85, 6.65, 9.35, 12.05]
        CW = [2.10, 2.70, 2.70, 2.10, 2.20]   # sütun genişlikleri

        col_hdr(CX[0], "① INPUTS", CA)
        col_hdr(CX[1], "② SHARED ACTORS", CA)
        col_hdr(CX[2], "③ MIXING NETWORK", CE)
        col_hdr(CX[3], "④ OPPONENT MODEL", C_TEAL)
        col_hdr(CX[4], "⑤ ROLE ASSIGNER", C_GRN)

        # ── COLUMN 1: Inputs ──────────────────────────────────────────────────
        for cy_i, lbl_i in [(5.65, "$o^1_{\\mathrm{ext}}$  60D\nbase(50) ‖ intent(6) ‖ role(4)"),
                             (4.40, "$o^2_{\\mathrm{ext}}$  60D\nbase(50) ‖ intent(6) ‖ role(4)")]:
            B(CX[0], cy_i, CW[0], 0.72, lbl_i, fc=C_LA, ec=CA, tc=CA, fs=8.5)

        # ── COLUMN 2: Shared Actor ────────────────────────────────────────────
        B(CX[1], 5.40, CW[1], 2.50,
          "Actor  $\\pi_\\theta$  — Split First Layer\n"
          "$W_{\\mathrm{base}}$   [50D]  pretrained  ✓\n"
          "$W_{\\mathrm{intent}}$  [6D]   zero-init\n"
          "$W_{\\mathrm{role}}$    [4D]   zero-init\n"
          "↓  FC 256 ReLU\n"
          "↓  FC 256 ReLU\n"
          "→  4D continuous action",
          fc=C_LA, ec=CA, tc=CA, fs=8.5)

        # Parametre paylaşım etiketi
        ax.text(CX[1], 6.76, "Parameter Sharing   $\\theta_1 = \\theta_2$",
                ha="center", va="bottom", fontsize=8.0,
                color=C_MED, style="italic", fontweight="bold")

        # Fire: kural tabanlı not
        ax.add_patch(FancyBboxPatch(
            (CX[1] - CW[1]/2, 3.75), CW[1], 0.70,
            boxstyle="round,pad=0.04", lw=1.0,
            facecolor=C_LRD, edgecolor=C_RD, zorder=3))
        ax.text(CX[1], 4.10,
                "Fire: rule-based\n"
                "$\\delta_f = 1$  iff  in\\_WEZ $\\wedge$ cooldown=0",
                ha="center", va="center", fontsize=7.8,
                color=C_RD, fontweight="bold", multialignment="center", zorder=4)

        # Input → Actor okları
        for cy_src in [5.65, 4.40]:
            AR(CX[0] + CW[0]/2 + 0.06, cy_src,
               CX[1] - CW[1]/2 - 0.06, 5.40,
               color=CA, lw=1.2, ms=10)

        # Actor → Mixing okları
        AR(CX[1] + CW[1]/2 + 0.06, 5.40,
           CX[2] - CW[2]/2 - 0.06, 5.40, color=CA)

        # ── COLUMN 3: Mixing Network ──────────────────────────────────────────
        B(CX[2], 5.90, CW[2], 0.55,
          "$Q_1(o^1, a^1)$  and  $Q_2(o^2, a^2)$",
          fc=C_LE, ec=CE, tc="#7D3A1A", fs=8.5)
        AR(CX[2], 5.625, CX[2], 5.43)
        B(CX[2], 5.10, CW[2], 0.72,
          r"Hypernetwork:  $\mathbf{s}_{\mathrm{global}} \to W_{\mathrm{mix}} \geq 0$"
          "\n"
          r"$\partial Q_{\mathrm{tot}} / \partial Q_i \geq 0$",
          fc=C_LE, ec=CE, tc="#7D3A1A", fs=8.5)
        AR(CX[2], 4.74, CX[2], 4.57)
        B(CX[2], 4.25, CW[2], 0.55,
          r"$\min(Q_{\mathrm{tot},1},\; Q_{\mathrm{tot},2})$  — Twin critics",
          fc=C_LE, ec=CE, tc="#7D3A1A", fs=8.5)
        ax.text(CX[2], 3.89,
                "TD3: target smoothing, delayed actor, Polyak avg",
                ha="center", va="top", fontsize=7.8,
                color=CE, style="italic")

        # Mixing → Mixing sonraki oka
        AR(CX[2] + CW[2]/2 + 0.06, 5.10,
           CX[3] - CW[3]/2 - 0.06, 5.10, color=CE)

        # Geri gradient oku (Actor'a → Mixing'ten)
        ax.annotate("", xy=(CX[1] + CW[1]/2 + 0.10, 3.50),
                    xytext=(CX[2] - CW[2]/2 - 0.10, 3.50),
                    arrowprops=dict(arrowstyle="-|>", color=CE,
                                    lw=1.5, mutation_scale=11, linestyle="dashed",
                                    connectionstyle="arc3,rad=0"), zorder=5)
        ax.text((CX[1] + CX[2]) / 2, 3.60,
                r"$\nabla_\theta$  actor gradient",
                ha="center", va="bottom", fontsize=8.0,
                color=CE, style="italic")

        # ── COLUMN 4: Opponent Model ──────────────────────────────────────────
        B(CX[3], 5.55, CW[3], 1.70,
          "Enemy history  960D\n"
          "→ FC 256 → FC 128\n"
          "→ 2 × Softmax₃\n"
          "Intent: [agg, def, eva] × 2",
          fc=C_LTEAL, ec=C_TEAL, tc=C_TEAL, fs=8.5)

        # OM → Actor (intent ekler)
        ax.annotate("", xy=(CX[1] + CW[1]/2 + 0.10, 5.10),
                    xytext=(CX[3] - CW[3]/2 - 0.10, 5.10),
                    arrowprops=dict(arrowstyle="-|>", color=C_TEAL,
                                    lw=1.2, mutation_scale=10, linestyle="dashed",
                                    connectionstyle="arc3,rad=0.18"), zorder=4)
        ax.text((CX[1] + CX[3]) / 2, 5.12,
                "+ intent (6D)", ha="center", va="bottom",
                fontsize=7.8, color=C_TEAL, style="italic")

        AR(CX[3] + CW[3]/2 + 0.06, 5.55,
           CX[4] - CW[4]/2 - 0.06, 5.55, color=C_TEAL)

        # ── COLUMN 5: Role Assigner ────────────────────────────────────────────
        B(CX[4], 5.55, CW[4], 1.90,
          "Intent(6D) + Team(6D)\n"
          "→ FC 64 → FC 12\n"
          "→ Gumbel-Softmax\n"
          "→ Role pair\n"
          "REINFORCE update",
          fc=C_LGRN, ec=C_GRN, tc=C_GRN, fs=8.5)

        # Role → Actor
        ax.annotate("", xy=(CX[1] + CW[1]/2 + 0.10, 4.70),
                    xytext=(CX[4] - CW[4]/2 - 0.10, 4.70),
                    arrowprops=dict(arrowstyle="-|>", color=C_GRN,
                                    lw=1.2, mutation_scale=10, linestyle="dashed",
                                    connectionstyle="arc3,rad=0.25"), zorder=4)
        ax.text((CX[1] + CX[4]) / 2 + 0.5, 4.72,
                "+ role (4D)", ha="center", va="bottom",
                fontsize=7.8, color=C_GRN, style="italic")

        # ── Alt çubuk (tam genişlik) ───────────────────────────────────────────
        ax.add_patch(FancyBboxPatch((0.20, 0.12), 13.60, 1.08,
                                    boxstyle="round,pad=0.05", lw=1.0,
                                    facecolor="#F8FAFC", edgecolor="#CBD5E0",
                                    zorder=2))
        ax.text(7.0, 1.08,
                r"Experience Replay Buffer  $|\mathcal{D}| = 50\,000$"
                "     |     Critic update: every step  (batch 256)"
                "     |     Actor update: every 4 steps  (delayed)\n"
                "OM update: cross-entropy, decoupled"
                "     |     Role update: REINFORCE, per episode",
                ha="center", va="top", fontsize=8.8, color=DARK,
                linespacing=1.55, multialignment="center", zorder=4)

        out = FIG_DIR / "facmac_pipeline.png"
        fig.savefig(out, dpi=DPI, facecolor="white")
        plt.close(fig)
    print(f"[OK] {out}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import sys
    targets = sys.argv[1:]   # python thesis_figures.py mdp  → sadece mdp
    run_all = not targets

    print(f"Çıktı klasörü: {FIG_DIR}\n")
    if run_all or "1" in targets or "paradigms" in targets:
        make_figure1()
    if run_all or "2" in targets or "activation" in targets:
        make_figure2()
    if run_all or "3" in targets or "gat" in targets:
        make_figure3()
    if run_all or "4" in targets or "mdp" in targets:
        make_mdp_figure()
    if run_all or "5" in targets or "ac" in targets:
        make_actor_critic_figure()
    if run_all or "6" in targets or "enu" in targets:
        make_enu_figure()
    if run_all or "7" in targets or "obs" in targets or "action" in targets:
        make_action_state_figure()
    if run_all or "8" in targets or "wez" in targets:
        make_wez_figure()
    if run_all or "9" in targets or "credit" in targets:
        make_credit_assignment_figure()
    if run_all or "10" in targets or "curriculum" in targets:
        make_curriculum_figure()
    if run_all or "11" in targets or "heading" in targets:
        make_heading_alignment_figure()
    if run_all or "12" in targets or "om" in targets or "opponent" in targets:
        make_opponent_modeling_figure()
    if run_all or "13" in targets or "role" in targets:
        make_role_assigner_figure()
    if run_all or "14" in targets or "qmix" in targets:
        make_qmix_figure()
    if run_all or "15" in targets or "td3" in targets:
        make_td3_figure()
    if run_all or "16" in targets or "facmac" in targets:
        make_facmac_pipeline_figure()
    print("\nTüm figürler kaydedildi.")

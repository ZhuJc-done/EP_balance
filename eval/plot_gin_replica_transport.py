#!/usr/bin/env python3
"""Schematic of the GIN replica transport: symmetric windows, transfers, fences, and state.

Draws the buffer roles exactly as ``eplb/integration/gin_weights.py`` implements them, for one
logical expert ``e`` whose main instance lives on GPU A while GPU B hosts a replica.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

# home / slot / scratch keep one colour each across both GPUs: the windows are symmetric, and the
# figure's main correction over an owner-vs-replica drawing is that both ranks hold all three.
C_HOME = "#cfe0f5"
C_SLOT = "#fbdfc0"
C_SCRATCH = "#cfe8d5"
C_DESC = "#e0e0e0"
C_EDGE = "#4a4a4a"
C_GIN = "#d9ead3"
C_FWD = "#1f4e79"
C_BWD = "#7d3c98"
C_LOCAL = "#5a5a5a"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="gin_replica_transport.pdf",
        help="Output figure; the extension selects PDF, PNG, or SVG",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--font-size",
        type=float,
        default=9.0,
        help="Base size in points for the buffer body text",
    )
    return parser.parse_args()


def _box(ax, x, y, w, h, face, *, edge=C_EDGE, lw=1.1, radius=1.2, z=2, ls="solid"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=face, edgecolor=edge, linewidth=lw, zorder=z, linestyle=ls,
        )
    )


def _buffer(ax, x, y, w, h, face, title, body, fs, *, idle=False):
    """One symmetric window: bold handle on the first line, role underneath."""
    _box(ax, x, y, w, h, face, lw=1.4 if not idle else 0.9,
         ls="solid" if not idle else (0, (3, 2)))
    colour = "#111111" if not idle else "#7a7a7a"
    ax.text(x + w / 2, y + h - 1.7, title, ha="center", va="top",
            fontsize=fs + 1.4, fontweight="bold", color=colour, zorder=3)
    ax.text(x + w / 2, y + h - 4.7, body, ha="center", va="top",
            fontsize=fs, color=colour, zorder=3, linespacing=1.45)


def _arrow(ax, xy_from, xy_to, *, rad=0.0, colour=C_FWD, lw=2.0, z=4):
    ax.add_patch(
        FancyArrowPatch(
            xy_from, xy_to,
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>", mutation_scale=16,
            linewidth=lw, color=colour, zorder=z,
            shrinkA=1.5, shrinkB=1.5,
        )
    )


def _badge(ax, x, y, glyph, fs, *, colour=C_FWD):
    """Circled step number, placed on the thing it describes rather than floating beside it."""
    ax.text(x, y, glyph, ha="center", va="center", fontsize=fs + 0.4, color="white",
            fontweight="bold", zorder=7,
            bbox=dict(boxstyle="circle,pad=0.28", facecolor=colour, edgecolor="white",
                      linewidth=1.1))


def build_figure(fs: float):
    fig, ax = plt.subplots(figsize=(13.6, 8.8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # ---- header -------------------------------------------------------------------------
    ax.text(50, 99.4, "Scale-EPLB replica transport over NCCL GIN", ha="center", va="top",
            fontsize=fs + 5.5, fontweight="bold")
    ax.text(50, 95.4,
            "Every rank registers all three symmetric windows and plays all three roles at once. "
            "The labels below are the roles for one expert $e$ with main($e$) = GPU A;\n"
            "GPU A also hosts replicas of other experts in its own slot[$j$], and GPU B is main of "
            "its own experts in its own home[$j$].",
            ha="center", va="top", fontsize=fs - 0.2, color="#444444", linespacing=1.5)

    pa_x, pb_x, pw = 2.5, 57.0, 40.5
    p_y, p_h = 36.8, 51.7
    mid = (pa_x + pw + pb_x) / 2.0

    for x, title, sub in (
        (pa_x, "GPU A", "main($e$) — owns the persistent training state"),
        (pb_x, "GPU B", "hosts a replica of $e$ at physical slot $s$"),
    ):
        _box(ax, x, p_y, pw, p_h, "white", edge="#333333", lw=1.7, radius=1.6, z=1)
        ax.text(x + pw / 2, p_y + p_h - 1.3, title, ha="center", va="top",
                fontsize=fs + 3.2, fontweight="bold")
        ax.text(x + pw / 2, p_y + p_h - 5.0, sub, ha="center", va="top",
                fontsize=fs + 0.2, color="#555555", style="italic")

    bw = pw - 5.0
    bh = 9.4
    y_home, y_slot, y_scratch = 70.0, 58.2, 46.4
    ax_l, ax_r = pa_x + 2.5, pb_x + 2.5

    # ---- GPU A windows ------------------------------------------------------------------
    _buffer(ax, ax_l, y_home, bw, bh, C_HOME, "home[$j$]",
            "$W_e$ re-published every forward at row local_slot($e$)\n"
            "capacity home_cap $\\times$ $|W_j|$", fs)
    _buffer(ax, ax_l, y_slot, bw, bh, C_SLOT, "slot[$j$]",
            "A's own physical slots: its mains gathered on device,\n"
            "plus replicas A hosts of other ranks' experts", fs)
    _buffer(ax, ax_l, y_scratch, bw, bh, C_SCRATCH, "scratch[$j$]",
            "gradient inbox, home_cap $\\times$ world columns;\n"
            "column (local_slot($e$), src rank)", fs)

    # ---- GPU B windows ------------------------------------------------------------------
    _buffer(ax, ax_r, y_home, bw, bh, C_HOME, "home[$j$]",
            "B's own mains ($e$ is not one of them)", fs, idle=True)
    _buffer(ax, ax_r, y_slot, bw, bh, C_SLOT, "slot[$j$]",
            "replica of $e$ at slot $s$ — runs the expert GEMM;\n"
            "backward reuses these bytes as the grad send area", fs)
    _buffer(ax, ax_r, y_scratch, bw, bh, C_SCRATCH, "scratch[$j$]",
            "idle for $e$ — receives grads only for B's own mains", fs, idle=True)

    # ---- device-resident descriptor (both ranks build it) --------------------------------
    dy, dh = 38.0, 6.2
    for x in (ax_l, ax_r):
        _box(ax, x, dy, bw, dh, C_DESC, lw=1.0)
        ax.text(x + bw / 2, dy + dh - 1.4, "device descriptor  (peers / offsets / nbytes)",
                ha="center", va="top", fontsize=fs - 0.4, fontweight="bold")
        ax.text(x + bw / 2, dy + dh - 4.1,
                "built on device from the plan — no D2H;  peers < 0 skipped in-kernel",
                ha="center", va="top", fontsize=fs - 1.3, color="#555555")

    # ---- transfers: badges ride the arrows, ordering lives in the strip below -------------
    _badge(ax, ax_l + bw - 2.4, y_home + bh - 1.5, "\u2460", fs, colour=C_LOCAL)

    _arrow(ax, (ax_l + bw + 0.4, y_home + bh * 0.42), (ax_r - 0.4, y_slot + bh * 0.80), rad=-0.13)
    _badge(ax, mid, y_home + 0.6, "\u2461", fs)
    ax.text(mid, y_home - 2.6, "FWD\nget_batched\npull $W_e$", ha="center", va="top",
            fontsize=fs - 0.8, color=C_FWD, fontweight="bold", linespacing=1.4)

    _arrow(ax, (ax_r - 0.4, y_slot + bh * 0.16), (ax_l + bw + 0.4, y_scratch + bh * 0.55),
           rad=-0.13, colour=C_BWD)
    _badge(ax, mid, y_scratch + bh - 1.4, "\u2462", fs, colour=C_BWD)
    ax.text(mid, y_scratch + bh - 3.8, "BWD\nput_batched\n$\\nabla W_e$", ha="center", va="top",
            fontsize=fs - 0.8, color=C_BWD, fontweight="bold", linespacing=1.4)

    _badge(ax, ax_l + bw - 2.4, y_scratch + 1.5, "\u2463", fs, colour=C_LOCAL)

    # ---- ordered sequence, fences included because they are load-bearing -----------------
    sy, sh = 22.4, 13.6
    _box(ax, 2.5, sy, 95.0, sh, "#fafafa", edge="#999999", lw=1.0)
    arrow_glyph = "  \u2500\u25b6  "
    ax.text(4.4, sy + sh - 1.6, "FORWARD", ha="left", va="top",
            fontsize=fs - 0.2, fontweight="bold", color=C_FWD)
    ax.text(15.0, sy + sh - 1.6,
            "\u2460 publish $W_e$ into home[$j$] (host loop, data-independent)" + arrow_glyph
            + "$\\it{fence(0)}$ homes visible" + arrow_glyph
            + "\u2461 get_batched pulls $W_e$ into slot[$j$];\nlocal slots gathered on device"
            + arrow_glyph + "$\\it{fence(1)}$ peers done reading home before it is recycled",
            ha="left", va="top", fontsize=fs - 1.0, color="#333333", linespacing=1.6)
    ax.text(4.4, sy + sh - 6.6, "BACKWARD", ha="left", va="top",
            fontsize=fs - 0.2, fontweight="bold", color=C_BWD)
    ax.text(15.0, sy + sh - 6.6,
            "$\\nabla$ staged into slot[$j$]" + arrow_glyph
            + "\u2462 put_batched into main's scratch column (local_slot($e$), src)"
            + arrow_glyph + "$\\it{fence(2)}$ puts landed\n"
            + "\u2463 main sums its world columns into $\\nabla W_e$" + arrow_glyph + "optimizer",
            ha="left", va="top", fontsize=fs - 1.0, color="#333333", linespacing=1.6)
    ax.text(4.4, sy + 1.0,
            "the three fences default to $\\mathtt{dist.barrier}$ (host-launched); "
            "$\\mathtt{EPLB\\_GIN\\_FENCE{=}signal}$ makes them device-stream and capture-safe",
            ha="left", va="bottom", fontsize=fs - 1.4, color="#b03a2e", style="italic")

    # ---- GIN primitive band --------------------------------------------------------------
    gy, gh = 14.6, 7.0
    _box(ax, 2.5, gy, 95.0, gh, C_GIN, lw=1.4)
    ax.text(50, gy + gh - 1.5,
            "one-sided GIN get / put issued from device kernels   \u21d2   NVLink  \u00b7  RDMA",
            ha="center", va="top", fontsize=fs + 1.6, fontweight="bold")
    ax.text(50, gy + gh - 4.4,
            "windows registered once with ncclMemAlloc + ncclCommWindowRegister and recycled across "
            "layers and micro-batches; the data-dependent schedule never reaches the host",
            ha="center", va="top", fontsize=fs - 1.0, color="#3a5a3a")

    # ---- training-state ledger -----------------------------------------------------------
    ly, lh = 1.5, 11.0
    _box(ax, 2.5, ly, 46.0, lh, "#eef4fb", lw=1.2)
    ax.text(2.5 + 46.0 / 2, ly + lh - 1.5,
            "persistent state — main($e$) only", ha="center", va="top",
            fontsize=fs + 0.8, fontweight="bold", color="#1f4e79")
    ax.text(2.5 + 46.0 / 2, ly + lh - 4.6,
            "BF16 $W$ 2  +  BF16 $\\nabla$ 2  +  FP32 master 4  +  FP32 $m,v$ 8\n"
            "= 16 bytes / parameter,  plus checkpoint ownership",
            ha="center", va="top", fontsize=fs - 0.2, color="#1f4e79", linespacing=1.5)

    _box(ax, 51.5, ly, 46.0, lh, "#fdf3e8", lw=1.2)
    ax.text(51.5 + 46.0 / 2, ly + lh - 1.5,
            "replica — transient, no optimizer state", ha="center", va="top",
            fontsize=fs + 0.8, fontweight="bold", color="#9c5518")
    ax.text(51.5 + 46.0 / 2, ly + lh - 4.6,
            "BF16 $W$ only = 2 bytes / parameter, into a preallocated slot;\n"
            "rebalancing moves bytes between fixed slots, never allocates",
            ha="center", va="top", fontsize=fs - 0.2, color="#9c5518", linespacing=1.5)

    fig.tight_layout(pad=0.3)
    return fig


def main() -> None:
    args = parse_args()
    fig = build_figure(args.font_size)
    out = Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

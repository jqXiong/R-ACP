from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT_DIR = Path("output/figures")
PNG_PATH = OUT_DIR / "abl6_refined_architecture_en.png"
SVG_PATH = OUT_DIR / "abl6_refined_architecture_en.svg"


def add_box(ax, x, y, w, h, title, lines, fc, ec="#1f2937", lw=1.5, title_size=11, body_size=9):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h - 0.018,
        title,
        ha="center",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color="#111827",
    )
    ax.text(
        x + w / 2,
        y + h / 2 - 0.012,
        "\n".join(lines),
        ha="center",
        va="center",
        fontsize=body_size,
        color="#1f2937",
        linespacing=1.3,
    )
    return patch


def connect(ax, xy1, xy2, text=None, color="#4b5563", lw=1.5, text_offset=(0, 0), style="-|>"):
    arrow = FancyArrowPatch(
        xy1,
        xy2,
        arrowstyle=style,
        mutation_scale=14,
        linewidth=lw,
        color=color,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arrow)
    if text:
        mx = (xy1[0] + xy2[0]) / 2 + text_offset[0]
        my = (xy1[1] + xy2[1]) / 2 + text_offset[1]
        ax.text(mx, my, text, fontsize=8.5, color=color, ha="center", va="center")


def section(ax, x, y, text, color):
    ax.text(x, y, text, fontsize=13, fontweight="bold", color=color, ha="left", va="center")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(24, 14), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    section(ax, 0.03, 0.95, "R-ACP Final Architecture for Paper", "#0f172a")
    ax.text(
        0.03,
        0.92,
        "Shared baseline backbone + SNR-aware refined transmission + strong cross-view BEV refinement",
        fontsize=10.5,
        color="#475569",
        ha="left",
        va="center",
    )

    input_box = add_box(
        ax, 0.04, 0.72, 0.12, 0.12, "Input Sequence",
        ["T synchronized frames", "N calibrated cameras", "RGB images 720 x 1280"],
        "#e0f2fe"
    )
    encoder_box = add_box(
        ax, 0.20, 0.69, 0.15, 0.18, "Shared View Encoder",
        ["Frozen ResNet18 split", "1 x 1 projection", "8-channel feature maps", "Quantized view features"],
        "#dbeafe"
    )
    hist_box = add_box(
        ax, 0.40, 0.78, 0.13, 0.10, "History Stack",
        ["tau2 past features", "per camera feature tensor"],
        "#ede9fe"
    )
    ent_box = add_box(
        ax, 0.57, 0.76, 0.14, 0.14, "Temporal Entropy Model",
        ["Predict scales_hat", "Predict means_hat", "Gaussian likelihood prior"],
        "#ede9fe"
    )
    selector_box = add_box(
        ax, 0.40, 0.57, 0.16, 0.15, "Refined Camera Selector",
        ["Current statistics", "History statistics", "Uncertainty statistics", "Camera score + channel scale"],
        "#fce7f3"
    )
    snr_ctrl_box = add_box(
        ax, 0.60, 0.56, 0.17, 0.16, "SNR-aware Transmission Controller",
        ["Adaptive camera keep", "Adaptive channel keep", "Low-SNR rescue camera", "Low-SNR scale recovery"],
        "#fae8ff"
    )
    bits_box = add_box(
        ax, 0.80, 0.60, 0.15, 0.11, "Communication Cost Branch",
        ["Masked likelihood", "Weighted entropy", "KB-level rate proxy"],
        "#fef3c7"
    )

    pred_box = add_box(
        ax, 0.39, 0.34, 0.16, 0.14, "Prediction Stack",
        ["tau1 + 1 recent features", "camera gating", "channel gating", "scale modulation"],
        "#dcfce7"
    )
    channel_box = add_box(
        ax, 0.60, 0.34, 0.16, 0.14, "Feature Channel",
        ["AWGN or Rayleigh", "SNR-dependent perturbation", "same physical channel for all methods"],
        "#d1fae5"
    )
    warp_box = add_box(
        ax, 0.80, 0.34, 0.15, 0.14, "Per-view BEV Projection",
        ["warp_perspective", "camera-wise BEV tensors", "token mask applied after projection"],
        "#dbeafe"
    )

    legacy_mask_box = add_box(
        ax, 0.05, 0.15, 0.16, 0.12, "Legacy Baseline Drop Module",
        ["baseline-compatible", "priority frame dropping", "used only in baseline path"],
        "#fee2e2"
    )
    fusion_box = add_box(
        ax, 0.28, 0.12, 0.18, 0.18, "Temporal BEV Fusion Head",
        ["AdaptiveTemporalFusionModule", "BEV tokens + coord map + token mask", "Baseline-compatible fusion output"],
        "#dbeafe"
    )

    attn_box = add_box(
        ax, 0.53, 0.12, 0.12, 0.10, "Optional BEV Attention",
        ["Channel attention", "view-wise feature emphasis"],
        "#fef9c3"
    )
    cross_box = add_box(
        ax, 0.68, 0.10, 0.15, 0.15, "Strong Cross-view Fusion",
        ["SNRGuidedCrossViewFusion", "attention fusion", "confidence fusion", "reliability-guided gating"],
        "#e9d5ff"
    )
    head_box = add_box(
        ax, 0.86, 0.10, 0.10, 0.15, "Enhanced Map Head",
        ["multi-dilation branches", "high-capacity BEV decoding"],
        "#ddd6fe"
    )

    ensemble_box = add_box(
        ax, 0.53, 0.80, 0.19, 0.10, "Output Ensemble",
        ["Final map = (1 - w) Temporal Head + w Strong Head", "w = refine_strong_head_weight"],
        "#e2e8f0"
    )
    output_box = add_box(
        ax, 0.79, 0.80, 0.17, 0.10, "Outputs",
        ["Final occupancy map", "Per-camera test maps", "Communication cost per frame"],
        "#cffafe"
    )
    loss_box = add_box(
        ax, 0.05, 0.44, 0.16, 0.11, "Training Objective",
        ["GaussianMSE map loss", "+ 1e-4 x communication cost", "best checkpoint selected by MODA"],
        "#f3e8ff"
    )

    connect(ax, (0.16, 0.78), (0.20, 0.78))
    connect(ax, (0.35, 0.80), (0.40, 0.83), "history path", text_offset=(0, 0.02))
    connect(ax, (0.35, 0.74), (0.40, 0.64), "statistics path", text_offset=(0, -0.02))
    connect(ax, (0.53, 0.83), (0.57, 0.83))
    connect(ax, (0.71, 0.83), (0.80, 0.66), "scales / means", text_offset=(0.02, 0.03))
    connect(ax, (0.56, 0.64), (0.60, 0.64))
    connect(ax, (0.77, 0.64), (0.80, 0.64))

    connect(ax, (0.35, 0.72), (0.39, 0.41), "recent features", text_offset=(-0.01, 0.00))
    connect(ax, (0.55, 0.41), (0.60, 0.41))
    connect(ax, (0.76, 0.41), (0.80, 0.41))

    connect(ax, (0.87, 0.34), (0.87, 0.25))
    connect(ax, (0.81, 0.25), (0.74, 0.25))
    connect(ax, (0.65, 0.25), (0.62, 0.22))
    connect(ax, (0.87, 0.25), (0.91, 0.25))

    connect(ax, (0.87, 0.34), (0.87, 0.17), "BEV view stack", text_offset=(0.05, 0.01))
    connect(ax, (0.53, 0.17), (0.46, 0.20), "baseline fusion path", text_offset=(0.00, -0.02))
    connect(ax, (0.21, 0.21), (0.28, 0.21))
    connect(ax, (0.65, 0.17), (0.68, 0.17))
    connect(ax, (0.83, 0.17), (0.86, 0.17))

    connect(ax, (0.37, 0.30), (0.58, 0.80), "temporal head", text_offset=(-0.03, 0.02), color="#2563eb")
    connect(ax, (0.91, 0.25), (0.66, 0.80), "strong head", text_offset=(0.00, 0.02), color="#7c3aed")
    connect(ax, (0.72, 0.85), (0.79, 0.85))
    connect(ax, (0.21, 0.50), (0.28, 0.24), "supervision", text_offset=(-0.02, 0.00), color="#9333ea")
    connect(ax, (0.88, 0.64), (0.13, 0.50), "rate regularization", text_offset=(0.00, 0.02), color="#d97706")

    ax.text(0.03, 0.37, "Shared baseline-compatible modules", fontsize=11, color="#2563eb", fontweight="bold")
    ax.text(0.39, 0.52, "Refined transmission modules", fontsize=11, color="#be185d", fontweight="bold")
    ax.text(0.67, 0.28, "Strong refined BEV enhancement", fontsize=11, color="#6d28d9", fontweight="bold")

    legend_x = 0.03
    legend_y = 0.03
    legend_items = [
        ("Blue", "shared baseline-compatible path"),
        ("Pink / Purple", "refined-only adaptive transmission path"),
        ("Green", "physical noisy channel"),
        ("Violet", "strong BEV refinement path"),
        ("Yellow", "cost / attention helper modules"),
    ]
    ax.text(legend_x, legend_y + 0.05, "Legend", fontsize=11, fontweight="bold", color="#111827")
    for idx, (_, text) in enumerate(legend_items):
        ax.text(legend_x, legend_y + 0.03 - idx * 0.015, text, fontsize=8.5, color="#374151")

    plt.tight_layout()
    fig.savefig(PNG_PATH, dpi=220, bbox_inches="tight")
    fig.savefig(SVG_PATH, bbox_inches="tight")


if __name__ == "__main__":
    main()

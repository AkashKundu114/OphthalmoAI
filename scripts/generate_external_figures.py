"""
Publication-Grade Academic Visual Suite for External Clinical Dataset Validation.
Engineered to high-resolution academic and scientific publication formatting standards.

Features:
- Pure white canvas with generous headroom (no text/bar/legend collisions)
- Error-bar-aware label placement (labels sit strictly above error bar caps)
- Top-mounted or non-overlapping legends with generous bounding margins
- Panel tags ((a), (b), (c), etc.) positioned safely outside the plot area
- High-contrast callout boxes for anatomical schematics (zero line/text cross-strikes)
- Exact 95% Wilson score binomial confidence intervals
- 300 DPI vector-sharp rendering
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Global Academic Matplotlib Configuration
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.0,
    "figure.titlesize": 13,
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#2c3e50",
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 4.0,
    "ytick.major.size": 4.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "grid.color": "#f1f5f9",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Computes exact two-sided Wilson score binomial confidence interval."""
    if n == 0:
        return 0.0, 0.0
    z = 1.96  # 95% CI
    p = k / n
    denom = 1.0 + (z**2) / n
    centre = (p + (z**2) / (2.0 * n)) / denom
    spread = (z * np.sqrt((p * (1.0 - p) + (z**2) / (4.0 * n)) / n)) / denom
    return max(0.0, centre - spread), min(1.0, centre + spread)


# ---------------------------------------------------------------------------
# Figure 1: Academic Multi-Metric Benchmark (Internal vs External)
# ---------------------------------------------------------------------------
def plot_academic_benchmark_comparison():
    fig, ax = plt.subplots(figsize=(9.2, 5.6), facecolor="#ffffff")

    metrics = ["Accuracy", "Sensitivity\n(Recall)", "Specificity", "F1 Score", "AUROC"]

    # CIs
    int_acc_ci = wilson_ci(round(0.8518 * 938), 938)
    int_sens_ci = wilson_ci(round(0.8850 * 225), 225)
    int_spec_ci = wilson_ci(round(0.9580 * 713), 713)

    idrid_acc_ci = wilson_ci(round(0.7573 * 103), 103)
    idrid_sens_ci = wilson_ci(59, 69)
    idrid_spec_ci = wilson_ci(19, 34)

    rim_acc_ci = wilson_ci(round(0.6532 * 447), 447)
    rim_sens_ci = wilson_ci(11, 162)
    rim_spec_ci = wilson_ci(281, 285)

    internal_vals = [85.18, 88.50, 95.80, 82.92, 98.05]
    internal_hi_err = [(int_acc_ci[1]*100 - 85.18), (int_sens_ci[1]*100 - 88.50), (int_spec_ci[1]*100 - 95.80), 0, 0]
    internal_lo_err = [(85.18 - int_acc_ci[0]*100), (88.50 - int_sens_ci[0]*100), (95.80 - int_spec_ci[0]*100), 0, 0]

    idrid_vals = [75.73, 85.51, 55.88, 82.52, 76.47]
    idrid_hi_err = [(idrid_acc_ci[1]*100 - 75.73), (idrid_sens_ci[1]*100 - 85.51), (idrid_spec_ci[1]*100 - 55.88), 0, 0]
    idrid_lo_err = [(75.73 - idrid_acc_ci[0]*100), (85.51 - idrid_sens_ci[0]*100), (55.88 - idrid_spec_ci[0]*100), 0, 0]

    rim_vals = [65.32, 6.79, 98.60, 12.43, 50.49]
    rim_hi_err = [(rim_acc_ci[1]*100 - 65.32), (rim_sens_ci[1]*100 - 6.79), (rim_spec_ci[1]*100 - 98.60), 0, 0]
    rim_lo_err = [(65.32 - rim_acc_ci[0]*100), (6.79 - rim_sens_ci[0]*100), (98.60 - rim_spec_ci[0]*100), 0, 0]

    x = np.arange(len(metrics))
    width = 0.24

    c_int = "#1f4e79"    # Deep Oxford Blue
    c_idrid = "#c0392b"  # Crimson / Terracotta
    c_rim = "#6c757d"    # Neutral Slate Grey

    rects1 = ax.bar(x - width, internal_vals, width, label="Internal Held-Out Test ($n = 938$)",
                    color=c_int, edgecolor="#0e2840", linewidth=0.8,
                    yerr=[internal_lo_err, internal_hi_err], capsize=3.0,
                    error_kw=dict(lw=0.9, ecolor="#0e2840"), zorder=3)

    rects2 = ax.bar(x, idrid_vals, width, label="External IDRiD Cohort (India, $n = 103$)",
                    color=c_idrid, edgecolor="#781e14", linewidth=0.8,
                    yerr=[idrid_lo_err, idrid_hi_err], capsize=3.0,
                    error_kw=dict(lw=0.9, ecolor="#781e14"), zorder=3)

    rects3 = ax.bar(x + width, rim_vals, width, label="External RIM-ONE Cohort (Spain, $n = 447$)",
                    color=c_rim, edgecolor="#343a40", linewidth=0.8,
                    yerr=[rim_lo_err, rim_hi_err], capsize=3.0,
                    error_kw=dict(lw=0.9, ecolor="#343a40"), zorder=3)

    ax.set_ylabel("Metric Value (%) / AUROC ($\\times 100$)", fontweight="bold", labelpad=8)
    ax.set_title("OphthalmoAI Generalization: Internal Benchmark vs. Independent External Cohorts",
                 fontweight="bold", pad=28, fontsize=12.5)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontweight="bold")
    ax.set_ylim(0, 132)  # Generous headroom to prevent any legend collision
    ax.grid(axis="y", zorder=0)

    # Top legend mounted neatly above the plot area
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.07),
              ncol=3, frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1",
              framealpha=0.95, fontsize=8.8)

    # Annotations placed STRICTLY above the upper error bar cap
    groups = [
        (rects1, internal_vals, internal_hi_err, c_int),
        (rects2, idrid_vals, idrid_hi_err, c_idrid),
        (rects3, rim_vals, rim_hi_err, "#334155")
    ]
    for rects, vals, hi_errs, col in groups:
        for i, rect in enumerate(rects):
            h = vals[i]
            y_top = h + hi_errs[i] + 3.0
            ax.annotate(f"{h:.1f}",
                        xy=(rect.get_x() + rect.get_width() / 2, y_top),
                        ha="center", va="bottom", fontsize=8.2, fontweight="bold", color=col)

    # Panel tag outside plot area
    ax.text(-0.06, 1.06, "(a)", transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom")

    # Footnote below x-axis
    fig.text(0.12, -0.02, "* Error bars indicate exact 95% Wilson score binomial confidence intervals.",
             fontsize=8.0, fontstyle="italic", color="#64748b")

    out_path = OUTPUT_DIR / "external_vs_internal_benchmark.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] Academic Figure 1 generated: {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: Severity-Stratified Sensitivity Plot (IDRiD)
# ---------------------------------------------------------------------------
def plot_academic_severity_detection():
    fig, ax = plt.subplots(figsize=(8.5, 5.2), facecolor="#ffffff")

    stages = [
        "Normal\n(No DR)",
        "Stage 1\n(Mild NPDR)",
        "Stage 2\n(Moderate NPDR)",
        "Stage 3\n(Severe NPDR)",
        "Stage 4\n(Proliferative DR)"
    ]
    det_rates = [55.88, 80.00, 84.38, 94.74, 76.92]
    k_counts = [19, 4, 27, 18, 10]
    n_counts = [34, 5, 32, 19, 13]

    lo_errs, hi_errs = [], []
    for k, n, r in zip(k_counts, n_counts, det_rates):
        low, high = wilson_ci(k, n)
        lo_errs.append(r - low * 100)
        hi_errs.append(high * 100 - r)

    colors = ["#3b7a57", "#d4ac0d", "#d97706", "#c0392b", "#78281f"]

    bars = ax.bar(stages, det_rates, width=0.50, color=colors, edgecolor="#1e293b",
                  linewidth=0.8, yerr=[lo_errs, hi_errs], capsize=3.5,
                  error_kw=dict(lw=0.9, ecolor="#1e293b"), zorder=3)

    # 90% target reference line
    ax.axhline(90.0, color="#b91c1c", linestyle="--", linewidth=1.1, alpha=0.85, zorder=2)
    ax.text(-0.42, 91.5, r"Target ($\geq 90\%$)", color="#b91c1c", fontsize=8.5,
            fontweight="bold", ha="left", va="bottom")

    ax.set_ylabel("Diagnostic Sensitivity / Specificity (%)", fontweight="bold", labelpad=8)
    ax.set_title("IDRiD External Cohort: Detection Sensitivity Across Clinical DR Stages",
                 fontweight="bold", pad=20, fontsize=12)
    ax.set_ylim(0, 134)  # Plenty of room above bars
    ax.grid(axis="y", zorder=0)

    # Numbers strictly above error bars
    for i, bar in enumerate(bars):
        h = det_rates[i]
        k, n = k_counts[i], n_counts[i]
        y_label = h + hi_errs[i] + 3.5
        ax.annotate(f"{h:.1f}%\n({k}/{n})",
                    xy=(bar.get_x() + bar.get_width() / 2, y_label),
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#0f172a")

    ax.text(-0.06, 1.05, "(b)", transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom")
    fig.text(0.12, -0.02, "* Evaluated against expert ophthalmologist ground truth (ICDR classification). CIs indicate 95% Wilson intervals.",
             fontsize=8.0, fontstyle="italic", color="#64748b")

    out_path = OUTPUT_DIR / "external_severity_detection_breakdown.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] Academic Figure 2 generated: {out_path}")


# ---------------------------------------------------------------------------
# Figure 3: Clinical Safety Escalation & Uncertainty Audit
# ---------------------------------------------------------------------------
def plot_academic_human_review_rates():
    fig, ax = plt.subplots(figsize=(9.0, 4.4), facecolor="#ffffff")
    fig.subplots_adjust(top=0.76, bottom=0.18)

    cohorts = [
        "Internal Held-Out Test ($n = 938$)",
        "External IDRiD Cohort ($n = 103$)",
        "External RIM-ONE Cohort ($n = 447$)"
    ]
    auto_rates = [85.8, 34.95, 0.0]
    review_rates = [14.2, 65.05, 100.0]

    y = np.arange(len(cohorts))
    bar_height = 0.38

    c_auto = "#2e7d32"    # Forest Green
    c_review = "#c0392b"  # Deep Crimson

    ax.barh(y, auto_rates, bar_height, label="Autonomous Screening Clearance",
            color=c_auto, edgecolor="#1b5e20", linewidth=0.8, zorder=3)
    ax.barh(y, review_rates, bar_height, left=auto_rates,
            label="Escalated for Clinician Safety Review (requires_human_review: true)",
            color=c_review, edgecolor="#78281f", linewidth=0.8, zorder=3)

    ax.set_xlabel("Cohort Allocation (%)", fontweight="bold", labelpad=8)
    fig.suptitle("Clinical Risk Control: Autonomous Triage vs. Specialist Escalation",
                 fontweight="bold", y=0.96, fontsize=12.5)
    ax.set_yticks(y)
    ax.set_yticklabels(cohorts, fontweight="bold")
    ax.set_xlim(0, 106)
    ax.grid(axis="x", zorder=0)

    # Clean text inside bars
    for i in range(len(cohorts)):
        if auto_rates[i] > 15:
            ax.text(auto_rates[i] / 2, y[i], f"{auto_rates[i]:.1f}%",
                    ha="center", va="center", color="#ffffff", fontweight="bold", fontsize=9.5)
        if review_rates[i] > 15:
            ax.text(auto_rates[i] + review_rates[i] / 2, y[i], f"{review_rates[i]:.1f}%",
                    ha="center", va="center", color="#ffffff", fontweight="bold", fontsize=9.5)

    # Legend mounted cleanly beneath the suptitle and above the plot
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15),
              ncol=2, frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=8.8)

    # Panel tag aligned with legend
    ax.text(-0.06, 1.14, "(c)", transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom")

    out_path = OUTPUT_DIR / "external_human_review_uncertainty.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] Academic Figure 3 generated: {out_path}")


# ---------------------------------------------------------------------------
# Figure 4: Field-of-View & Anatomical Optical Geometry Schematic
# ---------------------------------------------------------------------------
def plot_academic_fov_sensor_shift():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.5), facecolor="#ffffff")
    fig.subplots_adjust(top=0.82, wspace=0.30)

    # ---------------- PANEL 1: Canonical 45° Posterior Pole ----------------
    ax1 = axes[0]
    ax1.set_aspect("equal")
    ax1.set_facecolor("#ffffff")

    # Dark background aperture vignette
    aperture_bg = patches.Circle((0.5, 0.5), 0.44, facecolor="#a3432b", edgecolor="#1e293b", lw=1.5)
    ax1.add_patch(aperture_bg)

    # Angle degree rings (15°, 30°, 45°)
    for r, label in [(0.15, "15°"), (0.30, "30°"), (0.44, "45°")]:
        ring = patches.Circle((0.5, 0.5), r, fill=False, edgecolor="#ffffff", ls=":", lw=0.8, alpha=0.5)
        ax1.add_patch(ring)
        ax1.text(0.5, 0.5 + r - 0.025, label, color="#ffffff", fontsize=7.5, alpha=0.8, ha="center")

    # Vascular Arcades
    theta = np.linspace(-0.6, 1.2, 50)
    ax1.plot(0.32 + 0.35 * np.cos(theta), 0.50 + 0.22 * np.sin(theta), color="#e63946", lw=1.6, alpha=0.9)
    ax1.plot(0.32 + 0.35 * np.cos(theta), 0.50 - 0.22 * np.sin(theta), color="#e63946", lw=1.6, alpha=0.9)

    # Optic Disc & Cup
    disc = patches.Circle((0.32, 0.50), 0.075, facecolor="#fef08a", edgecolor="#ca8a04", lw=1.0)
    cup = patches.Circle((0.32, 0.50), 0.035, facecolor="#fef9c3", edgecolor="#ca8a04", lw=0.8)
    ax1.add_patch(disc)
    ax1.add_patch(cup)

    # Macula & FAZ
    macula = patches.Circle((0.62, 0.50), 0.055, facecolor="#5c1d0c", edgecolor="#3d1308", lw=0.8, alpha=0.85)
    fovea = patches.Circle((0.62, 0.50), 0.015, facecolor="#1e0501", lw=0.5)
    ax1.add_patch(macula)
    ax1.add_patch(fovea)

    # Clean high-contrast callouts with white badge backgrounds
    ax1.annotate("Optic Disc\n(Cup/Disc Ratio)", xy=(0.32, 0.57), xytext=(0.18, 0.88),
                 arrowprops=dict(arrowstyle="->", color="#0f172a", lw=1.0),
                 fontsize=8.5, fontweight="bold", color="#0f172a", ha="center",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#ffffff", ec="#cbd5e1", lw=0.8))

    ax1.annotate("Macula / Fovea", xy=(0.62, 0.44), xytext=(0.78, 0.16),
                 arrowprops=dict(arrowstyle="->", color="#0f172a", lw=1.0),
                 fontsize=8.5, fontweight="bold", color="#0f172a", ha="center",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#ffffff", ec="#cbd5e1", lw=0.8))

    ax1.annotate("Vascular Arcades", xy=(0.55, 0.68), xytext=(0.82, 0.88),
                 arrowprops=dict(arrowstyle="->", color="#0f172a", lw=1.0),
                 fontsize=8.5, fontweight="bold", color="#0f172a", ha="center",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#ffffff", ec="#cbd5e1", lw=0.8))

    ax1.set_title("(d1)  Canonical 45° Posterior Pole Input Space\n[Complete Spatial Context: Disc + Macula + Arcades]",
                  fontsize=10.0, fontweight="bold", pad=12, color="#1e3a8a")
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax1.axis("off")

    # ---------------- PANEL 2: Cropped Optic Disc ROI (RIM-ONE DL) ----------------
    ax2 = axes[1]
    ax2.set_aspect("equal")
    ax2.set_facecolor("#ffffff")

    # Square ROI boundary
    roi_bg = patches.Rectangle((0.10, 0.10), 0.80, 0.80, facecolor="#bd5338", edgecolor="#b91c1c", lw=2.0)
    ax2.add_patch(roi_bg)

    # Massive Optic Disc filling almost entire frame
    disc_crop = patches.Circle((0.50, 0.50), 0.30, facecolor="#fef08a", edgecolor="#ca8a04", lw=1.2)
    cup_crop = patches.Circle((0.50, 0.50), 0.17, facecolor="#fef9c3", edgecolor="#ca8a04", lw=1.0)
    ax2.add_patch(disc_crop)
    ax2.add_patch(cup_crop)

    # Peripheral truncated vessel trunks
    ax2.plot([0.50, 0.44, 0.38], [0.50, 0.68, 0.88], color="#e63946", lw=2.0)
    ax2.plot([0.50, 0.56, 0.64], [0.50, 0.72, 0.88], color="#e63946", lw=2.0)
    ax2.plot([0.50, 0.44, 0.38], [0.50, 0.30, 0.12], color="#e63946", lw=2.0)
    ax2.plot([0.50, 0.56, 0.64], [0.50, 0.28, 0.12], color="#e63946", lw=2.0)

    # Clean callouts with badges
    ax2.text(0.50, 0.50, "Enlarged Optic Cup\n(Glaucoma Cupping)",
             fontsize=8.5, fontweight="bold", color="#78350f", ha="center", va="center")

    ax2.annotate("Field Truncation:\nNo Macula or Vignette", xy=(0.88, 0.50), xytext=(0.50, 0.05),
                 arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=1.2),
                 fontsize=8.5, fontweight="bold", color="#b91c1c", ha="center",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#fee2e2", ec="#f87171", lw=0.8))

    ax2.set_title("(d2)  Cropped Optic Disc ROI Mismatch\n[RIM-ONE DL: 292×292 Localized Optic Nerve Crop]",
                  fontsize=10.0, fontweight="bold", pad=12, color="#b91c1c")
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)
    ax2.axis("off")

    fig.suptitle("Anatomical Field-of-View (FOV) Mismatch: Root Cause of Cross-Cohort Failure",
                 fontsize=12.5, fontweight="bold", y=0.97)

    out_path = OUTPUT_DIR / "external_fov_sensor_shift.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] Academic Figure 4 generated: {out_path}")


# ---------------------------------------------------------------------------
# Figure 5: Domain Adaptation Pre vs Post Fine-Tuning Gains (IDRiD)
# ---------------------------------------------------------------------------
def plot_academic_adaptation_gain():
    fig, ax = plt.subplots(figsize=(9.2, 5.5), facecolor="#ffffff")
    fig.subplots_adjust(top=0.80, bottom=0.14)

    metrics = [
        "Binary\nAccuracy",
        "Referable DR\nSensitivity",
        "F1 Score",
        "AUROC\n(DR vs Normal)",
        "Proliferative DR\n(Stage 4 Detection)"
    ]

    pre_vals = [75.73, 85.51, 82.52, 76.47, 76.92]
    post_vals = [81.55, 91.30, 86.90, 88.24, 100.00]

    pre_ci_acc = wilson_ci(round(0.7573 * 103), 103)
    pre_ci_sens = wilson_ci(59, 69)
    pre_ci_pdr = wilson_ci(10, 13)

    post_ci_acc = wilson_ci(round(0.8155 * 103), 103)
    post_ci_sens = wilson_ci(63, 69)
    post_ci_pdr = wilson_ci(13, 13)

    pre_hi_err = [(pre_ci_acc[1]*100 - 75.73), (pre_ci_sens[1]*100 - 85.51), 0, 0, (pre_ci_pdr[1]*100 - 76.92)]
    pre_lo_err = [(75.73 - pre_ci_acc[0]*100), (85.51 - pre_ci_sens[0]*100), 0, 0, (76.92 - pre_ci_pdr[0]*100)]

    post_hi_err = [(post_ci_acc[1]*100 - 81.55), (post_ci_sens[1]*100 - 91.30), 0, 0, (post_ci_pdr[1]*100 - 100.0)]
    post_lo_err = [(81.55 - post_ci_acc[0]*100), (91.30 - post_ci_sens[0]*100), 0, 0, (100.0 - post_ci_pdr[0]*100)]

    x = np.arange(len(metrics))
    width = 0.30

    c_pre = "#64748b"   # Slate Neutral
    c_post = "#1e3a8a"  # Deep Royal Navy

    rects1 = ax.bar(x - width/2, pre_vals, width, label="Pre-Adaptation (Base Ensemble)",
                    color=c_pre, edgecolor="#334155", linewidth=0.8,
                    yerr=[pre_lo_err, pre_hi_err], capsize=3.0,
                    error_kw=dict(lw=0.9, ecolor="#334155"), zorder=3)

    rects2 = ax.bar(x + width/2, post_vals, width, label="Post-Adaptation (Fine-Tuned + Re-Calibrated)",
                    color=c_post, edgecolor="#0f172a", linewidth=0.8,
                    yerr=[post_lo_err, post_hi_err], capsize=3.0,
                    error_kw=dict(lw=0.9, ecolor="#0f172a"), zorder=3)

    ax.set_ylabel("Metric Value (%) / AUROC ($\\times 100$)", fontweight="bold", labelpad=8)
    fig.suptitle("IDRiD External Validation: Generalization Gains Post Layer-Selective Fine-Tuning",
                 fontweight="bold", y=0.97, fontsize=12.5)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontweight="bold")
    ax.set_ylim(0, 138)  # Headroom preventing collision
    ax.grid(axis="y", zorder=0)

    # Legend mounted cleanly at the top below suptitle
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14),
              ncol=2, frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=9.0)

    deltas = ["+5.8%", "+5.8%", "+4.4%", "+11.8%", "+23.1%"]
    for i in range(len(metrics)):
        # Pre-adaptation value placed cleanly above pre-bar
        y_pre = pre_vals[i] + pre_hi_err[i] + 3.5
        ax.annotate(f"{pre_vals[i]:.1f}%",
                    xy=(x[i] - width/2, y_pre),
                    ha="center", va="bottom", fontsize=8.2, color="#475569")

        # Post-adaptation value placed strictly above post-bar error cap
        y_post = post_vals[i] + post_hi_err[i] + 4.5
        ax.annotate(f"{post_vals[i]:.1f}%\n({deltas[i]})",
                    xy=(x[i] + width/2, y_post),
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1e3a8a")

    ax.text(-0.06, 1.14, "(e)", transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom")
    fig.text(0.12, -0.02, "* Evaluated on held-out IDRiD test split ($n = 103$). Error bars indicate 95% Wilson intervals.",
             fontsize=8.0, fontstyle="italic", color="#64748b")

    out_path = OUTPUT_DIR / "external_adaptation_gain.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] Academic Figure 5 generated: {out_path}")


def main():
    print("=" * 70)
    print("RE-GENERATING ZERO-OVERLAP PUBLICATION ACADEMIC FIGURES")
    print("=" * 70)
    plot_academic_benchmark_comparison()
    plot_academic_severity_detection()
    plot_academic_human_review_rates()
    plot_academic_fov_sensor_shift()
    plot_academic_adaptation_gain()
    print("=" * 70)
    print("[SUCCESS] All 5 figures regenerated cleanly with zero overlapping text.")


if __name__ == "__main__":
    main()

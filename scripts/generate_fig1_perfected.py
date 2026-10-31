"""
Ultra-perfected Figure 1 System Architecture banner for OphthalmoAI.
Engineered for razor-sharp visual clarity, strict typographic hierarchy, and zero clipping.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_figure1():
    fig = plt.figure(figsize=(7.16, 3.10), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 44)

    # -------------------------------------------------------------------------
    # Color Palette (Publication Grade)
    # -------------------------------------------------------------------------
    c_t0_bg = '#f8fafc'
    c_t0_border = '#64748b'
    c_t0_hdr = '#334155'
    
    c_t1_bg = '#f0fdf4'
    c_t1_border = '#16a34a'
    c_t1_hdr = '#15803d'

    c_t2_bg = '#faf5ff'
    c_t2_border = '#9333ea'
    c_t2_hdr = '#7e22ce'

    # -------------------------------------------------------------------------
    # 1. Main Tier Outer Boxes
    # -------------------------------------------------------------------------
    tiers = [
        (0.8, 0.8, 22.0, 42.4, 'Tier 0: Optical Guardrail', c_t0_bg, c_t0_border, c_t0_hdr),
        (25.0, 0.8, 44.4, 42.4, 'Tier 1: Multi-Backbone Diagnostic Core', c_t1_bg, c_t1_border, c_t1_hdr),
        (73.4, 0.8, 25.8, 42.4, 'Tier 2: Conformal Triage & Policy', c_t2_bg, c_t2_border, c_t2_hdr)
    ]

    for x, y, w, h, title, bg, border, header_bg in tiers:
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=1.2",
                                     facecolor=bg, edgecolor=border, lw=1.1, zorder=1)
        ax.add_patch(rect)
        hdr_h = 3.6
        hdr_patch = patches.FancyBboxPatch((x, y + h - hdr_h), w, hdr_h,
                                          boxstyle="round,pad=0.0,rounding_size=1.2",
                                          facecolor=header_bg, edgecolor=header_bg, lw=0, zorder=2)
        ax.add_patch(hdr_patch)
        ax.add_patch(patches.Rectangle((x, y + h - hdr_h), w, 1.2, facecolor=header_bg, edgecolor=header_bg, lw=0, zorder=2))
        ax.text(x + w/2.0, y + h - hdr_h/2.0, title, ha='center', va='center',
                fontsize=7.2, fontweight='bold', color='#ffffff', zorder=3)

    # -------------------------------------------------------------------------
    # 2. Tier 0 Components
    # -------------------------------------------------------------------------
    # Card 1: Input Image
    c1_x, c1_y, c1_w, c1_h = 1.8, 30.6, 20.0, 8.0
    ax.add_patch(patches.FancyBboxPatch((c1_x, c1_y), c1_w, c1_h, boxstyle="round,pad=0.0,rounding_size=0.7",
                                       facecolor='#ffffff', edgecolor='#cbd5e1', lw=0.9, zorder=2))
    ax.text(c1_x + c1_w/2, c1_y + c1_h - 2.0, 'Input Fundus Photography', ha='center', va='center',
            fontsize=6.8, fontweight='bold', color='#0f172a', zorder=3)
    ax.text(c1_x + c1_w/2, c1_y + 3.0, 'RGB ($256 \\times 256$ / $384 \\times 384$ px)\nTele-Ophthalmic Cameras',
            ha='center', va='center', fontsize=5.8, color='#475569', zorder=3)

    # Card 2: Biophysical Gate
    c2_x, c2_y, c2_w, c2_h = 1.8, 15.2, 20.0, 11.8
    ax.add_patch(patches.FancyBboxPatch((c2_x, c2_y), c2_w, c2_h, boxstyle="round,pad=0.0,rounding_size=0.7",
                                       facecolor='#ffffff', edgecolor='#94a3b8', lw=1.0, zorder=2))
    ax.add_patch(patches.Rectangle((c2_x, c2_y + c2_h - 2.8), c2_w, 2.8, facecolor='#f1f5f9', edgecolor='none', zorder=2))
    ax.text(c2_x + c2_w/2, c2_y + c2_h - 1.4, 'Deterministic Optical Gate', ha='center', va='center',
            fontsize=6.6, fontweight='bold', color='#1e293b', zorder=3)
    
    t0_bullets = (
        r"$\bullet$ Hemoglobin: $\mu_R / \mu_G \in [1.25, 2.50]$" + "\n"
        r"$\bullet$ Aperture: Circularity $\geq 0.65$" + "\n"
        r"$\bullet$ Autocorrelation & Noise Filter" + "\n"
        r"$\bullet$ Latency: 5.87 ms (CPU-bound)"
    )
    ax.text(c2_x + 1.0, c2_y + 4.5, t0_bullets, ha='left', va='center', fontsize=5.6, color='#334155', zorder=3)

    # Card 3: Fail-Closed Reject
    c3_x, c3_y, c3_w, c3_h = 1.8, 1.8, 20.0, 9.4
    ax.add_patch(patches.FancyBboxPatch((c3_x, c3_y), c3_w, c3_h, boxstyle="round,pad=0.0,rounding_size=0.7",
                                       facecolor='#fef2f2', edgecolor='#ef4444', lw=1.1, zorder=2))
    ax.text(c3_x + c3_w/2, c3_y + c3_h - 2.4, 'FAIL-CLOSED: HTTP 422', ha='center', va='center',
            fontsize=6.8, fontweight='bold', color='#b91c1c', zorder=3)
    ax.text(c3_x + c3_w/2, c3_y + 3.4, '100.0% Distractor Rejection\nImmediate Non-Fundus Interception\n(X-Ray, OCT, Noise, Charts)',
            ha='center', va='center', fontsize=5.6, color='#991b1b', zorder=3)

    # Tier 0 Arrows
    # Input -> Gate
    ax.annotate('', xy=(c1_x + c1_w/2, c2_y + c2_h), xytext=(c1_x + c1_w/2, c1_y),
                arrowprops=dict(facecolor='#64748b', edgecolor='#64748b', arrowstyle="-|>", lw=1.2, mutation_scale=8), zorder=4)
    # Gate -> Reject
    ax.annotate('', xy=(c2_x + c2_w/2, c3_y + c3_h), xytext=(c2_x + c2_w/2, c2_y),
                arrowprops=dict(facecolor='#ef4444', edgecolor='#ef4444', arrowstyle="-|>", lw=1.3, mutation_scale=8), zorder=4)
    # Reject Pill
    ax.add_patch(patches.FancyBboxPatch((c2_x + c2_w/2 - 4.2, 12.1), 8.4, 2.2, boxstyle="round,pad=0.0,rounding_size=0.5",
                                       facecolor='#ffffff', edgecolor='#ef4444', lw=0.8, zorder=5))
    ax.text(c2_x + c2_w/2, 13.2, r"$\mathcal{S}(X) < 0.50$", ha='center', va='center',
            fontsize=5.8, fontweight='bold', color='#b91c1c', zorder=6)

    # -------------------------------------------------------------------------
    # 3. Tier 1 Components
    # -------------------------------------------------------------------------
    # Tier 1A: Server Deep Vision Ensemble Sub-box
    t1a_x, t1a_y, t1a_w, t1a_h = 26.0, 17.6, 42.4, 21.0
    ax.add_patch(patches.FancyBboxPatch((t1a_x, t1a_y), t1a_w, t1a_h, boxstyle="round,pad=0.0,rounding_size=0.8",
                                       facecolor='#ffffff', edgecolor='#86efac', lw=1.0, zorder=2))
    ax.add_patch(patches.Rectangle((t1a_x, t1a_y + t1a_h - 2.8), t1a_w, 2.8, facecolor='#dcfce7', edgecolor='none', zorder=2))
    ax.text(t1a_x + t1a_w/2, t1a_y + t1a_h - 1.4, 'Tier-1A: Server GPU Ensemble (Hospital Cloud / 124.3M)',
            ha='center', va='center', fontsize=6.7, fontweight='bold', color='#166534', zorder=3)

    # 3 Backbone Cards
    models = [
        ('ConvNeXt-Small', '50.2M Params', 'AUROC 0.9979', 'Macro-F1 0.9745'),
        ('DenseNet-201', '20.0M Params', 'AUROC 0.9920', 'Macro-F1 0.9679'),
        ('EfficientNet-V2-M', '54.1M Params', 'AUROC 0.9979', 'Macro-F1 0.9861')
    ]
    bw = 12.6
    for i, (m_name, m_par, m_auc, m_f1) in enumerate(models):
        bx = t1a_x + 1.2 + i * (bw + 1.2)
        by = t1a_y + 5.8
        bh = 11.4
        ax.add_patch(patches.FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.0,rounding_size=0.6",
                                           facecolor='#f8fafc', edgecolor='#cbd5e1', lw=0.8, zorder=3))
        ax.add_patch(patches.Rectangle((bx, by + bh - 2.4), bw, 2.4, facecolor='#e2e8f0', edgecolor='none', zorder=3))
        ax.text(bx + bw/2, by + bh - 1.2, m_name, ha='center', va='center',
                fontsize=6.2, fontweight='bold', color='#0f172a', zorder=4)
        ax.text(bx + bw/2, by + 6.6, m_par, ha='center', va='center', fontsize=5.6, color='#64748b', zorder=4)
        ax.text(bx + bw/2, by + 4.2, m_auc, ha='center', va='center', fontsize=5.8, fontweight='bold', color='#0f172a', zorder=4)
        ax.text(bx + bw/2, by + 1.8, m_f1, ha='center', va='center', fontsize=5.8, color='#15803d', zorder=4)

    # Fusion Bar
    fuse_x, fuse_y, fuse_w, fuse_h = t1a_x + 1.2, t1a_y + 0.8, t1a_w - 2.4, 4.2
    ax.add_patch(patches.FancyBboxPatch((fuse_x, fuse_y), fuse_w, fuse_h, boxstyle="round,pad=0.0,rounding_size=0.6",
                                       facecolor='#ecfdf5', edgecolor='#10b981', lw=0.9, zorder=3))
    ax.text(fuse_x + fuse_w/2, fuse_y + 2.7, 'Calibrated Probability Stacking & Temperature Scaling',
            ha='center', va='center', fontsize=6.3, fontweight='bold', color='#065f46', zorder=4)
    ax.text(fuse_x + fuse_w/2, fuse_y + 1.1, 'Accuracy: 98.70% (Soft Voting 98.56%)  |  AUROC: 0.9985  |  ECE: 0.0389',
            ha='center', va='center', fontsize=5.4, color='#047857', zorder=4)

    # Tier 1B: Client Edge Auxiliary Sub-box
    t1b_x, t1b_y, t1b_w, t1b_h = 26.0, 1.8, 42.4, 14.4
    ax.add_patch(patches.FancyBboxPatch((t1b_x, t1b_y), t1b_w, t1b_h, boxstyle="round,pad=0.0,rounding_size=0.8",
                                       facecolor='#ffffff', edgecolor='#fdba74', lw=1.0, zorder=2))
    ax.add_patch(patches.Rectangle((t1b_x, t1b_y + t1b_h - 2.8), t1b_w, 2.8, facecolor='#ffedd5', edgecolor='none', zorder=2))
    ax.text(t1b_x + t1b_w/2, t1b_y + t1b_h - 1.4, 'Tier-1B: Client Edge Auxiliary (Point-of-Care / Offline Tablet)',
            ha='center', va='center', fontsize=6.7, fontweight='bold', color='#9a3412', zorder=3)
    
    # Left pill for edge model
    ax.add_patch(patches.FancyBboxPatch((t1b_x + 1.2, t1b_y + 1.2), 11.2, 9.6, boxstyle="round,pad=0.0,rounding_size=0.6",
                                       facecolor='#fff7ed', edgecolor='#fb923c', lw=0.8, zorder=3))
    ax.text(t1b_x + 6.8, t1b_y + 7.8, 'MobileNetV3', ha='center', va='center', fontsize=6.6, fontweight='bold', color='#c2410c', zorder=4)
    ax.text(t1b_x + 6.8, t1b_y + 5.8, 'ONNX FP32', ha='center', va='center', fontsize=5.8, color='#7c2d12', zorder=4)
    ax.text(t1b_x + 6.8, t1b_y + 3.8, '5.82 MB | 2.54M', ha='center', va='center', fontsize=5.4, color='#9a3412', zorder=4)
    ax.text(t1b_x + 6.8, t1b_y + 1.8, 'P50: 1.48 ms (CPU)', ha='center', va='center', fontsize=5.4, fontweight='bold', color='#c2410c', zorder=4)

    # Edge description (concise with 8px buffer)
    t1b_text = (
        r"$\bullet$ Instant point-of-care triage without cloud uplink" + "\n"
        r"$\bullet$ 636.7 QPS on unaccelerated CPU / WebAssembly" + "\n"
        r"$\bullet$ Offline triage; syncs to Tier-2 when connected"
    )
    ax.text(t1b_x + 13.0, t1b_y + 5.8, t1b_text, ha='left', va='center', fontsize=5.6, color='#431407', zorder=3)

    # Connecting Arrows from Guardrail to Tier 1 via Admission Pill
    gate_out_x, gate_out_y = c2_x + c2_w, c2_y + c2_h/2.0
    pill_center_x, pill_center_y = 23.9, 21.1
    pill_w, pill_h = 2.4, 3.4
    
    # Gate -> Pill
    ax.annotate('', xy=(pill_center_x - pill_w/2, pill_center_y), xytext=(gate_out_x, gate_out_y),
                arrowprops=dict(facecolor='#16a34a', edgecolor='#16a34a', arrowstyle="-|>", lw=1.2, mutation_scale=7), zorder=4)
    # Admission Pill
    ax.add_patch(patches.FancyBboxPatch((pill_center_x - pill_w/2, pill_center_y - pill_h/2), pill_w, pill_h,
                                       boxstyle="round,pad=0.0,rounding_size=0.4",
                                       facecolor='#ecfdf5', edgecolor='#16a34a', lw=0.8, zorder=5))
    ax.text(pill_center_x, pill_center_y, r"$\mathcal{S}(X)$" + "\n" + r"$\geq 0.50$", ha='center', va='center',
            fontsize=5.2, fontweight='bold', color='#15803d', zorder=6)
    
    # Pill -> Tier 1A
    ax.annotate('', xy=(t1a_x, t1a_y + t1a_h/2.0), xytext=(pill_center_x + pill_w/2, pill_center_y + 0.8),
                arrowprops=dict(facecolor='#16a34a', edgecolor='#16a34a', arrowstyle="-|>", lw=1.2, mutation_scale=7), zorder=4)
    # Pill -> Tier 1B
    ax.annotate('', xy=(t1b_x, t1b_y + t1b_h/2.0), xytext=(pill_center_x + pill_w/2, pill_center_y - 0.8),
                arrowprops=dict(facecolor='#ea580c', edgecolor='#ea580c', arrowstyle="-|>", lw=1.0, mutation_scale=7, linestyle='--'), zorder=4)

    # -------------------------------------------------------------------------
    # 4. Tier 2 Components
    # -------------------------------------------------------------------------
    # Card 1: AW-CRC Conformal Engine
    c_crc_x, c_crc_y, c_crc_w, c_crc_h = 74.0, 21.6, 24.6, 17.0
    ax.add_patch(patches.FancyBboxPatch((c_crc_x, c_crc_y), c_crc_w, c_crc_h, boxstyle="round,pad=0.0,rounding_size=0.7",
                                       facecolor='#ffffff', edgecolor='#c084fc', lw=1.0, zorder=2))
    ax.add_patch(patches.Rectangle((c_crc_x, c_crc_y + c_crc_h - 2.8), c_crc_w, 2.8, facecolor='#f3e8ff', edgecolor='none', zorder=2))
    ax.text(c_crc_x + c_crc_w/2, c_crc_y + c_crc_h - 1.4, 'AW-CRC Conformal Engine', ha='center', va='center',
            fontsize=6.7, fontweight='bold', color='#6b21a8', zorder=3)

    t2_bullets = (
        r"$\bullet$ Score: $s_i(X) = (1 - \hat{P}_i) / \mathcal{S}(X)^\gamma$" + "\n"
        r"$\bullet$ Emergency ($\alpha = 0.01$): $q_{\mathrm{emerg}} = 0.7017$" + "\n"
        r"   $\rightarrow$ Empirical Coverage: 99.31%" + "\n"
        r"$\bullet$ Routine ($\alpha = 0.05$): $q_{\mathrm{rout}} = 0.3055$" + "\n"
        r"   $\rightarrow$ Empirical Coverage: 96.73%" + "\n"
        r"$\bullet$ Guaranteed Non-Empty Sets: $C(X) \neq \emptyset$"
    )
    ax.text(c_crc_x + 1.0, c_crc_y + 6.6, t2_bullets, ha='left', va='center', fontsize=5.6, color='#3b0764', zorder=3)

    # Card 2: Clinical Action Protocol
    c_act_x, c_act_y, c_act_w, c_act_h = 74.0, 1.8, 24.6, 18.2
    ax.add_patch(patches.FancyBboxPatch((c_act_x, c_act_y), c_act_w, c_act_h, boxstyle="round,pad=0.0,rounding_size=0.7",
                                       facecolor='#ffffff', edgecolor='#cbd5e1', lw=1.0, zorder=2))
    ax.add_patch(patches.Rectangle((c_act_x, c_act_y + c_act_h - 2.8), c_act_w, 2.8, facecolor='#f1f5f9', edgecolor='none', zorder=2))
    ax.text(c_act_x + c_act_w/2, c_act_y + c_act_h - 1.4, 'Clinical Action Protocol', ha='center', va='center',
            fontsize=6.7, fontweight='bold', color='#0f172a', zorder=3)

    actions = [
        ('|C(X)| = 1 (Singleton, 98.3%)', 'Autonomous Screening Report', '#ecfdf5', '#10b981', '#065f46'),
        ('|C(X)| >= 2 (Multi-Class, 1.1%)', 'Urgent Specialist Referral', '#fff7ed', '#f97316', '#9a3412'),
        ('Camera Shift / OOD Inputs', '100% Mandatory Clinician Review', '#fef2f2', '#ef4444', '#991b1b')
    ]
    aw_h = 4.4
    for j, (a_cond, a_act, abg, aborder, atxt) in enumerate(actions):
        ay = c_act_y + c_act_h - 3.4 - (j + 1) * 4.6
        ax.add_patch(patches.FancyBboxPatch((c_act_x + 0.8, ay), c_act_w - 1.6, aw_h, boxstyle="round,pad=0.0,rounding_size=0.5",
                                           facecolor=abg, edgecolor=aborder, lw=0.8, zorder=3))
        ax.text(c_act_x + 1.6, ay + 2.8, a_cond, ha='left', va='center', fontsize=5.8, fontweight='bold', color=atxt, zorder=4)
        ax.text(c_act_x + 1.6, ay + 1.2, r"$\rightarrow$ " + a_act, ha='left', va='center', fontsize=5.4, color=atxt, zorder=4)

    # Tier 1A -> Tier 2 Conformal Engine Connection with intermediate pill
    arr_y = 28.0
    pill2_w, pill2_h = 2.6, 3.2
    pill2_cx = (t1a_x + t1a_w + c_crc_x) / 2.0  # 71.2
    
    # Tier 1A -> Pill
    ax.annotate('', xy=(pill2_cx - pill2_w/2, arr_y), xytext=(t1a_x + t1a_w, arr_y),
                arrowprops=dict(facecolor='#8b5cf6', edgecolor='#8b5cf6', arrowstyle="-|>", lw=1.2, mutation_scale=7), zorder=4)
    # Pill
    ax.add_patch(patches.FancyBboxPatch((pill2_cx - pill2_w/2, arr_y - pill2_h/2), pill2_w, pill2_h,
                                       boxstyle="round,pad=0.0,rounding_size=0.4",
                                       facecolor='#f5f3ff', edgecolor='#8b5cf6', lw=0.8, zorder=5))
    ax.text(pill2_cx, arr_y, r"$\hat{P}(Y|X)$", ha='center', va='center',
            fontsize=5.4, fontweight='bold', color='#6b21a8', zorder=6)
    # Pill -> Tier 2
    ax.annotate('', xy=(c_crc_x, arr_y), xytext=(pill2_cx + pill2_w/2, arr_y),
                arrowprops=dict(facecolor='#8b5cf6', edgecolor='#8b5cf6', arrowstyle="-|>", lw=1.2, mutation_scale=7), zorder=4)

    # Conformal Engine -> Action Protocol Arrow
    ax.annotate('', xy=(c_act_x + c_act_w/2, c_act_y + c_act_h), xytext=(c_act_x + c_act_w/2, c_crc_y),
                arrowprops=dict(facecolor='#64748b', edgecolor='#64748b', arrowstyle="-|>", lw=1.1, mutation_scale=8), zorder=4)

    # Save to both locations
    p1_png = 'research/figures/fig1_system_architecture.png'
    p1_pdf = 'research/figures/fig1_system_architecture.pdf'
    p2_png = 'D:/OphthalmoAI-Paper/figures/fig1_system_architecture.png'
    p2_pdf = 'D:/OphthalmoAI-Paper/figures/fig1_system_architecture.pdf'
    
    fig.savefig(p1_png, dpi=300)
    fig.savefig(p1_pdf)
    fig.savefig(p2_png, dpi=300)
    fig.savefig(p2_pdf)
    plt.close(fig)
    print("Ultra-refined Fig 1 generated successfully!")

if __name__ == '__main__':
    generate_figure1()

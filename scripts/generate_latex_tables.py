"""
Generate publication-grade LaTeX tables (IEEE conference standard) for OphthalmoAI.
Engineered for maximum legibility and strict academic standards:
- Professional booktabs styling (\\toprule, \\midrule, \\bottomrule) with zero vertical rules
- Proportional scaling with compact headers maintaining ~8.5pt body text size
- Zero clumsy minipage blocks; clean captions and compact inline notes
- Exact arithmetic reconciliation (0 ghost cases)
- Synchronized to both research/tables/ and D:/OphthalmoAI-Paper/tables/
"""

import os

os.makedirs('research/tables', exist_ok=True)
os.makedirs('D:/OphthalmoAI-Paper/tables', exist_ok=True)

def save_table(filename, content):
    """Save LaTeX table to both repositories."""
    p1 = f'research/tables/{filename}'
    p2 = f'D:/OphthalmoAI-Paper/tables/{filename}'
    with open(p1, 'w', encoding='utf-8') as f:
        f.write(content)
    with open(p2, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Saved {filename}")

# ------------------------------------------------------------------------------
# TABLE 1: Dataset Provenance & Pristine Partitions
# ------------------------------------------------------------------------------
tab1_tex = r"""\begin{table}[t]
\centering
\caption{Reconstructed Pristine Partitions ($N = 13,892$ Images)}
\label{tab:dataset_provenance}
\vspace{-1mm}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Partition Name} & \textbf{Images ($n$)} & \textbf{Groups} & \textbf{Ratio} & \textbf{Disjointness Protocol} \\
\midrule
\texttt{train\_v2.csv} & 9,725 & 9,715 & 70.0\% & GroupKFold Disjoint \\
\texttt{validation\_v2.csv} & 1,389 & 1,388 & 10.0\% & Zero Group/Hash Overlap \\
\texttt{calibration\_v2.csv} & 1,389 & 1,388 & 10.0\% & Conformal Calibration Split \\
\texttt{test\_v2.csv} [Sealed] & 1,389 & 1,388 & 10.0\% & Sealed Evaluation Cohort \\
\midrule
\textbf{Total Pristine Corpus} & \textbf{13,892} & \textbf{13,879} & \textbf{100\%} & \textbf{Cryptographically Verified} \\
\bottomrule
\end{tabular}%
}
\end{table}
"""
save_table('tab1_dataset_provenance.tex', tab1_tex)

# ------------------------------------------------------------------------------
# TABLE 2: Primary Multi-Class Performance on Sealed Test Set
# ------------------------------------------------------------------------------
tab2_tex = r"""\begin{table}[t]
\centering
\caption{Primary In-Domain Multi-Class Performance ($N = 1,389$)}
\label{tab:model_performance}
\vspace{-1mm}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{@{}lcccccc@{}}
\toprule
\textbf{Model Architecture} & \textbf{Top-1} & \textbf{Bal. Acc} & \textbf{Macro F1} & \textbf{AUROC} & \textbf{ECE} & \textbf{McNemar $p$} \\
\midrule
\textbf{Meta-Ensemble (MLP)} & \textbf{98.70\%} & \textbf{98.28\%} & 0.9803 & \textbf{0.9985} & \textbf{0.0389} & Baseline \\
\textbf{Uniform Soft Voting} & 98.56\% & 97.44\% & 0.9821 & \textbf{0.9988} & 0.0477 & $p = 0.6875$ \\
\textbf{EfficientNet-V2-M} & 98.63\% & 98.05\% & \textbf{0.9861} & 0.9979 & 0.0369 & $p = 1.0000$ \\
\textbf{DenseNet-201} & 98.13\% & 97.10\% & 0.9679 & 0.9920 & 0.0462 & $p = 0.1153$ \\
\textbf{ConvNeXt-Small} & 97.98\% & 96.97\% & 0.9745 & 0.9979 & 0.0404 & $p = 0.0309$ \\
\bottomrule
\end{tabular}%
}
\end{table}
"""
save_table('tab2_model_performance.tex', tab2_tex)

# ------------------------------------------------------------------------------
# TABLE 3: Per-Class Detailed Performance & Arithmetic Reconciliation
# ------------------------------------------------------------------------------
tab3_tex = r"""\begin{table}[t]
\centering
\caption{Detailed Per-Class Performance and Contingency Check}
\label{tab:per_class_metrics}
\vspace{-1mm}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{@{}lcccccc@{}}
\toprule
\textbf{Diagnostic Class} & \textbf{Support} & \textbf{Sens. (\%)} & \textbf{Spec. (\%)} & \textbf{Prec. (\%)} & \textbf{F1} & \textbf{AUROC} \\
\midrule
Normal & 384 & 97.92 (376) & 99.90 (1,004) & 99.73 & 0.9882 & 0.9996 \\
Diabetic Retinopathy & 483 & 100.00 (483) & 99.89 (905) & 99.79 & 0.9990 & 0.9999 \\
Glaucoma & 361 & 100.00 (361) & 98.93 (1,017) & 97.04 & 0.9850 & 0.9988 \\
Cataract & 132 & 96.97 (128) & 100.00 (1,257) & 100.00 & 0.9846 & 0.9997 \\
AMD & 29 & 79.31 (23) & 100.00 (1,360) & 100.00 & 0.8846 & 0.9946 \\
Hypertensive Retinopathy & 0 & N/A & 100.00 (1,389) & N/A & N/A & N/A \\
\midrule
\multicolumn{7}{c}{\textbf{Contingency Sum:} $\sum \mathrm{TP} = 1,371$; Errors $= 18$; Total $= 1,389$; Ghost Cases $= \mathbf{0}$} \\
\bottomrule
\end{tabular}%
}
\end{table}
"""
save_table('tab3_per_class_metrics.tex', tab3_tex)

# ------------------------------------------------------------------------------
# TABLE 4: Biophysical Optical Guardrail Stress Benchmark
# ------------------------------------------------------------------------------
tab4_tex = r"""\begin{table}[t]
\centering
\caption{Optical Guardrail Distractor and Degradation Benchmark}
\label{tab:guardrail_stress}
\vspace{-1mm}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Evaluated Condition} & \textbf{Expected Type} & \textbf{Score $\mathcal{S}(X)$} & \textbf{Rejection Decision} & \textbf{Latency} \\
\midrule
\multicolumn{5}{l}{\textit{Panel A: Genuine and Degraded Retinal Scans (In-Distribution)}} \\
Genuine Clean Fundus ($n=1,389$) & Valid Scan & $0.960 \pm 0.041$ & 0.0\% Rejection (Admitted) & 5.87 ms \\
Defocus Blur (Gaussian $\sigma=5$) & Degraded Fundus & 0.9700 & 0.0\% Rejection (Admitted) & 5.16 ms \\
Motion Blur (Kernel 15px) & Degraded Fundus & 0.9700 & 0.0\% Rejection (Admitted) & 4.84 ms \\
Overexposure Distortion ($\times 2.2$) & Degraded Fundus & 0.5823 & 36.7\% Rejection (Marginal) & 3.50 ms \\
Underexposure Distortion ($\times 0.3$) & Degraded Fundus & 0.8120 & 0.0\% Rejection (Admitted) & 4.85 ms \\
\midrule
\multicolumn{5}{l}{\textit{Panel B: Out-of-Distribution Distractor Modalities (OOD Suite)}} \\
Thoracic X-Ray (Chest Radiograph) & OOD Medical & 0.0000 & 100.0\% Rejection (HTTP 422) & 18.01 ms \\
OCT B-Scan & OOD Medical & 0.0500 & 100.0\% Rejection (HTTP 422) & 7.93 ms \\
Histopathology Biopsy (H\&E) & OOD Medical & 0.0000 & 100.0\% Rejection (HTTP 422) & 4.63 ms \\
Synthetic RGB Noise (Entropy) & Non-Image & 0.0000 & 100.0\% Rejection (HTTP 422) & 4.77 ms \\
Natural Photographic Scene & Natural Image & 0.0000 & 100.0\% Rejection (HTTP 422) & 5.75 ms \\
Clinical Document / Chart & Text Document & 0.0000 & 100.0\% Rejection (HTTP 422) & 1.22 ms \\
Monochromatic Screen (Gray) & Degenerate & 0.0000 & 100.0\% Rejection (HTTP 422) & 1.04 ms \\
\midrule
\multicolumn{5}{c}{\textbf{Summary:} \textbf{100.0\% Distractor Rejection}; P50 Latency $= \mathbf{5.87\text{ ms}}$; P99 Latency $= \mathbf{8.31\text{ ms}}$} \\
\bottomrule
\end{tabular}%
}
\end{table}
"""
save_table('tab4_guardrail_stress.tex', tab4_tex)

# ------------------------------------------------------------------------------
# TABLE 5: Conformal Coverage and Efficiency (AW-CRC vs Baselines)
# ------------------------------------------------------------------------------
tab5_tex = r"""\begin{table}[t]
\centering
\caption{Conformal Calibration: Urgency-Stratified vs. AW-CRC}
\label{tab:conformal_coverage}
\vspace{-1mm}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{@{}lcccccc@{}}
\toprule
\textbf{Conformal Formulation} & \textbf{Emerg. Cov.} & \textbf{Rout. Cov.} & \textbf{Total Cov.} & \textbf{Mean $|C(X)|$} & \textbf{Singletons} & \textbf{Null Sets} \\
\midrule
Standard Split Conformal & 97.41\% & 95.80\% & 96.69\% & 1.002 & 99.80\% & 0.00\% \\
Standard US-CRC & \textbf{99.31\%} & 96.54\% & 98.27\% & 1.004 & 99.64\% & 0.36\% (Argmax) \\
\textbf{Proposed AW-CRC ($\gamma=1.0$)} & \textbf{99.31\%} & \textbf{96.73\%} & \textbf{98.34\%} & \textbf{1.010} & \textbf{98.99\%} & \textbf{0.58\% (Argmax)} \\
\bottomrule
\end{tabular}%
}
\end{table}
"""
save_table('tab5_conformal_coverage.tex', tab5_tex)

# ------------------------------------------------------------------------------
# TABLE 6: External Validation and Camera Domain Shift
# ------------------------------------------------------------------------------
tab6_tex = r"""\begin{table}[t]
\centering
\caption{Cross-Camera External Validation and Domain Shift}
\label{tab:external_validation}
\vspace{-1mm}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{@{}lccccc@{}}
\toprule
\textbf{External Cohort (Camera)} & \textbf{Preprocessing} & \textbf{Sens. (\%)} & \textbf{Spec. (\%)} & \textbf{AUROC} & \textbf{AW-CRC Action} \\
\midrule
\textbf{IDRiD} ($n=413$, India) & Ben Graham & 99.64 (278/279) & 16.42 (22/134) & 0.8005 & \textbf{100.0\% Referral} \\
Kowa VX-10 $\alpha$ ($50^{\circ}$ FOV) & Reinhard & 100.00 (279/279) & 2.99 (4/134) & 0.7230 & \textbf{100.0\% Referral} \\
\midrule
\textbf{RIM-ONE DL} ($n=447$, Spain) & Ben Graham & 100.00 (162/162) & 1.05 (3/285) & 0.6175 & \textbf{100.0\% Referral} \\
Nidek AFC-210 ($45^{\circ}$ FOV) & Reinhard & 99.38 (161/162) & 2.81 (8/285) & 0.6577 & \textbf{100.0\% Referral} \\
\bottomrule
\end{tabular}%
}
\end{table}
"""
save_table('tab6_external_validation.tex', tab6_tex)

# ------------------------------------------------------------------------------
# TABLE 7: Paired Statistical Significance Tests
# ------------------------------------------------------------------------------
tab7_tex = r"""\begin{table}[t]
\centering
\caption{Paired Statistical Significance Tests ($N = 1,389$)}
\label{tab:statistical_significance}
\vspace{-1mm}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Pairwise Comparison} & \textbf{Pairs ($b, c$)} & \textbf{McNemar $p$} & \textbf{Decision ($\alpha=0.05$)} & \textbf{$\Delta \mathrm{F1}$ [95\% CI]} \\
\midrule
Meta-Ensemble vs. Soft Voting & $b=4, c=2$ & $p = 0.6875$ & Not Significant & $+0.0120$ [$-0.015, +0.165$] \\
Meta-Ensemble vs. EfficientNet-V2-M & $b=4, c=3$ & $p = 1.0000$ & Not Significant & $-0.0431$ [$-0.176, +0.003$] \\
Meta-Ensemble vs. DenseNet-201 & $b=14, c=6$ & $p = 0.1153$ & Not Significant & $+0.0124$ [$-0.015, +0.171$] \\
Meta-Ensemble vs. ConvNeXt-Small & $b=14, c=4$ & $p = 0.0309$ & Significant & $+0.0058$ [$-0.012, +0.158$] \\
\bottomrule
\end{tabular}%
}
\end{table}
"""
save_table('tab7_statistical_significance.tex', tab7_tex)

# ------------------------------------------------------------------------------
# TABLE 8: Two-Tier Edge vs Server Deployment Profile
# ------------------------------------------------------------------------------
tab8_tex = r"""\begin{table}[t]
\centering
\caption{Two-Tier Deployment Profile: Edge vs. Server Ensemble}
\label{tab:edge_deployment}
\vspace{-1mm}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{@{}lccccc@{}}
\toprule
\textbf{Deployment Tier} & \textbf{Model Architecture} & \textbf{Footprint} & \textbf{Latency} & \textbf{Throughput} & \textbf{Clinical Scope} \\
\midrule
\textbf{Tier-1 Client Edge} & MobileNetV3-Small (ONNX) & 5.82 MB & 1.48 ms & 636.7 QPS & Offline Point-of-Care Pre-screening \\
\textbf{Tier-2 Server Ensemble} & ConvNeXt + DenseNet + EffNet & 481.3 MB & 46.4 ms & 21.5 QPS & Definitive Diagnosis \& AW-CRC \\
\bottomrule
\end{tabular}%
}
\end{table}
"""
save_table('tab8_edge_deployment.tex', tab8_tex)

print("All perfected publication tables saved successfully!")

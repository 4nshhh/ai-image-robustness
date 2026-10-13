"""Generate results/graphs/combined_c1_c2_c3_comparison.png (reproducible).

Two panels from saved comparison data (no new computation of metrics):
left = AI recall grouped bars (baseline vs robust) for C1/C2/C3;
right = false-positive counts grouped bars (separate scale, same order).
Run: conda run -n ml_clean python scripts/plot_combined_comparison.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "results" / "graphs" / "combined_c1_c2_c3_comparison.png"
BLUE, TEAL = "#2F5597", "#0E7C7B"

comp = pd.read_csv(REPO_ROOT / "results" / "metrics"
                   / "baseline_vs_robust_efficientnet_b0.csv").set_index("condition")
order = ["c1_resize_jpeg", "c2_jpeg_resize", "c3_resize_jpeg_jpeg"]
labels = ["C1\nResize → JPEG Q50", "C2\nJPEG Q50 → Resize 50%",
          "C3\nResize → JPEG → JPEG"]
assert all(c in comp.index for c in order)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
x = np.arange(len(order))
ax1.bar(x - 0.2, [comp.loc[c, "baseline_ai_recall"] for c in order], 0.4,
        label="Baseline", color=BLUE)
ax1.bar(x + 0.2, [comp.loc[c, "robust_ai_recall"] for c in order], 0.4,
        label="Robust", color=TEAL)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=9)
ax1.set_ylabel("AI recall (rate)", fontsize=10)
ax1.set_title("AI recall: identical ingredients, order-dependent outcome", fontsize=11)
ax1.set_ylim(0.5, 1.0)
ax1.legend(fontsize=9)
ax1.grid(axis="y", alpha=0.25)

ax2.bar(x - 0.2, [comp.loc[c, "baseline_fp"] for c in order], 0.4,
        label="Baseline", color=BLUE)
ax2.bar(x + 0.2, [comp.loc[c, "robust_fp"] for c in order], 0.4,
        label="Robust", color=TEAL)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=9)
ax2.set_ylabel("False positives on Real images (count, n=600)", fontsize=10)
ax2.set_title("False positives: C2 fails Real images instead", fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(axis="y", alpha=0.25)

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150)
print(f"saved {OUT} ({OUT.stat().st_size} B)")

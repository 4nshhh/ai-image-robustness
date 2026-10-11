"""Generate results/graphs/system_architecture_pipeline.png (reproducible).

Boxes-and-arrows diagram of the documented evaluation pipeline. No data,
no metrics, no training — layout only, faithful to src/ + scripts/.
Run: conda run -n ml_clean python scripts/plot_system_architecture.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "results" / "graphs" / "system_architecture_pipeline.png"

BLUE, TEAL, GREY = "#2F5597", "#0E7C7B", "#64748B"
BOXES = [
    (0, "Frozen splits\ntrain 5,600 / val 1,200\ntest 1,186 (600 R / 586 AI)", BLUE),
    (1, "Research transforms\nJPEG · resize · recomp ×2\nC1 / C2 / C3 (ordered)", TEAL),
    (2, "Preprocessing\nRGB → 224×224 → tensor\nImageNet norm", GREY),
    (3, "EfficientNet-B0\nbaseline + robust\n[B,2] logits", BLUE),
    (4, "Threshold 0.5\n+ shared metrics\nrecall, FNR, CM, AUC", TEAL),
]

fig, ax = plt.subplots(figsize=(12, 2.6))
ax.set_xlim(0, 5)
ax.set_ylim(0, 1)
ax.axis("off")
for i, (x, label, color) in enumerate([(b[0], b[1], b[2]) for b in BOXES]):
    box = FancyBboxPatch((x + 0.08, 0.22), 0.84, 0.56, boxstyle="round,pad=0.02",
                         facecolor="white", edgecolor=color, linewidth=2)
    ax.add_patch(box)
    ax.text(x + 0.5, 0.5, label, ha="center", va="center", fontsize=8.5)
    if x < 4:
        ax.annotate("", xy=(x + 1.02, 0.5), xytext=(x + 0.92, 0.5),
                    arrowprops={"arrowstyle": "->", "color": "#334155", "lw": 1.5})
ax.set_title("Evaluation pipeline (identical for every reported number)", fontsize=11, pad=8)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150)
print(f"saved {OUT} ({OUT.stat().st_size} B)")

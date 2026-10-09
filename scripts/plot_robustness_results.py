"""Experiment 7: figures + report summary from saved Experiment 6 CSVs.

Reads only committed/saved result files; no inference, no training, no
test-set access. All numbers are computed from the CSVs (percentage points
for rate deltas); nothing is hardcoded. Fails loudly on schema/consistency
problems instead of plotting around them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS_CSV = REPO_ROOT / "results" / "metrics" / "robust_evaluation_efficientnet_b0.csv"
COMP_CSV = REPO_ROOT / "results" / "metrics" / "baseline_vs_robust_efficientnet_b0.csv"
GRAPHS_DIR = REPO_ROOT / "results" / "graphs"
SUMMARY_MD = REPO_ROOT / "results" / "metrics" / "robustness_analysis_summary.md"

EXPECTED_CONDITIONS = [
    "clean", "jpeg_q90", "jpeg_q70", "jpeg_q50", "jpeg_q30",
    "resize_075", "resize_050", "resize_025",
    "recomp_q90_p2", "recomp_q70_p2", "recomp_q50_p2", "recomp_q30_p2",
    "c1_resize_jpeg", "c2_jpeg_resize", "c3_resize_jpeg_jpeg",
]
EXPECTED_MODELS = ["baseline", "robust"]
RATE_COLS = ["accuracy", "precision", "ai_recall", "ai_f1", "roc_auc",
             "ai_fnr", "real_recall", "fpr"]
COUNT_COLS = ["tn", "fp", "fn", "tp", "n_samples"]
JPEG_QS = [90, 70, 50, 30]
RESIZE_SCALES = [0.75, 0.50, 0.25]
UNSEEN = ["resize_025", "recomp_q50_p2", "recomp_q30_p2", "c3_resize_jpeg_jpeg"]
CM_FOCUS = ["clean", "jpeg_q30", "resize_025", "c3_resize_jpeg_jpeg"]

SHORT = {
    "clean": "Clean", "jpeg_q90": "JPEG Q90", "jpeg_q70": "JPEG Q70",
    "jpeg_q50": "JPEG Q50", "jpeg_q30": "JPEG Q30",
    "resize_075": "Resize .75", "resize_050": "Resize .50", "resize_025": "Resize .25",
    "recomp_q90_p2": "Recomp Q90", "recomp_q70_p2": "Recomp Q70",
    "recomp_q50_p2": "Recomp Q50", "recomp_q30_p2": "Recomp Q30",
    "c1_resize_jpeg": "C1 R→J", "c2_jpeg_resize": "C2 J→R", "c3_resize_jpeg_jpeg": "C3 R→J→J",
}


def fail(msg: str) -> None:
    print(f"VALIDATION FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def load_and_validate() -> tuple[pd.DataFrame, pd.DataFrame]:
    for path in (METRICS_CSV, COMP_CSV):
        if not path.is_file():
            fail(f"missing source file {path}")
    df = pd.read_csv(METRICS_CSV)
    comp = pd.read_csv(COMP_CSV)
    need = {"model", "condition", "pipeline", *RATE_COLS, *COUNT_COLS}
    if missing := need - set(df.columns):
        fail(f"metrics CSV missing columns {sorted(missing)}")
    if len(df) != 30:
        fail(f"expected 30 model-condition rows, got {len(df)}")
    if sorted(df["model"].unique().tolist()) != EXPECTED_MODELS:
        fail(f"unexpected models {df['model'].unique().tolist()}")
    want_order = [c for _ in EXPECTED_MODELS for c in EXPECTED_CONDITIONS]
    if df["condition"].tolist() != want_order:
        fail("condition order unexpected (want baseline block then robust block)")
    if comp["condition"].tolist() != EXPECTED_CONDITIONS:
        fail("comparison conditions do not match metrics conditions")
    rate_vals = [df[col].to_numpy(dtype=float) for col in RATE_COLS]
    rate_vals += [comp[f"baseline_{col}"].to_numpy(dtype=float) for col in RATE_COLS]
    rate_vals += [comp[f"robust_{col}"].to_numpy(dtype=float) for col in RATE_COLS]
    vals = np.concatenate(rate_vals)
    if not np.all(np.isfinite(vals)):
        fail("non-finite rate values")
    if not ((vals >= 0.0).all() and (vals <= 1.0).all()):
        fail("rate outside [0,1]")
    for col in COUNT_COLS:
        if (df[col] < 0).any():
            fail(f"negative counts in {col}")
    if not ((df["n_samples"] == 1186).all()):
        fail("n_samples != 1186 somewhere")
    for (model, cond), grp in df.groupby(["model", "condition"]):
        row = grp.iloc[0]
        if row["tn"] + row["fp"] + row["fn"] + row["tp"] != 1186:
            fail(f"CM cells do not sum to 1186 for {model}/{cond}")
    # cross-file consistency: recompute one delta per row from model metrics
    wide = df.pivot(index="condition", columns="model")
    for _, r in comp.iterrows():
        c = r["condition"]
        for k in RATE_COLS:
            expect = (wide[(k, "robust")][c] - wide[(k, "baseline")][c]) * 100.0
            if abs(r[f"delta_pp_{k}"] - expect) > 1e-9:
                fail(f"comparison delta mismatch for {c}/{k}")
    print("Source validation OK: 30 rows, 15 conditions x 2 models, "
          "finite [0,1] rates, n=1186, CM sums exact, deltas consistent.")
    return df, comp


def _bar(ax, labels, base, rob, title, ylabel):
    x = np.arange(len(labels))
    ax.bar(x - 0.2, base, 0.4, label="Baseline")
    ax.bar(x + 0.2, rob, 0.4, label="Robust")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def main() -> None:
    df, comp = load_and_validate()
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    base = df[df["model"] == "baseline"].set_index("condition").loc[EXPECTED_CONDITIONS]
    rob = df[df["model"] == "robust"].set_index("condition").loc[EXPECTED_CONDITIONS]
    labels = [SHORT[c] for c in EXPECTED_CONDITIONS]
    made: list[Path] = []

    def save(fig, name, tight=True):
        if tight:
            fig.tight_layout()
        path = GRAPHS_DIR / name
        fig.savefig(path, dpi=150)
        plt.close(fig)
        made.append(path)

    fig, ax = plt.subplots(figsize=(12, 5))
    _bar(ax, labels, base["ai_recall"], rob["ai_recall"],
         "AI recall: baseline vs robust (15 conditions)", "AI recall")
    save(fig, "ai_recall_all_conditions_baseline_vs_robust.png")

    fig, ax = plt.subplots(figsize=(12, 5))
    _bar(ax, labels, base["accuracy"], rob["accuracy"],
         "Accuracy: baseline vs robust (15 conditions)", "Accuracy")
    save(fig, "accuracy_all_conditions_baseline_vs_robust.png")

    fig, ax = plt.subplots(figsize=(12, 5))
    _bar(ax, labels, base["ai_fnr"], rob["ai_fnr"],
         "AI false-negative rate: baseline vs robust (15 conditions)", "AI FNR")
    save(fig, "ai_fnr_all_conditions_baseline_vs_robust.png")

    jq = [f"jpeg_q{q}" for q in JPEG_QS]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(JPEG_QS, base.loc[jq, "ai_recall"], marker="o", label="Baseline")
    ax.plot(JPEG_QS, rob.loc[jq, "ai_recall"], marker="s", label="Robust")
    ax.axhline(base.loc["clean", "ai_recall"], ls="--", c="gray", lw=1, label="Baseline clean ref")
    ax.axhline(rob.loc["clean", "ai_recall"], ls=":", c="gray", lw=1, label="Robust clean ref")
    ax.set_xlabel("JPEG quality")
    ax.set_ylabel("AI recall")
    ax.set_title("AI recall vs JPEG quality")
    ax.invert_xaxis()
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save(fig, "ai_recall_vs_jpeg_quality.png")

    rs = ["resize_075", "resize_050", "resize_025"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(RESIZE_SCALES, base.loc[rs, "ai_recall"], marker="o", label="Baseline")
    ax.plot(RESIZE_SCALES, rob.loc[rs, "ai_recall"], marker="s", label="Robust")
    ax.axhline(base.loc["clean", "ai_recall"], ls="--", c="gray", lw=1, label="Baseline clean ref")
    ax.axhline(rob.loc["clean", "ai_recall"], ls=":", c="gray", lw=1, label="Robust clean ref")
    ax.set_xlabel("Resize scale")
    ax.set_ylabel("AI recall")
    ax.set_title("AI recall vs resize scale")
    ax.invert_xaxis()
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save(fig, "ai_recall_vs_resize_scale.png")

    for model_id, frame in (("baseline", base), ("robust", rob)):
        fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        for ax, cond in zip(axes, CM_FOCUS):
            cm = np.array([[frame.loc[cond, "tn"], frame.loc[cond, "fp"]],
                            [frame.loc[cond, "fn"], frame.loc[cond, "tp"]]])
            im = ax.matshow(cm, cmap="Blues")
            for (i, j), v in np.ndenumerate(cm):
                ax.text(j, i, f"{v}", ha="center", va="center")
            ax.set_title(f"{model_id}: {SHORT[cond]}", fontsize=9)
            ax.set_xlabel("Predicted (Real, AI)", fontsize=8)
            ax.set_ylabel("True (Real, AI)", fontsize=8)
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Real", "AI"], fontsize=8)
            ax.set_yticklabels(["Real", "AI"], fontsize=8)
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="Count")
        fig.suptitle(f"Confusion matrices ({model_id}): clean, JPEG Q30, resize 0.25, C3")
        save(fig, f"confusion_matrices_key_conditions_{model_id}.png", tight=False)

    # ---- report summary, computed from the CSVs ----
    comp_i = comp.set_index("condition").loc[EXPECTED_CONDITIONS]
    gains = comp_i["delta_pp_ai_recall"].sort_values(ascending=False)
    regs = comp_i[comp_i["delta_pp_ai_recall"] < 0].sort_values("delta_pp_ai_recall")
    lines = [
        "# Robustness Analysis Summary (generated from saved CSVs)",
        "",
        "Source: `results/metrics/robust_evaluation_efficientnet_b0.csv` and",
        "`results/metrics/baseline_vs_robust_efficientnet_b0.csv` "
        "(30 model-condition rows, n=1,186 each). Deltas are percentage points.",
        "",
        "## Clean-performance trade-off (EfficientNet-B0, threshold 0.5)",
        "",
        f"- Baseline clean accuracy {comp_i.loc['clean', 'baseline_accuracy']:.4f} "
        f"-> robust {comp_i.loc['clean', 'robust_accuracy']:.4f} "
        f"({comp_i.loc['clean', 'delta_pp_accuracy']:+.2f} pp).",
        f"- Baseline clean AI recall {comp_i.loc['clean', 'baseline_ai_recall']:.4f} "
        f"-> robust {comp_i.loc['clean', 'robust_ai_recall']:.4f} "
        f"({comp_i.loc['clean', 'delta_pp_ai_recall']:+.2f} pp); "
        f"AI FNR {comp_i.loc['clean', 'baseline_ai_fnr']:.4f} -> "
        f"{comp_i.loc['clean', 'robust_ai_fnr']:.4f}.",
        f"- Clean ROC-AUC {comp_i.loc['clean', 'baseline_roc_auc']:.4f} -> "
        f"{comp_i.loc['clean', 'robust_roc_auc']:.4f} "
        f"({comp_i.loc['clean', 'delta_pp_roc_auc']:+.2f} pp).",
        "",
        "## Largest AI-recall gains (robust minus baseline, pp)",
        "",
    ]
    for cond, val in gains.head(5).items():
        lines.append(f"- {cond}: {val:+.2f} pp.")
    lines += ["", "## Regressions / negligible change (AI recall delta < 0 pp)", ""]
    if len(regs):
        for cond, val in regs["delta_pp_ai_recall"].items():
            lines.append(f"- {cond}: {val:+.2f} pp.")
    else:
        lines.append("- None: robust AI recall is at or above baseline everywhere.")
    lines += ["", "## Unseen-by-training conditions (excluded from the training policy)", ""]
    for cond in UNSEEN:
        lines.append(
            f"- {cond}: baseline AI recall "
            f"{comp_i.loc[cond, 'baseline_ai_recall']:.4f} -> robust "
            f"{comp_i.loc[cond, 'robust_ai_recall']:.4f} "
            f"({comp_i.loc[cond, 'delta_pp_ai_recall']:+.2f} pp).")
    lines += [
        "",
        "## Interpretation limits",
        "",
        "- Observed patterns only; no causal mechanism is claimed.",
        "- Known confound: Real images are file-format/metadata-wise JPEG while "
        "AI images are predominantly PNG, so JPEG-related changes may reflect "
        "both compression effects and format/distribution bias.",
        "- Results cover 15 fixed conditions on one frozen test set; they do not "
        "demonstrate universal real-world robustness.",
        "",
    ]
    SUMMARY_MD.write_text("\n".join(lines))
    print(f"Summary written: {SUMMARY_MD}")

    for path in made + [SUMMARY_MD]:
        size = path.stat().st_size
        assert size > 0, f"empty output {path}"
        print(f"OK {path} ({size} B)")


if __name__ == "__main__":
    main()

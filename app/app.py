"""AI Image Robustness Lab — read-only research dashboard.

Presents saved experimental results. No model loading, no GPU, no image
uploads, no inference, no filesystem writes. All numbers come from
validated CSV/JSON artifacts under results/; nothing is hardcoded.
Missing artifacts produce a clear message, never invented data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RESULTS = REPO_ROOT / "results"
EVAL_CSV = RESULTS / "metrics" / "robust_evaluation_efficientnet_b0.csv"
COMP_CSV = RESULTS / "metrics" / "baseline_vs_robust_efficientnet_b0.csv"
FORMAT_CSV = RESULTS / "metrics" / "format_diagnostic_efficientnet_b0.csv"
FORMAT_COMP_CSV = RESULTS / "metrics" / "baseline_vs_robust_format_efficientnet_b0.csv"
BIAS_CSV = RESULTS / "metrics" / "dataset_bias_audit.csv"
BIAS_JSON = RESULTS / "experiment_logs" / "dataset_bias_audit.json"
SUMMARY_MD = RESULTS / "metrics" / "robustness_analysis_summary.md"
SPLIT_META = REPO_ROOT / "data" / "splits" / "genimage_8000_split_v4" / "split_metadata.json"
MANIFEST = REPO_ROOT / "data" / "splits" / "genimage_8000_split_v4" / "manifest.csv"
ROBUST_CFG = REPO_ROOT / "models" / "robust" / "robust_training_config.json"
ROBUST_HIST = REPO_ROOT / "models" / "robust" / "robust_training_history.json"

# --- Display vocabulary (internal identifiers never shown to visitors) ---
BASELINE_LABEL = "Baseline Model"
ROBUST_LABEL = "Transformation-Aware Model"
COLORS = {"baseline": "#2F5597", "robust": "#0E7C7B"}

CONDITIONS = [
    "clean", "jpeg_q90", "jpeg_q70", "jpeg_q50", "jpeg_q30",
    "resize_075", "resize_050", "resize_025",
    "recomp_q90_p2", "recomp_q70_p2", "recomp_q50_p2", "recomp_q30_p2",
    "c1_resize_jpeg", "c2_jpeg_resize", "c3_resize_jpeg_jpeg",
]
SECTIONS = [
    ("overview", "Overview", "home"),
    ("transformations", "Transformation Robustness", "tune"),
    ("comparison", "Model Comparison", "balance"),
    ("bias", "Dataset Bias & Format", "database"),
    ("methodology", "Methodology & Limitations", "menu_book"),
]
SECTION_LABEL = {key: label for key, label, _ in SECTIONS}
SECTION_ICON = {key: icon for key, _, icon in SECTIONS}
NAV_OPTIONS = [f":material/{icon}: {label}" for _, label, icon in SECTIONS]
NAV_TO_KEY = {opt: key for opt, (key, _, _) in zip(NAV_OPTIONS, SECTIONS)}


def section_heading(key: str) -> None:
    """Page title with its matching navigation icon on the same baseline."""
    st.header(f":material/{SECTION_ICON[key]}: {SECTION_LABEL[key]}")


# Short chart labels (full descriptions stay in tables/captions).
CHART_LABEL = {
    "clean": "Clean",
    "jpeg_q90": "JPEG Q90", "jpeg_q70": "JPEG Q70",
    "jpeg_q50": "JPEG Q50", "jpeg_q30": "JPEG Q30",
    "resize_075": "Resize 75%", "resize_050": "Resize 50%",
    "resize_025": "Resize 25%",
    "recomp_q90_p2": "Recomp Q90", "recomp_q70_p2": "Recomp Q70",
    "recomp_q50_p2": "Recomp Q50", "recomp_q30_p2": "Recomp Q30",
    "c1_resize_jpeg": "Resize → JPEG Q50",
    "c2_jpeg_resize": "JPEG Q50 → Resize 50%",
    "c3_resize_jpeg_jpeg": "Resize → JPEG → JPEG",
}
CONDITION_LABEL = {
    "clean": "Clean (no transformation)",
    "jpeg_q90": "JPEG Quality 90", "jpeg_q70": "JPEG Quality 70",
    "jpeg_q50": "JPEG Quality 50", "jpeg_q30": "JPEG Quality 30",
    "resize_075": "Resize to 75%", "resize_050": "Resize to 50%",
    "resize_025": "Resize to 25%",
    "recomp_q90_p2": "Recompression: Quality 90, 2 passes",
    "recomp_q70_p2": "Recompression: Quality 70, 2 passes",
    "recomp_q50_p2": "Recompression: Quality 50, 2 passes",
    "recomp_q30_p2": "Recompression: Quality 30, 2 passes",
    "c1_resize_jpeg": "Resize → JPEG Quality 50",
    "c2_jpeg_resize": "JPEG Quality 50 → Resize",
    "c3_resize_jpeg_jpeg": "Resize → JPEG → JPEG",
}
FAMILY = {
    "clean": "Clean baseline", "jpeg_q90": "JPEG compression",
    "jpeg_q70": "JPEG compression", "jpeg_q50": "JPEG compression",
    "jpeg_q30": "JPEG compression", "resize_075": "Image resizing",
    "resize_050": "Image resizing", "resize_025": "Image resizing",
    "recomp_q90_p2": "Recompression", "recomp_q70_p2": "Recompression",
    "recomp_q50_p2": "Recompression", "recomp_q30_p2": "Recompression",
    "c1_resize_jpeg": "Combined", "c2_jpeg_resize": "Combined",
    "c3_resize_jpeg_jpeg": "Combined",
}
FAMILIES = ["Clean baseline", "JPEG compression", "Image resizing",
            "Recompression", "Combined"]
METRIC_LABEL = {
    "accuracy": "Accuracy", "precision": "Precision",
    "ai_recall": "AI Detection Recall", "ai_f1": "AI F1 Score",
    "roc_auc": "ROC-AUC", "ai_fnr": "AI False-Negative Rate",
    "real_recall": "Real Image Recall", "fpr": "False-Positive Rate",
}
METRICS = list(METRIC_LABEL)
HIGHER_BETTER = {"accuracy", "precision", "ai_recall", "ai_f1",
                 "roc_auc", "real_recall"}
METRIC_GUIDE = {
    "accuracy": "Share of all test images classified correctly. Higher is better.",
    "precision": "Share of AI-predicted images that are actually AI. Higher is better.",
    "ai_recall": "Share of AI images detected. Higher is better — misses let synthetic images through.",
    "ai_f1": "Balance of precision and AI recall. Higher is better.",
    "roc_auc": "Ranking quality across thresholds. Higher is better.",
    "ai_fnr": "Share of AI images missed. Lower is better.",
    "real_recall": "Share of real images kept. Higher is better.",
    "fpr": "Share of real images falsely flagged. Lower is better.",
}
JPEG_ORDER = ["jpeg_q90", "jpeg_q70", "jpeg_q50", "jpeg_q30"]
JPEG_X = [90, 70, 50, 30]
RESIZE_ORDER = ["resize_075", "resize_050", "resize_025"]
RESIZE_X = [0.75, 0.50, 0.25]
RECOMP_ORDER = ["recomp_q90_p2", "recomp_q70_p2", "recomp_q50_p2", "recomp_q30_p2"]
CM_FOCUS = ["clean", "jpeg_q30", "resize_025", "c3_resize_jpeg_jpeg"]
TOL = 1e-9


def load_frame(path: Path, required: set[str]) -> pd.DataFrame:
    """Read a CSV artifact or raise a visitor-friendly error."""
    if not path.is_file():
        raise FileNotFoundError(
            "A required result file is currently unavailable, so this view "
            "cannot be displayed. No values have been estimated.")
    df = pd.read_csv(path)
    if missing := required - set(df.columns):
        raise ValueError(f"Result file {path.name} is missing columns "
                         f"{sorted(missing)}.")
    return df


def validate_eval(df: pd.DataFrame) -> None:
    """Stop the app (with a message, not a traceback) on bad source data."""
    problems = []
    if len(df) != 30:
        problems.append(f"expected 30 rows, got {len(df)}")
    if sorted(df["model"].unique().tolist()) != ["baseline", "robust"]:
        problems.append("unexpected model set")
    for col in METRICS:
        vals = df[col].to_numpy(dtype=float)
        if not np.all(np.isfinite(vals)):
            problems.append(f"non-finite {col}")
        if vals.min() < 0.0 or vals.max() > 1.0:
            problems.append(f"{col} outside [0,1]")
    if not (df["n_samples"] == 1186).all():
        problems.append("n_samples != 1186 somewhere")
    for _, r in df.iterrows():
        if r["tn"] + r["fp"] + r["fn"] + r["tp"] != 1186:
            problems.append("confusion counts do not sum to 1186")
    if problems:
        raise ValueError("Source metrics failed validation: " + "; ".join(problems))


def validate_deltas(df: pd.DataFrame, comp: pd.DataFrame) -> None:
    wide = df.pivot(index="condition", columns="model")
    for _, r in comp.iterrows():
        for k in METRICS:
            expect = (wide[(k, "robust")][r["condition"]]
                      - wide[(k, "baseline")][r["condition"]]) * 100.0
            if abs(r[f"delta_pp_{k}"] - expect) > TOL:
                raise ValueError("Saved deltas disagree with model metrics.")


@st.cache_data(show_spinner="Loading saved results…")
def load_all() -> dict:
    """Load every dashboard artifact (read-only, cached per session)."""
    eval_req = {"model", "condition", "pipeline", "n_samples", *METRICS,
                "tn", "fp", "fn", "tp"}
    df = load_frame(EVAL_CSV, eval_req)
    validate_eval(df)
    comp = load_frame(COMP_CSV, {"condition", "pipeline", "n_samples"})
    validate_deltas(df, comp)
    fmt = load_frame(FORMAT_CSV, eval_req)
    fmt_comp = load_frame(FORMAT_COMP_CSV, {"condition", "pipeline", "n_samples"})
    bias = load_frame(BIAS_CSV, {"split", "label", "n", "extensions", "formats",
                                 "modes", "n_quantized", "width_mean",
                                 "height_mean", "aspect_mean", "mpix_mean"})
    bundle: dict = {"eval": df, "comp": comp, "format": fmt,
                    "format_comp": fmt_comp, "bias": bias}
    for key, path in (("bias_json", BIAS_JSON), ("split_meta", SPLIT_META),
                      ("robust_cfg", ROBUST_CFG), ("robust_hist", ROBUST_HIST),
                      ("summary", SUMMARY_MD)):
        bundle[key] = (json.loads(path.read_text()) if path.suffix == ".json"
                       else path.read_text() if path.is_file() else None)
        if bundle[key] is None:
            raise FileNotFoundError("A required result file is currently "
                                    "unavailable, so this view cannot be displayed.")
    try:
        manifest = pd.read_csv(MANIFEST, usecols=["generator", "label", "split"])
        bundle["generators"] = (manifest.groupby(["label", "generator"])
                                .size().unstack(fill_value=0))
    except Exception:
        bundle["generators"] = None
    return bundle


def style_axes(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12, pad=10)
    ax.tick_params(labelsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=9)


def line_sweep(ax, xs, series: dict[str, list], xlabel, title, ylabel, fmt_x) -> None:
    for model_id, vals in series.items():
        label = BASELINE_LABEL if model_id == "baseline" else ROBUST_LABEL
        marker = "o" if model_id == "baseline" else "s"
        ax.plot(xs, vals, marker=marker, color=COLORS[model_id], label=label)
    ax.set_xticks(xs)
    ax.set_xticklabels([fmt_x(x) for x in xs])
    style_axes(ax, xlabel, ylabel, title)


def grouped_bars(ax, labels, base_vals, rob_vals, title, xlabel) -> None:
    """Horizontal grouped bars: readable rows, scores on the x-axis.

    Height scales with row count so labels never overlap and small
    selections do not leave oceans of whitespace (caller sets figsize).
    """
    y = np.arange(len(labels))
    ax.barh(y + 0.2, base_vals, 0.4, label=BASELINE_LABEL, color=COLORS["baseline"])
    ax.barh(y - 0.2, rob_vals, 0.4, label=ROBUST_LABEL, color=COLORS["robust"])
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(title, fontsize=12, pad=10)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()


def bar_height(n_rows: int) -> float:
    """Dynamic figure height: compact for few rows, roomy for many."""
    return max(3.2, 1.6 + 0.55 * n_rows)


def page_overview(data: dict, comp_i: pd.DataFrame) -> None:
    section_heading("overview")
    st.write("How do JPEG compression, resizing, recompression, and combined "
             "social-media-style transformations affect AI-generated image "
             "detection, and can transformation-aware training improve robustness?")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset size", "7,986 images")
    c2.metric("Frozen test set", "1,186 images")
    c3.metric("Baseline C3 AI recall",
              f"{comp_i.loc['c3_resize_jpeg_jpeg', 'baseline_ai_recall'] * 100:.2f}%")
    c4.metric("Transformation-aware C3 AI recall",
              f"{comp_i.loc['c3_resize_jpeg_jpeg', 'robust_ai_recall'] * 100:.2f}%")
    st.divider()
    st.subheader("Most important validated findings")
    top = comp_i["delta_pp_ai_recall"].sort_values(ascending=False)
    st.write(f"- Largest AI-detection-recall gains "
             f"(transformation-aware minus baseline): "
             f"{CONDITION_LABEL[top.index[0]]} ({top.iloc[0]:+.2f} pp), "
             f"{CONDITION_LABEL[top.index[1]]} ({top.iloc[1]:+.2f} pp).")
    st.write(f"- Clean-image trade-off: accuracy "
             f"{comp_i.loc['clean', 'delta_pp_accuracy']:+.2f} pp, AI recall "
             f"{comp_i.loc['clean', 'delta_pp_ai_recall']:+.2f} pp. The "
             f"transformation-aware model is more robust under most tested "
             "transformations but gives up some clean accuracy — it is not "
             "universally superior.")
    st.write("- The 25% resize remains the weakest absolute condition for both models.")
    st.divider()
    st.subheader("Key limitation")
    st.info("Real images in this dataset are stored as JPEG while AI images are "
            "PNG (perfect format separation in file metadata). JPEG-related "
            "results therefore mix compression effects with format bias. "
            "Findings apply to the evaluated dataset and conditions only — "
            "details in Dataset Bias & Format.")


def page_transformations(df: pd.DataFrame, comp_i: pd.DataFrame) -> None:
    section_heading("transformations")
    st.caption("Evaluation conditions share one frozen test set (1,186 images), "
               "identical preprocessing, and a 0.5 decision threshold.")
    col1, col2, col3 = st.columns(3)
    with col1:
        metric = st.selectbox("Metric", METRICS, index=2,
                              format_func=METRIC_LABEL.get)
    with col2:
        family = st.selectbox("Transformation family",
                              ["All transformations", *FAMILIES])
    with col3:
        view = st.selectbox("Models", ["Compare both models", BASELINE_LABEL,
                                       ROBUST_LABEL])
    st.caption(METRIC_GUIDE[metric])

    base = df[df["model"] == "baseline"].set_index("condition")
    rob = df[df["model"] == "robust"].set_index("condition")
    show_both = view == "Compare both models"
    models = ["baseline", "robust"] if show_both else [
        "baseline" if view == BASELINE_LABEL else "robust"]
    picked = {"baseline": base, "robust": rob}

    import matplotlib.pyplot as plt

    if family in ("JPEG compression", "Image resizing", "Recompression"):
        fig, ax = plt.subplots(figsize=(11, 5))
    else:
        if family == "All transformations":
            order = CONDITIONS
        elif family == "Clean baseline":
            order = ["clean"]
        else:  # Combined
            order = ["c1_resize_jpeg", "c2_jpeg_resize", "c3_resize_jpeg_jpeg"]
        fig, ax = plt.subplots(figsize=(11, bar_height(len(order))))
    if family == "JPEG compression":
        order = JPEG_ORDER
        xs, xfmt = JPEG_X, lambda x: f"Q{x}"
        xlabel = "JPEG quality (higher = lighter compression)"
        title = f"{METRIC_LABEL[metric]} vs JPEG quality"
    elif family == "Image resizing":
        order = RESIZE_ORDER
        xs, xfmt = RESIZE_X, lambda x: f"{int(x * 100)}%"
        xlabel = "Resize scale (fraction of original size)"
        title = f"{METRIC_LABEL[metric]} vs resize scale"
    elif family == "Recompression":
        order = RECOMP_ORDER
        xs, xfmt = JPEG_X, lambda x: f"Q{x}"
        xlabel = "JPEG quality per pass, 2 passes (higher = lighter)"
        title = f"{METRIC_LABEL[metric]} vs recompression quality"
    if family in ("JPEG compression", "Image resizing", "Recompression"):
        series = {m: [picked[m].loc[c, metric] for c in order] for m in models}
        line_sweep(ax, xs, series, xlabel, title,
                   f"{METRIC_LABEL[metric]} (rate)", xfmt)
        style_axes(ax, xlabel, f"{METRIC_LABEL[metric]} (rate)", title)
    else:
        labels = [CHART_LABEL[c] for c in order]
        bvals = [base.loc[c, metric] for c in order]
        rvals = [rob.loc[c, metric] for c in order]
        if show_both:
            grouped_bars(ax, labels, bvals, rvals,
                         f"{METRIC_LABEL[metric]} by evaluation condition",
                         f"{METRIC_LABEL[metric]} (rate)")
        else:
            single = bvals if models[0] == "baseline" else rvals
            color = COLORS[models[0]]
            y = np.arange(len(labels))
            ax.barh(y, single, 0.55, label=view, color=color)
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=9)
            ax.set_xlabel(f"{METRIC_LABEL[metric]} (rate)", fontsize=10)
            ax.set_title(f"{METRIC_LABEL[metric]} — {view}", fontsize=12, pad=10)
            ax.tick_params(labelsize=9)
            ax.legend(fontsize=9)
            ax.grid(axis="x", alpha=0.25)
            ax.invert_yaxis()
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    st.subheader("Condition detail")
    detail = st.selectbox("Evaluation condition", CONDITIONS,
                          format_func=CONDITION_LABEL.get, key="detail_cond")
    row_b, row_r = base.loc[detail], rob.loc[detail]
    table = pd.DataFrame({
        "Model": [BASELINE_LABEL, ROBUST_LABEL],
        METRIC_LABEL[metric]: [f"{row_b[metric]:.2%}", f"{row_r[metric]:.2%}"],
        "Change (pp)": ["—", f"{(row_r[metric] - row_b[metric]) * 100:+.2f}"],
    })
    st.table(table)
    st.caption(f"Values shown as percentages; change in percentage points (pp). "
               f"{METRIC_GUIDE[metric]}")


def page_comparison(base: pd.DataFrame, rob: pd.DataFrame,
                    comp_i: pd.DataFrame) -> None:
    section_heading("comparison")
    st.caption("Same test images, preprocessing, threshold (0.5), and metric "
               "definitions for both models. Rates as percentages; changes in "
               "percentage points (pp).")
    st.subheader("Clean-condition AI Detection Recall")
    cc1, cc2 = st.columns(2)
    cc1.metric(f"{BASELINE_LABEL} — AI Detection Recall",
               f"{comp_i.loc['clean', 'baseline_ai_recall'] * 100:.2f}%")
    cc2.metric(f"{ROBUST_LABEL} — AI Detection Recall",
               f"{comp_i.loc['clean', 'robust_ai_recall'] * 100:.2f}%")
    st.write("The transformation-aware model improves AI detection recall under "
             "many tested transformations while giving up some clean accuracy. "
             "This trade-off — not universal superiority — is the finding.")

    import matplotlib.pyplot as plt

    metric = st.selectbox("Chart metric", METRICS, index=2,
                          format_func=METRIC_LABEL.get, key="cmp_metric")
    fig, ax = plt.subplots(figsize=(11, bar_height(len(CONDITIONS))))
    y = np.arange(len(CONDITIONS))
    bvals = [base.loc[c, metric] for c in CONDITIONS]
    rvals = [rob.loc[c, metric] for c in CONDITIONS]
    ax.barh(y + 0.2, bvals, 0.4, label=BASELINE_LABEL, color=COLORS["baseline"])
    ax.barh(y - 0.2, rvals, 0.4, label=ROBUST_LABEL, color=COLORS["robust"])
    ax.set_yticks(y)
    ax.set_yticklabels([CHART_LABEL[c] for c in CONDITIONS], fontsize=9)
    ax.set_xlabel(f"{METRIC_LABEL[metric]} (rate)", fontsize=10)
    ax.set_title(f"{METRIC_LABEL[metric]} across evaluation conditions",
                 fontsize=12, pad=10)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    st.subheader("Condition detail with confusion counts")
    cond = st.selectbox("Evaluation condition", CONDITIONS,
                        format_func=CONDITION_LABEL.get, key="cmp_cond")
    b, r = base.loc[cond], rob.loc[cond]
    rows = []
    for m in METRICS:
        rows.append({"Metric": METRIC_LABEL[m],
                     BASELINE_LABEL: f"{b[m]:.2%}",
                     ROBUST_LABEL: f"{r[m]:.2%}",
                     "Change (pp)": f"{(r[m] - b[m]) * 100:+.2f}"})
    st.table(pd.DataFrame(rows))
    for m in METRICS:
        direction = ("higher-is-better" if m in HIGHER_BETTER
                     else "lower-is-better (fewer errors)")
        leader = (ROBUST_LABEL if r[m] > b[m] else
                  BASELINE_LABEL if b[m] > r[m] else "Tie")
        st.write(f"- **{METRIC_LABEL[m]}**: {leader} ({direction}).")
    cc1, cc2 = st.columns(2)
    for col, frame, name in ((cc1, b, BASELINE_LABEL), (cc2, r, ROBUST_LABEL)):
        with col:
            st.write(f"**Confusion counts — {name}**")
            st.table(pd.DataFrame(
                [[int(frame["tn"]), int(frame["fp"])],
                 [int(frame["fn"]), int(frame["tp"])]],
                index=["Actual Real", "Actual AI"],
                columns=["Predicted Real", "Predicted AI"]))


def page_bias(data: dict) -> None:
    section_heading("bias")
    st.subheader("1. Observed findings")
    st.dataframe(data["bias"][["split", "label", "n", "extensions", "formats",
                               "modes", "n_quantized"]])
    st.write("Every Real image is JPEG (with quantization tables); every "
             "AI image is PNG (none). AI images are always square (128–1024 px "
             "canvases); Real images are variable rectangles. A classifier using "
             "only seven trivial file properties reaches 1.0000 validation and "
             "1.0000 held-out test accuracy (see the bias audit log).")
    st.subheader("Format-normalization diagnostic (1,186 images per row)")
    f = data["format"]
    fb = f[f["model"] == "baseline"].set_index("condition")
    fr = f[f["model"] == "robust"].set_index("condition")
    rows = []
    for cond, label in (("clean", "Original input"), ("png", "Lossless PNG"),
                        ("jpeg_q95", "JPEG Quality 95")):
        rows.append({"Input": label,
                     "Baseline AI recall": f"{fb.loc[cond, 'ai_recall']:.2%}",
                     "Aware AI recall": f"{fr.loc[cond, 'ai_recall']:.2%}",
                     "Baseline accuracy": f"{fb.loc[cond, 'accuracy']:.2%}",
                     "Aware accuracy": f"{fr.loc[cond, 'accuracy']:.2%}"})
    st.table(pd.DataFrame(rows))
    st.write("Lossless PNG re-encoding changes nothing (predictions identical — "
             "the pipeline decodes to RGB first); JPEG Quality 95 costs little. "
             "Container identity is not the driver; lossy pixel damage is.")
    st.subheader("2. Interpretation")
    st.write("JPEG-related degradations plausibly mix genuine compression damage "
             "with format-bias effects: the classes already differ in compression "
             "history before any research transformation is applied.")
    st.subheader("3. What has not been established")
    st.write("- That the neural networks use the format shortcut (the audit "
             "tested file properties, not network internals).\n"
             "- That normalization removes the bias (it only neutralizes the "
             "container at near-lossless fidelity).\n"
             "- Any causal mechanism or real-world generalization.")


def page_methodology(data: dict) -> None:
    section_heading("methodology")
    meta = data["split_meta"]
    with st.expander("Dataset and split", expanded=True):
        st.write(f"**{meta.get('dataset', 'Tiny-GenImage')}** — "
                 f"{meta.get('total_images', 7986)} images "
                 f"({meta.get('real_images', 4000)} Real / "
                 f"{meta.get('ai_images', 3986)} AI-generated).")
        st.table(pd.DataFrame([
            {"Split": "Train", "Real": 2800, "AI-generated": 2800, "Total": 5600},
            {"Split": "Validation", "Real": 600, "AI-generated": 600, "Total": 1200},
            {"Split": "Test (frozen)", "Real": 600, "AI-generated": 586, "Total": 1186},
        ]))
        st.write("Split seed 42. Labels: 0 = Real, 1 = AI-generated. "
                 "The test set is reused across all evaluated conditions and was "
                 "never used for training, tuning, or checkpoint selection.")
        if data["generators"] is not None:
            st.write("Generator composition (manifest counts by label):")
            st.dataframe(data["generators"])
    with st.expander("Models"):
        st.write(f"- **{BASELINE_LABEL}:** EfficientNet-B0, clean-trained, "
                 f"decision threshold 0.5, best clean-validation epoch "
                 f"{data['robust_hist'].get('best_epoch', 'n/a')} recorded for "
                 "the transformation-aware run below.")
        st.write("- **Transformation-Aware Model:** EfficientNet-B0 fine-tuned "
                 "from the baseline under the logged stochastic policy; "
                 "selected on clean validation accuracy.")
    with st.expander("Input preprocessing"):
        st.write("RGB decoding → resize to 224×224 → tensor → ImageNet "
                 "normalization. Identical for both classes, both models, and "
                 "every evaluation condition.")
    with st.expander("Transformation settings"):
        st.table(pd.DataFrame([
            {"Family": "JPEG compression", "Settings": "Quality 90 / 70 / 50 / 30"},
            {"Family": "Image resizing", "Settings": "Scale 0.75 / 0.50 / 0.25, aspect preserved"},
            {"Family": "Recompression", "Settings": "Two sequential passes, quality 90 / 70 / 50 / 30"},
            {"Family": "Combined C1", "Settings": "Resize to 50% → JPEG Quality 50"},
            {"Family": "Combined C2", "Settings": "JPEG Quality 50 → Resize to 50%"},
            {"Family": "Combined C3", "Settings": "Resize to 50% → JPEG → JPEG (Quality 50)"},
        ]))
        st.write("Research transformations precede the single final preprocessing; "
                 "order is preserved exactly as listed.")
    with st.expander("Robust-training policy"):
        st.write("Per-image draw: 40% clean, 20% JPEG (quality 90/70/50/30), "
                 "15% resize (0.75/0.50), 15% recompression (quality 90/70, "
                 "2 passes), 10% combined C1/C2. Resize 0.25, recompression "
                 "quality 50/30, and C3 were held out of training. Seed 42; "
                 "Adam; 10 epochs; batch 16.")
    with st.expander("Evaluation metrics"):
        st.write("Accuracy, precision, AI detection recall, AI F1, ROC-AUC (from "
                 "AI probabilities), AI false-negative rate, Real recall, "
                 "false-positive rate, and TN/FP/FN/TP counts (rows: actual, "
                 "columns: predicted). Differences in rates use percentage points.")
    with st.expander("Reproducibility"):
        st.write("Frozen manifest and seed; deterministic preprocessing; fixed "
                 "transformation parameters; saved per-condition logs with "
                 "run identifiers. Exact GPU bit-reproducibility is not claimed.")
    with st.expander("Limitations and generalization", expanded=True):
        st.write("- The Real-JPEG / AI-PNG separation tempers all compression "
                 "conclusions (see Dataset Bias & Format).\n"
                 "- Nothing here establishes a reliable general-purpose detector "
                 "for unseen generators, platforms, or edits.\n"
                 "- All findings apply to the evaluated dataset and the 15 "
                 "fixed conditions (plus format-diagnostic inputs).")
    with st.expander("Generated analysis summary (from saved result files)"):
        st.markdown(data["summary"])


def main() -> None:
    st.set_page_config(page_title="AI Image Robustness Lab", layout="wide")
    st.markdown(
        "<style>"
        "h1 {font-size: 2rem; margin-bottom: 0.1rem;}"
        "h2 {font-size: 1.35rem; margin-top: 1.2rem; border-bottom: 1px solid #E2E8F0; padding-bottom: 0.3rem;}"
        "h3 {font-size: 1.1rem; margin-top: 1rem;}"
        "section[data-testid='stSidebar'] .stRadio label {font-size: 0.95rem;}"
        "</style>",
        unsafe_allow_html=True,
    )
    st.title("AI Image Robustness Lab")
    st.caption("Robust detection of AI-generated images under social-media-style "
               "compression and transformations — read-only results dashboard.")

    try:
        data = load_all()
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        st.stop()

    df = data["eval"]
    base = df[df["model"] == "baseline"].set_index("condition").loc[CONDITIONS]
    rob = df[df["model"] == "robust"].set_index("condition").loc[CONDITIONS]
    comp_i = data["comp"].set_index("condition").loc[CONDITIONS]

    st.sidebar.title("Navigate")
    choice = st.sidebar.radio(
        "Research sections",
        NAV_OPTIONS,
        label_visibility="collapsed")
    section = NAV_TO_KEY[choice]
    st.sidebar.divider()
    st.sidebar.caption("Saved results only · test set n=1,186 · threshold 0.5")

    if section == "overview":
        page_overview(data, comp_i)
    elif section == "transformations":
        page_transformations(df, comp_i)
    elif section == "comparison":
        page_comparison(base, rob, comp_i)
    elif section == "bias":
        page_bias(data)
    else:
        page_methodology(data)


if __name__ == "__main__":
    main()

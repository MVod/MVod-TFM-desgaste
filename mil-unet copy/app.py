"""
app.py — Streamlit dashboard for wear classification results.

Usage (local):
    uv run streamlit run app.py

Requirements:
    uv add streamlit
    # MLflow results must be present locally (download from Colab)
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import sys
import yaml
from pathlib import Path as _Path

_SRC = _Path(__file__).parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from inference_utils import SOURCES, build_inference_index, get_tool_ids, model_path
from pipeline import load_branch as _load_branch_obj

st.set_page_config(
    page_title="Wear Classification — TFM",
    page_icon="🔩",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------
st.sidebar.title("Configuration")

try:
    import mlflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
except ImportError:
    st.error("mlflow not installed. Run: uv add mlflow")
    st.stop()

BRANCH_NAMES = {"A": "MIL + EfficientNet-B0", "B": "U-Net + ISO masks", "C": "ISO Profile Comparison"}
BRANCH_COLORS = {"A": "#4e79a7", "B": "#59a14f", "C": "#f28e2b"}

# ---------------------------------------------------------------------------
# Helper — load MLflow runs
# ---------------------------------------------------------------------------
@st.cache_data(ttl=60)
def get_optimal_threshold(branch_letter: str) -> float:
    """Read f1_optimal_threshold from latest MLflow summary run. Falls back to 0.5."""
    try:
        runs = mlflow.search_runs(
            experiment_names=[f"branch_{branch_letter}"],
            filter_string="tags.run_type = 'summary'",
        )
        if runs.empty:
            return 0.5
        r = runs.sort_values("start_time", ascending=False).iloc[0]
        t = r.get("metrics.f1_optimal_threshold", float("nan"))
        if math.isnan(t):
            return 0.5
        return float(t)
    except Exception:
        return 0.5


@st.cache_data(ttl=30)
def load_summary_runs() -> pd.DataFrame:
    rows = []
    for letter in ["A", "B", "C"]:
        exp_name = f"branch_{letter}"
        try:
            runs = mlflow.search_runs(
                experiment_names=[exp_name],
                filter_string="tags.run_type = 'summary'",
            )
            if runs.empty:
                continue
            r = runs.sort_values("start_time", ascending=False).iloc[0]
            row = {"Branch": letter, "Name": BRANCH_NAMES[letter]}
            for metric in ["auc_roc", "auprc", "accuracy"]:
                row[f"{metric}_mean"] = r.get(f"metrics.{metric}_mean", float("nan"))
                row[f"{metric}_std"] = r.get(f"metrics.{metric}_std", float("nan"))
            # Prefer optimal-threshold F1 (global, meaningful for LOO-CV with 1 sample/fold)
            f1_opt = r.get("metrics.f1_optimal_mean", float("nan"))
            f1_avg = r.get("metrics.f1_mean", float("nan"))
            row["f1_mean"] = f1_opt if not math.isnan(f1_opt) else f1_avg
            row["f1_std"] = float("nan")
            rows.append(row)
        except Exception:
            pass
    return pd.DataFrame(rows)


@st.cache_data(ttl=30)
def load_all_summary_runs() -> pd.DataFrame:
    rows = []
    for letter in ["A", "B", "C"]:
        try:
            runs = mlflow.search_runs(
                experiment_names=[f"branch_{letter}"],
                filter_string="tags.run_type = 'summary'",
            )
            if runs.empty:
                continue
            for _, r in runs.iterrows():
                f1_opt = r.get("metrics.f1_optimal_mean", float("nan"))
                f1_avg = r.get("metrics.f1_mean", float("nan"))
                rows.append({
                    "Branch": letter,
                    "Name": BRANCH_NAMES[letter],
                    "Fecha": pd.to_datetime(r["start_time"], unit="ms").strftime("%Y-%m-%d %H:%M"),
                    "start_ts": r["start_time"],
                    "F1": f1_opt if not math.isnan(f1_opt) else f1_avg,
                    "AUC-ROC": r.get("metrics.auc_roc_mean", float("nan")),
                    "AUPRC": r.get("metrics.auprc_mean", float("nan")),
                    "run_id": r["run_id"],
                })
        except Exception:
            pass
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("start_ts", ascending=False).reset_index(drop=True)
    return df


@st.cache_data(ttl=30)
def load_fold_runs(branch_letter: str) -> pd.DataFrame:
    try:
        runs = mlflow.search_runs(
            experiment_names=[f"branch_{branch_letter}"],
            filter_string="tags.run_type = 'fold'",
        )
        if runs.empty:
            return pd.DataFrame()
        runs = runs.sort_values("tags.fold_id")
        keep = ["tags.fold_id", "tags.test_tool_id", "metrics.f1",
                "metrics.auc_roc", "metrics.auprc", "metrics.accuracy"]
        return runs[[c for c in keep if c in runs.columns]].rename(
            columns=lambda c: c.replace("tags.", "").replace("metrics.", "")
        )
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
st.title("🔩 Wear Classification — TFM Dashboard")
st.caption("LOO-CV results across 3 classification branches | threading tap inspection")

BRANCH_NAMES_FULL = {
    "A": "A — MIL + EfficientNet-B0",
    "B": "B — U-Net + ISO masks",
    "C": "C — ISO Profile Comparison",
}

DATA_DIR = Path("../data")


@st.cache_resource
def load_final_model(branch_letter: str):
    """Load and cache a final trained branch model. Returns None if file missing."""
    path = model_path(branch_letter)
    if not path.exists():
        return None
    branch = _load_branch_obj(branch_letter)
    branch.load(path)
    return branch


@st.cache_data
def run_inference(branch_letter: str, tool_id: str, source_subdir: str, label: int):
    """Run inference on all images of a tool. Cached per (branch, tool, source)."""
    branch = load_final_model(branch_letter)
    if branch is None:
        return None
    source_dir = DATA_DIR / source_subdir
    index = build_inference_index(source_dir, label)
    if tool_id not in index:
        return None
    config = yaml.safe_load(Path(f"configs/branch_{branch_letter.lower()}.yaml").read_text())
    t0 = time.perf_counter()
    tool_score, image_scores = branch.predict(tool_id, index, config)
    infer_total_ms = (time.perf_counter() - t0) * 1_000
    n_images = len(image_scores)
    return {
        "tool_score": float(tool_score),
        "image_scores": [float(s) for s in image_scores],
        "image_paths": [str(p) for p in index[tool_id][0]],
        "infer_total_ms": infer_total_ms,
        "infer_per_image_ms": infer_total_ms / n_images if n_images else 0.0,
        "n_images": n_images,
    }


tab_metrics, tab_history, tab_inference = st.tabs(["📊 Métricas", "📈 Histórico", "🔍 Inferencia"])

with tab_metrics:
    st.warning(
        "**Métricas LOO-CV — evaluación honesta del sistema.** "
        "En cada fold el modelo fue evaluado sobre una herramienta que nunca había visto. "
        "Las predicciones sobre herramientas conocidas (tab Inferencia) son orientativas "
        "— el modelo ya las vio durante el entrenamiento."
    )

    # ---------------------------------------------------------------------------
    # Section 1 — Comparison table
    # ---------------------------------------------------------------------------
    st.header("Branch Comparison (LOO-CV means)")

    summary = load_summary_runs()

    if summary.empty:
        st.warning("No completed runs found. Run the pipeline first, then download `mlflow.db`.")
        st.code(
            "# In Kaggle, after training:\n"
            "# Output tab → download tfm_results.zip\n\n"
            "# Then locally:\n"
            "cd jordi/ && unzip ~/Downloads/tfm_results.zip"
        )
        st.stop()

    # Format table
    display = summary.copy()
    for metric in ["f1", "auc_roc", "auprc", "accuracy"]:
        if f"{metric}_mean" in display.columns and f"{metric}_std" in display.columns:
            display[metric.upper()] = display.apply(
                lambda r, m=metric: (
                    f"{r[f'{m}_mean']:.3f} ± {r[f'{m}_std']:.3f}"
                    if not math.isnan(r[f"{m}_mean"]) and not math.isnan(r[f"{m}_std"])
                    else f"{r[f'{m}_mean']:.3f}"
                    if not math.isnan(r[f"{m}_mean"])
                    else "—"
                ),
                axis=1,
            )

    table_cols = ["Branch", "Name"] + [m.upper() for m in ["f1", "auc_roc", "auprc"] if m.upper() in display.columns]
    st.dataframe(
        display[table_cols].set_index("Branch"),
        use_container_width=True,
    )

    # Best branch callout
    if "f1_mean" in summary.columns:
        best = summary.dropna(subset=["f1_mean"]).sort_values("f1_mean", ascending=False)
        if not best.empty:
            b = best.iloc[0]
            st.success(
                f"**Best branch: {b['Branch']} — {b['Name']}** | "
                f"F1 = {b['f1_mean']:.3f} | "
                f"AUC-ROC = {b['auc_roc_mean']:.3f}"
            )

    # ---------------------------------------------------------------------------
    # Section 2 — Bar chart with error bars [COMMENTED OUT]
    # ---------------------------------------------------------------------------
    # st.header("Metric Comparison Chart")
    #
    # metrics_to_plot = st.multiselect(
    #     "Metrics", ["f1", "auc_roc", "auprc"], default=["f1", "auc_roc", "auprc"]
    # )
    #
    # if metrics_to_plot and not summary.empty:
    #     fig, ax = plt.subplots(figsize=(9, 4))
    #     n_branches = len(summary)
    #     n_metrics = len(metrics_to_plot)
    #     x = np.arange(n_branches)
    #     width = 0.8 / n_metrics
    #
    #     for i, metric in enumerate(metrics_to_plot):
    #         col_mean = f"{metric}_mean"
    #         col_std = f"{metric}_std"
    #         if col_mean not in summary.columns:
    #             continue
    #         means = summary[col_mean].fillna(0).values
    #         stds = summary[col_std].fillna(0).values
    #         offset = (i - n_metrics / 2 + 0.5) * width
    #         bars = ax.bar(x + offset, means, width, yerr=stds, capsize=4,
    #                       label=metric.upper().replace("_", "-"), alpha=0.85)
    #
    #     ax.set_xticks(x)
    #     ax.set_xticklabels([f"{r['Branch']}\n{r['Name']}" for _, r in summary.iterrows()], fontsize=9)
    #     ax.set_ylim(0, 1.05)
    #     ax.set_ylabel("Score")
    #     ax.set_title("LOO-CV metrics by branch (mean ± std)")
    #     ax.legend(loc="lower right")
    #     ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    #     ax.grid(axis="y", alpha=0.3)
    #     fig.tight_layout()
    #     st.pyplot(fig)
    #     plt.close(fig)

    # ---------------------------------------------------------------------------
    # Section 3 — Per-fold detail [COMMENTED OUT]
    # ---------------------------------------------------------------------------
    # st.header("Per-Fold Detail")
    #
    # col1, col2 = st.columns([1, 3])
    # with col1:
    #     selected_branch = st.selectbox("Branch", ["A", "B", "C"], format_func=lambda x: f"{x} — {BRANCH_NAMES[x]}")
    #
    # folds = load_fold_runs(selected_branch)
    #
    # if folds.empty:
    #     st.info(f"No fold runs found for Branch {selected_branch}.")
    # else:
    #     with col2:
    #         st.dataframe(folds.set_index("fold_id"), use_container_width=True)
    #
    #     # Boxplot per metric
    #     metric_cols = [c for c in ["f1", "auc_roc", "auprc"] if c in folds.columns]
    #     if metric_cols:
    #         fig2, ax2 = plt.subplots(figsize=(7, 3.5))
    #         data = [folds[m].dropna().values for m in metric_cols]
    #         bp = ax2.boxplot(data, labels=[m.upper().replace("_", "-") for m in metric_cols],
    #                          patch_artist=True, medianprops=dict(color="black", linewidth=2))
    #         color = BRANCH_COLORS.get(selected_branch, "#888")
    #         for patch in bp["boxes"]:
    #             patch.set_facecolor(color)
    #             patch.set_alpha(0.7)
    #         ax2.set_ylim(-0.05, 1.05)
    #         ax2.set_title(f"Branch {selected_branch} — LOO-CV distribution ({len(folds)} folds)")
    #         ax2.grid(axis="y", alpha=0.3)
    #         ax2.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    #         fig2.tight_layout()
    #         st.pyplot(fig2)
    #         plt.close(fig2)

    # ---------------------------------------------------------------------------
    # Section 4 — Fold scores scatter (all branches overlay) [COMMENTED OUT]
    # ---------------------------------------------------------------------------
    # st.header("F1 Score Distribution — All Branches")
    #
    # fig3, ax3 = plt.subplots(figsize=(9, 4))
    # any_data = False
    # for letter in ["A", "B", "C"]:
    #     df = load_fold_runs(letter)
    #     if df.empty or "f1" not in df.columns:
    #         continue
    #     f1_vals = df["f1"].dropna().values
    #     if len(f1_vals) == 0:
    #         continue
    #     ax3.scatter(
    #         range(len(f1_vals)), f1_vals,
    #         label=f"{letter} — {BRANCH_NAMES[letter]}",
    #         color=BRANCH_COLORS[letter], alpha=0.8, s=50,
    #     )
    #     ax3.axhline(f1_vals.mean(), color=BRANCH_COLORS[letter],
    #                 linestyle="--", linewidth=1.0, alpha=0.6)
    #     any_data = True
    #
    # if any_data:
    #     ax3.set_xlabel("Fold index")
    #     ax3.set_ylabel("F1 Score")
    #     ax3.set_ylim(-0.05, 1.05)
    #     ax3.set_title("Per-fold F1 score by branch (dashed = mean)")
    #     ax3.legend(fontsize=8)
    #     ax3.grid(alpha=0.3)
    #     fig3.tight_layout()
    #     st.pyplot(fig3)
    #     plt.close(fig3)
    # else:
    #     st.info("Run at least one branch to see the scatter plot.")

    # ---------------------------------------------------------------------------
    # Section 5 — Prediction visualizations
    # ---------------------------------------------------------------------------
    st.header("Prediction Visualizations")

    VIS_DIR = Path("output/vis")

    vis_branches = [d.name for d in sorted(VIS_DIR.iterdir()) if d.is_dir()] if VIS_DIR.exists() else []

    if not vis_branches:
        st.info("No visualizations yet. Run the pipeline with `vis_dir: output/vis` in config.")
    else:
        vcol1, vcol2 = st.columns([1, 3])
        with vcol1:
            vis_branch = st.selectbox("Branch", vis_branches, key="vis_branch")
            branch_vis_dir = VIS_DIR / vis_branch
            all_imgs = sorted(branch_vis_dir.glob("*.png"))
            tool_ids = sorted({p.stem.rsplit("_", 1)[0] for p in all_imgs})
            vis_tool = st.selectbox("Tool", tool_ids, key="vis_tool")

        imgs_for_tool = sorted(branch_vis_dir.glob(f"{vis_tool}_*.png"))

        caption = "Left→Right: tool ROI | predicted mask | ideal mask | deviation map"

        st.caption(caption)

        cols = st.columns(min(len(imgs_for_tool), 3))
        for i, img_path in enumerate(imgs_for_tool):
            with cols[i % 3]:
                st.image(str(img_path), caption=img_path.stem, use_container_width=True)

    # ---------------------------------------------------------------------------
    # Footer
    # ---------------------------------------------------------------------------
    st.divider()
    st.caption(
        "Tracking URI: `sqlite:///mlflow.db` — "
        "Refresh page to reload data (cache TTL: 30s)"
    )

with tab_history:
    st.header("Histórico de experimentos")
    st.caption("Todos los runs de tipo 'summary' en la base de datos, ordenados por fecha.")

    all_runs = load_all_summary_runs()

    if all_runs.empty:
        st.info("No hay runs en la base de datos.")
    else:
        # --- Table ---
        display_cols = ["Fecha", "Branch", "Name", "F1", "AUC-ROC", "AUPRC", "run_id"]
        st.dataframe(
            all_runs[[c for c in display_cols if c in all_runs.columns]]
            .style.format({"F1": "{:.3f}", "AUC-ROC": "{:.3f}", "AUPRC": "{:.3f}"}, na_rep="—"),
            use_container_width=True,
            hide_index=True,
        )

        # --- Evolution chart (only if >1 run exists for any branch) ---
        runs_per_branch = all_runs.groupby("Branch").size()
        has_history = (runs_per_branch > 1).any()

        metric_choice = st.selectbox("Métrica", ["F1", "AUC-ROC", "AUPRC"], key="hist_metric")

        if has_history:
            fig_h, ax_h = plt.subplots(figsize=(10, 4))
            for letter in ["A", "B", "C"]:
                branch_df = all_runs[all_runs["Branch"] == letter].sort_values("start_ts")
                if branch_df.empty or branch_df[metric_choice].isna().all():
                    continue
                ax_h.plot(
                    range(len(branch_df)), branch_df[metric_choice],
                    marker="o", label=f"{letter} — {BRANCH_NAMES[letter]}",
                    color=BRANCH_COLORS[letter],
                )
                for i, (_, row) in enumerate(branch_df.iterrows()):
                    ax_h.annotate(
                        row["Fecha"][5:16],
                        (i, row[metric_choice]),
                        textcoords="offset points", xytext=(0, 8),
                        fontsize=7, ha="center", color=BRANCH_COLORS[letter],
                    )
            ax_h.set_ylabel(metric_choice)
            ax_h.set_xlabel("Ejecución (cronológico)")
            ax_h.set_ylim(0, 1.05)
            ax_h.set_title(f"{metric_choice} — evolución entre ejecuciones")
            ax_h.legend(fontsize=8)
            ax_h.grid(alpha=0.3)
            ax_h.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
            fig_h.tight_layout()
            st.pyplot(fig_h)
            plt.close(fig_h)
        else:
            st.info(
                "Solo hay una ejecución por branch. El gráfico de evolución aparece "
                "cuando hay más de un run por branch en la base de datos."
            )

        # --- Side-by-side comparison of selected runs ---
        st.subheader("Comparar dos ejecuciones")
        run_labels = all_runs["Fecha"] + " — Branch " + all_runs["Branch"]
        col_a, col_b = st.columns(2)
        with col_a:
            sel_a = st.selectbox("Ejecución A", run_labels, index=0, key="cmp_a")
        with col_b:
            sel_b = st.selectbox("Ejecución B", run_labels,
                                 index=min(1, len(run_labels) - 1), key="cmp_b")

        idx_a = run_labels[run_labels == sel_a].index[0]
        idx_b = run_labels[run_labels == sel_b].index[0]
        cmp = pd.DataFrame([all_runs.iloc[idx_a], all_runs.iloc[idx_b]])[
            ["Fecha", "Branch", "Name", "F1", "AUC-ROC", "AUPRC"]
        ].set_index("Fecha")
        st.dataframe(
            cmp.style.format({"F1": "{:.3f}", "AUC-ROC": "{:.3f}", "AUPRC": "{:.3f}"}, na_rep="—"),
            use_container_width=True,
        )

with tab_inference:
    left_col, right_col = st.columns([1, 2])

    with left_col:
        branch_choice = st.selectbox(
            "Branch",
            list(BRANCH_NAMES_FULL.keys()),
            format_func=lambda x: BRANCH_NAMES_FULL[x],
            key="inf_branch",
        )

        source_label = st.radio(
            "Fuente de imágenes",
            list(SOURCES.keys()),
            index=2,
            key="inf_source",
        )
        source_subdir, label_int = SOURCES[source_label]
        source_dir = DATA_DIR / source_subdir

        if not source_dir.exists():
            st.error(f"Carpeta no encontrada: `{source_dir}`")
        else:
            index_preview = build_inference_index(source_dir, label_int)
            tool_ids = get_tool_ids(index_preview)

            if not tool_ids:
                st.warning("No se encontraron herramientas en esta carpeta.")
            else:
                selected_tool = st.selectbox("Herramienta", tool_ids, key="inf_tool")

                preview_paths = index_preview[selected_tool][0]
                st.caption(f"{len(preview_paths)} imágenes")
                preview_cols = st.columns(min(len(preview_paths), 3))
                for i, p in enumerate(preview_paths):
                    with preview_cols[i % 3]:
                        st.image(str(p), use_container_width=True)

                run_btn = st.button("Analizar", type="primary", key="inf_run")

    with right_col:
        if not source_dir.exists():
            st.error(f"Carpeta no encontrada: `{source_dir}`")
        elif not tool_ids:
            st.warning("No hay herramientas disponibles.")
        else:
            # Disclaimer for known images (normal/worn, not intermediate)
            if source_label != "Intermedias (36)":
                st.warning(
                    "**Herramienta conocida** — el modelo la vio durante el entrenamiento. "
                    "Resultado orientativo. Las métricas reales están en el tab 📊 Métricas."
                )

            # Model availability check
            m_path = model_path(branch_choice)
            if not m_path.exists():
                st.error(
                    f"Modelo no encontrado: `{m_path}`\n\n"
                    f"Ejecuta: `uv run python train_final.py --branch {branch_choice}`"
                )
            elif run_btn:
                with st.spinner(f"Analizando {selected_tool}..."):
                    result = run_inference(
                        branch_choice, selected_tool, source_subdir, label_int
                    )

                if result is None:
                    st.error("Error al ejecutar la inferencia.")
                else:
                    score = result["tool_score"]
                    threshold = get_optimal_threshold(branch_choice)
                    is_worn = score >= threshold
                    label_text = "DESGASTADA" if is_worn else "NORMAL"
                    label_icon = "⚠️" if is_worn else "✅"

                    st.subheader(f"{label_icon} {label_text}")
                    col_score, col_thr = st.columns(2)
                    col_score.metric("Score global", f"{score:.3f}")
                    col_thr.metric("Umbral (LOO-CV óptimo)", f"{threshold:.3f}")
                    st.progress(min(score, 1.0))

                    st.subheader("⏱️ Tiempos de inferencia")
                    col_t1, col_t2, col_t3 = st.columns(3)
                    col_t1.metric(
                        "Total herramienta",
                        f"{result['infer_total_ms']:.0f} ms",
                        help=f"Tiempo total de predict() sobre {result['n_images']} imágenes",
                    )
                    col_t2.metric(
                        "Por imagen (media)",
                        f"{result['infer_per_image_ms']:.1f} ms",
                        help="infer_total_ms / n_images",
                    )
                    col_t3.metric(
                        "Imágenes procesadas",
                        str(result["n_images"]),
                    )
                    st.caption(
                        "Tiempo medido con `time.perf_counter()` en CPU "
                        f"({'primera ejecución real' if not st.session_state.get(f'cached_{branch_choice}_{selected_tool}') else 'resultado cacheado — tiempo de la primera ejecución'})."
                    )
                    st.session_state[f"cached_{branch_choice}_{selected_tool}"] = True

                    img_names = [Path(p).name for p in result["image_paths"]]
                    scores_df = pd.DataFrame({
                        "Imagen": img_names,
                        "Score": [f"{s:.3f}" for s in result["image_scores"]],
                    })
                    st.dataframe(scores_df.set_index("Imagen"), use_container_width=True)

                    vis_dir = Path("output/vis") / branch_choice
                    heatmap_imgs = (
                        sorted(vis_dir.glob(f"{selected_tool}_*.png"))
                        if vis_dir.exists() else []
                    )
                    if heatmap_imgs:
                        st.subheader("Heatmaps")
                        h_cols = st.columns(min(len(heatmap_imgs), 3))
                        for i, hp in enumerate(heatmap_imgs):
                            with h_cols[i % 3]:
                                st.image(str(hp), caption=hp.stem, use_container_width=True)
            else:
                st.info("Selecciona una herramienta y pulsa **Analizar**.")

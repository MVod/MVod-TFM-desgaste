# app/app.py
import os

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import json as _json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

matplotlib.use("Agg")

from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from config import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
    MODELS_DIR,
    NUM_EPOCHS,
    ROSCAS_BUENAS,
    ROSCAS_MALAS,
    es_buena,
    get_roscas_anotadas,
    get_ruta_rosca,
)
from logger import get_logger

log = get_logger("app")
from db import (
    actualizar_metricas_avanzadas,
    actualizar_tiempos,
    borrar_experimento,
    borrar_todos_experimentos,
    buscar_experimento,
    get_accuracy,
    get_mejor_modelo,
    get_resultados,
    guardar_experimento,
    guardar_resultado,
    init_db,
    listar_experimentos,
    migrar_db,
    set_mejor_modelo,
)
from inference import (
    UMBRAL_ROSCA,
    diagnosticar_rosca,
    generar_gradcam,
    generar_overlay_iso,
    overlay_gradcam,
)
from model import cargar_modelo, nombre_checkpoint
from training import entrenar

GLOBAL_CSS = """
<style>
.stApp { background: #0f1117; }
section[data-testid="stSidebar"] {
    background: #13151f !important;
    border-right: 1px solid #2d3148;
}
#MainMenu, footer { visibility: hidden; }
div[data-testid="stDecoration"] { display: none; }

div[data-testid="stMetric"] {
    background: #1a1d27;
    border: 1px solid #2d3148;
    border-radius: 10px;
    padding: 16px 20px;
}
div[data-testid="stMetricLabel"] p { color: #94a3b8 !important; font-size: 12px !important; }
div[data-testid="stMetricValue"] { color: #f1f5f9 !important; }

div[data-testid="stButton"] button[kind="primary"] {
    background: #4f8ef7 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
div[data-testid="stButton"] button[kind="primary"]:hover {
    background: #3b7de8 !important;
}
div[data-testid="stExpander"] {
    background: #1a1d27 !important;
    border: 1px solid #2d3148 !important;
    border-radius: 10px !important;
}
div[data-testid="stButton"] button {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    width: 100% !important;
}
</style>
"""

st.set_page_config(page_title="LoRA + SAM", page_icon="🔩", layout="wide")

init_db()
migrar_db()
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

def _buscar_exp_por_id(exp_id: int) -> dict | None:
    return next((e for e in listar_experimentos() if e["id"] == exp_id), None)


def _cargar_history(modelo_path: str | None) -> dict | None:
    if not modelo_path:
        return None
    p = Path(modelo_path).with_suffix(".json")
    if not p.exists():
        return None
    try:
        with open(p) as _f:
            return _json.load(_f)
    except Exception:
        return None


def calcular_metricas_clasificacion(resultados: list[dict]) -> dict:
    labels = [0 if r["es_buena"] else 1 for r in resultados]
    scores = [r["score_medio"] for r in resultados]
    preds = [0 if r["veredicto"] == "BUENA" else 1 for r in resultados]
    if len(set(labels)) < 2:
        return {"auc_roc": None, "f1": None, "auprc": None}
    try:
        return {
            "auc_roc": round(roc_auc_score(labels, scores), 4),
            "f1": round(f1_score(labels, preds, zero_division=0), 4),
            "auprc": round(average_precision_score(labels, scores), 4),
        }
    except Exception:
        return {"auc_roc": None, "f1": None, "auprc": None}


def _auto_run_combo(
    roscas_train: list[str],
    roscas_eval: list[str],
    epochs: int,
    lr: float,
    batch: int,
) -> dict:
    """Entrena (o reutiliza checkpoint) y devuelve métricas para un combo dado."""
    ckpt_path = MODELS_DIR / nombre_checkpoint(roscas_train)
    exp = buscar_experimento(roscas_train)


    if (
        exp
        and ckpt_path.exists()
        and exp.get("auc_roc") is not None
        and exp.get("tipo") == "auto"
    ):
        acc_db = get_accuracy(exp["id"])
        return {
            "exp_id": exp["id"],
            "auc": exp["auc_roc"],
            "f1": exp.get("f1"),
            "acc": acc_db,
            "roscas_train": roscas_train,
        }

    if not ckpt_path.exists() or (ckpt_path.exists() and exp is None):
        if ckpt_path.exists():
            ckpt_path.unlink(missing_ok=True)
        res_t = entrenar(
            roscas_train=roscas_train, epochs=epochs, lr=lr, batch_size=batch
        )
        exp_id = guardar_experimento(
            roscas_train=roscas_train,
            modelo_path=res_t["modelo_path"],
            iou_cresta=res_t["iou_cresta"],
            iou_paso=res_t["iou_paso"],
            loss_final=res_t["loss_final"],
            epochs=res_t["epochs"],
            nombre=f"auto_{'_'.join(sorted(roscas_train))}",
            roscas_eval=roscas_eval,
            tipo="auto",
            tiempo_train_s=res_t.get("tiempo_train_s"),
        )
    else:

        if exp.get("tipo") == "auto":
            exp_id = exp["id"]
        else:
            exp_id = guardar_experimento(
                roscas_train=roscas_train,
                modelo_path=str(ckpt_path),
                iou_cresta=exp.get("iou_cresta"),
                iou_paso=exp.get("iou_paso"),
                loss_final=exp.get("loss_final"),
                epochs=exp.get("epochs"),
                nombre=f"auto_{'_'.join(sorted(roscas_train))}",
                roscas_eval=roscas_eval,
                tipo="auto",
                tiempo_train_s=exp.get("tiempo_train_s"),
            )


    sam_a, head_a, _ = cargar_modelo(str(ckpt_path))
    res_inf: list[dict] = []
    t_inf_start = time.time()
    for rosca_id in roscas_eval:
        try:
            r = diagnosticar_rosca(get_ruta_rosca(rosca_id), sam_a, head_a)
            r["rosca_id"] = rosca_id
            r["es_buena"] = es_buena(rosca_id)
            res_inf.append(r)
            if exp_id:
                guardar_resultado(
                    experimento_id=exp_id,
                    rosca_id=rosca_id,
                    es_buena=es_buena(rosca_id),
                    cv_cresta=r["cv_medio"],
                    score=r["score_medio"],
                    diagnostico=r["veredicto"],
                )
        except Exception as _exc:
            log.warning("Auto combo %s - eval %s: %s", roscas_train, rosca_id, _exc)
    t_inf_s = round(time.time() - t_inf_start, 1)

    del sam_a, head_a
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mc = calcular_metricas_clasificacion(res_inf)
    if exp_id:
        actualizar_metricas_avanzadas(
            exp_id, mc.get("auc_roc"), mc.get("f1"), mc.get("auprc")
        )
        actualizar_tiempos(exp_id, tiempo_inf_s=t_inf_s)

    n_ok = sum(1 for r in res_inf if (r["veredicto"] == "BUENA") == r["es_buena"])
    acc = n_ok / len(res_inf) if res_inf else 0.0
    return {
        "exp_id": exp_id,
        "auc": mc.get("auc_roc"),
        "f1": mc.get("f1"),
        "acc": acc,
        "roscas_train": roscas_train,
    }


def render_empty_state():
    st.markdown(
        """
    <div style="text-align:center; padding:60px 20px;">
        <div style="font-size:48px; margin-bottom:16px;">🔩</div>
        <div style="font-size:18px; font-weight:600; color:#f1f5f9; margin-bottom:8px;">
            Sin resultados todavía
        </div>
        <div style="font-size:14px; color:#94a3b8;">
            Configura y ejecuta un experimento para ver el diagnóstico aquí
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_progress_panel(epoch, total, loss, iou_f, iou_p, placeholder):
    pct = int(epoch / total * 100)
    placeholder.markdown(
        f"""
    <div style="background:#1a1d27; border:1px solid #2d3148; border-radius:10px; padding:20px;">
        <div style="font-size:12px; color:#94a3b8; margin-bottom:8px;">
            Entrenando — época {epoch}/{total}
        </div>
        <div style="background:#222536; border-radius:4px; height:6px; margin-bottom:16px;">
            <div style="background:#4f8ef7; width:{pct}%; height:6px; border-radius:4px;"></div>
        </div>
        <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px; text-align:center;">
            <div>
                <div style="font-size:10px; color:#94a3b8;">Loss</div>
                <div style="font-size:20px; font-weight:700; color:#f1f5f9;">{loss:.4f}</div>
            </div>
            <div>
                <div style="font-size:10px; color:#94a3b8;">IoU Cresta</div>
                <div style="font-size:20px; font-weight:700; color:#4f8ef7;">{iou_f:.3f}</div>
            </div>
            <div>
                <div style="font-size:10px; color:#94a3b8;">IoU Paso</div>
                <div style="font-size:20px; font-weight:700; color:#f59e0b;">{iou_p:.3f}</div>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_rosca_card(col, f):
    correcto = f["correcto"]
    es_b = f["es_buena"]
    veredicto = f["veredicto"]
    selected = st.session_state.get("rosca_sel") == f["rosca_id"]

    borde = "#22c55e" if correcto else "#ef4444"
    badge_bg = "#14532d" if correcto else "#7f1d1d"
    badge_tx = "#22c55e" if correcto else "#ef4444"
    icono = "✅" if correcto else "❌"
    tipo_col = "#22c55e" if es_b else "#ef4444"
    ring = "box-shadow: 0 0 0 3px #f1f5f9;" if selected else ""

    with col:
        st.markdown(
            f"""
        <div style="border:2px solid {borde}; border-radius:10px; padding:14px;
                    background:#1a1d27; text-align:center; margin-bottom:4px; {ring}">
            <div style="width:8px; height:8px; border-radius:50%;
                        background:{borde}; margin:0 auto 8px;"></div>
            <div style="font-size:16px; font-weight:700; color:#f1f5f9;">{f['rosca_id']}</div>
            <div style="font-size:11px; color:{tipo_col}; font-weight:600; margin-bottom:6px;">
                {'BUENA' if es_b else 'DESGASTE'}
            </div>
            <div style="font-size:10px; color:#94a3b8;">CV medio</div>
            <div style="font-size:18px; font-weight:700; color:#f1f5f9;">{f['cv_medio']:.4f}</div>
            <div style="font-size:10px; color:#94a3b8; margin-top:2px;">
                {f['n_alertas']}/{f['n_imagenes']} alertas
            </div>
            <div style="margin-top:8px; padding:3px 8px; border-radius:6px;
                        background:{badge_bg}; color:{badge_tx};
                        font-size:10px; font-weight:700; display:inline-block;">
                {icono} {veredicto}
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        if st.button("ver", key=f"btn_{f['rosca_id']}", width="stretch"):
            st.session_state.rosca_sel = f["rosca_id"]
            st.rerun()


def _dark_ax(ax):
    ax.set_facecolor("#1a1d27")
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2d3148")


def _render_analisis_roc(resultados: list[dict], key_prefix: str = "") -> None:
    """
    Análisis de sensibilidad del umbral CV con curva ROC interactiva.
    resultados: lista de dicts con keys 'cv_medio'|'cv_cresta' y 'es_buena'.
    """
    cv_arr = np.array(
        [float(r.get("cv_medio") or r.get("cv_cresta") or 0) for r in resultados]
    )
    labels = np.array([0 if bool(r["es_buena"]) else 1 for r in resultados])

    if len(set(labels.tolist())) < 2:
        st.info(
            "Se necesitan roscas buenas **y** con desgaste para trazar la curva ROC."
        )
        return

    thr_range = np.unique(
        np.concatenate([np.linspace(0, cv_arr.max() * 1.2, 500), cv_arr])
    )
    thr_range = np.sort(thr_range)
    tprs, fprs = [], []
    for t in thr_range:
        pred = (cv_arr > t).astype(int)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        tn = int(((pred == 0) & (labels == 0)).sum())
        tprs.append(tp / (tp + fn + 1e-9))
        fprs.append(fp / (fp + tn + 1e-9))
    tprs, fprs = np.array(tprs), np.array(fprs)

    sort_idx = np.argsort(fprs)
    auc_cv = float(np.trapezoid(tprs[sort_idx], fprs[sort_idx]))

    j_idx = int(np.argmax(tprs - fprs))
    opt_thr = float(thr_range[j_idx])

    umbral_sel = float(UMBRAL_ROSCA)
    st.caption(
        f"Umbral fijo del sistema: **{UMBRAL_ROSCA:.3f}**. "
        f"El umbral óptimo de Youden J ({opt_thr:.3f}) se muestra solo como referencia post-hoc "
        f"y no modifica los resultados guardados."
    )


    pred_sel = (cv_arr > umbral_sel).astype(int)
    tp = int(((pred_sel == 1) & (labels == 1)).sum())
    fn = int(((pred_sel == 0) & (labels == 1)).sum())
    fp = int(((pred_sel == 1) & (labels == 0)).sum())
    tn = int(((pred_sel == 0) & (labels == 0)).sum())
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-9)
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1_ = 2 * prec * rec / (prec + rec + 1e-9)
    spec = tn / (tn + fp + 1e-9)
    fpr_sel = fp / (fp + tn + 1e-9)

    col_roc, col_cm = st.columns([3, 2], gap="large")

    with col_roc:
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        fig.patch.set_facecolor("#1a1d27")
        _dark_ax(ax)
        ax.plot(
            fprs[sort_idx],
            tprs[sort_idx],
            color="#4f8ef7",
            lw=2,
            label=f"Curva ROC (AUC={auc_cv:.3f})",
        )
        ax.plot(
            [0, 1],
            [0, 1],
            color="#475569",
            lw=1,
            linestyle="--",
            label="Clasificador aleatorio",
        )
        ax.scatter(
            [fpr_sel],
            [rec],
            color="#ef4444",
            s=110,
            zorder=5,
            label=f"Umbral fijo={umbral_sel:.3f}",
        )
        ax.scatter(
            [float(fprs[j_idx])],
            [float(tprs[j_idx])],
            color="#f59e0b",
            s=140,
            marker="*",
            zorder=6,
            label=f"Óptimo J={opt_thr:.3f}",
        )
        ax.set_xlabel("FPR (1 − Especificidad)", color="#94a3b8")
        ax.set_ylabel("TPR (Sensibilidad)", color="#94a3b8")
        ax.set_title("Curva ROC — umbral CV", color="#f1f5f9")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.15, color="#2d3148")
        ax.legend(facecolor="#1a1d27", labelcolor="#94a3b8", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_cm:
        st.markdown("**Matriz de confusión**")
        st.markdown(
            f"""
        <table style="border-collapse:separate; border-spacing:4px; width:100%;
                      font-family:monospace; margin-bottom:12px;">
            <tr>
                <td></td>
                <td style="text-align:center; color:#94a3b8; font-size:11px;
                           padding:4px;">Pred: BUENA</td>
                <td style="text-align:center; color:#94a3b8; font-size:11px;
                           padding:4px;">Pred: DESGASTE</td>
            </tr>
            <tr>
                <td style="color:#94a3b8; font-size:11px; white-space:nowrap;
                           padding-right:8px;">Real: BUENA</td>
                <td style="background:#14532d; padding:14px 10px; text-align:center;
                           border-radius:8px; color:#22c55e; font-size:22px;
                           font-weight:700;">{tn}<br>
                    <span style="font-size:10px;">TN ✅</span></td>
                <td style="background:#7f1d1d; padding:14px 10px; text-align:center;
                           border-radius:8px; color:#ef4444; font-size:22px;
                           font-weight:700;">{fp}<br>
                    <span style="font-size:10px;">FP ❌</span></td>
            </tr>
            <tr>
                <td style="color:#94a3b8; font-size:11px; white-space:nowrap;
                           padding-right:8px;">Real: DESGASTE</td>
                <td style="background:#7f1d1d; padding:14px 10px; text-align:center;
                           border-radius:8px; color:#ef4444; font-size:22px;
                           font-weight:700;">{fn}<br>
                    <span style="font-size:10px;">FN ❌</span></td>
                <td style="background:#14532d; padding:14px 10px; text-align:center;
                           border-radius:8px; color:#22c55e; font-size:22px;
                           font-weight:700;">{tp}<br>
                    <span style="font-size:10px;">TP ✅</span></td>
            </tr>
        </table>
        """,
            unsafe_allow_html=True,
        )

        ma, mb = st.columns(2)
        ma.metric("Accuracy", f"{acc*100:.0f}%")
        mb.metric("F1", f"{f1_:.3f}")
        mc, md = st.columns(2)
        mc.metric(
            "Sensibilidad",
            f"{rec*100:.0f}%",
            help="TP / (TP+FN) — % de roscas con desgaste detectadas",
        )
        md.metric(
            "Especificidad",
            f"{spec*100:.0f}%",
            help="TN / (TN+FP) — % de roscas buenas correctamente clasificadas",
        )
        st.markdown(
            f"<div style='margin-top:8px; padding:10px; background:#13151f; "
            f"border:1px solid #2d3148; border-radius:8px; font-size:12px;'>"
            f"<span style='color:#94a3b8;'>Umbral óptimo (Youden J):</span> "
            f"<span style='color:#f59e0b; font-weight:700;'>{opt_thr:.3f}</span>"
            f"&ensp;·&ensp;"
            f"<span style='color:#94a3b8;'>AUC-ROC:</span> "
            f"<span style='color:#4f8ef7; font-weight:700;'>{auc_cv:.3f}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )


def _init_state():
    defaults = {
        "experimento_id": None,
        "modelo_path": None,
        "sam_model": None,
        "head_model": None,
        "resultados": [],
        "history": None,
        "pagina": "experimentos",
        "modo": "auto",
        "nombre_exp_guardado": "",
        "roscas_train_info": [],
        "metricas_clas": {},
        "rosca_sel": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


with st.sidebar:
    st.markdown("## LoRA + SAM")
    st.caption("Detección de desgaste en roscas")
    st.markdown("---")
    pagina = st.radio(
        "Navegación",
        ["🧪 Experimentos", "📊 Resultados"],
        label_visibility="collapsed",
        key="nav_radio",
    )
    st.session_state.pagina = {
        "🧪 Experimentos": "experimentos",
        "📊 Resultados": "comparativa",
    }[pagina]
    st.markdown("---")

    exp_id_activo = st.session_state.get("experimento_id")
    if exp_id_activo:
        st.caption("Experimento activo")
        st.markdown(f"**#{exp_id_activo}**")
        if st.session_state.get("nombre_exp_guardado"):
            st.caption(st.session_state.nombre_exp_guardado)
        exp_activo_sb = _buscar_exp_por_id(exp_id_activo)
        if exp_activo_sb:
            _iou_c = (
                f"{exp_activo_sb['iou_cresta']:.3f}"
                if exp_activo_sb.get("iou_cresta") is not None
                else "—"
            )
            _iou_p = (
                f"{exp_activo_sb['iou_paso']:.3f}"
                if exp_activo_sb.get("iou_paso") is not None
                else "—"
            )
            acc_sb = get_accuracy(exp_id_activo)
            _acc_s = f"{acc_sb*100:.0f}%" if acc_sb is not None else "—"
            st.markdown(
                f"""<div style="display:flex;gap:6px;margin-top:4px;">
  <div style="flex:1;background:#1a1d27;border-radius:6px;padding:6px 8px;text-align:center;">
    <div style="font-size:10px;color:#64748b;">IoU Cresta</div>
    <div style="font-size:13px;font-weight:700;color:#f1f5f9;">{_iou_c}</div>
  </div>
  <div style="flex:1;background:#1a1d27;border-radius:6px;padding:6px 8px;text-align:center;">
    <div style="font-size:10px;color:#64748b;">IoU Paso</div>
    <div style="font-size:13px;font-weight:700;color:#f1f5f9;">{_iou_p}</div>
  </div>
  <div style="flex:1;background:#1a1d27;border-radius:6px;padding:6px 8px;text-align:center;">
    <div style="font-size:10px;color:#64748b;">Accuracy</div>
    <div style="font-size:13px;font-weight:700;color:#f1f5f9;">{_acc_s}</div>
  </div>
</div>""",
                unsafe_allow_html=True,
            )


@st.dialog("⏳ Cargando experimento", width="large")
def _dialog_carga_modelo(exp_sel_id: int, mp_det: str, roscas_reinf: list[str]) -> None:
    """Modal de carga: carga el modelo, ejecuta inferencia y pre-computa visualizaciones."""
    st.markdown(f"Preparando experimento **#{exp_sel_id}**…")
    with st.spinner("Cargando pesos del modelo…"):
        _sam_d, _head_d, _ = cargar_modelo(mp_det)

    _run_d: list[dict] = []
    _prog_d = st.progress(0, text="Iniciando inferencia…")
    _n_d = len(roscas_reinf)
    for _ji_d, _rid_d in enumerate(roscas_reinf):
        _prog_d.progress(
            int(_ji_d / _n_d * 100),
            text=f"Procesando {_rid_d}… ({_ji_d + 1}/{_n_d})",
        )
        try:
            _rl_d = diagnosticar_rosca(get_ruta_rosca(_rid_d), _sam_d, _head_d)
            _rl_d["rosca_id"] = _rid_d
            _rl_d["es_buena"] = es_buena(_rid_d)
            _pxmm_d = _rl_d.get("px_por_mm")
            for _img_d in _rl_d.get("imagenes", []):
                try:
                    _cam_d = generar_gradcam(
                        _sam_d, _head_d, _img_d["img_gray"], canal=0
                    )
                    _img_d["gradcam_rgb"] = cv2.cvtColor(
                        overlay_gradcam(_img_d["img_gray"], _cam_d), cv2.COLOR_BGR2RGB
                    )
                except Exception:
                    _img_d["gradcam_rgb"] = None
                try:
                    _iso_d = generar_overlay_iso(
                        _img_d["img_gray"], _img_d["probs"], px_por_mm=_pxmm_d
                    )
                    _img_d["iso_rgb"] = (
                        cv2.cvtColor(_iso_d, cv2.COLOR_BGR2RGB)
                        if _iso_d is not None
                        else None
                    )
                except Exception:
                    _img_d["iso_rgb"] = None
            _run_d.append(_rl_d)
        except Exception as _exc_d:
            log.error("dialog carga %s: %s", _rid_d, _exc_d, exc_info=True)
            st.warning(f"Error en {_rid_d}: {_exc_d}")

    _prog_d.progress(100, text="✅ Listo")
    st.session_state.sam_model = _sam_d
    st.session_state.head_model = _head_d
    st.session_state.modelo_path = mp_det
    st.session_state.resultados = _run_d
    st.session_state.experimento_id = exp_sel_id
    st.session_state.metricas_clas = calcular_metricas_clasificacion(_run_d)
    st.session_state.history = None
    st.rerun()


if st.session_state.pagina == "experimentos":

    anotadas = get_roscas_anotadas()
    if not anotadas:
        st.error(
            "No hay roscas anotadas disponibles. Añade JSONs COCO en `annotations/`."
        )
        st.stop()

    nombre_exp = st.text_input(
        "Nombre del experimento",
        placeholder="Ej: RB01-RB09 baseline",
        max_chars=80,
        key="nombre_exp",
    )
    st.title(f"🧪 {nombre_exp}" if nombre_exp else "🧪 Experimentos")

    st.markdown("### Modo de entrenamiento")
    modos_def = [
        (
            "⚡",
            "Auto",
            "Evalúa combinaciones de roscas en 3 fases (individuales, pares, greedy) y selecciona la de mayor AUC-ROC.",
            "3 fases · greedy",
            "auto",
        ),
        (
            "⚙️",
            "Manual",
            "Elige exactamente qué roscas van a train y cuáles a evaluación.",
            "selección manual",
            "manual",
        ),
    ]
    cols_modo = st.columns(2)
    for col, (icono, titulo, desc, tag, key_modo) in zip(cols_modo, modos_def):
        sel = st.session_state.modo == key_modo
        border = "2px solid #4f8ef7" if sel else "1px solid #2d3148"
        bg = "#1e3a6e" if sel else "#1a1d27"
        with col:
            st.markdown(
                f"""
            <div style="background:{bg}; border:{border}; border-radius:10px;
                        padding:16px; min-height:120px;">
                <div style="font-size:20px; margin-bottom:6px;">{icono}</div>
                <div style="font-size:14px; font-weight:600; color:#f1f5f9;
                            margin-bottom:4px;">{titulo}</div>
                <div style="font-size:12px; color:#94a3b8; line-height:1.4;
                            margin-bottom:8px;">{desc}</div>
                <span style="font-size:10px; background:#222536; color:#94a3b8;
                             padding:2px 8px; border-radius:20px;">{tag}</span>
            </div>
            """,
                unsafe_allow_html=True,
            )
            if st.button("Seleccionar", key=f"modo_{key_modo}", width="stretch"):
                st.session_state.modo = key_modo
                st.rerun()

    modo = st.session_state.modo
    st.markdown("---")

    roscas_train: list[str] = []
    roscas_eval: list[str] = []

    if modo == "auto":
        _anotadas_auto = anotadas
        st.markdown(
            f"""
        <div style="background:#1a1d27; border:1px solid #2d3148; border-radius:10px; padding:16px;">
            <div style="font-size:13px; font-weight:600; color:#f1f5f9; margin-bottom:8px;">
                Búsqueda automática en 3 fases
            </div>
            <div style="font-size:12px; color:#94a3b8; line-height:1.6;">
                <b style="color:#4f8ef7;">Fase 1</b> &mdash; Entrena cada rosca por separado y las rankea por AUC.<br>
                <b style="color:#f59e0b;">Fase 2</b> &mdash; Evalúa todos los pares formados con el top-K de la fase 1.<br>
                <b style="color:#22c55e;">Fase 3</b> &mdash; Expansión greedy: parte del mejor par y añade roscas mientras mejore el AUC.
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        st.markdown("")
        _ac1, _ac2 = st.columns(2)
        with _ac1:
            _auto_top_k = st.slider(
                "Top-K para fase 2 (pares)",
                min_value=2,
                max_value=min(5, len(_anotadas_auto)),
                value=min(3, len(_anotadas_auto)),
                key="auto_top_k",
            )
        with _ac2:
            _auto_max_sz = st.slider(
                "Tamaño máximo combo (fase 3)",
                min_value=2,
                max_value=min(6, len(_anotadas_auto)),
                value=min(4, len(_anotadas_auto)),
                key="auto_max_size",
            )
        n_f1 = len(_anotadas_auto)
        n_f2 = _auto_top_k * (_auto_top_k - 1) // 2
        n_f3_approx = sum(n_f1 - 2 - k for k in range(_auto_max_sz - 2))
        st.caption(
            f"Estimación: ~{n_f1 + n_f2 + max(n_f3_approx,0)} evaluaciones "
            f"({n_f1} fase 1 + {n_f2} fase 2 + ~{max(n_f3_approx,0)} fase 3). "
            f"Las existentes en DB se reutilizan sin reentrenar."
        )

    else:  # manual
        st.markdown("#### Train *(con anotación)*")
        roscas_train = (
            st.pills(
                "Roscas train",
                options=anotadas,
                selection_mode="multi",
                label_visibility="collapsed",
                key="pills_train",
            )
            or []
        )

        roscas_eval = [r for r in ROSCAS_BUENAS if r not in roscas_train] + list(ROSCAS_MALAS)

        st.markdown(
            f"""
        <div style="background:#1a1d27; border:1px solid #2d3148; border-radius:8px;
                    padding:10px 16px; display:flex; gap:24px; font-size:12px; margin-top:8px;">
            <span style="color:#94a3b8;">Train:</span>
            <span style="color:#f1f5f9; font-weight:500;">{', '.join(roscas_train) or '—'}</span>
            <span style="color:#94a3b8; margin-left:auto;">Eval:</span>
            <span style="color:#f1f5f9; font-weight:500;">{', '.join(roscas_eval)}</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

    _lr_opts = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    _lr_default = LEARNING_RATE if LEARNING_RATE in _lr_opts else 1e-4
    _bs_opts = [1, 2, 4]
    _bs_default = BATCH_SIZE if BATCH_SIZE in _bs_opts else 2

    with st.expander("⚙️ Configuración avanzada", expanded=False):
        ca1, ca2, ca3 = st.columns(3)
        with ca1:
            epochs = st.slider(
                "Épocas", 10, 100, NUM_EPOCHS if NUM_EPOCHS <= 100 else 40, 5
            )
        with ca2:
            lr = st.select_slider(
                "Learning rate",
                options=_lr_opts,
                value=_lr_default,
                format_func=lambda x: f"{x:.0e}",
            )
        with ca3:
            batch = st.select_slider("Batch size", options=_bs_opts, value=_bs_default)
    if modo == "manual":
        modelo_existe = bool(roscas_train and buscar_experimento(roscas_train))
        label_btn = (
            "▶ Cargar experimento existente"
            if modelo_existe
            else "🚀 Entrenar y evaluar"
        )
        btn_disabled = not roscas_train
    else:
        label_btn = "Lanzar búsqueda automática"
        btn_disabled = False

    st.markdown("")
    ejecutar = st.button(
        label_btn, type="primary", width="stretch", disabled=btn_disabled
    )

    if ejecutar and modo == "manual":
        log.info(
            "Experimento manual iniciado | train=%s eval=%s epochs=%d lr=%g",
            roscas_train,
            roscas_eval,
            int(epochs),
            float(lr),
        )
        ckpt_path = MODELS_DIR / nombre_checkpoint(roscas_train)
        exp_existente = buscar_experimento(roscas_train)

        if exp_existente and ckpt_path.exists():
            st.session_state.experimento_id = exp_existente["id"]
            st.session_state.modelo_path = exp_existente["modelo_path"]
            st.session_state.history = None
            st.success(f"Experimento existente #{exp_existente['id']} cargado.")
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric(
                "Loss",
                (
                    f"{exp_existente['loss_final']:.4f}"
                    if exp_existente.get("loss_final")
                    else "—"
                ),
            )
            col_m2.metric(
                "IoU Cresta",
                (
                    f"{exp_existente['iou_cresta']:.3f}"
                    if exp_existente.get("iou_cresta")
                    else "—"
                ),
            )
            col_m3.metric(
                "IoU Paso",
                (
                    f"{exp_existente['iou_paso']:.3f}"
                    if exp_existente.get("iou_paso")
                    else "—"
                ),
            )
        else:
            prog_ph = st.empty()

            def _train_cb(epoch, total, loss, iou_f, iou_p):
                render_progress_panel(epoch, total, loss, iou_f, iou_p, prog_ph)

            with st.spinner("Entrenando..."):
                res_train = entrenar(
                    roscas_train=roscas_train,
                    epochs=int(epochs),
                    lr=float(lr),
                    batch_size=int(batch),
                    callback=_train_cb,
                )
            prog_ph.empty()
            st.session_state.history = res_train["history"]

            exp_id = guardar_experimento(
                roscas_train=roscas_train,
                modelo_path=res_train["modelo_path"],
                iou_cresta=res_train["iou_cresta"],
                iou_paso=res_train["iou_paso"],
                loss_final=res_train["loss_final"],
                epochs=res_train["epochs"],
                nombre=nombre_exp,
                roscas_eval=roscas_eval,
                tiempo_train_s=res_train.get("tiempo_train_s"),
            )
            st.session_state.experimento_id = exp_id
            st.session_state.modelo_path = str(res_train["modelo_path"])

            t_s = res_train.get("tiempo_train_s", 0)
            t_ep = res_train.get("tiempo_epoca_media_s", 0)
            _tm = [
                ("Loss final", f"{res_train['loss_final']:.4f}"),
                ("IoU Cresta", f"{res_train['iou_cresta']:.3f}"),
                ("IoU Paso", f"{res_train['iou_paso']:.3f}"),
                ("Tiempo total", f"{t_s:.0f} s"),
                ("T/época", f"{t_ep:.1f} s"),
            ]
            _tm_html = "".join(
                f"""<div style="flex:1;background:#1a1d27;border-radius:6px;padding:8px 10px;">
  <div style="font-size:10px;color:#64748b;margin-bottom:3px;">{_lbl}</div>
  <div style="font-size:12px;font-weight:700;color:#f1f5f9;white-space:nowrap;">{_val}</div>
</div>"""
                for _lbl, _val in _tm
            )
            st.markdown(
                f'<div style="display:flex;gap:8px;margin:8px 0;">{_tm_html}</div>',
                unsafe_allow_html=True,
            )

        with st.spinner("Cargando modelo..."):
            sam, head, _ = cargar_modelo(st.session_state.modelo_path)
            st.session_state.sam_model = sam
            st.session_state.head_model = head
        log.info("Modelo cargado: %s", st.session_state.modelo_path)

        _es_exp_nuevo = not exp_existente
        resultados_run: list[dict] = []
        prog_inf = st.progress(0, text="Procesando roscas...")
        for i, rosca_id in enumerate(roscas_eval):
            prog_inf.progress(
                int(i / len(roscas_eval) * 100),
                text=f"Procesando {rosca_id}… ({i+1}/{len(roscas_eval)})",
            )
            ruta = get_ruta_rosca(rosca_id)
            try:
                res = diagnosticar_rosca(ruta, sam, head)
            except Exception as exc:
                log.error("Error diagnosticando %s: %s", rosca_id, exc, exc_info=True)
                st.warning(f"Error en {rosca_id}: {exc}")
                continue
            res["rosca_id"] = rosca_id
            res["es_buena"] = es_buena(rosca_id)
            resultados_run.append(res)
            if _es_exp_nuevo:
                guardar_resultado(
                    experimento_id=st.session_state.experimento_id,
                    rosca_id=rosca_id,
                    es_buena=es_buena(rosca_id),
                    cv_cresta=res["cv_medio"],
                    score=res["score_medio"],
                    diagnostico=res["veredicto"],
                )
        prog_inf.progress(100, text="✅ Inferencia completada")
        st.session_state.resultados = resultados_run

        t_inf_ms = sum(r.get("tiempo_total_ms", 0) for r in resultados_run)
        mc = calcular_metricas_clasificacion(resultados_run)
        st.session_state.metricas_clas = mc
        if _es_exp_nuevo:
            actualizar_metricas_avanzadas(
                st.session_state.experimento_id,
                mc.get("auc_roc"),
                mc.get("f1"),
                mc.get("auprc"),
            )
            actualizar_tiempos(
                st.session_state.experimento_id, tiempo_inf_s=round(t_inf_ms / 1000, 1)
            )

        n_ok_run = sum(
            1
            for r in resultados_run
            if (r["veredicto"] == "BUENA" and r["es_buena"])
            or (r["veredicto"] == "POSIBLE DESGASTE" and not r["es_buena"])
        )
        n_alert = sum(1 for r in resultados_run if r["veredicto"] == "POSIBLE DESGASTE")
        _v_acc = f"{n_ok_run}/{len(resultados_run)} ({100*n_ok_run/len(resultados_run):.0f}%)"
        _v_auc = f"{mc['auc_roc']:.3f}" if mc.get("auc_roc") is not None else "—"
        _v_f1 = f"{mc['f1']:.3f}" if mc.get("f1") is not None else "—"
        _v_alt = f"{n_alert} alertas"
        st.markdown(
            f"""<div style="display:flex;gap:8px;margin:8px 0;">
  <div style="flex:1;background:#1a1d27;border-radius:6px;padding:8px 10px;">
    <div style="font-size:10px;color:#64748b;margin-bottom:3px;">Accuracy</div>
    <div style="font-size:12px;font-weight:700;color:#f1f5f9;white-space:nowrap;">{_v_acc}</div>
  </div>
  <div style="flex:1;background:#1a1d27;border-radius:6px;padding:8px 10px;">
    <div style="font-size:10px;color:#64748b;margin-bottom:3px;">AUC-ROC</div>
    <div style="font-size:12px;font-weight:700;color:#f1f5f9;">{_v_auc}</div>
  </div>
  <div style="flex:1;background:#1a1d27;border-radius:6px;padding:8px 10px;">
    <div style="font-size:10px;color:#64748b;margin-bottom:3px;">F1</div>
    <div style="font-size:12px;font-weight:700;color:#f1f5f9;">{_v_f1}</div>
  </div>
  <div style="flex:1;background:#1a1d27;border-radius:6px;padding:8px 10px;">
    <div style="font-size:10px;color:#64748b;margin-bottom:3px;">Alertas</div>
    <div style="font-size:12px;font-weight:700;color:#f1f5f9;">{_v_alt}</div>
  </div>
</div>""",
            unsafe_allow_html=True,
        )

        log.info(
            "Inferencia completada | %d/%d roscas correctas AUC=%.3f F1=%.3f",
            n_ok_run,
            len(resultados_run),
            mc.get("auc_roc") or 0.0,
            mc.get("f1") or 0.0,
        )
        st.session_state.nombre_exp_guardado = nombre_exp
        st.session_state.roscas_train_info = roscas_train


    elif ejecutar and modo == "auto":
        from itertools import combinations as _combinations

        _anotadas_auto = anotadas  # get_roscas_anotadas() ya excluye RB10/RB11
        _top_k = int(st.session_state.get("auto_top_k", 3))
        _max_sz = int(st.session_state.get("auto_max_size", 4))

        log.info(
            "Auto iniciado | roscas=%s top_k=%d max_sz=%d epochs=%d lr=%g",
            _anotadas_auto,
            _top_k,
            _max_sz,
            int(epochs),
            float(lr),
        )

        _auto_prog = st.progress(0, text="Iniciando búsqueda automática...")
        _n_f1 = len(_anotadas_auto)

        st.markdown("#### Fase 1 — Roscas individuales")
        _f1_res: list[dict] = []
        for _i, _r in enumerate(_anotadas_auto):
            _auto_prog.progress(
                int(_i / _n_f1 * 33),
                text=f"Fase 1 — {_r} ({_i+1}/{_n_f1})",
            )
            _eval_c = [x for x in _anotadas_auto if x != _r] + list(ROSCAS_MALAS)
            try:
                _res = _auto_run_combo(
                    [_r], _eval_c, int(epochs), float(lr), int(batch)
                )
                _f1_res.append(_res)
            except Exception as _exc:
                log.error("Auto fase1 %s: %s", _r, _exc, exc_info=True)
                st.warning(f"Error en {_r}: {_exc}")

        _f1_sorted = sorted(_f1_res, key=lambda x: x.get("auc") or 0, reverse=True)
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Rosca": _r["roscas_train"][0],
                        "AUC": round(_r["auc"], 3) if _r.get("auc") else None,
                        "F1": round(_r["f1"], 3) if _r.get("f1") else None,
                        "Acc": (
                            f"{(_r['acc'] or 0)*100:.0f}%"
                            if _r.get("acc") is not None
                            else "—"
                        ),
                        "Exp #": _r.get("exp_id") or "—",
                    }
                    for _r in _f1_sorted
                ]
            ),
            width="stretch",
            hide_index=True,
        )


        _top_roscas = [_r["roscas_train"][0] for _r in _f1_sorted[:_top_k]]
        _pares_f2 = list(_combinations(_top_roscas, 2))
        st.markdown(f"#### Fase 2 — Pares (top {_top_k}: {', '.join(_top_roscas)})")
        _f2_res: list[dict] = []
        for _i, _par in enumerate(_pares_f2):
            _auto_prog.progress(
                33 + int(_i / max(len(_pares_f2), 1) * 33),
                text=f"Fase 2 — {'+'.join(_par)} ({_i+1}/{len(_pares_f2)})",
            )
            _train_c = list(_par)
            _eval_c = [x for x in _anotadas_auto if x not in _train_c] + list(
                ROSCAS_MALAS
            )
            try:
                _res = _auto_run_combo(
                    _train_c, _eval_c, int(epochs), float(lr), int(batch)
                )
                _f2_res.append(_res)
            except Exception as _exc:
                log.error("Auto fase2 %s: %s", _par, _exc, exc_info=True)
                st.warning(f"Error en {'+'.join(_par)}: {_exc}")

        _f2_sorted = sorted(_f2_res, key=lambda x: x.get("auc") or 0, reverse=True)
        if _f2_sorted:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Combo": "+".join(_r["roscas_train"]),
                            "AUC": round(_r["auc"], 3) if _r.get("auc") else None,
                            "F1": round(_r["f1"], 3) if _r.get("f1") else None,
                            "Acc": (
                                f"{(_r['acc'] or 0)*100:.0f}%"
                                if _r.get("acc") is not None
                                else "—"
                            ),
                            "Exp #": _r.get("exp_id") or "—",
                        }
                        for _r in _f2_sorted
                    ]
                ),
                width="stretch",
                hide_index=True,
            )

        _current = (
            _f2_sorted[0] if _f2_sorted else (_f1_sorted[0] if _f1_sorted else None)
        )
        _f3_log: list[str] = []
        _f3_res: list[dict] = []

        if (
            _current
            and len(_current["roscas_train"]) < _max_sz
            and len(_anotadas_auto) > 2
        ):
            st.markdown(f"#### Fase 3 — Expansión greedy (hasta {_max_sz} roscas)")
            _n_f3_est = sum(len(_anotadas_auto) - 2 - _k for _k in range(_max_sz - 2))
            _step_f3 = 0

            while len(_current["roscas_train"]) < _max_sz:
                _remaining = [
                    x for x in _anotadas_auto if x not in _current["roscas_train"]
                ]
                if not _remaining:
                    break
                _cands: list[dict] = []
                for _r in _remaining:
                    _step_f3 += 1
                    _pct = 66 + int(_step_f3 / max(_n_f3_est, 1) * 33)
                    _auto_prog.progress(
                        min(_pct, 99),
                        text=f"Fase 3 — probando +{_r} a {'+'.join(_current['roscas_train'])}",
                    )
                    _train_c = _current["roscas_train"] + [_r]
                    _eval_c = [x for x in _anotadas_auto if x not in _train_c] + list(
                        ROSCAS_MALAS
                    )
                    try:
                        _res = _auto_run_combo(
                            _train_c, _eval_c, int(epochs), float(lr), int(batch)
                        )
                        _cands.append(_res)
                        _f3_res.append(_res)
                    except Exception as _exc:
                        log.error("Auto fase3 +%s: %s", _r, _exc, exc_info=True)

                if not _cands:
                    break
                _best_cand = max(_cands, key=lambda x: x.get("auc") or 0)
                if (_best_cand.get("auc") or 0) > (_current.get("auc") or 0):
                    _current = _best_cand
                    _f3_log.append(
                        f"+{_current['roscas_train'][-1]} → {'+'.join(_current['roscas_train'])} "
                        f"| AUC {_current.get('auc'):.3f}"
                    )
                else:
                    _f3_log.append(
                        "Sin mejora al ampliar el conjunto. Búsqueda terminada."
                    )
                    break

            if _f3_log:
                st.markdown("\n".join(f"- {_l}" for _l in _f3_log))

        _auto_prog.progress(100, text="Búsqueda completada")

        _all_res = _f1_res + _f2_res + _f3_res
        _best = max(_all_res, key=lambda x: x.get("auc") or 0) if _all_res else None

        if _best:
            _best_tr = _best["roscas_train"]
            _best_eid = _best.get("exp_id")

            st.session_state.experimento_id = _best_eid
            st.session_state["comp_sel_id"] = _best_eid  # pre-seleccionar en Resultados
            st.session_state.nombre_exp_guardado = f"auto_{'+'.join(_best_tr)}"
            st.session_state.roscas_train_info = _best_tr
            st.session_state.history = None

            st.success(
                f"Mejor combinación: **{'+'.join(_best_tr)}** | "
                f"AUC={_best.get('auc'):.3f} | F1={_best.get('f1'):.3f}"
                if _best.get("auc")
                else f"Mejor combinación: **{'+'.join(_best_tr)}**"
            )
            log.info("Auto completado | mejor=%s AUC=%s", _best_tr, _best.get("auc"))

            _best_ckpt = MODELS_DIR / nombre_checkpoint(_best_tr)
            if _best_ckpt.exists():
                with st.spinner("Cargando modelo..."):
                    _sam_b, _head_b, _ = cargar_modelo(str(_best_ckpt))
                    st.session_state.sam_model = _sam_b
                    st.session_state.head_model = _head_b
                    st.session_state.modelo_path = str(_best_ckpt)

                _eval_best = [x for x in _anotadas_auto if x not in _best_tr] + list(
                    ROSCAS_MALAS
                )
                resultados_eval: list[dict] = []
                _prog_b = st.progress(0, text="Evaluando roscas...")
                for _j, _rid in enumerate(_eval_best):
                    _prog_b.progress(int(_j / len(_eval_best) * 100))
                    try:
                        _r = diagnosticar_rosca(get_ruta_rosca(_rid), _sam_b, _head_b)
                        _r["rosca_id"] = _rid
                        _r["es_buena"] = es_buena(_rid)
                        resultados_eval.append(_r)
                    except Exception as _exc:
                        log.error("Auto final eval %s: %s", _rid, _exc)
                _prog_b.progress(100, text="Evaluación completada")
                metricas_eval = calcular_metricas_clasificacion(resultados_eval)
                st.session_state.resultados = resultados_eval
                st.session_state.metricas_clas = metricas_eval
                if _best_eid:
                    actualizar_metricas_avanzadas(
                        _best_eid,
                        metricas_eval.get("auc_roc"),
                        metricas_eval.get("f1"),
                        metricas_eval.get("auprc"),
                    )
                    _t_inf_final = round(
                        sum(r.get("tiempo_total_ms", 0) for r in resultados_eval) / 1000, 1
                    )
                    actualizar_tiempos(_best_eid, tiempo_inf_s=_t_inf_final)

    resultados = st.session_state.get("resultados", [])
    if not resultados:
        if not ejecutar:
            st.markdown("---")
            render_empty_state()
        st.stop()

    st.markdown("---")
    _n_ok_r = sum(
        1
        for r in resultados
        if (r["veredicto"] == "BUENA" and r["es_buena"])
        or (r["veredicto"] == "POSIBLE DESGASTE" and not r["es_buena"])
    )
    _n_al_r = sum(1 for r in resultados if r["veredicto"] == "POSIBLE DESGASTE")
    _exp_kpi = (
        _buscar_exp_por_id(st.session_state.experimento_id)
        if st.session_state.experimento_id
        else None
    )
    _sk1, _sk2, _sk3, _sk4, _sk5, _sk6 = st.columns(6)
    _sk1.metric(
        "Accuracy", f"{_n_ok_r}/{len(resultados)} ({100*_n_ok_r/len(resultados):.0f}%)"
    )
    _sk2.metric(
        "IoU Cresta",
        (
            f"{_exp_kpi['iou_cresta']:.3f}"
            if _exp_kpi and _exp_kpi.get("iou_cresta") is not None
            else "—"
        ),
    )
    _sk3.metric(
        "IoU Paso",
        (
            f"{_exp_kpi['iou_paso']:.3f}"
            if _exp_kpi and _exp_kpi.get("iou_paso") is not None
            else "—"
        ),
    )
    _sk4.metric("Alertas", f"{_n_al_r} roscas")
    _sk5.metric(
        "T. entreno",
        (
            f"{_exp_kpi['tiempo_train_s']:.0f} s"
            if _exp_kpi and _exp_kpi.get("tiempo_train_s") is not None
            else "—"
        ),
    )
    _n_imgs_kpi = (
        sum(
            len(list(get_ruta_rosca(r).glob("*.jpg")))
            for r in _json.loads(_exp_kpi["roscas_eval"])
        )
        if _exp_kpi and _exp_kpi.get("roscas_eval")
        else None
    )
    _sk6.metric(
        "T. inf.",
        (
            f"{_exp_kpi['tiempo_inf_s']:.1f} s"
            if _exp_kpi and _exp_kpi.get("tiempo_inf_s")
            else "—"
        ),
        help=(
            f"{_exp_kpi['tiempo_inf_s'] / _n_imgs_kpi:.2f} s/img"
            if _exp_kpi and _exp_kpi.get("tiempo_inf_s") and _n_imgs_kpi
            else None
        ),
    )
    st.info(
        "Ve a **📊 Resultados** para ver imágenes, segmentación y métricas."
    )
    if st.button("→ Ir a Resultados", type="primary", key="btn_ir_resultados"):
        st.session_state.pagina = "comparativa"
        st.rerun()


elif st.session_state.pagina == "comparativa":
    st.title("📊 Resultados")

    experimentos = listar_experimentos()
    if not experimentos:
        render_empty_state()
        st.stop()

    _accs_all = [a for e in experimentos if (a := get_accuracy(e["id"])) is not None]
    _aucs_all = [e["auc_roc"] for e in experimentos if e.get("auc_roc")]
    km1, km2, km3 = st.columns(3)
    km1.metric("Experimentos", len(experimentos))
    km2.metric("Mejor accuracy", f"{max(_accs_all)*100:.0f}%" if _accs_all else "—")
    km3.metric("Mejor AUC-ROC", f"{max(_aucs_all):.3f}" if _aucs_all else "—")

    _exps_sorted = sorted(
        experimentos, key=lambda e: get_accuracy(e["id"]) or 0, reverse=True
    )
    _filas_all = []
    for _e in _exps_sorted:
        _acc_e = get_accuracy(_e["id"])
        _rtr_e = _json.loads(_e["roscas_train"])
        _res_e   = get_resultados(_e["id"])
        _res_b   = [r for r in _res_e if r["es_buena"]]
        _res_m   = [r for r in _res_e if not r["es_buena"]]
        _ok_b    = sum(1 for r in _res_b if r["correcto"])
        _ok_m    = sum(1 for r in _res_m if r["correcto"])
        _filas_all.append(
            {
                "⭐": "⭐" if _e.get("es_mejor") else "",
                "Roscas train": _e.get("nombre") or " + ".join(_rtr_e),
                "N roscas train": len(_rtr_e),
                "Buenas": f"{_ok_b}/{len(_res_b)} ({_ok_b/len(_res_b)*100:.0f}%)" if _res_b else None,
                "Desgaste": f"{_ok_m}/{len(_res_m)} ({_ok_m/len(_res_m)*100:.0f}%)" if _res_m else None,
                "Accuracy": _acc_e if _acc_e is not None else None,
                "AUC-ROC": round(_e["auc_roc"], 3) if _e.get("auc_roc") else None,
                "F1": round(_e["f1"], 3) if _e.get("f1") else None,
                "IoU Cresta": (
                    round(_e["iou_cresta"], 3) if _e.get("iou_cresta") else None
                ),
                "T. entreno": (
                    round(_e["tiempo_train_s"]) if _e.get("tiempo_train_s") else None
                ),
                "T. inf.": (
                    f"{_e['tiempo_inf_s']:.1f} s"
                    f" ({_e['tiempo_inf_s'] / max(sum(len(list(get_ruta_rosca(r).glob('*.jpg'))) for r in _json.loads(_e['roscas_eval'])), 1):.2f} s/img)"
                    if _e.get("tiempo_inf_s") and _e.get("roscas_eval") else None
                ),
            }
        )
    _ev_all = st.dataframe(
        pd.DataFrame(_filas_all),
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="tabla_res_all",
        column_config={
            "Accuracy": st.column_config.ProgressColumn(
                "Accuracy", format="%.0f%%", min_value=0, max_value=1
            ),
            "AUC-ROC": st.column_config.NumberColumn("AUC-ROC", format="%.3f"),
            "F1": st.column_config.NumberColumn("F1", format="%.3f"),
            "IoU Cresta": st.column_config.NumberColumn("IoU Cresta", format="%.3f"),
        },
    )
    _sel_rows = _ev_all.selection.rows if hasattr(_ev_all, "selection") else []
    if _sel_rows:
        _clicked_id = _exps_sorted[_sel_rows[0]]["id"]
        st.session_state["comp_sel_id"] = _clicked_id
        st.session_state["_pending_load"] = _clicked_id

    _sel_id = st.session_state.get("comp_sel_id") or (
        _exps_sorted[0]["id"] if _exps_sorted else None
    )
    _sel_nombre = (
        next(
            (
                e.get("nombre") or "+".join(_json.loads(e["roscas_train"]))
                for e in experimentos
                if e["id"] == _sel_id
            ),
            str(_sel_id),
        )
        if _sel_id
        else ""
    )

    with st.expander("⚙️ Gestión de experimentos", expanded=False):
        _act1, _act2, _act3 = st.columns(3)
        if _act1.button(
            "⭐ Marcar como mejor",
            key="btn_mejor_tbl",
            width="stretch",
            disabled=not _sel_id,
        ):
            set_mejor_modelo(_sel_id)
            st.rerun()
        if _act2.button(
            "🗑️ Borrar experimento",
            key="btn_borrar_tbl",
            width="stretch",
            disabled=not _sel_id,
        ):
            st.session_state["confirmar_borrar"] = _sel_id
        if _act3.button(
            "⚠️ Borrar todos los experimentos",
            key="btn_borrar_todos",
            width="stretch",
            disabled=len(experimentos) == 0,
        ):
            st.session_state["confirmar_borrar_todos"] = True

    if _sel_id and st.session_state.get("confirmar_borrar") == _sel_id:
        st.warning(
            f"¿Borrar **#{_sel_id} — {_sel_nombre}**? Se eliminarán todos sus resultados. Irreversible."
        )
        _cb1, _cb2, _ = st.columns([1, 1, 4])
        if _cb1.button("Sí, borrar", key="btn_confirm_del", type="primary"):
            borrar_experimento(_sel_id)
            if st.session_state.get("experimento_id") == _sel_id:
                st.session_state.experimento_id = None
                st.session_state.resultados = []
                st.session_state.metricas_clas = {}
            st.session_state.pop("confirmar_borrar", None)
            st.session_state.pop("comp_sel_id", None)
            st.rerun()
        if _cb2.button("Cancelar", key="btn_cancel_del"):
            st.session_state.pop("confirmar_borrar", None)
            st.rerun()

    if st.session_state.get("confirmar_borrar_todos"):
        st.warning(
            f"¿Borrar **todos los experimentos** ({len(experimentos)})? Esta acción es irreversible."
        )
        _bt1, _bt2, _ = st.columns([1, 1, 4])
        if _bt1.button("Sí, borrar todos", key="btn_todos_confirm", type="primary"):
            borrar_todos_experimentos()
            st.session_state.experimento_id = None
            st.session_state.resultados = []
            st.session_state.metricas_clas = {}
            st.session_state.pop("confirmar_borrar_todos", None)
            st.session_state.pop("comp_sel_id", None)
            st.rerun()
        if _bt2.button("Cancelar", key="btn_todos_cancel"):
            st.session_state.pop("confirmar_borrar_todos", None)
            st.rerun()

    st.markdown("---")

    exp_sel_id = _sel_id or _exps_sorted[0]["id"]
    exp_det = next((e for e in experimentos if e["id"] == exp_sel_id), {})
    resultados_db = get_resultados(exp_sel_id)

    _tipo_lbl = exp_det.get("tipo", "manual").upper()
    _nombre_lbl = exp_det.get("nombre") or "+".join(
        _json.loads(exp_det.get("roscas_train", "[]"))
    )
    st.markdown(f"#### #{exp_sel_id} · {_tipo_lbl} — {_nombre_lbl}")

    _dm1, _dm2, _dm3, _dm4, _dm5, _dm6, _dm7 = st.columns(7)
    _acc_det = get_accuracy(exp_sel_id)
    _dm1.metric("Accuracy", f"{_acc_det*100:.0f}%" if _acc_det is not None else "—")
    _dm2.metric(
        "AUC-ROC", f"{exp_det['auc_roc']:.3f}" if exp_det.get("auc_roc") else "—"
    )
    _dm3.metric("F1", f"{exp_det['f1']:.3f}" if exp_det.get("f1") else "—")
    _dm4.metric("AUPRC", f"{exp_det['auprc']:.3f}" if exp_det.get("auprc") else "—")
    _dm5.metric(
        "IoU Cresta",
        f"{exp_det['iou_cresta']:.3f}" if exp_det.get("iou_cresta") else "—",
    )
    _dm6.metric(
        "T. entreno",
        f"{exp_det['tiempo_train_s']:.0f} s" if exp_det.get("tiempo_train_s") else "—",
    )
    _n_imgs_det = (
        sum(
            len(list(get_ruta_rosca(r).glob("*.jpg")))
            for r in _json.loads(exp_det["roscas_eval"])
        )
        if exp_det.get("roscas_eval")
        else None
    )
    _dm7.metric(
        "T. inf.",
        f"{exp_det['tiempo_inf_s']:.1f} s" if exp_det.get("tiempo_inf_s") else "—",
        help=(
            f"{exp_det['tiempo_inf_s'] / _n_imgs_det:.2f} s/img"
            if exp_det.get("tiempo_inf_s") and _n_imgs_det
            else None
        ),
    )

    if not resultados_db:
        st.info("Este experimento no tiene resultados guardados.")
    else:
        _df_r = pd.DataFrame(resultados_db)
        _df_r["Tipo"] = _df_r["es_buena"].map({1: "Buena", 0: "Con desgaste"})
        _df_r["CV"] = _df_r["cv_cresta"].round(4)
        _df_r["Score"] = _df_r["score"].round(4)
        _df_r["✓/✗"] = _df_r["correcto"].map({1: "✓", 0: "✗"})
        st.dataframe(
            _df_r[["rosca_id", "Tipo", "CV", "Score", "diagnostico", "✓/✗"]],
            width="stretch",
            hide_index=True,
        )

        _hist_d = _cargar_history(exp_det.get("modelo_path"))
        if _hist_d:
            if "iou_cresta" not in _hist_d and "iou_filete" in _hist_d:
                _hist_d["iou_cresta"] = _hist_d["iou_filete"]
            if "val_iou_cresta" not in _hist_d and "val_iou_filete" in _hist_d:
                _hist_d["val_iou_cresta"] = _hist_d["val_iou_filete"]
            with st.expander("📈 Curvas de entrenamiento", expanded=False):
                _hv = bool(_hist_d.get("val_iou_cresta"))
                _nc = 3 if _hv else 2
                _fhc, _axhc = plt.subplots(1, _nc, figsize=(4 * _nc, 2.8))
                _fhc.patch.set_facecolor("#1a1d27")
                for _a in (_axhc if _nc > 1 else [_axhc]):
                    _dark_ax(_a)
                _axhc[0].plot(_hist_d["loss"], color="#ef4444")
                _axhc[0].set_title("Loss", color="#f1f5f9")
                _axhc[0].set_xlabel("Época", color="#94a3b8")
                _axhc[0].grid(alpha=0.2, color="#2d3148")
                _axhc[1].plot(
                    _hist_d["iou_cresta"], color="#4f8ef7", label="IoU Cresta"
                )
                _axhc[1].plot(_hist_d["iou_paso"], color="#f59e0b", label="IoU Paso")
                _axhc[1].set_title("IoU", color="#f1f5f9")
                _axhc[1].set_xlabel("Época", color="#94a3b8")
                _axhc[1].grid(alpha=0.2, color="#2d3148")
                _axhc[1].legend(facecolor="#1a1d27", labelcolor="#94a3b8")
                if _hv:
                    _axhc[2].plot(
                        _hist_d["val_iou_cresta"],
                        color="#4f8ef7",
                        linestyle="--",
                        label="Val Cresta",
                    )
                    _axhc[2].plot(
                        _hist_d["val_iou_paso"],
                        color="#f59e0b",
                        linestyle="--",
                        label="Val Paso",
                    )
                    _axhc[2].set_title("IoU val.", color="#f1f5f9")
                    _axhc[2].set_xlabel("Época", color="#94a3b8")
                    _axhc[2].grid(alpha=0.2, color="#2d3148")
                    _axhc[2].legend(facecolor="#1a1d27", labelcolor="#94a3b8")
                if _hist_d.get("best_epoch"):
                    st.caption(
                        f"Mejor época: {_hist_d['best_epoch']} · {len(_hist_d['loss'])} épocas"
                    )
                plt.tight_layout()
                st.pyplot(_fhc)
                plt.close(_fhc)

        if len({r["es_buena"] for r in resultados_db}) == 2:
            with st.expander("🔬 ROC (umbral CV)", expanded=False):
                _render_analisis_roc(resultados_db, key_prefix="res_db")

    _modelo_en_memoria = (
        st.session_state.get("experimento_id") == exp_sel_id
        and bool(st.session_state.get("resultados"))
        and st.session_state.get("sam_model") is not None
    )
    _mp_det = exp_det.get("modelo_path")
    _mp_existe = bool(_mp_det and Path(_mp_det).exists())
    _roscas_reinf = [r["rosca_id"] for r in (resultados_db or [])]

    if not _mp_existe:
        st.warning("Modelo no encontrado en disco.")
    elif (
        st.session_state.get("_pending_load") == exp_sel_id
        and not _modelo_en_memoria
        and _roscas_reinf
    ):
        st.session_state.pop("_pending_load", None)
        _dialog_carga_modelo(exp_sel_id, _mp_det, _roscas_reinf)

    if _modelo_en_memoria:
        resultados = st.session_state.get("resultados", [])
        filas = []
        for r in resultados:
            _c = (r["veredicto"] == "BUENA" and r["es_buena"]) or (
                r["veredicto"] == "POSIBLE DESGASTE" and not r["es_buena"]
            )
            filas.append({**r, "correcto": _c})

        st.markdown("##### Diagnóstico por rosca")
        if st.session_state.get("rosca_sel") not in [f["rosca_id"] for f in filas]:
            st.session_state.rosca_sel = filas[0]["rosca_id"]
        for _rs in range(0, len(filas), 5):
            _rf = filas[_rs : _rs + 5]
            _cc = st.columns(len(_rf))
            for _col, _f in zip(_cc, _rf):
                render_rosca_card(_col, _f)

        rosca_sel = st.session_state.rosca_sel
        res_sel = next(r for r in resultados if r["rosca_id"] == rosca_sel)
        veredicto = res_sel["veredicto"]
        correcto = (veredicto == "BUENA" and res_sel["es_buena"]) or (
            veredicto == "POSIBLE DESGASTE" and not res_sel["es_buena"]
        )
        borde_d = "#22c55e" if correcto else "#ef4444"
        tipo_txt = "Rosca buena" if res_sel["es_buena"] else "Con desgaste"
        estado = "✅ Correcto" if correcto else "❌ Incorrecto"

        st.markdown(
            f"""<div style="background:#1a1d27; border-left:4px solid {borde_d};
                    border-radius:0 10px 10px 0; padding:10px 16px; margin:12px 0 8px;">
            <span style="font-size:17px; font-weight:700; color:#f1f5f9;">{rosca_sel}</span>
            <span style="color:#94a3b8; margin-left:8px; font-size:12px;">{tipo_txt}</span>
            <span style="float:right; color:{borde_d}; font-weight:600;">{estado}</span>
        </div>""",
            unsafe_allow_html=True,
        )

        dc1, dc2, dc3, dc4 = st.columns(4)
        dc1.metric("CV medio", f"{res_sel['cv_medio']:.4f}")
        dc2.metric("Score", f"{res_sel.get('score_medio', 0):.4f}")
        dc3.metric("Alertas", f"{res_sel['n_alertas']}/{res_sel['n_imagenes']}")
        _pxmm = res_sel.get("px_por_mm")
        dc4.metric("Escala", f"{_pxmm:.1f} px/mm" if _pxmm else "—")

        d2c = res_sel.get("diagnostico_2c")
        if d2c:
            c1d = d2c["capa1"]
            c2d = d2c["capa2"]
            vfd = d2c["veredicto_final"]
            _v1c = "#22c55e" if c1d["veredicto"] == "BUENA" else "#ef4444"
            _v1i = "✅" if c1d["veredicto"] == "BUENA" else "⚠️"
            _v2_lut = {
                "CONFORME": ("✅", "#22c55e", "CONFORME"),
                "DESVIACION_LEVE": ("⚠️", "#f59e0b", "LEVE"),
                "DESVIACION_SIGNIFICATIVA": ("❌", "#ef4444", "SIGNIFICATIVA"),
            }
            _v2i, _v2c, _v2l = _v2_lut.get(
                c2d["veredicto"], ("—", "#94a3b8", c2d["veredicto"])
            )
            _vfc = "#22c55e" if vfd == "BUENA" else "#ef4444"
            _vfi = "✅" if vfd == "BUENA" else "⚠️"
            _c2det = (
                f"{c2d.get('n_fuera',0)} fuera · {'cal.' if c2d['calibrado'] else 'sin px/mm'}"
                if c2d["disponible"]
                else "sin datos de cresta"
            )
            st.markdown(
                f"""<div style="display:flex;gap:8px;align-items:stretch;background:#1a1d27;border:1px solid #2d3148;border-radius:10px;padding:10px 14px;margin-bottom:8px;flex-wrap:wrap;">
                <div style="background:#13151f;border-radius:8px;padding:8px 12px;flex:1;min-width:130px;"><div style="font-size:10px;color:#64748b;margin-bottom:3px;">Capa 1 · Variabilidad</div><div style="font-size:13px;font-weight:700;color:{_v1c};">{_v1i} {c1d["veredicto"]}</div><div style="font-size:10px;color:#94a3b8;">CV {c1d["cv_medio"]:.4f}</div></div>
                <div style="color:#475569;align-self:center;font-size:16px;">›</div>
                <div style="background:#13151f;border-radius:8px;padding:8px 12px;flex:1;min-width:130px;"><div style="font-size:10px;color:#64748b;margin-bottom:3px;">Capa 2 · ISO M10×1.5</div><div style="font-size:13px;font-weight:700;color:{_v2c};">{_v2i} {_v2l}</div><div style="font-size:10px;color:#94a3b8;">{_c2det}</div></div>
                <div style="color:#475569;align-self:center;font-size:16px;">›</div>
                <div style="background:{_vfc}18;border:1px solid {_vfc}44;border-radius:8px;padding:8px 14px;flex-shrink:0;min-width:110px;display:flex;flex-direction:column;justify-content:center;"><div style="font-size:10px;color:#64748b;margin-bottom:3px;">Veredicto</div><div style="font-size:14px;font-weight:700;color:{_vfc};">{_vfi} {vfd}</div></div>
            </div>""",
                unsafe_allow_html=True,
            )

        imgs_rosca = res_sel.get("imagenes", [])
        tab_imgs, tab_diag = st.tabs(["🖼️ Imágenes", "📊 Diagnóstico"])

        with tab_imgs:
            if not imgs_rosca:
                st.info("No hay imágenes en memoria.")
            else:
                _fil_col, _ = st.columns([2, 4])
                _filtro_img = _fil_col.radio(
                    "Mostrar",
                    ["Todas", "Solo alertas ⚠️", "Solo buenas ✅"],
                    horizontal=True,
                    key="filtro_img",
                )
                if _filtro_img == "Solo alertas ⚠️":
                    _imgs_vis = [
                        i
                        for i in imgs_rosca
                        if i.get("veredicto") == "POSIBLE DESGASTE"
                    ]
                elif _filtro_img == "Solo buenas ✅":
                    _imgs_vis = [
                        i
                        for i in imgs_rosca
                        if i.get("veredicto") != "POSIBLE DESGASTE"
                    ]
                else:
                    _imgs_vis = imgs_rosca
                st.caption(
                    f"{len(_imgs_vis)} imágenes · Segmentación · ISO M10×1.5 · GradCAM"
                )
                if not _imgs_vis:
                    st.info("Ninguna imagen coincide con el filtro seleccionado.")
                for _ir in _imgs_vis:
                    _alerta = _ir.get("veredicto") == "POSIBLE DESGASTE"
                    _hdr = f"**{_ir['imagen']}** · CV={_ir['cv_cresta']:.3f}" + (
                        "  ⚠️" if _alerta else ""
                    )
                    st.markdown(_hdr)
                    _col_s, _col_i, _col_g = st.columns(3)
                    _col_s.image(
                        cv2.cvtColor(_ir["overlay"], cv2.COLOR_BGR2RGB),
                        caption="Segmentación",
                        width="stretch",
                    )
                    if _ir.get("iso_rgb") is not None:
                        _col_i.image(
                            _ir["iso_rgb"], caption="ISO M10×1.5", width="stretch"
                        )
                    else:
                        _col_i.caption("ISO: no disponible\n(< 2 pasos detectados)")
                    if _ir.get("gradcam_rgb") is not None:
                        _col_g.image(
                            _ir["gradcam_rgb"],
                            caption="GradCAM cresta",
                            width="stretch",
                        )
                    else:
                        _col_g.caption("GradCAM: no disponible")
                    st.markdown("---")

        with tab_diag:
            if d2c:
                c1 = d2c["capa1"]
                c2 = d2c["capa2"]
                _col_c1, _col_c2 = st.columns(2, gap="large")
                _v1c2 = "#22c55e" if c1["veredicto"] == "BUENA" else "#ef4444"
                _v1i2 = "✅" if c1["veredicto"] == "BUENA" else "⚠️"
                with _col_c1:
                    st.markdown(
                        f"""<div style="background:#1a1d27;border:1px solid #2d3148;border-radius:10px;padding:16px;">
                        <div style="font-size:11px;color:#94a3b8;font-weight:600;text-transform:uppercase;margin-bottom:8px;">Capa 1 — Diagnóstico relativo</div>
                        <div style="font-size:12px;color:#cbd5e1;margin-bottom:12px;">Variabilidad entre imágenes de la misma rosca</div>
                        <div style="display:flex;justify-content:space-between;margin-bottom:6px;"><span style="color:#94a3b8;font-size:12px;">CV medio</span><span style="color:#f1f5f9;font-size:12px;font-weight:600;">{c1['cv_medio']:.4f}</span></div>
                        <div style="display:flex;justify-content:space-between;margin-bottom:14px;"><span style="color:#94a3b8;font-size:12px;">Umbral</span><span style="color:#94a3b8;font-size:12px;">{c1['umbral']}</span></div>
                        <div style="border-top:1px solid #2d3148;padding-top:12px;font-size:16px;font-weight:700;color:{_v1c2};">{_v1i2} {c1['veredicto']}</div>
                    </div>""",
                        unsafe_allow_html=True,
                    )
                _v2m2 = {
                    "CONFORME": ("#22c55e", "✅ CONFORME"),
                    "DESVIACION_LEVE": ("#f59e0b", "⚠️ DESVIACIÓN LEVE"),
                    "DESVIACION_SIGNIFICATIVA": (
                        "#ef4444",
                        "❌ DESVIACIÓN SIGNIFICATIVA",
                    ),
                }
                _v2c2b, _v2lbl2 = _v2m2.get(
                    c2["veredicto"], ("#94a3b8", c2["veredicto"])
                )
                with _col_c2:
                    _cal_b = (
                        '<span style="font-size:10px;background:#14532d;color:#22c55e;padding:2px 6px;border-radius:4px;margin-left:6px;">calibrado px/mm</span>'
                        if c2["calibrado"]
                        else '<span style="font-size:10px;background:#422006;color:#f59e0b;padding:2px 6px;border-radius:4px;margin-left:6px;">sin calibración</span>'
                    )
                    st.markdown(
                        f"""<div style="background:#1a1d27;border:1px solid #2d3148;border-radius:10px;padding:16px;">
                        <div style="font-size:11px;color:#94a3b8;font-weight:600;text-transform:uppercase;margin-bottom:4px;">Capa 2 — Perfil ISO {_cal_b}</div>
                        <div style="font-size:12px;color:#cbd5e1;margin-bottom:12px;">vs. ISO M10×1.5</div>""",
                        unsafe_allow_html=True,
                    )
                    if c2["disponible"]:
                        for _chk in c2["checks"]:
                            _ic2 = "✅" if _chk["ok"] else "❌"
                            _cl2 = "#22c55e" if _chk["ok"] else "#ef4444"
                            _vl2 = (
                                f"{_chk['medido']} {_chk['unidad']}"
                                if _chk["unidad"] != "—"
                                else str(_chk["medido"])
                            )
                            _tl2 = (
                                f"{_chk['teo']} {_chk['unidad']}"
                                if _chk["unidad"] != "—"
                                else str(_chk["teo"])
                            )
                            st.markdown(
                                f"<div style='font-size:12px;margin-bottom:4px;'>{_ic2} <span style='color:#94a3b8;'>{_chk['nombre']}:</span> <span style='color:#f1f5f9;'>{_vl2}</span> <span style='color:#475569;'>(teo: {_tl2}) → <span style='color:{_cl2};'>±{_chk['desv_pct']}%</span></span></div>",
                                unsafe_allow_html=True,
                            )
                        st.markdown(
                            f"<div style='border-top:1px solid #2d3148;padding-top:12px;font-size:16px;font-weight:700;color:{_v2c2b};margin-top:8px;'>{_v2lbl2}</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.caption("No se detectaron crestas en ninguna imagen.")
                    st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("---")
                _vfc2 = "#22c55e" if d2c["veredicto_final"] == "BUENA" else "#ef4444"
                _vfi2 = "✅" if d2c["veredicto_final"] == "BUENA" else "⚠️"
                _acrd2 = d2c["acuerdo"]
                _itp2 = (
                    (
                        "Ambas capas confirman normalidad."
                        if d2c["veredicto_final"] == "BUENA"
                        else "Ambas capas detectan anomalías. Desgaste probable."
                    )
                    if _acrd2
                    else (
                        "Variabilidad baja pero dimensiones fuera de ISO — desgaste uniforme posible."
                        if d2c["capa1"]["veredicto"] == "BUENA"
                        else "Alta variabilidad pero dimensiones ajustadas al ISO — posible defecto localizado."
                    )
                )
                st.markdown(
                    f"""<div style="background:#13151f;border:1px solid #2d3148;border-radius:10px;padding:14px;display:flex;align-items:center;gap:16px;">
                    <div style="font-size:20px;font-weight:700;color:{_vfc2};white-space:nowrap;">{_vfi2} {d2c["veredicto_final"]}</div>
                    <div style="font-size:13px;color:#94a3b8;line-height:1.5;">{_itp2}</div>
                </div>""",
                    unsafe_allow_html=True,
                )

                _LABEL_MAP = {
                    "cv_area_f": "cv_area_cresta",
                    "cv_ancho_f": "cv_ancho_cresta",
                    "cv_alto_f": "cv_alto_cresta",
                    "cv_solid_f": "cv_solid_cresta",
                    "solid_f_media": "solid_cresta_media",
                    "rect_f_media": "rect_cresta_media",
                }
                if imgs_rosca:
                    _img_max = max(imgs_rosca, key=lambda x: x.get("score", 0))
                    _det_ej = _img_max.get("detalles", {})
                    _contribs = {
                        _LABEL_MAP.get(k, k): v["contrib"]
                        for k, v in _det_ej.items()
                        if v.get("contrib", 0) > 0
                    }
                    from inference import _INVERSAS, UMBRALES_BUENOS

                    with st.expander("🔍 Valores vs umbrales", expanded=False):
                        _rows = []
                        for _k, _det_v in _img_max.get("detalles", {}).items():
                            _lbl = _LABEL_MAP.get(_k, _k)
                            _val = _det_v.get("valor", 0)
                            _umb = _det_v.get("umbral", 0)
                            _inv = _k in _INVERSAS
                            _ok = (_val < _umb) if not _inv else (_val >= _umb)
                            _rows.append(
                                {
                                    "Métrica": _lbl,
                                    "Valor": round(_val, 4),
                                    "Umbral": _umb,
                                    "OK": "✅" if _ok else "❌",
                                }
                            )
                        st.dataframe(
                            pd.DataFrame(_rows), hide_index=True, width="stretch"
                        )
                    if _contribs:
                        st.markdown("**Métricas que disparan el diagnóstico:**")
                        _df_c = pd.DataFrame(
                            {
                                "Métrica": list(_contribs.keys()),
                                "Contribución": list(_contribs.values()),
                            }
                        ).sort_values("Contribución", ascending=True)
                        _fig_c, _ax_c = plt.subplots(
                            figsize=(6, max(2.5, len(_df_c) * 0.35))
                        )
                        _fig_c.patch.set_facecolor("#1a1d27")
                        _dark_ax(_ax_c)
                        _ax_c.barh(
                            _df_c["Métrica"], _df_c["Contribución"], color="#ef4444"
                        )
                        _ax_c.set_xlabel("Contribución", color="#94a3b8")
                        _ax_c.grid(alpha=0.2, color="#2d3148", axis="x")
                        plt.tight_layout()
                        st.pyplot(_fig_c)
                        plt.close(_fig_c)
            else:
                st.info("Diagnóstico en dos capas no disponible.")

        with st.expander("📐 Análisis técnico avanzado", expanded=False):
            _tab_adv_m, _tab_adv_r = st.tabs(["📐 Métricas geométricas", "🔬 ROC"])
            with _tab_adv_m:
                if imgs_rosca:
                    _m = imgs_rosca[-1]["metricas"]
                    _mg1, _mg2, _mg3, _mg4 = st.columns(4)
                    with _mg1:
                        st.markdown("**Cresta**")
                        st.write(
                            {
                                k.replace("_f", "_c"): round(v, 4)
                                for k, v in _m.items()
                                if "_f" in k or "cresta" in k
                            }
                        )
                    with _mg2:
                        st.markdown("**Paso**")
                        st.write(
                            {
                                k: round(v, 4)
                                for k, v in _m.items()
                                if any(x in k for x in ["_p", "paso"])
                            }
                        )
                    with _mg3:
                        st.markdown("**Teórico M10×1.5**")
                        st.write(
                            {
                                "ar_cresta_media": round(
                                    _m.get("ar_cresta_media", 0), 4
                                ),
                                "desv_ar_teo": round(_m.get("desv_ar_teo", 0), 4),
                                "angulo_flanco_medio": round(
                                    _m.get("angulo_flanco_medio", 0), 4
                                ),
                                "desv_angulo_teo": round(
                                    _m.get("desv_angulo_teo", 0), 4
                                ),
                            }
                        )
                    with _mg4:
                        st.markdown("**Escala real (mm)**")
                        _pxmm_m = _m.get("px_por_mm")
                        if _pxmm_m:
                            _alto_mm = _m.get("alto_cresta_mm")
                            _ancho_mm = _m.get("ancho_cresta_mm")
                            st.write(
                                {
                                    "px/mm": round(_pxmm_m, 2),
                                    "alto_cresta_mm": (
                                        round(_alto_mm, 3) if _alto_mm else "—"
                                    ),
                                    "ancho_cresta_mm": (
                                        round(_ancho_mm, 3) if _ancho_mm else "—"
                                    ),
                                }
                            )
                            if _alto_mm:
                                _dp = abs(_alto_mm - 0.812) / 0.812 * 100
                                st.markdown(
                                    f"<span style='font-size:11px;color:{'#ef4444' if _dp > 15 else '#22c55e'};'>Desv. vs ISO H≈0.812 mm: {_dp:.1f}%</span>",
                                    unsafe_allow_html=True,
                                )
                        else:
                            st.caption(
                                "Sin calibración — se necesitan ≥ 2 pasos detectados."
                            )
                else:
                    st.info("No hay imágenes en memoria.")

            with _tab_adv_r:
                if (
                    len(resultados) >= 2
                    and len({r["es_buena"] for r in resultados}) == 2
                ):
                    st.caption("Curva ROC barriendo el umbral CV.")
                    _render_analisis_roc(resultados, key_prefix="res_mem")
                else:
                    st.info(
                        "Se necesitan roscas buenas **y** con desgaste para el análisis ROC."
                    )

    _combo_exps = []  # ocultar en la UI final — datos heredados de script auxiliar
    _auto_exps = [e for e in experimentos if e.get("tipo") == "auto"]
    if not _combo_exps and not _auto_exps:
        st.stop()

    st.markdown("---")
    with st.expander("Resumen de experimentos Auto", expanded=False):
        _tab_cmp_labels = []
        if _auto_exps:
            _tab_cmp_labels.append("Auto")
        if _combo_exps:
            _tab_cmp_labels.append("🗺️ Combos")
        _tabs_cmp = st.tabs(_tab_cmp_labels)

        if _auto_exps:
            _auto_tab_idx = _tab_cmp_labels.index("Auto")
            with _tabs_cmp[_auto_tab_idx]:
                st.caption(
                    "Experimentos generados por el modo Auto (3 fases: individuales → pares → expansión greedy)."
                )

                _a_mejor = get_mejor_modelo()
                _a_mejor_id = _a_mejor["id"] if _a_mejor else None

                _a_filas = []
                for _ae in sorted(
                    _auto_exps, key=lambda e: get_accuracy(e["id"]) or 0, reverse=True
                ):
                    _a_rtr = _json.loads(_ae["roscas_train"])
                    _a_acc = get_accuracy(_ae["id"])
                    _a_filas.append(
                        {
                            "⭐": "⭐" if _ae["id"] == _a_mejor_id else "",
                            "ID": f"#{_ae['id']}",
                            "Combo": " + ".join(_a_rtr),
                            "Tamaño": len(_a_rtr),
                            "Accuracy": _a_acc if _a_acc is not None else None,
                            "AUC-ROC": (
                                round(_ae["auc_roc"], 3) if _ae.get("auc_roc") else None
                            ),
                            "F1": round(_ae["f1"], 3) if _ae.get("f1") else None,
                            "IoU Cresta": (
                                round(_ae["iou_cresta"], 3)
                                if _ae.get("iou_cresta")
                                else None
                            ),
                            "T. entreno": (
                                round(_ae["tiempo_train_s"])
                                if _ae.get("tiempo_train_s")
                                else None
                            ),
                        }
                    )
                _ev_auto = st.dataframe(
                    pd.DataFrame(_a_filas),
                    width="stretch",
                    hide_index=True,
                    on_select="rerun",
                    selection_mode="single-row",
                    key="tabla_auto_cmp",
                    column_config={
                        "Accuracy": st.column_config.ProgressColumn(
                            "Accuracy", format="%.0f%%", min_value=0, max_value=1
                        ),
                        "AUC-ROC": st.column_config.NumberColumn(
                            "AUC-ROC", format="%.3f"
                        ),
                        "F1": st.column_config.NumberColumn("F1", format="%.3f"),
                        "IoU Cresta": st.column_config.NumberColumn(
                            "IoU Cresta", format="%.3f"
                        ),
                    },
                )
                _auto_sel = (
                    _ev_auto.selection.rows if hasattr(_ev_auto, "selection") else []
                )
                _auto_exps_sorted = sorted(
                    _auto_exps, key=lambda e: get_accuracy(e["id"]) or 0, reverse=True
                )
                if _auto_sel:
                    st.session_state["comp_sel_id"] = _auto_exps_sorted[_auto_sel[0]][
                        "id"
                    ]

                _a_sizes = sorted(
                    {len(_json.loads(e["roscas_train"])) for e in _auto_exps}
                )
                if len(_a_sizes) > 1:
                    st.markdown("**AUC-ROC por tamaño de conjunto train:**")
                    _col_aa, _col_ab = st.columns(2)
                    with _col_aa:
                        _fig_as, _ax_as = plt.subplots(figsize=(6, 3.5))
                        _fig_as.patch.set_facecolor("#1a1d27")
                        _dark_ax(_ax_as)
                        for _sz in _a_sizes:
                            _ae_sz = [
                                e
                                for e in _auto_exps
                                if len(_json.loads(e["roscas_train"])) == _sz
                            ]
                            _aucs_sz = [
                                e["auc_roc"] for e in _ae_sz if e.get("auc_roc")
                            ]
                            if _aucs_sz:
                                _ax_as.scatter(
                                    [_sz] * len(_aucs_sz),
                                    _aucs_sz,
                                    color="#4f8ef7",
                                    alpha=0.7,
                                    s=60,
                                    zorder=3,
                                )
                                _ax_as.scatter(
                                    _sz,
                                    max(_aucs_sz),
                                    color="gold",
                                    s=90,
                                    zorder=4,
                                    marker="*",
                                    label=(
                                        f"max sz={_sz}"
                                        if _sz == max(_a_sizes)
                                        else None
                                    ),
                                )
                        _ax_as.set_xlabel("Nº roscas train", color="#94a3b8")
                        _ax_as.set_ylabel("AUC-ROC", color="#94a3b8")
                        _ax_as.set_ylim(0, 1.05)
                        _ax_as.set_xticks(_a_sizes)
                        _ax_as.set_title(
                            "AUC-ROC vs. tamaño train", color="#f1f5f9", fontsize=11
                        )
                        _ax_as.grid(alpha=0.15, color="#2d3148")
                        plt.tight_layout()
                        st.pyplot(_fig_as)
                        plt.close(_fig_as)

                    with _col_ab:
                        _fig_ab, _ax_ab = plt.subplots(figsize=(6, 3.5))
                        _fig_ab.patch.set_facecolor("#1a1d27")
                        _dark_ax(_ax_ab)
                        _best_per_sz = []
                        for _sz in _a_sizes:
                            _ae_sz = [
                                e
                                for e in _auto_exps
                                if len(_json.loads(e["roscas_train"])) == _sz
                            ]
                            _best = max(
                                _ae_sz,
                                key=lambda e: e.get("auc_roc") or 0,
                                default=None,
                            )
                            if _best:
                                _best_per_sz.append(
                                    (
                                        _sz,
                                        _best.get("auc_roc") or 0,
                                        _best.get("f1") or 0,
                                        get_accuracy(_best["id"]) or 0,
                                    )
                                )
                        if _best_per_sz:
                            _szs, _b_auc, _b_f1, _b_acc = zip(*_best_per_sz)
                            _xb = np.arange(len(_szs))
                            _ax_ab.plot(
                                _xb,
                                _b_auc,
                                "o-",
                                color="#4f8ef7",
                                label="AUC-ROC (mejor)",
                            )
                            _ax_ab.plot(
                                _xb, _b_f1, "s-", color="#22c55e", label="F1 (mejor)"
                            )
                            _ax_ab.plot(
                                _xb,
                                _b_acc,
                                "^-",
                                color="#f59e0b",
                                label="Accuracy (mejor)",
                            )
                            _ax_ab.set_xticks(_xb)
                            _ax_ab.set_xticklabels(
                                [f"sz={s}" for s in _szs], color="#94a3b8", fontsize=9
                            )
                            _ax_ab.set_ylim(0, 1.05)
                            _ax_ab.set_title(
                                "Mejor combo por tamaño", color="#f1f5f9", fontsize=11
                            )
                            _ax_ab.grid(alpha=0.15, color="#2d3148", axis="y")
                            _ax_ab.legend(
                                facecolor="#1a1d27", labelcolor="#94a3b8", fontsize=8
                            )
                        plt.tight_layout()
                        st.pyplot(_fig_ab)
                        plt.close(_fig_ab)

        if _combo_exps:
            with _tabs_cmp[-1]:
                st.caption(
                    "Generado con `train_all_combos.py`. Verde = diagnóstico correcto · Rojo = incorrecto."
                )

                sizes_available = sorted(
                    {len(_json.loads(e["roscas_train"])) for e in _combo_exps}
                )
                if len(sizes_available) > 1:
                    size_sel = st.select_slider(
                        "Tamaño del conjunto train",
                        options=sizes_available,
                        value=sizes_available[0],
                        key="hm_size",
                    )
                else:
                    size_sel = sizes_available[0]
                    st.caption(f"Tamaño del conjunto train: **{size_sel}**")
                exps_filt = [
                    e
                    for e in _combo_exps
                    if len(_json.loads(e["roscas_train"])) == size_sel
                ]

                all_eval: list[str] = []
                exp_res_map: dict[int, dict] = {}
                for exp in exps_filt:
                    res = get_resultados(exp["id"])
                    exp_res_map[exp["id"]] = {
                        r["rosca_id"]: int(r["correcto"]) for r in res
                    }
                    for r in res:
                        if r["rosca_id"] not in all_eval:
                            all_eval.append(r["rosca_id"])
                all_eval = sorted(
                    all_eval, key=lambda x: (0 if x.startswith("RB") else 1, x)
                )

                if all_eval and exps_filt:
                    row_labels = [
                        "+".join(_json.loads(e["roscas_train"])) for e in exps_filt
                    ]
                    import numpy as _np_hm

                    mat = _np_hm.full((len(exps_filt), len(all_eval)), _np_hm.nan)
                    for ri, exp in enumerate(exps_filt):
                        for ci, rosca in enumerate(all_eval):
                            v = exp_res_map[exp["id"]].get(rosca)
                            if v is not None:
                                mat[ri, ci] = v
                    acc_row = _np_hm.nanmean(mat, axis=1)

                    fig_hm, ax_hm = plt.subplots(
                        figsize=(
                            max(8, len(all_eval) * 0.65),
                            max(3, len(exps_filt) * 0.45 + 1),
                        )
                    )
                    fig_hm.patch.set_facecolor("#1a1d27")
                    _dark_ax(ax_hm)
                    im = ax_hm.imshow(
                        mat,
                        aspect="auto",
                        cmap="RdYlGn",
                        vmin=0,
                        vmax=1,
                        interpolation="nearest",
                    )
                    ax_hm.set_xticks(range(len(all_eval)))
                    ax_hm.set_xticklabels(
                        all_eval, rotation=45, ha="right", color="#94a3b8", fontsize=9
                    )
                    ylabels = [
                        f"{lbl}  ({acc_row[i]*100:.0f}%)"
                        for i, lbl in enumerate(row_labels)
                    ]
                    ax_hm.set_yticks(range(len(exps_filt)))
                    ax_hm.set_yticklabels(ylabels, color="#94a3b8", fontsize=9)
                    ax_hm.set_title(
                        f"Diagnóstico correcto — train size={size_sel}",
                        color="#f1f5f9",
                        fontsize=12,
                    )
                    n_buenas = sum(1 for r in all_eval if r.startswith("RB"))
                    if 0 < n_buenas < len(all_eval):
                        ax_hm.axvline(
                            n_buenas - 0.5,
                            color="#f59e0b",
                            linewidth=1.5,
                            linestyle="--",
                            alpha=0.7,
                        )
                        ax_hm.text(
                            n_buenas / 2 - 0.5,
                            -0.8,
                            "Buenas",
                            ha="center",
                            color="#f59e0b",
                            fontsize=8,
                        )
                        ax_hm.text(
                            n_buenas + (len(all_eval) - n_buenas) / 2 - 0.5,
                            -0.8,
                            "Con desgaste",
                            ha="center",
                            color="#ef4444",
                            fontsize=8,
                        )
                    for ri in range(len(exps_filt)):
                        for ci in range(len(all_eval)):
                            v = mat[ri, ci]
                            if not _np_hm.isnan(v):
                                ax_hm.text(
                                    ci,
                                    ri,
                                    "✓" if v == 1 else "✗",
                                    ha="center",
                                    va="center",
                                    color="white",
                                    fontsize=10,
                                    fontweight="bold",
                                )
                    plt.colorbar(
                        im, ax=ax_hm, fraction=0.02, pad=0.02, label="Accuracy"
                    ).ax.yaxis.label.set_color("#94a3b8")
                    plt.tight_layout()
                    st.pyplot(fig_hm)
                    plt.close(fig_hm)
                else:
                    st.info(
                        "No hay resultados disponibles para los experimentos de tipo 'combo'."
                    )


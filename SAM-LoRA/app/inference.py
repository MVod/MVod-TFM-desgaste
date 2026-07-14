# app/inference.py
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from config import (
    CANAL_CRESTA,
    CANAL_PASO,
    CV_UMBRAL,
    M10_AR_THEO,
    M10_FLANK_DEG,
    M10_H_CRESTA_MM,
    NUM_CLASSES,
)
from logger import get_logger
from model import device

log = get_logger("inference")
from segment_anything.utils.transforms import ResizeLongestSide

IMG_H, IMG_W = 1024, 1024
THRESHOLD = 0.5

transform = ResizeLongestSide(1024)

UMBRALES_BUENOS = {
    "cv_area_f": 0.033,
    "cv_ancho_f": 0.030,
    "cv_alto_f": 0.030,
    "solid_f_media": 0.900,
    "cv_solid_f": 0.050,
    "rect_f_media": 0.700,
    "cv_area_p": 0.150,
    "cv_ancho_p": 0.150,
    "solid_p_media": 0.850,
    "horiz_p_media": 0.050,
    "desv_ar_teo": 0.120,
    "desv_angulo_teo": 8.000,
}

PESOS = {
    "cv_area_f": 2.0,
    "cv_ancho_f": 1.0,
    "cv_alto_f": 1.0,
    "solid_f_media": 2.0,
    "cv_solid_f": 1.0,
    "rect_f_media": 2.0,
    "cv_area_p": 1.0,
    "cv_ancho_p": 1.0,
    "solid_p_media": 1.5,
    "horiz_p_media": 1.5,
    "desv_ar_teo": 1.5,
    "desv_angulo_teo": 1.5,
}

UMBRAL_IMG = 0.033
UMBRAL_ROSCA = CV_UMBRAL

_INVERSAS = {"solid_f_media", "rect_f_media", "solid_p_media"}


def inferir_mascara(sam_model, head_model, img_gray: np.ndarray) -> np.ndarray:
    """Devuelve probabilidades de segmentación (NUM_CLASSES, H, W) en [0, 1]."""
    h, w = img_gray.shape
    img_resized = cv2.resize(img_gray, (IMG_W, IMG_H))
    img_rgb = cv2.cvtColor(cv2.bitwise_not(img_resized), cv2.COLOR_GRAY2RGB)
    inp = transform.apply_image(img_rgb)
    inp = torch.as_tensor(inp, device=device).permute(2, 0, 1).contiguous()
    inp = sam_model.preprocess(inp.unsqueeze(0))

    with torch.no_grad():
        emb = sam_model.image_encoder(inp)
        logits = head_model(emb)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    return np.stack(
        [
            cv2.resize(probs[c], (w, h), interpolation=cv2.INTER_LINEAR)
            for c in range(NUM_CLASSES)
        ],
        axis=0,
    )


def _estimar_px_por_mm(pasos: list[dict]) -> float | None:
    """
    Estima la escala px/mm usando el paso M10×1.5 (P=1.5mm) como calibrador.
    Proyecta los centroides de los pasos sobre el eje principal del tornillo
    y mide la distancia mediana entre pasos consecutivos.
    """
    if len(pasos) < 2:
        log.debug(
            "px_por_mm: insuficientes pasos para calibrar (%d detectados)", len(pasos)
        )
        return None
    pts = np.array([[p["cx"], p["cy"]] for p in pasos], dtype=np.float32)
    mean_pt = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - mean_pt)
    axis = vt[0]
    projs = sorted(float(np.dot(pt - mean_pt, axis)) for pt in pts)
    diffs = np.diff(projs)
    diffs = diffs[diffs > 2]
    if len(diffs) == 0:
        log.debug(
            "px_por_mm: separaciones entre pasos demasiado pequeñas, no calibrable"
        )
        return None
    pitch_px = float(np.median(diffs))
    return pitch_px / 1.5


def _estimar_angulo_flanco(cnt: np.ndarray, bbox: tuple) -> float:
    """
    Estima el ángulo del flanco del diente respecto a la vertical (grados).
    Teórico para M10x1.5: 30° (= 90° - 60° de ángulo de flanco).
    """
    x, y, w, h = bbox
    if h < 12 or w < 6:
        return M10_FLANK_DEG

    pts = cnt.reshape(-1, 2).astype(np.float32)
    x_center = x + w * 0.5
    y_lo, y_hi = y + h * 0.15, y + h * 0.85
    mid = pts[(pts[:, 1] >= y_lo) & (pts[:, 1] <= y_hi)]
    if len(mid) < 6:
        return M10_FLANK_DEG

    angles = []
    for lado in [mid[mid[:, 0] < x_center], mid[mid[:, 0] >= x_center]]:
        if len(lado) >= 3:
            [vx, vy, _, _] = cv2.fitLine(
                lado.astype(np.int32).reshape(-1, 1, 2), cv2.DIST_L2, 0, 0.01, 0.01
            )
            vx_s = float(np.asarray(vx).flat[0])
            vy_s = float(np.asarray(vy).flat[0])
            angle = float(np.degrees(np.arctan2(abs(vx_s), abs(vy_s) + 1e-6)))
            angles.append(angle)

    return float(np.mean(angles)) if angles else M10_FLANK_DEG


def extraer_metricas(img_gray: np.ndarray, probs: np.ndarray) -> dict:
    """
    Extrae métricas geométricas + desviaciones del perfil teórico M10x1.5.
    """
    mask_cresta = (probs[CANAL_CRESTA] > THRESHOLD).astype(np.uint8) * 255
    mask_paso = (probs[CANAL_PASO] > THRESHOLD).astype(np.uint8) * 255

    def analizar_contornos(mask: np.ndarray, area_min: float) -> list[dict]:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [c for c in cnts if cv2.contourArea(c) > area_min]
        items = []
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            M_cnt = cv2.moments(cnt)
            cx = float(M_cnt["m10"] / (M_cnt["m00"] + 1e-6))
            cy = float(M_cnt["m01"] / (M_cnt["m00"] + 1e-6))
            hull = cv2.convexHull(cnt)
            area_hull = cv2.contourArea(hull)
            solidity = area / (area_hull + 1e-6)
            rect = area / (float(w * h) + 1e-6)
            pts = cnt.reshape(-1, 2)
            y_top = pts[pts[:, 1] < (y + h * 0.2), 1]
            horiz = float(np.std(y_top)) / (h + 1e-6) if len(y_top) > 1 else 0.0
            ar = float(h) / (float(w) + 1e-6)
            angulo = _estimar_angulo_flanco(cnt, (x, y, w, h))
            items.append(
                {
                    "area": area,
                    "ancho": float(w),
                    "alto": float(h),
                    "cx": cx,
                    "cy": cy,
                    "solidity": solidity,
                    "rect": rect,
                    "horiz": horiz,
                    "ar": ar,
                    "angulo_flanco": angulo,
                }
            )
        return items

    def stats(values: list) -> tuple[float, float, float]:
        if not values:
            return 0.0, 0.0, 0.0
        arr = np.array(values, dtype=float)
        m, s = arr.mean(), arr.std()
        return float(m), float(s), float(s / (m + 1e-6))

    crestas = analizar_contornos(mask_cresta, 500)
    pasos = analizar_contornos(mask_paso, 200)

    area_f_m, _, cv_area_f = stats([c["area"] for c in crestas])
    ancho_f_m, _, cv_ancho_f = stats([c["ancho"] for c in crestas])
    alto_f_m, _, cv_alto_f = stats([c["alto"] for c in crestas])
    solid_f_m, _, cv_solid_f = stats([c["solidity"] for c in crestas])
    rect_f_m, _, cv_rect_f = stats([c["rect"] for c in crestas])
    ar_f_m, _, _ = stats([c["ar"] for c in crestas])
    ang_f_m, _, _ = stats([c["angulo_flanco"] for c in crestas])

    area_p_m, _, cv_area_p = stats([p["area"] for p in pasos])
    ancho_p_m, _, cv_ancho_p = stats([p["ancho"] for p in pasos])
    solid_p_m, _, cv_solid_p = stats([p["solidity"] for p in pasos])
    horiz_p_m, _, cv_horiz_p = stats([p["horiz"] for p in pasos])

    desv_ar_teo = abs(ar_f_m - M10_AR_THEO) if crestas else 0.0
    desv_angulo_teo = abs(ang_f_m - M10_FLANK_DEG) if crestas else 0.0

    px_por_mm = _estimar_px_por_mm(pasos)
    alto_cresta_mm = alto_f_m / px_por_mm if px_por_mm else None
    ancho_cresta_mm = ancho_f_m / px_por_mm if px_por_mm else None

    return {
        "n_crestas": len(crestas),
        "n_pasos": len(pasos),
        "area_f_media": area_f_m,
        "cv_area_f": cv_area_f,
        "ancho_f_media": ancho_f_m,
        "cv_ancho_f": cv_ancho_f,
        "alto_f_media": alto_f_m,
        "cv_alto_f": cv_alto_f,
        "solid_f_media": solid_f_m,
        "cv_solid_f": cv_solid_f,
        "rect_f_media": rect_f_m,
        "cv_rect_f": cv_rect_f,
        "ar_cresta_media": ar_f_m,
        "angulo_flanco_medio": ang_f_m,
        "desv_ar_teo": desv_ar_teo,
        "desv_angulo_teo": desv_angulo_teo,
        "area_p_media": area_p_m,
        "cv_area_p": cv_area_p,
        "ancho_p_media": ancho_p_m,
        "cv_ancho_p": cv_ancho_p,
        "solid_p_media": solid_p_m,
        "cv_solid_p": cv_solid_p,
        "horiz_p_media": horiz_p_m,
        "cv_horiz_p": cv_horiz_p,
        "ratio_pf": area_p_m / (area_f_m + 1e-6),
        "cobertura_f": mask_cresta.sum() / (mask_cresta.size * 255),
        "cobertura_p": mask_paso.sum() / (mask_paso.size * 255),
        "px_por_mm": px_por_mm,
        "alto_cresta_mm": alto_cresta_mm,
        "ancho_cresta_mm": ancho_cresta_mm,
    }



def calcular_score(m: dict) -> tuple[float, dict]:
    """Score ponderado geométrico + teórico. Mayor valor = más anomalía."""
    score_total, peso_total = 0.0, 0.0
    detalles = {}

    for key, umbral in UMBRALES_BUENOS.items():
        val = m.get(key, 0.0)
        peso = PESOS.get(key, 1.0)

        if key in _INVERSAS:
            contrib = max(0.0, (umbral - val) / (umbral + 1e-6))
        else:
            contrib = max(0.0, (val - umbral) / (umbral + 1e-6))

        score_total += peso * contrib
        peso_total += peso
        detalles[key] = {"valor": val, "umbral": umbral, "contrib": contrib * peso}

    return score_total / (peso_total + 1e-6), detalles


_DESC_MODOS = {
    "Desgaste de flancos": "Alta variabilidad de área y baja solidez de la cresta.",
    "Pérdida de cresta": "Alta variabilidad de altura y baja rectangularidad.",
    "Deformación geométrica": "Perfil se aleja del teórico M10x1.5 (ratio alto/ancho o ángulo).",
    "Irregularidad de paso": "Alta variabilidad en el espacio entre crestas.",
    "Desgaste generalizado": "Múltiples métricas simultáneamente fuera de rango.",
}


def clasificar_modo_desgaste(
    metricas: dict, detalles: dict, score: float
) -> list[dict]:
    """
    Retorna lista de modos detectados: [{modo, descripcion, severidad}].
    Solo se llama cuando el veredicto ya es POSIBLE DESGASTE.
    """
    m = metricas
    modos = []

    if (
        m.get("cv_area_f", 0) > UMBRALES_BUENOS["cv_area_f"]
        and m.get("solid_f_media", 1) < UMBRALES_BUENOS["solid_f_media"]
    ):
        sev = (
            detalles.get("cv_area_f", {}).get("contrib", 0)
            + detalles.get("solid_f_media", {}).get("contrib", 0)
        ) / 2
        modos.append(
            {
                "modo": "Desgaste de flancos",
                "descripcion": _DESC_MODOS["Desgaste de flancos"],
                "severidad": min(sev, 1.0),
            }
        )

    if (
        m.get("cv_alto_f", 0) > UMBRALES_BUENOS["cv_alto_f"]
        and m.get("rect_f_media", 1) < UMBRALES_BUENOS["rect_f_media"]
    ):
        sev = (
            detalles.get("cv_alto_f", {}).get("contrib", 0)
            + detalles.get("rect_f_media", {}).get("contrib", 0)
        ) / 2
        modos.append(
            {
                "modo": "Pérdida de cresta",
                "descripcion": _DESC_MODOS["Pérdida de cresta"],
                "severidad": min(sev, 1.0),
            }
        )

    if (
        m.get("desv_ar_teo", 0) > UMBRALES_BUENOS["desv_ar_teo"]
        or m.get("desv_angulo_teo", 0) > UMBRALES_BUENOS["desv_angulo_teo"]
    ):
        sev = (
            detalles.get("desv_ar_teo", {}).get("contrib", 0)
            + detalles.get("desv_angulo_teo", {}).get("contrib", 0)
        ) / 2
        modos.append(
            {
                "modo": "Deformación geométrica",
                "descripcion": _DESC_MODOS["Deformación geométrica"],
                "severidad": min(sev, 1.0),
            }
        )

    if m.get("cv_area_p", 0) > UMBRALES_BUENOS["cv_area_p"]:
        sev = detalles.get("cv_area_p", {}).get("contrib", 0)
        modos.append(
            {
                "modo": "Irregularidad de paso",
                "descripcion": _DESC_MODOS["Irregularidad de paso"],
                "severidad": min(sev, 1.0),
            }
        )

    n_activas = sum(1 for d in detalles.values() if d["contrib"] > 0.05)
    if score > 0.4 and n_activas >= 4:
        modos.append(
            {
                "modo": "Desgaste generalizado",
                "descripcion": _DESC_MODOS["Desgaste generalizado"],
                "severidad": min(score, 1.0),
            }
        )

    return modos



def generar_gradcam(
    sam_model, head_model, img_gray: np.ndarray, canal: int = CANAL_CRESTA
) -> np.ndarray:
    """
    GradCAM sobre la capa up2 del SegHead para el canal indicado.
    Desacopla el embedding del encoder para que el grafo solo recorra el SegHead.
    Retorna heatmap normalizado (H, W) en [0, 1].
    """
    h, w = img_gray.shape
    img_resized = cv2.resize(img_gray, (IMG_W, IMG_H))
    img_rgb = cv2.cvtColor(cv2.bitwise_not(img_resized), cv2.COLOR_GRAY2RGB)
    inp = transform.apply_image(img_rgb)
    inp_t = torch.as_tensor(inp, device=device).permute(2, 0, 1).contiguous()
    inp_t = sam_model.preprocess(inp_t.unsqueeze(0))

    _acts = [None]
    _grads = [None]

    def fwd_hook(module, inp_, out):
        _acts[0] = out

    def bwd_hook(module, grad_in, grad_out):
        _grads[0] = grad_out[0]

    fh = head_model.up2.register_forward_hook(fwd_hook)
    bh = head_model.up2.register_full_backward_hook(bwd_hook)

    sam_model.image_encoder.eval()
    head_model.eval()

    with torch.enable_grad():
        emb = sam_model.image_encoder(inp_t).detach().requires_grad_(True)
        logits = head_model(emb)
        score = logits[0, canal].mean()
        score.backward()

    fh.remove()
    bh.remove()

    acts = _acts[0].detach().squeeze(0)
    grads = _grads[0].detach().squeeze(0)
    weights = grads.mean(dim=(1, 2), keepdim=True)
    cam = torch.relu((weights * acts).sum(dim=0)).cpu().numpy()

    cam = cv2.resize(cam, (w, h))
    if cam.max() > 0:
        cam = (cam / cam.max()).astype(np.float32)
    else:
        cam = np.zeros((h, w), dtype=np.float32)

    return cam


def overlay_gradcam(
    img_gray: np.ndarray, cam: np.ndarray, alpha: float = 0.45
) -> np.ndarray:
    """Superpone el heatmap GradCAM sobre la imagen en escala de grises."""
    img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 1 - alpha, heatmap, alpha, 0)


def _draw_dashed(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 1,
    dash_len: int = 8,
    gap_len: int = 5,
) -> None:
    """Dibuja una línea discontinua entre pt1 y pt2."""
    x1, y1 = float(pt1[0]), float(pt1[1])
    x2, y2 = float(pt2[0]), float(pt2[1])
    length = np.hypot(x2 - x1, y2 - y1)
    if length < 1:
        return
    dx, dy = (x2 - x1) / length, (y2 - y1) / length
    t, drawing = 0.0, True
    while t < length:
        seg_len = dash_len if drawing else gap_len
        t_end = min(t + seg_len, length)
        if drawing:
            p0 = (int(x1 + t * dx), int(y1 + t * dy))
            p1 = (int(x1 + t_end * dx), int(y1 + t_end * dy))
            cv2.line(img, p0, p1, color, thickness, lineType=cv2.LINE_AA)
        t = t_end
        drawing = not drawing


def generar_overlay_iso(
    img_gray: np.ndarray,
    probs: np.ndarray,
    px_por_mm: float | None = None,
) -> np.ndarray | None:
    """
    Superpone el perfil teórico ISO M10×1.5 sobre la segmentación.

    - Líneas amarillas:  centros de pasos detectados (referencia de calibración).
    - Líneas naranja:    posición teórica de crestas (exactamente entre pasos consecutivos).
    - Líneas blancas:    ancho teórico de cresta ±P/8 (solo si px_por_mm disponible).

    Retorna imagen BGR o None si no hay suficientes pasos detectados (< 2).
    """
    h, w = img_gray.shape
    mask_f = cv2.resize(
        (probs[CANAL_CRESTA] > THRESHOLD).astype(np.uint8) * 255,
        (w, h),
        interpolation=cv2.INTER_NEAREST,
    )
    mask_p = cv2.resize(
        (probs[CANAL_PASO] > THRESHOLD).astype(np.uint8) * 255,
        (w, h),
        interpolation=cv2.INTER_NEAREST,
    )

    cnts_p, _ = cv2.findContours(mask_p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_p = [c for c in cnts_p if cv2.contourArea(c) > 200]
    if len(cnts_p) < 2:
        log.warning("overlay_iso: < 2 pasos detectados, no se puede generar el overlay")
        return None

    pasos_pts = []
    for cnt in cnts_p:
        M_ = cv2.moments(cnt)
        if M_["m00"] < 1:
            continue
        pasos_pts.append(
            np.array([M_["m10"] / M_["m00"], M_["m01"] / M_["m00"]], dtype=np.float32)
        )
    if len(pasos_pts) < 2:
        return None

    pts = np.stack(pasos_pts)
    mean_pt = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - mean_pt)
    axis = vt[0]
    perp = vt[1]

    projs_sorted = sorted((float(np.dot(p - mean_pt, axis)), p) for p in pasos_pts)

    all_mask = cv2.bitwise_or(mask_f, mask_p)
    ys, xs = np.where(all_mask > 0)
    if len(ys) == 0:
        return None
    thread_pts = np.stack([xs, ys], axis=1).astype(np.float32)
    perp_projs = thread_pts @ perp - float(mean_pt @ perp)
    p_min, p_max = float(perp_projs.min()) - 12, float(perp_projs.max()) + 12

    def _endpoints(center: np.ndarray) -> tuple[tuple, tuple]:
        p0 = center + p_min * perp
        p1 = center + p_max * perp
        return (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1]))

    img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    result = img_bgr.copy()

    cnts_f, _ = cv2.findContours(mask_f, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(
        result, cnts_f, -1, (200, 100, 100), 1
    )
    cv2.drawContours(
        result, cnts_p, -1, (100, 200, 100), 1
    )

    for _, paso_center in projs_sorted:
        cv2.line(result, *_endpoints(paso_center), color=(0, 210, 210), thickness=2)


    for i in range(len(projs_sorted) - 1):
        _, p1 = projs_sorted[i]
        _, p2 = projs_sorted[i + 1]
        crest_center = (p1 + p2) / 2.0
        cv2.line(result, *_endpoints(crest_center), color=(0, 100, 255), thickness=2)


    if px_por_mm:
        half_crest_px = 0.1875 * px_por_mm  
        for i in range(len(projs_sorted) - 1):
            proj1, _ = projs_sorted[i]
            proj2, _ = projs_sorted[i + 1]
            mid_proj = (proj1 + proj2) / 2.0
            for sign in (-1.0, +1.0):
                edge = mean_pt + (mid_proj + sign * half_crest_px) * axis
                _draw_dashed(
                    result,
                    *_endpoints(edge),
                    color=(220, 220, 220),
                    thickness=1,
                    dash_len=5,
                    gap_len=4,
                )

    log.info(
        "overlay_iso generado | %d pasos | px_por_mm=%s",
        len(projs_sorted),
        f"{px_por_mm:.1f}" if px_por_mm else "N/A",
    )
    return result


_AR_UMBRAL_PCT = 12.0
_ANG_UMBRAL_DEG = 8.0
_H_UMBRAL_PCT = 20.0


def calcular_diagnostico_dos_capas(
    imagenes: list[dict],
    cv_medio: float,
    px_por_mm: float | None,
) -> dict:
    """
    Diagnóstico en dos capas para una rosca completa.

    - Capa 1 (relativa): variabilidad inter-imagen (CV > UMBRAL_ROSCA).
    - Capa 2 (absoluta): métricas promedio vs perfil teórico ISO M10×1.5.

    Retorna dict con capa1, capa2, veredicto_final, acuerdo.
    """

    v_c1 = "BUENA" if cv_medio <= UMBRAL_ROSCA else "POSIBLE DESGASTE"


    imgs_f = [img for img in imagenes if img["metricas"].get("n_crestas", 0) > 0]
    checks: list[dict] = []

    if imgs_f:

        ar_med = float(np.mean([img["metricas"]["ar_cresta_media"] for img in imgs_f]))
        d_ar = abs(ar_med - M10_AR_THEO) / (M10_AR_THEO + 1e-6) * 100
        checks.append(
            {
                "nombre": "Aspect ratio cresta (h/w)",
                "medido": round(ar_med, 3),
                "teo": round(M10_AR_THEO, 3),
                "desv_abs": round(abs(ar_med - M10_AR_THEO), 3),
                "desv_pct": round(d_ar, 1),
                "umbral_pct": _AR_UMBRAL_PCT,
                "ok": d_ar <= _AR_UMBRAL_PCT,
                "unidad": "—",
            }
        )


        ang_med = float(
            np.mean([img["metricas"]["angulo_flanco_medio"] for img in imgs_f])
        )
        d_ang = abs(ang_med - M10_FLANK_DEG)
        checks.append(
            {
                "nombre": "Ángulo de flanco",
                "medido": round(ang_med, 1),
                "teo": M10_FLANK_DEG,
                "desv_abs": round(d_ang, 1),
                "desv_pct": round(d_ang / M10_FLANK_DEG * 100, 1),
                "umbral_pct": round(_ANG_UMBRAL_DEG / M10_FLANK_DEG * 100, 1),
                "ok": d_ang <= _ANG_UMBRAL_DEG,
                "unidad": "°",
            }
        )

    if px_por_mm:
        altos = [
            img["metricas"]["alto_cresta_mm"]
            for img in imagenes
            if img["metricas"].get("alto_cresta_mm") is not None
        ]
        if altos:
            alto_med = float(np.mean(altos))
            d_h_pct = abs(alto_med - M10_H_CRESTA_MM) / (M10_H_CRESTA_MM + 1e-6) * 100
            checks.append(
                {
                    "nombre": "Alto de cresta",
                    "medido": round(alto_med, 3),
                    "teo": round(M10_H_CRESTA_MM, 3),
                    "desv_abs": round(abs(alto_med - M10_H_CRESTA_MM), 3),
                    "desv_pct": round(d_h_pct, 1),
                    "umbral_pct": _H_UMBRAL_PCT,
                    "ok": d_h_pct <= _H_UMBRAL_PCT,
                    "unidad": "mm",
                }
            )

    n_fuera = sum(1 for c in checks if not c["ok"])
    if n_fuera == 0:
        v_c2 = "CONFORME"
    elif n_fuera == 1:
        v_c2 = "DESVIACION_LEVE"
    else:
        v_c2 = "DESVIACION_SIGNIFICATIVA"

    c2_alerta = v_c2 != "CONFORME"
    v_final = (
        "POSIBLE DESGASTE" if (v_c1 == "POSIBLE DESGASTE" or c2_alerta) else "BUENA"
    )
    acuerdo = (v_c1 == "POSIBLE DESGASTE") == c2_alerta

    log.info(
        "Diagnóstico 2 capas | C1=%s C2=%s final=%s acuerdo=%s n_fuera=%d",
        v_c1,
        v_c2,
        v_final,
        acuerdo,
        n_fuera,
    )
    return {
        "capa1": {
            "veredicto": v_c1,
            "cv_medio": round(cv_medio, 4),
            "umbral": UMBRAL_ROSCA,
        },
        "capa2": {
            "disponible": bool(checks),
            "calibrado": bool(px_por_mm),
            "veredicto": v_c2,
            "n_fuera": n_fuera,
            "checks": checks,
        },
        "veredicto_final": v_final,
        "acuerdo": acuerdo,
    }


def diagnosticar_imagen(
    ruta_img: str | Path,
    sam_model,
    head_model,
) -> dict:
    """Diagnóstico completo de una sola imagen con modos de desgaste."""
    t0 = time.perf_counter()

    ruta_img = Path(ruta_img)
    img_gray = cv2.imread(str(ruta_img), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError(f"Imagen no encontrada: {ruta_img}")

    probs = inferir_mascara(sam_model, head_model, img_gray)
    metricas = extraer_metricas(img_gray, probs)
    score, detalles = calcular_score(metricas)

    veredicto = "BUENA" if metricas["cv_area_f"] <= UMBRAL_IMG else "POSIBLE DESGASTE"
    modos = (
        clasificar_modo_desgaste(metricas, detalles, score)
        if veredicto == "POSIBLE DESGASTE"
        else []
    )

    h, w = img_gray.shape
    mask_f = cv2.resize(
        (probs[CANAL_CRESTA] > THRESHOLD).astype(np.uint8) * 255,
        (w, h),
        interpolation=cv2.INTER_NEAREST,
    )
    mask_p = cv2.resize(
        (probs[CANAL_PASO] > THRESHOLD).astype(np.uint8) * 255,
        (w, h),
        interpolation=cv2.INTER_NEAREST,
    )
    img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    overlay = img_bgr.copy()
    overlay[mask_f > 0] = [255, 80, 80]
    overlay[mask_p > 0] = [80, 200, 80]
    blended = cv2.addWeighted(img_bgr, 0.45, overlay, 0.55, 0)

    tiempo_ms = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "veredicto": veredicto,
        "score": score,
        "metricas": metricas,
        "detalles": detalles,
        "modos_desgaste": modos,
        "overlay": blended,
        "cv_cresta": metricas["cv_area_f"],
        "img_gray": img_gray,
        "probs": probs,
        "tiempo_ms": tiempo_ms,
    }


def diagnosticar_rosca(
    ruta_rosca: str | Path,
    sam_model,
    head_model,
    excluir: set[str] = None,
) -> dict:
    """
    Diagnostica una rosca completa procesando todas sus imágenes.
    Agrega modos de desgaste por frecuencia entre imágenes con alerta.
    """
    ruta_rosca = Path(ruta_rosca)
    excluir = excluir or set()

    rutas = sorted([r for r in ruta_rosca.glob("*.jpg") if r.name not in excluir])
    if not rutas:
        raise FileNotFoundError(f"No hay imágenes en: {ruta_rosca}")

    log.info("Diagnóstico rosca: %s (%d imágenes)", ruta_rosca.name, len(rutas))
    imagenes, cv_values, scores, tiempos_ms, px_mm_vals = [], [], [], [], []

    t_rosca = time.perf_counter()
    for ruta in rutas:
        try:
            resultado = diagnosticar_imagen(ruta, sam_model, head_model)
        except Exception as exc:
            log.error("Error procesando imagen %s: %s", ruta.name, exc, exc_info=True)
            continue
        resultado["imagen"] = ruta.name
        imagenes.append(resultado)
        cv_values.append(resultado["cv_cresta"])
        scores.append(resultado["score"])
        tiempos_ms.append(resultado["tiempo_ms"])
        pxmm = resultado["metricas"].get("px_por_mm")
        if pxmm is not None:
            px_mm_vals.append(pxmm)
    tiempo_total_inf_ms = round((time.perf_counter() - t_rosca) * 1000, 1)

    if not imagenes:
        log.error("Rosca %s: ninguna imagen procesada correctamente", ruta_rosca.name)
        raise RuntimeError(f"Ninguna imagen procesada correctamente en {ruta_rosca}")

    px_por_mm_rosca = float(np.median(px_mm_vals)) if px_mm_vals else None

    cv_medio = float(np.mean(cv_values))
    cv_max = float(np.max(cv_values))
    n_alertas = sum(1 for r in imagenes if r["veredicto"] == "POSIBLE DESGASTE")

    modos_counter: Counter = Counter()
    for img_res in imagenes:
        if img_res["veredicto"] == "POSIBLE DESGASTE":
            for m in img_res.get("modos_desgaste", []):
                modos_counter[m["modo"]] += 1
    modos_rosca = [
        {"modo": modo, "n_imagenes": cnt} for modo, cnt in modos_counter.most_common(3)
    ]

    veredicto = "BUENA" if cv_medio <= UMBRAL_ROSCA else "POSIBLE DESGASTE"
    diagnostico2c = calcular_diagnostico_dos_capas(imagenes, cv_medio, px_por_mm_rosca)
    pxmm_str = f"{px_por_mm_rosca:.1f}" if px_por_mm_rosca else "N/A"
    log.info(
        "Rosca %s → %s | CV_medio=%.4f alertas=%d/%d px/mm=%s tiempo=%.0fms",
        ruta_rosca.name,
        veredicto,
        cv_medio,
        n_alertas,
        len(imagenes),
        pxmm_str,
        tiempo_total_inf_ms,
    )

    return {
        "veredicto": veredicto,
        "cv_medio": cv_medio,
        "cv_max": cv_max,
        "score_medio": float(np.mean(scores)),
        "n_alertas": n_alertas,
        "n_imagenes": len(imagenes),
        "imagenes": imagenes,
        "modos_desgaste": modos_rosca,
        "px_por_mm": px_por_mm_rosca,
        "diagnostico_2c": diagnostico2c,
        "tiempo_total_ms": tiempo_total_inf_ms,
        "tiempo_img_media_ms": round(float(np.mean(tiempos_ms)), 1),
        "tiempo_img_max_ms": round(float(np.max(tiempos_ms)), 1),
    }

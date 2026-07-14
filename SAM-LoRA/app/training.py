# app/training.py
import json
import time
from collections import Counter
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
import torch.nn as nn
from config import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_R,
    MODELS_DIR,
    NUM_CLASSES,
    NUM_EPOCHS,
    SAM_CHECKPOINT,
    SAM_MODEL_TYPE,
    get_ruta_anotacion,
    get_ruta_rosca,
)
from logger import get_logger
from model import SegHead, device, nombre_checkpoint
from peft import LoraConfig, get_peft_model
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

log = get_logger("training")


IMG_H, IMG_W = 1024, 1024
OUT_H, OUT_W = 256, 256
WEIGHT_DECAY = 0.01
CATEGORY_MAP = {0: 0, 1: 1}

_transform = ResizeLongestSide(1024)


def fusionar_cocos(roscas_train: list[str]) -> dict:
    """Fusiona los JSONs COCO de las roscas indicadas renumerando IDs."""
    coco_fusionado = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 0, "name": "Cresta"},
            {"id": 1, "name": "Paso"},
        ],
    }

    img_id_offset = 0
    ann_id_offset = 0

    for rosca_id in roscas_train:
        json_path = get_ruta_anotacion(rosca_id)
        if not json_path.exists():
            raise FileNotFoundError(f"Anotación no encontrada: {json_path}")

        with open(json_path) as f:
            coco = json.load(f)

        clase = "Buenas" if rosca_id.startswith("RB") else "Malas"
        ruta_rosca = get_ruta_rosca(rosca_id)
        for img in coco["images"]:
            nombre = Path(img["file_name"]).name
            if "-" in nombre and len(nombre.split("-")[0]) == 8:
                nombre = nombre.split("-", 1)[1]
            img["file_name"] = nombre
            img["rosca"] = rosca_id
            img["clase"] = clase

        n_total = len(coco["images"])
        imgs_ok = [
            img for img in coco["images"] if (ruta_rosca / img["file_name"]).exists()
        ]
        n_faltantes = n_total - len(imgs_ok)
        if n_faltantes:
            log.warning(
                "%s: %d imagen(es) no encontrada(s) en disco, se omiten",
                rosca_id,
                n_faltantes,
            )
        if not imgs_ok:
            log.warning("%s: ninguna imagen disponible, se omite esta rosca", rosca_id)
            continue

        ids_validos = {img["id"] for img in imgs_ok}
        anns_ok = [ann for ann in coco["annotations"] if ann["image_id"] in ids_validos]

        img_id_map = {}
        for img in imgs_ok:
            old_id = img["id"]
            new_id = old_id + img_id_offset
            img_id_map[old_id] = new_id
            img["id"] = new_id
            coco_fusionado["images"].append(img)

        for ann in anns_ok:
            ann["image_id"] = img_id_map[ann["image_id"]]
            ann["id"] = ann["id"] + ann_id_offset
            coco_fusionado["annotations"].append(ann)

        max_img_id = max(img_id_map.values())
        max_ann_id = max((ann["id"] for ann in anns_ok), default=0)
        img_id_offset = max_img_id + 1
        ann_id_offset = max_ann_id + 1

        conteo = Counter(a["category_id"] for a in anns_ok)
        log.info(
            "%s: %d/%d imgs válidas | Cresta=%d Paso=%d",
            rosca_id,
            len(imgs_ok),
            n_total,
            conteo.get(0, 0),
            conteo.get(1, 0),
        )

        coco_fusionado["categories"] = coco["categories"]

    log.info(
        "Total fusionado: %d imgs | %d anotaciones",
        len(coco_fusionado["images"]),
        len(coco_fusionado["annotations"]),
    )
    return coco_fusionado


class RoscaDataset(Dataset):
    """Dataset multi-rosca a partir de un COCO fusionado."""

    def __init__(self, coco_fusionado: dict):
        self.coco = coco_fusionado
        self.imagenes = coco_fusionado["images"]
        self.ann_by_img: dict[int, list] = {}
        for ann in coco_fusionado["annotations"]:
            self.ann_by_img.setdefault(ann["image_id"], []).append(ann)

        log.info(
            "Dataset: %d imágenes, %d anotaciones",
            len(self.imagenes),
            len(self.coco["annotations"]),
        )

    def __len__(self) -> int:
        return len(self.imagenes)

    def _cargar_imagen(self, img_info: dict) -> np.ndarray:
        ruta = get_ruta_rosca(img_info["rosca"]) / img_info["file_name"]
        img = cv2.imread(str(ruta), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Imagen no encontrada: {ruta}")
        img_resized = cv2.resize(img, (IMG_W, IMG_H))
        return cv2.cvtColor(cv2.bitwise_not(img_resized), cv2.COLOR_GRAY2RGB)

    def _construir_mascara(self, img_info: dict) -> np.ndarray:
        h_orig = img_info["height"]
        w_orig = img_info["width"]
        mask = np.zeros((NUM_CLASSES, h_orig, w_orig), dtype=np.uint8)

        for ann in self.ann_by_img.get(img_info["id"], []):
            cat_idx = CATEGORY_MAP.get(ann["category_id"])
            if cat_idx is None:
                continue
            seg = ann["segmentation"][0]
            pts = np.array(seg).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(mask[cat_idx], [pts], 1)

        return np.stack(
            [
                cv2.resize(mask[c], (OUT_W, OUT_H), interpolation=cv2.INTER_NEAREST)
                for c in range(NUM_CLASSES)
            ],
            axis=0,
        )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_info = self.imagenes[idx]
        img_rgb = self._cargar_imagen(img_info)
        img_tensor = torch.tensor(img_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
        mask_np = self._construir_mascara(img_info)
        mask_gt = torch.tensor(mask_np, dtype=torch.float32)
        return img_tensor, mask_gt


def _preprocess_batch(sam, imgs_tensor: torch.Tensor) -> torch.Tensor:
    imgs_np = (imgs_tensor.cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)
    processed = []
    for img_np in imgs_np:
        t = _transform.apply_image(img_np)
        t = torch.as_tensor(t, device=device).permute(2, 0, 1).contiguous()
        t = sam.preprocess(t.unsqueeze(0))
        processed.append(t)
    return torch.cat(processed, dim=0)


def _batch_iou(
    pred_logits: torch.Tensor, gt: torch.Tensor, threshold: float = 0.5
) -> torch.Tensor:
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    inter = (pred * gt).sum(dim=(0, 2, 3))
    union = (pred + gt).clamp(0, 1).sum(dim=(0, 2, 3))
    return inter / (union + 1e-6)


def _evaluar_en_rosca(
    sam, seg_head, rosca_val: str, batch_size: int = 1
) -> tuple[float, float]:
    """Calcula (iou_cresta, iou_paso) sobre la rosca de validación."""
    coco_val = fusionar_cocos([rosca_val])
    ds_val = RoscaDataset(coco_val)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=0)

    sam.image_encoder.eval()
    seg_head.eval()
    all_ious = []

    with torch.no_grad():
        for imgs, masks_gt in dl_val:
            masks_gt = masks_gt.to(device)
            inp = _preprocess_batch(sam, imgs)
            emb = sam.image_encoder(inp)
            logits = seg_head(emb)
            iou = _batch_iou(logits, masks_gt)
            all_ious.append(iou.cpu().numpy())

    sam.image_encoder.train()
    seg_head.train()

    mean_iou = np.mean(all_ious, axis=0)
    return float(mean_iou[0]), float(mean_iou[1])


def entrenar(
    roscas_train: list[str],
    epochs: int = NUM_EPOCHS,
    lr: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    callback: Callable | None = None,
    rosca_val: str | None = None,
    paciencia: int = EARLY_STOPPING_PATIENCE,
) -> dict:
    """
    Entrena SAM+LoRA+SegHead con las roscas indicadas.

    Parameters
    ----------
    rosca_val  : si se indica, activa early stopping usando esa rosca como
                 validación (debe tener anotación COCO disponible).
    paciencia  : épocas sin mejora en val IoU antes de detener.

    Returns
    -------
    dict con modelo_path, iou_cresta, iou_paso, loss_final, epochs, history
    """
    log.info(
        "Iniciando entrenamiento | roscas=%s epochs=%d lr=%g batch=%d",
        roscas_train,
        epochs,
        lr,
        batch_size,
    )
    if rosca_val:
        log.info(
            "Early stopping activo | rosca_val=%s paciencia=%d", rosca_val, paciencia
        )

    coco = fusionar_cocos(roscas_train)
    dataset = RoscaDataset(coco)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=str(SAM_CHECKPOINT))
    sam.to(device)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["qkv"],
        lora_dropout=0.0,
        bias="none",
    )
    sam.image_encoder = get_peft_model(sam.image_encoder, lora_config)

    for name, param in sam.named_parameters():
        param.requires_grad = "lora" in name.lower()

    seg_head = SegHead().to(device)

    n_lora = sum(p.numel() for p in sam.parameters() if p.requires_grad)
    n_head = sum(p.numel() for p in seg_head.parameters())
    log.info("Parámetros entrenables — LoRA: %d | SegHead: %d", n_lora, n_head)

    optimizer = AdamW(
        list(filter(lambda p: p.requires_grad, sam.image_encoder.parameters()))
        + list(seg_head.parameters()),
        lr=lr,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.BCEWithLogitsLoss()

    ckpt_name = nombre_checkpoint(roscas_train)
    ckpt_path = MODELS_DIR / ckpt_name
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    history = {
        "loss": [],
        "iou_cresta": [],
        "iou_paso": [],
        "val_iou_cresta": [],
        "val_iou_paso": [],
        "tiempo_epoch_s": [],
    }
    best_loss = float("inf")
    best_meta = {}
    best_val_iou = -1.0
    epochs_sin_mejora = 0

    sam.image_encoder.train()
    seg_head.train()

    t_train_inicio = time.time()

    for epoch in range(1, epochs + 1):
        t_epoch = time.time()
        epoch_losses, epoch_ious = [], []

        for imgs, mascaras_gt in dataloader:
            mascaras_gt = mascaras_gt.to(device)
            optimizer.zero_grad()
            input_batch = _preprocess_batch(sam, imgs)
            embeddings = sam.image_encoder(input_batch)
            pred_logits = seg_head(embeddings)
            loss = criterion(pred_logits, mascaras_gt)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                iou = _batch_iou(pred_logits, mascaras_gt)
            epoch_losses.append(loss.item())
            epoch_ious.append(iou.cpu().numpy())

        scheduler.step()

        mean_loss = float(np.mean(epoch_losses))
        mean_iou = np.mean(epoch_ious, axis=0)
        iou_f = float(mean_iou[0])
        iou_p = float(mean_iou[1])

        history["loss"].append(mean_loss)
        history["iou_cresta"].append(iou_f)
        history["iou_paso"].append(iou_p)
        history["tiempo_epoch_s"].append(round(time.time() - t_epoch, 2))

        val_iou_f, val_iou_p = 0.0, 0.0
        if rosca_val:
            val_iou_f, val_iou_p = _evaluar_en_rosca(
                sam, seg_head, rosca_val, batch_size
            )
            history["val_iou_cresta"].append(val_iou_f)
            history["val_iou_paso"].append(val_iou_p)
            val_score = (val_iou_f + val_iou_p) / 2.0
            if val_score > best_val_iou:
                best_val_iou = val_score
                epochs_sin_mejora = 0
            else:
                epochs_sin_mejora += 1

        if mean_loss < best_loss:
            best_loss = mean_loss
            best_meta = {
                "epoch": epoch,
                "loss": mean_loss,
                "iou_cresta": iou_f,
                "iou_paso": iou_p,
            }
            torch.save(
                {
                    "image_encoder": sam.image_encoder.state_dict(),
                    "seg_head": seg_head.state_dict(),
                    "epoch": epoch,
                    "loss": mean_loss,
                    "iou_cresta": iou_f,
                    "iou_paso": iou_p,
                    "roscas": sorted(roscas_train),
                },
                ckpt_path,
            )
            marker = " ✓"
        else:
            marker = ""

        val_str = f" | Val IoU C:{val_iou_f:.3f} P:{val_iou_p:.3f}" if rosca_val else ""
        log.info(
            "Época %3d/%d | Loss: %.4f | IoU C:%.3f P:%.3f%s%s",
            epoch,
            epochs,
            mean_loss,
            iou_f,
            iou_p,
            val_str,
            marker,
        )

        if callback:
            callback(epoch, epochs, mean_loss, iou_f, iou_p)

        if rosca_val and epochs_sin_mejora >= paciencia:
            log.info(
                "Early stopping en época %d (sin mejora en %d épocas)", epoch, paciencia
            )
            break

    tiempo_train_s = round(time.time() - t_train_inicio, 1)
    t_epoca_media = (
        round(float(np.mean(history["tiempo_epoch_s"])), 2)
        if history["tiempo_epoch_s"]
        else 0.0
    )
    log.info(
        "Entrenamiento completado | loss=%.4f IoU_C=%.3f IoU_P=%.3f → %s | %.0fs (%.1fs/época)",
        best_loss,
        best_meta["iou_cresta"],
        best_meta["iou_paso"],
        ckpt_path.name,
        tiempo_train_s,
        t_epoca_media,
    )

    history_path = ckpt_path.with_suffix(".json")
    try:
        with open(history_path, "w") as _hf:
            json.dump(
                {
                    **history,
                    "roscas": sorted(roscas_train),
                    "best_epoch": best_meta["epoch"],
                },
                _hf,
            )
        log.info("Historial guardado → %s", history_path.name)
    except Exception as _exc:
        log.warning("No se pudo guardar historial: %s", _exc)

    return {
        "modelo_path": ckpt_path,
        "iou_cresta": best_meta["iou_cresta"],
        "iou_paso": best_meta["iou_paso"],
        "loss_final": best_meta["loss"],
        "epochs": best_meta["epoch"],
        "history": history,
        "tiempo_train_s": tiempo_train_s,
        "tiempo_epoca_media_s": t_epoca_media,
    }

# app/model.py
from pathlib import Path

import torch
import torch.nn as nn
from config import LORA_ALPHA, LORA_R, NUM_CLASSES, SAM_CHECKPOINT, SAM_MODEL_TYPE
from peft import LoraConfig, get_peft_model
from segment_anything import sam_model_registry

device = "cuda" if torch.cuda.is_available() else "cpu"


class SegHead(nn.Module):
    """
    Cabeza de segmentación sobre los embeddings del image_encoder de SAM.
    Entrada:  (B, 256, 64, 64)  — salida del ViT-B
    Salida:   (B, NUM_CLASSES, 256, 256)
    """

    def __init__(self, in_channels: int = 256, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.up2(self.up1(x)))


def cargar_modelo(ckpt_path: str | Path) -> tuple:
    """
    Carga SAM ViT-B + LoRA + SegHead desde un checkpoint .pth.

    Returns
    -------
    sam   : SAM model con image_encoder LoRA, en eval mode
    head  : SegHead, en eval mode
    meta  : dict con epoch, loss y roscas del checkpoint
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint no encontrado: {ckpt_path}")

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

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sam.image_encoder.load_state_dict(ckpt["image_encoder"])

    head = SegHead().to(device)
    head.load_state_dict(ckpt["seg_head"])

    sam.image_encoder.eval()
    head.eval()

    meta = {
        "epoch": ckpt.get("epoch", None),
        "loss": ckpt.get("loss", None),
        "roscas": ckpt.get("roscas", None),
    }

    print(
        f"[model] Cargado: {ckpt_path.name} | "
        f"época {meta['epoch']} | "
        f"loss {meta['loss']:.4f} | "
        f"roscas: {meta['roscas']}"
    )

    return sam, head, meta


def nombre_checkpoint(roscas_train: list[str]) -> str:
    """
    Genera el nombre canónico del .pth para una combinación de roscas.
    Ej: ["RB01","RB03","RB02"] → "sam_lora_rb01_rb02_rb03.pth"
    """
    sufijo = "_".join(r.lower() for r in sorted(roscas_train))
    return f"sam_lora_{sufijo}.pth"

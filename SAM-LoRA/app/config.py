# app/config.py
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = BASE_DIR / "app"
IMAGES_DIR = BASE_DIR / "images" / "roscas_limpias"
MODELS_DIR = BASE_DIR / "models"
ANNOTATIONS_DIR = BASE_DIR / "annotations"
DB_PATH = APP_DIR / "experiments.db"

BUENAS_DIR = IMAGES_DIR / "Buenas"
MALAS_DIR = IMAGES_DIR / "Malas"

ROSCAS_BUENAS = [f"RB{i:02d}" for i in range(1, 12)]  # RB01..RB11
ROSCAS_MALAS = [f"RM{i:02d}" for i in range(1, 12)]  # RM01..RM11
TODAS_ROSCAS = ROSCAS_BUENAS + ROSCAS_MALAS


def get_ruta_rosca(rosca_id: str) -> Path:
    """Devuelve la ruta a la carpeta de imágenes de una rosca."""
    if rosca_id.startswith("RB"):
        return BUENAS_DIR / rosca_id
    elif rosca_id.startswith("RM"):
        return MALAS_DIR / rosca_id
    raise ValueError(f"ID de rosca no reconocido: {rosca_id}")


def get_ruta_anotacion(rosca_id: str) -> Path:
    """Devuelve la ruta al JSON COCO de una rosca."""
    return ANNOTATIONS_DIR / f"{rosca_id}_coco.json"


def get_roscas_anotadas() -> list[str]:
    """Devuelve las roscas buenas que tienen anotación COCO disponible."""
    return [r for r in ROSCAS_BUENAS if get_ruta_anotacion(r).exists()]


def es_buena(rosca_id: str) -> bool:
    return rosca_id.startswith("RB")


SAM_CHECKPOINT = MODELS_DIR / "sam_vit_b.pth"
SAM_MODEL_TYPE = "vit_b"

LORA_R = 8
LORA_ALPHA = 16
LORA_TARGET = "image_encoder.qkv"

IMG_SIZE = 256
NUM_CLASSES = 2
CANAL_CRESTA = 0
CANAL_PASO = 1

CV_UMBRAL = 0.025  # Umbral fijo utilizado para el diagnóstico final por rosca

LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
BATCH_SIZE = 1
EARLY_STOPPING_PATIENCE = 10

M10_AR_THEO = 3**0.5 / 2  # ≈ 0.866 — ratio alto/ancho del diente
M10_FLANK_DEG = 30.0  # ángulo del flanco respecto a la vertical
M10_SOLIDITY_THEO = 0.74  # solidez esperada del perfil truncado
M10_H_CRESTA_MM = 5 / 8 * (3**0.5 / 2 * 1.5)  # ≈ 0.812 mm — alto de cresta (5H/8)

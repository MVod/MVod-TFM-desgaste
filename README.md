# TFM-desgaste

# SAM-LoRA — Detección de desgaste en roscas métricas M10×1.5

Módulo del TFM *"Estrategia inteligente para la detección del desgaste en imágenes BLMD"* (Máster en Inteligencia Artificial, UNIR). Implementa un pipeline de segmentación y clasificación de desgaste en herramientas de rosca mediante fine-tuning eficiente de SAM con adaptadores LoRA y una cabeza de segmentación personalizada.

---

## Trabajo relacionado en el grupo TFM

Este módulo forma parte de un proyecto grupal donde cada miembro implementa una estrategia diferente sobre el mismo problema:


Daniel Alcalde Martin Calero — PatchCore (detección de anomalías sin supervisión)

Jordi Peiro Castello — Comparativa multi-modelo (U-Net, DeepLab, otros)

Miguel Gonzalez Sanchez — SAM + LoRA (este módulo)

---

## Contexto

Las herramientas de corte de rosca métrica M10×1.5 sufren desgaste progresivo que, si no se detecta a tiempo, compromete la calidad dimensional de la pieza y genera rechazos en línea. Este módulo aborda ese problema desde visión por computador: dada una imagen BLMD de la cresta de la rosca, el sistema segmenta la región de interés, extrae el coeficiente de variación (CV) de esa área y lo contrasta con los parámetros geométricos ISO para emitir un diagnóstico en dos niveles —herramienta operativa o desgastada.

El enfoque técnico se apoya en tres decisiones clave. Primero, usar SAM (ViT-B) como backbone pretrained para no partir de cero con un dataset pequeño. Segundo, congelar los pesos originales y entrenar únicamente los adaptadores LoRA (r=8, alpha=16) en las capas de atención del encoder, lo que reduce drásticamente los parámetros entrenables. Tercero, añadir una cabeza de segmentación propia (`SegHead`) entrenada sobre las máscaras anotadas en Label Studio en formato COCO.

---

## Arquitectura

```
Imagen BLMD
    │
    ▼
┌─────────────────────────────┐
│  SAM Image Encoder (ViT-B)  │  ← pesos congelados
│  + LoRA r=8 / alpha=16      │  ← únicos parámetros entrenados
└────────────┬────────────────┘
             │  embeddings
             ▼
┌─────────────────────────────┐
│         SegHead             │  ← cabeza personalizada (2 capas conv)
└────────────┬────────────────┘
             │  máscara binaria
             ▼
┌─────────────────────────────┐
│   Extracción CV cresta      │  ← discriminador de anomalía
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Diagnóstico ISO M10×1.5    │  ← comparación geométrica 2 niveles
└─────────────────────────────┘
```

---

## Dataset

Las imágenes provienen de capturas BLMD de herramientas de rosca etiquetadas con identificadores `RB01`–`RB09`. Las anotaciones se generaron en **Label Studio** y se exportaron en formato COCO estándar.

La imagen `Imagen_000512` fue excluida como outlier tras análisis exploratorio. El espécimen `RM08` presenta un comportamiento sistemático de falso negativo documentado en la memoria del TFM.

La validación se realizó mediante **Leave-One-Out cross-validation (LOO-CV)**, obteniendo un AUC-ROC medio de **~0.758**.

---

## Estructura del repositorio

```
SAM-LoRA/
├── data/
│   ├── raw/                   # Imágenes BLMD originales (RB01–RB09)
│   ├── annotations/           # Anotaciones COCO (Label Studio export)
│   └── cleaned/               # Imágenes tras pipeline de limpieza
│
├── src/
│   ├── preprocessing.py       # Pipeline limpiar_imagen() y normalización
│   ├── dataset.py             # Dataset PyTorch con carga COCO
│   ├── model.py               # SAMLoRA + SegHead (arquitectura completa)
│   ├── lora.py                # Implementación adaptadores LoRA sobre ViT-B
│   ├── train.py               # Bucle de entrenamiento y LOO-CV
│   ├── evaluate.py            # Métricas, AUC-ROC, Grad-CAM
│   └── diagnostics.py         # Sistema diagnóstico ISO dos niveles + CV
│
├── app/
│   ├── streamlit_app.py       # Interfaz Streamlit con tracking de experimentos
│   └── db/
│       └── experiments.db     # SQLite — historial de experimentos
│
├── checkpoints/               # Pesos entrenados (.pth)
├── notebooks/                 # Exploración, visualizaciones, ablaciones
├── requirements.txt
└── README.md
```

> **Nota:** los checkpoints y el dataset no se incluyen en el repositorio por su tamaño. Ver sección de instalación para obtenerlos.

---

## Instalación

El proyecto usa Python 3.12 y gestión de entornos con `uv`. Se recomienda GPU con soporte CUDA (probado en RTX 4090).

```bash
# Clonar el repositorio
git clone https://github.com/MVod/MVod-TFM-desgaste.git
cd MVod-TFM-desgaste/SAM-LoRA

# Crear entorno virtual e instalar dependencias
uv venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

uv pip install -r requirements.txt
```

Dependencias principales: `torch>=2.6` con CUDA, `segment-anything`, `opencv-python`, `streamlit`, `scikit-learn`, `matplotlib`.

Descargar el checkpoint base de SAM (ViT-B):

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
     -P checkpoints/
```

---

## Licencia

Uso académico. El código base de SAM pertenece a Meta AI y se distribuye bajo su licencia Apache 2.0 original.

"""Configuración central del proyecto (rutas, constantes y semilla).

Todas las rutas se pueden sobrescribir con variables de entorno, útil para
correr lo mismo en la laptop y en un Notebook de Kaggle con GPU.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Datos crudos de Kaggle: debe contener train.csv y train_images/
RAW_DIR = Path(os.environ.get("SEVERSTAL_RAW", ROOT / "data" / "raw"))
SPLIT_DIR = ROOT / "data" / "splits"
SPLIT_CSV = SPLIT_DIR / "split.csv"
# Dataset convertido a formato YOLO-seg (tiles)
YOLO_DIR = Path(os.environ.get("SEVERSTAL_YOLO", ROOT / "data" / "yolo"))
RUNS_DIR = Path(os.environ.get("SEVERSTAL_RUNS", ROOT / "runs"))
RESULTS_DIR = ROOT / "eval" / "results"
FIG_DIR = ROOT / "docs" / "figuras"

# Geometría de las imágenes de Severstal
IMG_H, IMG_W = 256, 1600
TILE_W = 320                      # 1600 / 320 = 5 tiles por imagen, sin solapamiento
N_TILES = IMG_W // TILE_W

NUM_CLASSES = 4
# ClassId 1..4 de Kaggle -> id YOLO 0..3
CLASS_NAMES = ["defecto_1", "defecto_2", "defecto_3", "defecto_4"]
# Colores BGR para visualizar cada clase
CLASS_COLORS = [(0, 200, 255), (255, 0, 255), (0, 0, 255), (0, 255, 0)]

SEED = 42

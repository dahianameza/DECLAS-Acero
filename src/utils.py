"""Utilidades compartidas: semillas, RLE de Kaggle y construcción de máscaras."""
import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd

from config import IMG_H, IMG_W, NUM_CLASSES


def set_seed(seed: int) -> None:
    """Fija semillas de Python, NumPy y PyTorch (si está instalado)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def rle_decode(rle, h: int = IMG_H, w: int = IMG_W) -> np.ndarray:
    """Decodifica el RLE de Severstal a una máscara (h, w) uint8.

    OJO: los píxeles están numerados en orden de COLUMNA (de arriba hacia abajo
    y luego de izquierda a derecha) y empiezan en 1. Por eso se usa order="F".
    """
    mask = np.zeros(h * w, dtype=np.uint8)
    if isinstance(rle, str) and rle.strip():
        s = np.asarray(rle.split(), dtype=np.int64)
        starts, lengths = s[0::2] - 1, s[1::2]
        for st, ln in zip(starts, lengths):
            mask[st:st + ln] = 1
    return mask.reshape((h, w), order="F")


def rle_encode(mask: np.ndarray) -> str:
    """Inversa de rle_decode (orden de columna, índices desde 1)."""
    pixels = np.concatenate([[0], mask.flatten(order="F").astype(np.uint8), [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[0::2]
    return " ".join(map(str, runs))


def load_annotations(csv_path) -> pd.DataFrame:
    """Lee train.csv en formato nuevo (ImageId, ClassId, EncodedPixels) o
    antiguo (ImageId_ClassId, EncodedPixels). Devuelve solo filas con defecto."""
    df = pd.read_csv(csv_path)
    if "ImageId_ClassId" in df.columns:
        parts = df["ImageId_ClassId"].str.rsplit("_", n=1, expand=True)
        df["ImageId"], df["ClassId"] = parts[0], parts[1].astype(int)
    df = df.dropna(subset=["EncodedPixels"])
    df = df[df["EncodedPixels"].astype(str).str.strip() != ""]
    return df[["ImageId", "ClassId", "EncodedPixels"]].reset_index(drop=True)


def group_annotations(df: pd.DataFrame) -> dict:
    """{ImageId: [(ClassId, rle), ...]}"""
    out = defaultdict(list)
    for img, cls, rle in df[["ImageId", "ClassId", "EncodedPixels"]].itertuples(index=False):
        out[img].append((int(cls), rle))
    return dict(out)


def build_masks(ann_by_image: dict, image_id: str) -> np.ndarray:
    """Máscaras (NUM_CLASSES, H, W) uint8 de una imagen (ceros si no tiene defectos)."""
    masks = np.zeros((NUM_CLASSES, IMG_H, IMG_W), dtype=np.uint8)
    for cls, rle in ann_by_image.get(image_id, []):
        masks[cls - 1] |= rle_decode(rle)
    return masks


def overlay_masks(gray: np.ndarray, masks: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Superpone máscaras por clase (C, H, W) sobre una imagen gris -> BGR."""
    import cv2
    from config import CLASS_COLORS
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if gray.ndim == 2 else gray.copy()
    out = bgr.copy()
    for c in range(masks.shape[0]):
        m = masks[c].astype(bool)
        if m.any():
            out[m] = (out[m] * (1 - alpha) + np.array(CLASS_COLORS[c]) * alpha).astype(np.uint8)
            contours, _ = cv2.findContours(masks[c].astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, CLASS_COLORS[c], 1)
    return out

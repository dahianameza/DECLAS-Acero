"""Convierte Severstal (RLE) a formato YOLO-seg con tiles de 320x256.

Por qué tiles: las imágenes miden 1600x256. Si YOLO las redimensiona a un
cuadrado de 320 o 640, los defectos finos se pierden. Cortando en 5 tiles de
320x256 se conserva la resolución nativa (1 píxel = 1 píxel) y el
entrenamiento en CPU es mucho más barato. Como el modelo aprende a escala
nativa, en inferencia se puede procesar la imagen completa (1600x256) de una vez.

Conversión de etiquetas por tile:
  máscara de clase -> componentes conexos (cv2.findContours) -> un polígono por
  instancia -> coordenadas normalizadas.  Línea YOLO-seg: "cls x1 y1 x2 y2 ..."

Negativos: en val y test se guardan TODOS los tiles (evaluación realista). En
train se conserva solo una fracción de los tiles sin defecto (--neg-keep) para
que el entrenamiento no esté dominado por fondo.

Uso:  python src/make_yolo_dataset.py [--neg-keep 0.25] [--min-area 30]
"""
import argparse
import shutil

import cv2
import numpy as np
import pandas as pd
import yaml

from config import (CLASS_NAMES, IMG_H, N_TILES, NUM_CLASSES, RAW_DIR, SEED,
                    SPLIT_CSV, TILE_W, YOLO_DIR)
from utils import build_masks, group_annotations, load_annotations, set_seed


def mask_to_polygons(mask: np.ndarray, min_area: int, eps: float = 1.0):
    """Polígonos (lista de arrays Nx2 en píxeles) de cada componente conexo."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
        if len(approx) >= 3:
            polys.append(approx)
    return polys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neg-keep", type=float, default=0.25,
                    help="fracción de tiles SIN defecto que se conservan en train")
    ap.add_argument("--min-area", type=int, default=30,
                    help="área mínima (px) de una instancia; menores se consideran ruido")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = ap.parse_args()
    set_seed(SEED)
    rng = np.random.default_rng(SEED)

    split_df = pd.read_csv(SPLIT_CSV)
    ann_by_image = group_annotations(load_annotations(RAW_DIR / "train.csv"))

    for split in args.splits:
        for sub in ("images", "labels"):
            d = YOLO_DIR / sub / split
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)

    records = []
    for split in args.splits:
        ids = split_df.loc[split_df.split == split, "ImageId"].tolist()
        print(f"[{split}] {len(ids)} imágenes -> tiles...")
        for k, image_id in enumerate(ids, 1):
            img = cv2.imread(str(RAW_DIR / "train_images" / image_id), cv2.IMREAD_GRAYSCALE)
            masks = build_masks(ann_by_image, image_id)
            stem = image_id.rsplit(".", 1)[0]
            for t in range(N_TILES):
                x0, x1 = t * TILE_W, (t + 1) * TILE_W
                lines, classes = [], []
                for c in range(NUM_CLASSES):
                    m = masks[c, :, x0:x1]
                    if not m.any():
                        continue
                    for poly in mask_to_polygons(m, args.min_area):
                        xy = np.clip(poly / [TILE_W, IMG_H], 0, 1)
                        lines.append(f"{c} " + " ".join(f"{x:.5f} {y:.5f}" for x, y in xy))
                        classes.append(c)
                if not lines and split == "train" and rng.random() > args.neg_keep:
                    continue
                name = f"{stem}_t{t}"
                cv2.imwrite(str(YOLO_DIR / "images" / split / f"{name}.jpg"),
                            img[:, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 95])
                (YOLO_DIR / "labels" / split / f"{name}.txt").write_text("\n".join(lines))
                records.append({"tile": name, "ImageId": image_id, "split": split, "tile_idx": t,
                                "n_instances": len(lines),
                                "classes": " ".join(str(c + 1) for c in sorted(set(classes)))})
            if k % 1000 == 0:
                print(f"  {k}/{len(ids)}")

    tiles = pd.DataFrame(records)
    tiles.to_csv(YOLO_DIR / "tiles.csv", index=False)

    cfg = {"path": str(YOLO_DIR.resolve()), "train": "images/train", "val": "images/val",
           "test": "images/test", "names": {i: n for i, n in enumerate(CLASS_NAMES)}}
    with open(YOLO_DIR / "dataset.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    print("\nTiles por partición (total / con defecto / instancias):")
    for split, g in tiles.groupby("split"):
        print(f"  {split:5s}: {len(g):6d} / {(g.n_instances > 0).sum():6d} / {g.n_instances.sum():6d}")
    print(f"dataset.yaml -> {YOLO_DIR / 'dataset.yaml'}")


if __name__ == "__main__":
    main()

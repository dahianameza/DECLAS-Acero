"""Particiones train / val / test SIN fuga de información.

Problema: Severstal no trae ID de bobina ni de lámina, y existen imágenes casi
duplicadas (recortes contiguos o repetidos de la misma lámina). Si una cae en
train y su gemela en test, la métrica de prueba se infla.

Estrategia:
  1. Hash perceptual (dHash de 256 bits) de cada imagen.
  2. Dos imágenes con distancia de Hamming <= umbral se consideran el mismo
     grupo (union-find), así que los grupos son componentes conexos.
  3. StratifiedGroupKFold (20 folds): todo un grupo cae en la misma partición y
     se estratifica por la clase MÁS RARA presente en la imagen (o "sin defecto"),
     para que la clase 2 (muy escasa) aparezca en las tres particiones.
  4. 3 folds -> test (15 %), 3 folds -> val (15 %), 14 folds -> train (70 %).

El conjunto de prueba se usa SOLO para el reporte final; umbrales e
hiperparámetros se eligen en validación.

Uso:  python src/prepare_split.py [--hash-thresh 12]
"""
import argparse

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from config import NUM_CLASSES, RAW_DIR, SEED, SPLIT_CSV, SPLIT_DIR
from utils import load_annotations, set_seed

POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def dhash(path, w: int = 33, h: int = 8) -> np.ndarray:
    """Difference hash de (w-1)*h = 256 bits, devuelto como 32 bytes."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    small = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA).astype(np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return np.packbits(bits)


def find_groups(hashes: np.ndarray, thresh: int, chunk: int = 128) -> np.ndarray:
    """Union-find sobre pares con Hamming <= thresh. Devuelve id de grupo por imagen."""
    n = len(hashes)
    parent = np.arange(n)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    n_pairs = 0
    for s in range(0, n, chunk):
        block = hashes[s:s + chunk]
        dist = POPCOUNT[np.bitwise_xor(block[:, None, :], hashes[None, :, :])].sum(-1, dtype=np.int32)
        ii, jj = np.nonzero(dist <= thresh)
        for i, j in zip(ii + s, jj):
            if j > i:
                n_pairs += 1
                ri, rj = root(i), root(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
    groups = np.array([root(i) for i in range(n)])
    print(f"Pares casi duplicados (Hamming <= {thresh}): {n_pairs}")
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hash-thresh", type=int, default=12,
                    help="distancia de Hamming máxima (de 256 bits) para considerar duplicado")
    ap.add_argument("--n-folds", type=int, default=20)
    ap.add_argument("--test-folds", type=int, default=3)
    ap.add_argument("--val-folds", type=int, default=3)
    args = ap.parse_args()
    set_seed(SEED)

    ann = load_annotations(RAW_DIR / "train.csv")
    images = sorted(p.name for p in (RAW_DIR / "train_images").glob("*.jpg"))
    if not images:
        raise SystemExit(f"No hay imágenes en {RAW_DIR / 'train_images'}. Corre data/download.py")
    print(f"Imágenes: {len(images)} | filas con defecto: {len(ann)}")

    df = pd.DataFrame({"ImageId": images})
    for c in range(1, NUM_CLASSES + 1):
        ids = set(ann.loc[ann.ClassId == c, "ImageId"])
        df[f"has_c{c}"] = df.ImageId.isin(ids).astype(int)
    has = df[[f"has_c{c}" for c in range(1, NUM_CLASSES + 1)]].to_numpy()
    df["n_classes"] = has.sum(1)

    # Etiqueta de estratificación: clase más rara presente (por frecuencia global)
    freq = has.sum(0)
    rare_order = np.argsort(freq)                      # índices de la más rara a la más común
    strat = np.zeros(len(df), dtype=int)               # 0 = sin defecto
    for c in rare_order[::-1]:                         # de común a rara: la rara sobrescribe
        strat[has[:, c] == 1] = c + 1
    df["strat"] = strat

    print("Calculando hashes perceptuales...")
    hashes = np.stack([dhash(RAW_DIR / "train_images" / f) for f in images])
    df["group"] = find_groups(hashes, args.hash_thresh)
    sizes = df.group.value_counts()
    print(f"Grupos: {len(sizes)} | grupos con >1 imagen: {(sizes > 1).sum()} | "
          f"grupo más grande: {sizes.max()} imágenes")
    if sizes.max() > 0.01 * len(df):
        print("AVISO: hay un grupo muy grande; considera bajar --hash-thresh y revisar.")

    sgkf = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True, random_state=SEED)
    fold = np.empty(len(df), dtype=int)
    for k, (_, idx) in enumerate(sgkf.split(np.zeros(len(df)), df.strat, df.group)):
        fold[idx] = k
    df["fold"] = fold
    t, v = args.test_folds, args.test_folds + args.val_folds
    df["split"] = np.where(fold < t, "test", np.where(fold < v, "val", "train"))

    # Verificación de no-fuga: ningún grupo en dos particiones
    leaks = df.groupby("group").split.nunique()
    assert (leaks == 1).all(), "Fuga: un grupo de duplicados quedó en dos particiones"

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["strat"]).to_csv(SPLIT_CSV, index=False)

    summary = df.groupby("split").agg(
        imagenes=("ImageId", "size"),
        con_defecto=("n_classes", lambda s: int((s > 0).sum())),
        **{f"clase_{c}": (f"has_c{c}", "sum") for c in range(1, NUM_CLASSES + 1)},
    ).loc[["train", "val", "test"]]
    summary["pct_con_defecto"] = (100 * summary.con_defecto / summary.imagenes).round(1)
    summary.to_csv(SPLIT_DIR / "split_summary.csv")
    print("\nResumen de particiones (imágenes por clase):")
    print(summary.to_string())
    print(f"\nGuardado en {SPLIT_CSV}")


if __name__ == "__main__":
    main()

"""Evaluación cuantitativa (indicadores D1, D3 y parte de C2).

Mide tres niveles:
  1. Tiles (val de Ultralytics): mAP@0.5 y mAP@0.5:0.95 de cajas y de máscaras.
  2. Imagen completa 1600x256, segmentación: Dice por clase (estilo Kaggle y
     solo en imágenes positivas) e IoU.
  3. Imagen completa, clasificación multietiqueta (¿la imagen tiene la clase k?):
     F1 macro, reporte por clase y matrices de confusión.
  + Latencia media por imagen en el dispositivo actual (FPS).
  + Lista de errores agrupados por tipo y visualizaciones (D3).

Protocolo sin fuga:
  python src/evaluate.py --weights <best.pt> --split val    # elige umbrales por clase
  python src/evaluate.py --weights <best.pt> --split test   # reporta con esos umbrales
"""
import argparse
import json
import time

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import (classification_report, confusion_matrix,  # noqa: E402
                             f1_score, multilabel_confusion_matrix)

from config import (CLASS_NAMES, FIG_DIR, IMG_H, IMG_W, NUM_CLASSES, RAW_DIR,  # noqa: E402
                    RESULTS_DIR, RUNS_DIR, SEED, SPLIT_CSV, TILE_W, YOLO_DIR)
from train_yolo import pick_device  # noqa: E402
from utils import build_masks, group_annotations, load_annotations, overlay_masks, set_seed  # noqa: E402

THR_GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)


def tile_level_metrics(model, split, device, name):
    m = model.val(data=str(YOLO_DIR / "dataset.yaml"), split=split, imgsz=TILE_W, batch=16,
                  device=device, project=str(RUNS_DIR / "val"), name=f"{name}_{split}",
                  exist_ok=True, plots=True, verbose=False)
    out = {"box_mAP50": float(m.box.map50), "box_mAP50_95": float(m.box.map),
           "mask_mAP50": float(m.seg.map50), "mask_mAP50_95": float(m.seg.map), "por_clase": {}}
    try:
        for i, c in enumerate(m.ap_class_index):
            out["por_clase"][CLASS_NAMES[int(c)]] = {
                "box_AP50": float(m.box.ap50[i]), "box_AP50_95": float(m.box.ap[i]),
                "mask_AP50": float(m.seg.ap50[i]), "mask_AP50_95": float(m.seg.ap[i])}
    except Exception as e:  # la API por clase cambia entre versiones
        print(f"(no se pudieron extraer métricas por clase: {e})")
    return out


def collect_predictions(model, image_ids, device, conf_floor):
    """Predice sobre imágenes completas. Guarda cada detección recortada a su caja
    para no tener en memoria máscaras de 1600x256 por detección."""
    preds, times = {}, []
    for k, image_id in enumerate(image_ids):
        img = cv2.imread(str(RAW_DIR / "train_images" / image_id))
        t0 = time.perf_counter()
        r = model.predict(img, imgsz=(IMG_H, IMG_W), conf=conf_floor, device=device,
                          retina_masks=True, verbose=False)[0]
        dt = time.perf_counter() - t0
        if k >= 5:                                   # descarta calentamiento
            times.append(dt)
        dets = []
        if r.masks is not None and len(r.boxes):
            cls = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy()
            md = r.masks.data.cpu().numpy() > 0.5
            for c, s, m in zip(cls, conf, md):
                ys, xs = np.nonzero(m)
                if len(ys) == 0:
                    continue
                y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                dets.append((int(c), float(s), y0, y1, x0, x1, m[y0:y1, x0:x1].copy()))
        preds[image_id] = dets
        if (k + 1) % 200 == 0:
            print(f"  {k + 1}/{len(image_ids)}")
    return preds, (np.mean(times) if times else float("nan"))


def image_scores(preds, image_ids):
    s = np.zeros((len(image_ids), NUM_CLASSES))
    for i, img in enumerate(image_ids):
        for c, conf, *_ in preds[img]:
            s[i, c] = max(s[i, c], conf)
    return s


def tune_thresholds(y_true, scores):
    thr = []
    for c in range(NUM_CLASSES):
        f1s = [f1_score(y_true[:, c], scores[:, c] >= t, zero_division=0) for t in THR_GRID]
        thr.append(float(THR_GRID[int(np.argmax(f1s))]))
    return thr


def predicted_masks(dets, thr):
    pm = np.zeros((NUM_CLASSES, IMG_H, IMG_W), dtype=bool)
    for c, s, y0, y1, x0, x1, crop in dets:
        if s >= thr[c]:
            pm[c, y0:y1, x0:x1] |= crop
    return pm


def plot_confusions(y_true, y_pred, path, title):
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    bin_cm = confusion_matrix(y_true.any(1), y_pred.any(1), labels=[False, True])
    fig, axes = plt.subplots(1, NUM_CLASSES + 1, figsize=(4 * (NUM_CLASSES + 1), 3.8))
    mats = list(mcm) + [bin_cm]
    titles = CLASS_NAMES + ["cualquier defecto"]
    for ax, cm, t in zip(axes, mats, titles):
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black", fontsize=12)
        ax.set_xticks([0, 1], ["no", "sí"])
        ax.set_yticks([0, 1], ["no", "sí"])
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        ax.set_title(t)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--name", default="e1_yolo26n_seg")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--conf", type=float, default=None,
                    help="umbral único para todas las clases (si no, se usan los de val)")
    ap.add_argument("--limit", type=int, default=None, help="evaluar solo N imágenes (prueba rápida)")
    ap.add_argument("--n-error-figs", type=int, default=12)
    ap.add_argument("--skip-tiles", action="store_true")
    args = ap.parse_args()
    set_seed(SEED)

    from ultralytics import YOLO
    device = pick_device(args.device)
    model = YOLO(args.weights)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.name}_{args.split}"
    results = {"experimento": args.name, "split": args.split, "pesos": args.weights,
               "dispositivo": device}

    # 1) Métricas de detección/segmentación a nivel tile
    if not args.skip_tiles:
        print("== Métricas por tile (Ultralytics val) ==")
        results["tiles"] = tile_level_metrics(model, args.split, device, args.name)
        print(json.dumps(results["tiles"], indent=2))

    # 2) y 3) Imagen completa
    split_df = pd.read_csv(SPLIT_CSV)
    image_ids = split_df.loc[split_df.split == args.split, "ImageId"].tolist()
    if args.limit:
        image_ids = image_ids[:args.limit]
    ann_by_image = group_annotations(load_annotations(RAW_DIR / "train.csv"))
    y_true = split_df.set_index("ImageId").loc[image_ids, [f"has_c{c}" for c in range(1, 5)]].to_numpy()

    print(f"== Predicción sobre {len(image_ids)} imágenes completas 1600x256 ==")
    preds, sec = collect_predictions(model, image_ids, device, conf_floor=0.05)
    results["latencia_ms_por_imagen"] = round(1000 * sec, 1)
    results["fps"] = round(1 / sec, 1) if sec > 0 else None

    scores = image_scores(preds, image_ids)
    thr_path = RESULTS_DIR / f"thresholds_{args.name}.json"
    if args.conf is not None:
        thr = [args.conf] * NUM_CLASSES
    elif args.split == "val":
        thr = tune_thresholds(y_true, scores)
        thr_path.write_text(json.dumps(dict(zip(CLASS_NAMES, thr)), indent=2))
        print(f"Umbrales elegidos en validación: {thr} -> {thr_path}")
    else:
        if not thr_path.exists():
            raise SystemExit("Primero corre --split val para elegir umbrales (no se ajustan en test).")
        thr = list(json.loads(thr_path.read_text()).values())
    results["umbrales"] = dict(zip(CLASS_NAMES, thr))

    y_pred = (scores >= np.array(thr)).astype(int)
    results["clasificacion"] = {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "reporte": classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                          output_dict=True, zero_division=0),
        "acc_binaria_defecto": float((y_true.any(1) == y_pred.any(1)).mean()),
    }
    plot_confusions(y_true, y_pred, FIG_DIR / f"{tag}_confusion.png",
                    f"{args.name} · {args.split} · F1 macro = {results['clasificacion']['f1_macro']:.3f}")

    # Segmentación a imagen completa + registro de errores
    dice_all = np.zeros((len(image_ids), NUM_CLASSES))
    inter = np.zeros(NUM_CLASSES)
    union = np.zeros(NUM_CLASSES)
    errors = []
    for i, img_id in enumerate(image_ids):
        gt = build_masks(ann_by_image, img_id).astype(bool)
        pm = predicted_masks(preds[img_id], thr)
        for c in range(NUM_CLASSES):
            a, b = gt[c].sum(), pm[c].sum()
            it = np.logical_and(gt[c], pm[c]).sum()
            dice_all[i, c] = 1.0 if a + b == 0 else 2 * it / (a + b)
            inter[c] += it
            union[c] += a + b - it
            if y_true[i, c] and not y_pred[i, c]:
                errors.append((img_id, "falso_negativo", CLASS_NAMES[c], scores[i, c], dice_all[i, c]))
            elif y_pred[i, c] and not y_true[i, c]:
                errors.append((img_id, "falso_positivo", CLASS_NAMES[c], scores[i, c], dice_all[i, c]))
            elif y_true[i, c] and y_pred[i, c] and dice_all[i, c] < 0.3:
                errors.append((img_id, "mala_localizacion", CLASS_NAMES[c], scores[i, c], dice_all[i, c]))

    seg = {"dice_kaggle": float(dice_all.mean()), "por_clase": {}}
    for c in range(NUM_CLASSES):
        pos = y_true[:, c] == 1
        seg["por_clase"][CLASS_NAMES[c]] = {
            "dice_kaggle": float(dice_all[:, c].mean()),
            "dice_en_positivas": float(dice_all[pos, c].mean()) if pos.any() else None,
            "iou_global": float(inter[c] / union[c]) if union[c] else None,
            "imagenes_positivas": int(pos.sum())}
    results["segmentacion"] = seg

    err_df = pd.DataFrame(errors, columns=["ImageId", "tipo", "clase", "score", "dice"])
    err_df.to_csv(RESULTS_DIR / f"{tag}_errores.csv", index=False)
    results["errores_por_tipo"] = err_df.groupby(["tipo", "clase"]).size().to_dict() if len(err_df) else {}
    results["errores_por_tipo"] = {f"{k[0]}|{k[1]}": int(v) for k, v in results["errores_por_tipo"].items()}

    # Figuras de errores: arriba la verdad, abajo la predicción
    err_dir = FIG_DIR / f"{tag}_errores"
    err_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    for tipo, g in err_df.groupby("tipo"):
        pick = g.iloc[rng.permutation(len(g))[:max(1, args.n_error_figs // 3)]]
        for row in pick.itertuples():
            gray = cv2.imread(str(RAW_DIR / "train_images" / row.ImageId), cv2.IMREAD_GRAYSCALE)
            top = overlay_masks(gray, build_masks(ann_by_image, row.ImageId))
            bot = overlay_masks(gray, predicted_masks(preds[row.ImageId], thr).astype(np.uint8))
            cv2.putText(top, f"REAL  {row.ImageId}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(bot, f"PRED  {tipo} {row.clase} score={row.score:.2f}", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imwrite(str(err_dir / f"{tipo}_{row.clase}_{row.ImageId}"), np.vstack([top, bot]))

    out = RESULTS_DIR / f"{tag}.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=float))
    print("\n== Resumen ==")
    print(f"F1 macro (imagen): {results['clasificacion']['f1_macro']:.3f}")
    print(f"Dice Kaggle: {seg['dice_kaggle']:.3f}")
    for c, v in seg["por_clase"].items():
        print(f"  {c}: dice_pos={v['dice_en_positivas']} iou={v['iou_global']}")
    print(f"Latencia: {results['latencia_ms_por_imagen']} ms/imagen ({results['fps']} FPS) en {device}")
    print(f"Resultados -> {out}")


if __name__ == "__main__":
    main()

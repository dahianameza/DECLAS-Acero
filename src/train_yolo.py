"""Experimento 1 (E1): fine-tuning de YOLO26n-seg sobre tiles 320x256.

Ejemplos:
  # Prueba rápida en CPU (25 % de train, 10 épocas) para validar el pipeline
  python src/train_yolo.py --fraction 0.25 --epochs 10 --name e1_smoke

  # Entrenamiento completo de E1
  python src/train_yolo.py

Control de sobreajuste (D2): early stopping (--patience), weight decay,
aumentos de datos y selección del mejor checkpoint por métrica de validación.
Ultralytics guarda las curvas en runs/segment/<name>/results.csv y results.png.
"""
import argparse
import json
import platform
import time

from config import RUNS_DIR, SEED, YOLO_DIR
from utils import set_seed


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26n-seg.pt", help="checkpoint preentrenado en COCO")
    ap.add_argument("--name", default="e1_yolo26n_seg")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=10, help="early stopping")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--fraction", type=float, default=1.0, help="fracción de train a usar")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="auto", help="auto | cpu | 0 | mps")
    args = ap.parse_args()

    set_seed(SEED)
    from ultralytics import YOLO
    import ultralytics

    device = pick_device(args.device)
    print(f"Dispositivo: {device} | ultralytics {ultralytics.__version__}")

    model = YOLO(args.model)
    t0 = time.time()
    model.train(
        data=str(YOLO_DIR / "dataset.yaml"),
        epochs=args.epochs, patience=args.patience, imgsz=args.imgsz, batch=args.batch,
        fraction=args.fraction, workers=args.workers, device=device,
        seed=SEED, deterministic=True,
        project=str(RUNS_DIR / "segment"), name=args.name, exist_ok=True,
        # Aumentos coherentes con acero en escala de grises
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.3,      # solo brillo
        fliplr=0.5, flipud=0.5,               # la lámina no tiene "arriba" ni "izquierda"
        degrees=0.0, translate=0.1, scale=0.25,
        mosaic=1.0, close_mosaic=5,
        weight_decay=5e-4,
        plots=True,
    )
    minutes = (time.time() - t0) / 60

    run_dir = RUNS_DIR / "segment" / args.name
    meta = {**vars(args), "device": device, "seed": SEED, "train_minutes": round(minutes, 1),
            "ultralytics": ultralytics.__version__, "python": platform.python_version(),
            "machine": platform.platform(), "processor": platform.processor()}
    (run_dir / "experiment.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\nEntrenamiento terminado en {minutes:.1f} min. Pesos: {run_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()

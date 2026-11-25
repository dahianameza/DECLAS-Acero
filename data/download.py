"""Descarga los datos de la competencia Severstal desde Kaggle.

Requisitos (una sola vez):
  1. Tener cuenta de Kaggle y ACEPTAR las reglas de la competencia en
     https://www.kaggle.com/competitions/severstal-steel-defect-detection/rules
  2. Crear un token en Kaggle > Settings > API > "Create New Token" y guardar
     kaggle.json en ~/.kaggle/ (Linux/Mac) o C:\\Users\\<usuario>\\.kaggle\\ (Windows).

Uso:
  python data/download.py            # descarga train.csv + train_images/
  python data/download.py --all      # también test_images/ (no trae etiquetas)

Los datos NO se suben al repositorio: las reglas de la competencia limitan su
redistribución, por eso cada usuario los descarga con su propia cuenta.
"""
import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import RAW_DIR  # noqa: E402

COMPETITION = "severstal-steel-defect-detection"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="extraer también test_images/")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if (RAW_DIR / "train.csv").exists() and (RAW_DIR / "train_images").exists():
        n = len(list((RAW_DIR / "train_images").glob("*.jpg")))
        print(f"Datos ya presentes en {RAW_DIR} ({n} imágenes). Nada que hacer.")
        return

    zip_path = RAW_DIR / f"{COMPETITION}.zip"
    if not zip_path.exists():
        kaggle = shutil.which("kaggle")
        if kaggle is None:
            sys.exit("No se encontró el comando 'kaggle'. Instala requirements.txt primero.")
        print("Descargando desde Kaggle (~1.6 GB)...")
        subprocess.run([kaggle, "competitions", "download", "-c", COMPETITION,
                        "-p", str(RAW_DIR)], check=True)

    print("Extrayendo...")
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist()
                   if args.all or m == "train.csv" or m.startswith("train_images/")]
        zf.extractall(RAW_DIR, members=members)
    n = len(list((RAW_DIR / "train_images").glob("*.jpg")))
    print(f"Listo: {n} imágenes de entrenamiento en {RAW_DIR}")
    zip_path.unlink()


if __name__ == "__main__":
    main()

# data/

Aquí **no** se versionan imágenes. Esta carpeta solo contiene:

| Ruta | Contenido | ¿En git? |
|---|---|---|
| `download.py` | Script de descarga desde Kaggle | Sí |
| `splits/split.csv` | Asignación de cada imagen a train / val / test (sin fuga) | Sí (es pequeño y garantiza reproducibilidad) |
| `raw/` | `train.csv` + `train_images/` de Kaggle | No |
| `yolo/` | Tiles 320×256 y etiquetas YOLO-seg generadas | No |

## Descarga

1. Acepta las reglas de la competencia: <https://www.kaggle.com/competitions/severstal-steel-defect-detection/rules>
2. Coloca tu `kaggle.json` en `~/.kaggle/` (Windows: `C:\Users\<usuario>\.kaggle\`).
3. `python data/download.py`

Si trabajas en un Notebook de Kaggle, no descargues nada: agrega la competencia
como input y define `SEVERSTAL_RAW=/kaggle/input/severstal-steel-defect-detection`.

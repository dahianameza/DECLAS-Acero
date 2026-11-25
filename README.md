# Detección y clasificación de defectos en láminas de acero

> Estado: **en desarrollo, Experimento 1 (YOLO26n-seg)**. Las secciones marcadas con `[POR COMPLETAR]` se llenan conforme avanzan los experimentos.

## 1. Problema y usuario

[POR COMPLETAR] Inspección visual automática de láminas de acero planas en línea de producción. Usuario: operador / ingeniero de calidad que necesita detectar, localizar y clasificar defectos superficiales en tiempo real para separar material defectuoso.

## 2. Integrantes

| Nombre completo | Código |
|---|---|
| [POR COMPLETAR] | [POR COMPLETAR] |

## 3. Conjunto de datos

- **Fuente:** [Severstal: Steel Defect Detection (Kaggle)](https://www.kaggle.com/competitions/severstal-steel-defect-detection/data). 12,568 imágenes etiquetadas de 1600×256 px en escala de grises, con 4 clases de defecto anotadas como máscaras RLE.
- **Licencia de uso:** reglas de la competencia de Kaggle. Los datos no se redistribuyen; se descargan con `data/download.py`.
- **Particiones:** 70/15/15 agrupando casi duplicados por hash perceptual y estratificando por clase rara (ver `src/prepare_split.py`). El conjunto de prueba solo se usa para el reporte final.
- Detalle completo: [`docs/datasheet.md`](docs/datasheet.md) [POR COMPLETAR]

## 4. Arquitectura

[POR COMPLETAR: diagrama]

- **Principal:** YOLO26-seg (segmentación de instancias: clase + caja + máscara + confianza en un solo modelo, sin NMS), con fine-tuning desde COCO sobre tiles de 320×256 a resolución nativa.
- **Alternativa comparada:** clasificador + U-Net (ResNet34), en dos etapas.
- Justificación y experimentos: [`docs/experimentos.md`](docs/experimentos.md)

## 5. Resultados principales

[POR COMPLETAR: tabla de métricas en test]

| Modelo | mask mAP@0.5 | mask mAP@0.5:0.95 | box mAP@0.5 | Dice | F1 macro | ms/imagen (CPU) |
|---|---|---|---|---|---|---|
| E1 YOLO26n-seg | | | | | | |

## 6. Instalación y reproducción

Hardware usado: [POR COMPLETAR: CPU, RAM, sistema operativo]

```bash
python -m venv .venv && .venv\Scripts\activate        # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
python data/download.py
python src/prepare_split.py && python src/make_yolo_dataset.py
python src/train_yolo.py && python src/evaluate.py --weights runs/segment/e1_yolo26n_seg/weights/best.pt --split val
```

[POR COMPLETAR: consolidar en ≤5 comandos con un script `reproduce`]

## 7. Demo en tiempo real

[POR COMPLETAR] `python app/demo.py`

## 8. Video de respaldo

[POR COMPLETAR]

## 9. Limitaciones

[POR COMPLETAR]

## 10. Uso de IA generativa y créditos

[POR COMPLETAR] Se usó Claude (Anthropic) como apoyo para estructurar el repositorio y escribir código base; el equipo revisó, ejecutó y validó todo el código. Créditos: Ultralytics YOLO (AGPL-3.0), Severstal / Kaggle por el conjunto de datos.

# Bitácora de experimentos

Todas las métricas finales se reportan sobre **test**. Los umbrales de confianza por clase se eligen en **val**.

| ID | Modelo | Entrada | Épocas (mejor) | mask mAP@0.5 | mask mAP@0.5:0.95 | box mAP@0.5 | Dice Kaggle | F1 macro | ms/img CPU | Tamaño (MB) |
|---|---|---|---|---|---|---|---|---|---|---|
| E1 | YOLO26n-seg | tiles 320×256 | | | | | | | | |
| E2 | YOLO26s-seg | tiles 320×256 | | | | | | | | |
| E3 | ResNet18 + U-Net ResNet34 | 1600×256 | | | | | | | | |
| E4 | Ganador exportado (ONNX FP16 / INT8) | | | | | | | | | |

## E1: YOLO26n-seg

- **Hipótesis:** un modelo de segmentación de instancias ligero alcanza un Dice competitivo y más de 10 FPS en CPU.
- **Configuración:** `src/train_yolo.py` (semilla 42; aumentos: flips H/V, brillo, mosaic; early stopping con patience 10).
- **Hardware:** [POR COMPLETAR]
- **Tiempo de entrenamiento:** [POR COMPLETAR] (ver `runs/segment/e1_yolo26n_seg/experiment.json`)
- **Curvas:** `runs/segment/e1_yolo26n_seg/results.png` → copiar a `docs/figuras/`
- **Resultados en val:** `eval/results/e1_yolo26n_seg_val.json`
- **Observaciones:** [POR COMPLETAR]

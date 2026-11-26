# Cassava AI

Sistema inteligente de clasificación de enfermedades de hojas de yuca mediante aprendizaje profundo, aprendizaje federado y ensembles heterogéneos.

## Descripción

El proyecto integra tres arquitecturas CNN y dos arquitecturas Transformer para clasificar cinco categorías de hojas de yuca.

Se comparan modelos centralizados y federados, incluyendo ensembles heterogéneos mediante Soft Voting.

## Arquitecturas

| Arquitectura | Familia |
|---|---|
| MobileNetV3-Large | CNN |
| EfficientNet-B0 | CNN |
| ResNet50 | CNN |
| ViT-B/16 | Transformer |
| Swin-Tiny | Transformer |

## Metodología

1. Preparación y partición del dataset.
2. Entrenamiento inicial y optimización.
3. Validación y entrenamiento federado.
4. Ensemble e integración.
5. Evaluación final y documentación.

## Instalación

Crear un entorno virtual e instalar las dependencias compatibles con el equipo.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

La instalación de PyTorch con CUDA debe ajustarse a la GPU y a las versiones utilizadas durante los experimentos.

## Ejecución

```powershell
python -m streamlit run app.py
```

Abrir `http://localhost:8501`.

La ejecución requiere los checkpoints entrenados y los registros de modelos incluidos en la estructura del proyecto.

## Estructura

- `dataset/`: dataset y características precalculadas.
- `notebooks/`: desarrollo experimental.
- `src/`: módulos reutilizables.
- `configs/`: configuraciones y registros de modelos.
- `models/`: checkpoints entrenados.
- `reports/`: métricas, predicciones y gráficos.
- `experiments/`: registros de reproducibilidad.
- `docs/`: documentación técnica y manuales.
- `app.py`: aplicación Streamlit.

## Resultados

Los resultados definitivos se encuentran en:

`reports/ensemble/ensemble_cached_v1/all_test_results.csv`

La métrica principal es F1 Macro.

## Documentación

- Documentación técnica: `docs/technical_documentation.md`.
- Manual de usuario: `docs/user_manual.md`.
- Model Card: `docs/model_card.md`.

## Limitaciones

El sistema constituye un prototipo académico. El aprendizaje federado es simulado y las probabilidades de clasificación no están calibradas.

No debe utilizarse como sustituto de una evaluación fitosanitaria profesional.
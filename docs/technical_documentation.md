
# Documentación técnica - Cassava AI

## 1. Descripción del proyecto

Sistema de clasificación de enfermedades de
hojas de yuca mediante aprendizaje profundo,
aprendizaje federado y ensembles heterogéneos.

## 2. Dataset

- Total de imágenes: 21397
- Desarrollo: 18187
- Test independiente: 3210
- Número de clases: 5
- Semilla experimental: 42

La partición se realizó de forma estratificada.

## 3. Preprocesamiento

- Conversión a RGB.
- Corrección de orientación EXIF.
- Redimensionamiento a 224 x 224.
- Normalización ImageNet.

En la versión optimizada mediante Feature
Caching se utilizaron características
precalculadas de extractores congelados.

## 4. Arquitecturas

- MobileNetV3-Large.
- EfficientNet-B0.
- ResNet50.
- Vision Transformer B/16.
- Swin Transformer Tiny.

## 5. Optimización de hiperparámetros

Se utilizó Optuna con Hyperband y un presupuesto
reducido de búsqueda para limitar el costo
computacional.

Los presupuestos efectivos y configuraciones
seleccionadas se conservan en los registros
experimentales.

## 6. Validación cruzada

Se utilizaron tres folds estratificados.

| architecture | folds | f1_macro_mean | f1_macro_std | f1_macro_min | f1_macro_max |
| --- | --- | --- | --- | --- | --- |
| efficientnet_b0 | 3 | 0.5869 | 0.0046 | 0.5824 | 0.5916 |
| mobilenet_v3_large | 3 | 0.5847 | 0.0028 | 0.5815 | 0.5864 |
| resnet50 | 3 | 0.5789 | 0.0074 | 0.5737 | 0.5874 |
| swin_t | 3 | 0.6212 | 0.0112 | 0.6084 | 0.6291 |
| vit_b_16 | 3 | 0.5816 | 0.0038 | 0.5783 | 0.5858 |

## 7. Aprendizaje federado

Se utilizó una simulación con un servidor
central y dos clientes con distribución
aproximadamente IID.

El algoritmo de agregación fue FedAvg.

### Distribución de clientes

| class | 1 | 2 | All |
| --- | --- | --- | --- |
| bacterial_blight | 462 | 462 | 924 |
| brown_streak_disease | 930 | 931 | 1861 |
| green_mottle | 1014 | 1014 | 2028 |
| healthy | 1095 | 1095 | 2190 |
| mosaic_disease | 5592 | 5592 | 11184 |
| All | 9093 | 9094 | 18187 |

## 8. Ensemble heterogéneo

Se implementaron dos ensembles independientes:

- Ensemble centralizado.
- Ensemble federado.

Cada ensemble combina cinco arquitecturas
mediante Soft Voting con pesos uniformes.

## 9. Evaluación final

La métrica principal es F1 Macro debido
al desbalance entre clases.

| model | approach | accuracy | precision_macro | recall_macro | f1_macro |
| --- | --- | --- | --- | --- | --- |
| heterogeneous_ensemble | centralized | 0.7701 | 0.6273 | 0.6983 | 0.6549 |
| heterogeneous_ensemble | federated | 0.7539 | 0.6097 | 0.6799 | 0.6383 |
| swin_t | centralized | 0.7312 | 0.5922 | 0.6837 | 0.6216 |
| swin_t | federated | 0.7361 | 0.5919 | 0.6563 | 0.6183 |
| vit_b_16 | federated | 0.7212 | 0.5767 | 0.6433 | 0.6022 |
| vit_b_16 | centralized | 0.7109 | 0.5660 | 0.6539 | 0.5956 |
| resnet50 | centralized | 0.7199 | 0.5721 | 0.6272 | 0.5939 |
| resnet50 | federated | 0.7171 | 0.5664 | 0.6269 | 0.5898 |
| efficientnet_b0 | federated | 0.7062 | 0.5603 | 0.6274 | 0.5868 |
| mobilenet_v3_large | centralized | 0.7137 | 0.5578 | 0.6298 | 0.5806 |
| efficientnet_b0 | centralized | 0.7037 | 0.5526 | 0.6256 | 0.5771 |
| mobilenet_v3_large | federated | 0.6629 | 0.5230 | 0.6210 | 0.5492 |

## 10. Comparación de ensembles

| approach | best_individual_model | best_individual_f1 | mean_individual_f1 | ensemble_f1 | ensemble_accuracy | ensemble_gain_pp |
| --- | --- | --- | --- | --- | --- | --- |
| centralized | swin_t | 0.6216 | 0.5937 | 0.6549 | 0.7701 | 3.3213 |
| federated | swin_t | 0.6183 | 0.5893 | 0.6383 | 0.7539 | 1.9966 |

## 11. Optimización computacional

Se reutilizaron características de extractores
congelados para reducir operaciones repetidas.

La evaluación compartida centralizada y federada
evitó duplicar la extracción de características
cuando los backbones eran idénticos.

## 12. Producto

La aplicación utiliza Streamlit como interfaz
y CassavaInferenceEngine como motor de inferencia.

Permite seleccionar modelos individuales,
ensembles y enfoques de aprendizaje.

## 13. Limitaciones

- Dataset desbalanceado.
- Aprendizaje federado simulado.
- Distribución aproximadamente IID.
- Probabilidades no calibradas.
- Presupuesto reducido de hiperparámetros.
- No se implementó detección fuera del dominio.
- No se realizó validación externa en campo.
- La validación cruzada no fue anidada.
- Feature Caching limita el ajuste del backbone.

## 14. Reproducibilidad

Se conservaron configuraciones JSON, particiones
CSV, checkpoints, semillas, historiales y
registros experimentales.

Fecha de generación:
2026-11-26T04:00:42.470446

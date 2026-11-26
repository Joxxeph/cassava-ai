# Manual de usuario — Cassava AI

## 1. Introducción

Cassava AI es una aplicación experimental que permite clasificar imágenes de hojas de yuca utilizando modelos de inteligencia artificial.

## 2. Requisitos

- Python y las dependencias del proyecto.
- Modelos entrenados y archivos de configuración.
- Navegador web.
- GPU compatible con CUDA, opcional para acelerar la inferencia.

## 3. Inicio de la aplicación

Abrir una terminal en la carpeta principal del proyecto, activar el entorno virtual y ejecutar:

```powershell
python -m streamlit run app.py
```

Acceder desde el navegador a `http://localhost:8501`.

## 4. Clasificación de una imagen

1. Seleccionar la sección Clasificación.
2. Elegir el enfoque centralizado, federado o ambos.
3. Seleccionar una arquitectura individual o el ensemble heterogéneo.
4. Cargar una fotografía en formato JPG, JPEG o PNG.
5. Presionar el botón Clasificar imagen.
6. Esperar a que termine la inferencia.
7. Consultar la clase predicha y las probabilidades.
8. Descargar el resultado JSON si se desea conservar la predicción.

## 5. Interpretación de resultados

La aplicación muestra una clase predicha y cinco probabilidades correspondientes a las categorías conocidas.

La probabilidad máxima representa la puntuación asignada por el modelo, pero no garantiza que la predicción sea correcta.

Cuando se selecciona el enfoque Ambos, se presentan los resultados centralizados y federados para facilitar su comparación.

## 6. Resultados experimentales

La sección Resultados experimentales permite consultar las métricas obtenidas sobre el conjunto de test independiente y las matrices de confusión de los ensembles.

## 7. Recomendaciones

Utilizar fotografías claras, bien iluminadas y centradas en una hoja de yuca. Evitar imágenes borrosas, fotografías de otros cultivos o imágenes donde la hoja no sea visible.

## 8. Advertencia

Cassava AI es un prototipo académico de apoyo a la clasificación. Sus resultados no sustituyen la evaluación de un especialista en sanidad vegetal.

## 9. Cierre de la aplicación

Cerrar la pestaña del navegador y presionar Ctrl + C en la terminal donde se está ejecutando Streamlit.
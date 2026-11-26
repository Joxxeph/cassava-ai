
import gc
import hashlib
import io
import json
import threading
from pathlib import Path

import pandas as pd
import streamlit as st
import torch

from PIL import Image, ImageOps
from src.inference import CassavaInferenceEngine


# ==========================================================
# 1. CONFIGURACIÓN GENERAL
# ==========================================================

PROJECT_DIR = Path(__file__).resolve().parent

REPORTS_DIR = (
    PROJECT_DIR
    / "reports"
    / "ensemble"
    / "ensemble_cached_v1"
)

MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000

st.set_page_config(
    page_title="Cassava AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================================
# 2. NOMBRES DE MODELOS Y CLASES
# ==========================================================

MODEL_NAMES = {
    "ensemble": "Ensemble heterogéneo",
    "mobilenet_v3_large": "MobileNetV3-Large",
    "efficientnet_b0": "EfficientNet-B0",
    "resnet50": "ResNet50",
    "vit_b_16": "ViT-B/16",
    "swin_t": "Swin-Tiny",
}

APPROACH_NAMES = {
    "centralized": "Centralizado",
    "federated": "Federado",
    "both": "Ambos enfoques",
}

CLASS_NAMES = {
    "bacterial_blight": "Tizón bacteriano",
    "brown_streak_disease": "Enfermedad de estría marrón",
    "green_mottle": "Moteado verde",
    "healthy": "Hoja sana",
    "mosaic_disease": "Enfermedad del mosaico",
}


# ==========================================================
# 3. MOTOR DE INFERENCIA COMPARTIDO
# ==========================================================

@st.cache_resource(show_spinner=False)
def get_engine():

    torch.set_num_threads(1)

    engine = CassavaInferenceEngine(
        project_dir=PROJECT_DIR,
        device=torch.device("cpu"),
    )

    lock = threading.Lock()

    return engine, lock


# ==========================================================
# 4. CARGA DE RESULTADOS EXPERIMENTALES
# ==========================================================

@st.cache_data
def load_experimental_results():

    results_path = REPORTS_DIR / "all_test_results.csv"

    if not results_path.exists():
        return None

    return pd.read_csv(results_path)


# ==========================================================
# 5. FUNCIONES AUXILIARES
# ==========================================================

def translate_class(class_name):

    return CLASS_NAMES.get(class_name, class_name)


def translate_model(model_name):

    if model_name == "heterogeneous_ensemble":
        return "Ensemble heterogéneo"

    return MODEL_NAMES.get(model_name, model_name)


def load_uploaded_image(uploaded_file):

    image_bytes = uploaded_file.getvalue()

    if len(image_bytes) > MAX_FILE_SIZE:
        raise ValueError(
            "La imagen supera el límite de 10 MB."
        )

    with Image.open(io.BytesIO(image_bytes)) as source:

        if source.format not in {"JPEG", "PNG"}:
            raise ValueError(
                "El archivo debe ser una imagen JPEG o PNG."
            )

        width, height = source.size

        if width * height > MAX_IMAGE_PIXELS:
            raise ValueError(
                "La resolución de la imagen es demasiado alta."
            )

        image = ImageOps.exif_transpose(source)
        image = image.convert("RGB")

    return image, image_bytes


def build_prediction_key(
    image_bytes,
    approach,
    model_name,
):

    image_hash = hashlib.sha256(
        image_bytes
    ).hexdigest()

    return (
        f"{image_hash}_{approach}_{model_name}"
    )


# ==========================================================
# 6. VISUALIZACIÓN DE PREDICCIONES
# ==========================================================

def display_prediction(prediction, approach):

    st.subheader(
        APPROACH_NAMES.get(approach, approach)
    )

    predicted_class = prediction["predicted_class"]

    confidence = float(
        prediction["confidence"]
    )

    probabilities = prediction["probabilities"]

    st.metric(
        label="Clase predicha",
        value=translate_class(predicted_class),
    )

    st.metric(
        label="Probabilidad máxima",
        value=f"{confidence * 100:.2f}%",
    )

    st.markdown("#### Probabilidades por clase")

    sorted_probabilities = sorted(
        probabilities.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    for class_name, probability in sorted_probabilities:

        probability = float(probability)

        st.write(
            f"**{translate_class(class_name)}**"
            f" — {probability * 100:.2f}%"
        )

        st.progress(
            max(0.0, min(1.0, probability))
        )


def display_individual_predictions(result):

    individual = result.get(
        "individual_predictions",
        {},
    )

    if not individual:
        return

    with st.expander(
        "Consultar predicciones individuales"
    ):

        st.json(individual, expanded=False)


# ==========================================================
# 7. PÁGINA DE CLASIFICACIÓN
# ==========================================================

def classification_page():

    st.title("🌿 Cassava AI")

    st.markdown(
        """
        Sistema inteligente para la clasificación
        de enfermedades de hojas de yuca mediante
        aprendizaje profundo, aprendizaje federado
        y ensembles heterogéneos.
        """
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        approach = st.selectbox(
            "Enfoque de aprendizaje",
            options=list(APPROACH_NAMES.keys()),
            format_func=lambda x: APPROACH_NAMES[x],
        )

    with col2:

        model_name = st.selectbox(
            "Modelo de clasificación",
            options=list(MODEL_NAMES.keys()),
            format_func=lambda x: MODEL_NAMES[x],
        )

    uploaded_file = st.file_uploader(
        "Cargar una imagen de una hoja de yuca",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )

    if uploaded_file is None:

        st.info(
            "Selecciona una imagen para iniciar "
            "la clasificación."
        )

        return

    try:

        image, image_bytes = load_uploaded_image(
            uploaded_file
        )

    except Exception as error:

        st.error(
            f"No fue posible abrir la imagen: {error}"
        )

        return

    preview = image.copy()
    preview.thumbnail((600, 600))

    st.image(
        preview,
        caption="Imagen seleccionada",
    )

    prediction_key = build_prediction_key(
        image_bytes,
        approach,
        model_name,
    )

    if model_name == "ensemble":

        st.info(
            "El ensemble ejecuta cinco arquitecturas "
            "de forma secuencial. La primera predicción "
            "puede tardar mientras se descargan los "
            "pesos preentrenados."
        )

    if st.button(
        "Clasificar imagen",
        type="primary",
        use_container_width=True,
    ):

        st.session_state.pop(
            "prediction_result",
            None,
        )

        st.session_state.pop(
            "prediction_key",
            None,
        )

        try:

            with st.spinner(
                "Procesando imagen y ejecutando modelos..."
            ):

                engine, lock = get_engine()

                with lock:

                    try:

                        result = engine.predict(
                            image=image,
                            approach=approach,
                            model_name=model_name,
                        )

                    finally:

                        # Liberar el modelo activo después
                        # de cada solicitud para reducir RAM.
                        engine.unload()
                        gc.collect()

                st.session_state[
                    "prediction_result"
                ] = result

                st.session_state[
                    "prediction_key"
                ] = prediction_key

        except Exception as error:

            st.error(
                "No fue posible completar la clasificación."
            )

            with st.expander("Detalles técnicos"):

                st.exception(error)

    # Evita mostrar resultados correspondientes
    # a una imagen o selección anterior.

    if (
        st.session_state.get("prediction_key")
        != prediction_key
    ):

        return

    result = st.session_state.get(
        "prediction_result"
    )

    if result is None:
        return

    st.divider()
    st.header("Resultados de clasificación")

    predictions = result["results"]

    if approach == "both":

        central_col, federal_col = st.columns(2)

        with central_col:

            display_prediction(
                predictions["centralized"],
                "centralized",
            )

        with federal_col:

            display_prediction(
                predictions["federated"],
                "federated",
            )

    else:

        display_prediction(
            predictions[approach],
            approach,
        )

    inference_seconds = result.get(
        "inference_seconds"
    )

    if inference_seconds is not None:

        st.caption(
            "Tiempo de inferencia reportado por el motor: "
            f"{float(inference_seconds):.2f} segundos"
        )

    display_individual_predictions(result)

    st.download_button(
        label="Descargar resultados JSON",
        data=json.dumps(
            result,
            ensure_ascii=False,
            indent=4,
            default=str,
        ),
        file_name="cassava_prediction.json",
        mime="application/json",
    )

    st.warning(
        "Este sistema es un prototipo académico. "
        "Las probabilidades no representan una certeza "
        "diagnóstica y no sustituyen la evaluación "
        "de un especialista agrícola."
    )


# ==========================================================
# 8. PÁGINA DE RESULTADOS EXPERIMENTALES
# ==========================================================

def experimental_results_page():

    st.title("📊 Resultados experimentales")

    st.markdown(
        """
        Comparación del rendimiento de los modelos
        centralizados, federados y ensembles sobre
        el conjunto independiente de evaluación.
        """
    )

    df = load_experimental_results()

    if df is None:

        st.warning(
            "El archivo de resultados experimentales "
            "no está disponible en el repositorio."
        )

        st.code(
            str(
                REPORTS_DIR
                / "all_test_results.csv"
            )
        )

        return

    required_columns = {
        "model",
        "approach",
        "f1_macro",
    }

    if not required_columns.issubset(df.columns):

        st.error(
            "El archivo de resultados no tiene "
            "las columnas esperadas."
        )

        st.dataframe(df)

        return

    ensemble_df = df[
        df["model"].isin(
            ["ensemble", "heterogeneous_ensemble"]
        )
    ]

    col1, col2 = st.columns(2)

    with col1:

        central = ensemble_df[
            ensemble_df["approach"] == "centralized"
        ]

        if not central.empty:

            st.metric(
                "F1 Macro — Ensemble centralizado",
                f"{central.iloc[0]['f1_macro']:.4f}",
            )

    with col2:

        federated = ensemble_df[
            ensemble_df["approach"] == "federated"
        ]

        if not federated.empty:

            st.metric(
                "F1 Macro — Ensemble federado",
                f"{federated.iloc[0]['f1_macro']:.4f}",
            )

    metric_names = {
        "f1_macro": "F1 Macro",
        "accuracy": "Accuracy",
        "precision_macro": "Precisión Macro",
        "recall_macro": "Recall Macro",
        "f1_weighted": "F1 ponderado",
        "balanced_accuracy": "Balanced Accuracy",
    }

    available_metrics = [
        metric
        for metric in metric_names
        if metric in df.columns
    ]

    selected_metric = st.selectbox(
        "Métrica de comparación",
        options=available_metrics,
        format_func=lambda x: metric_names[x],
    )

    chart_df = df.pivot_table(
        index="model",
        columns="approach",
        values=selected_metric,
        aggfunc="first",
    )

    chart_df.index = [
        translate_model(name)
        for name in chart_df.index
    ]

    chart_df = chart_df.rename(
        columns=APPROACH_NAMES
    )

    st.bar_chart(chart_df)

    st.subheader("Tabla de resultados")

    display_df = df.copy()

    display_df["model"] = (
        display_df["model"]
        .map(translate_model)
    )

    display_df["approach"] = (
        display_df["approach"]
        .map(
            lambda x: APPROACH_NAMES.get(x, x)
        )
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Matrices de confusión")

    confusion_images = sorted(
        REPORTS_DIR.rglob("*confusion*.png")
    )

    if confusion_images:

        for image_path in confusion_images:

            st.image(
                str(image_path),
                caption=image_path.stem,
            )

    else:

        st.caption(
            "No se encontraron matrices de confusión "
            "en los archivos publicados."
        )

    st.download_button(
        "Descargar resultados experimentales",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="cassava_test_results.csv",
        mime="text/csv",
    )


# ==========================================================
# 9. PÁGINA DE INFORMACIÓN
# ==========================================================

def about_page():

    st.title("Acerca de Cassava AI")

    st.markdown(
        """
        ## Objetivo

        Desarrollar un sistema de clasificación de
        enfermedades de hojas de yuca y comparar
        diferentes estrategias de aprendizaje profundo.

        ## Dataset

        - 21.397 imágenes.
        - Cinco categorías.
        - 85 % para desarrollo.
        - 15 % para evaluación independiente.
        - Validación cruzada estratificada de tres folds.

        ## Arquitecturas

        **Redes neuronales convolucionales:**

        - MobileNetV3-Large.
        - EfficientNet-B0.
        - ResNet50.

        **Transformers:**

        - ViT-B/16.
        - Swin-Tiny.

        ## Metodología

        - Transfer Learning.
        - Optimización mediante Hyperband.
        - Feature Caching.
        - Aprendizaje centralizado.
        - Aprendizaje federado mediante FedAvg.
        - Ensemble heterogéneo mediante Soft Voting.

        ## Limitaciones

        - Aprendizaje federado simulado.
        - Inferencia en CPU en la versión web.
        - Recursos computacionales limitados.
        - No incluye detección de imágenes ajenas
          a las categorías del dataset.
        - No sustituye el diagnóstico profesional.
        """
    )


# ==========================================================
# 10. MENÚ LATERAL
# ==========================================================

def main():

    with st.sidebar:

        st.title("🌿 Cassava AI")

        st.caption(
            "Inteligencia artificial aplicada "
            "al sector agrícola"
        )

        st.divider()

        page = st.radio(
            "Navegación",
            options=[
                "Clasificación",
                "Resultados experimentales",
                "Acerca del proyecto",
            ],
        )

        st.divider()

        st.caption("Dispositivo de inferencia: CPU")

        st.caption("Versión web del proyecto académico")

    if page == "Clasificación":

        classification_page()

    elif page == "Resultados experimentales":

        experimental_results_page()

    else:

        about_page()


if __name__ == "__main__":

    main()

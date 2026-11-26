import os

os.environ.setdefault(
    "CUBLAS_WORKSPACE_CONFIG",
    ":4096:8"
)

from pathlib import Path
from io import BytesIO

import json
import hashlib

import numpy as np
import pandas as pd
import streamlit as st
import torch

from PIL import Image, ImageOps

from src.inference import CassavaInferenceEngine


PROJECT_DIR = Path(__file__).resolve().parent

REPORTS_DIR = (
    PROJECT_DIR
    / "reports"
    / "ensemble"
    / "ensemble_cached_v1"
)

MODEL_NAMES = {

    "ensemble": "Ensemble heterogéneo",

    "mobilenet_v3_large": "MobileNetV3-Large",

    "efficientnet_b0": "EfficientNet-B0",

    "resnet50": "ResNet50",

    "vit_b_16": "ViT-B/16",

    "swin_t": "Swin-Tiny"

}

APPROACH_NAMES = {

    "centralized": "Centralizado",

    "federated": "Federado",

    "both": "Ambos"

}

CLASS_NAMES = {

    "bacterial_blight":
        "Tizón bacteriano",

    "brown_streak_disease":
        "Enfermedad de la estría marrón",

    "green_mottle":
        "Moteado verde",

    "healthy":
        "Hoja sana",

    "mosaic_disease":
        "Enfermedad del mosaico"

}


st.set_page_config(

    page_title="Cassava AI",

    page_icon="🌿",

    layout="wide"

)


def get_engine():

    if "inference_engine" not in st.session_state:

        device = torch.device(

            "cuda"

            if torch.cuda.is_available()

            else "cpu"

        )

        st.session_state["inference_engine"] = (

            CassavaInferenceEngine(

                project_dir=PROJECT_DIR,

                device=device

            )

        )

    return st.session_state["inference_engine"]


@st.cache_data
def load_experimental_results():

    return pd.read_csv(

        REPORTS_DIR / "all_test_results.csv"

    )


def display_prediction(approach, result):

    st.subheader(
        APPROACH_NAMES[approach]
    )

    predicted_class = result["predicted_class"]

    st.metric(

        "Clase predicha",

        CLASS_NAMES.get(
            predicted_class,
            predicted_class
        )

    )

    st.metric(

        "Probabilidad máxima",

        f"{result['confidence']:.2%}"

    )

    probabilities = sorted(

        result["probabilities"].items(),

        key=lambda item: item[1],

        reverse=True

    )

    for class_name, probability in probabilities:

        label = CLASS_NAMES.get(

            class_name,

            class_name

        )

        st.write(
            f"{label}: {probability:.2%}"
        )

        st.progress(
            float(np.clip(probability, 0, 1))
        )

    with st.expander(
        "Predicciones individuales"
    ):

        rows = []

        for architecture, prediction in (

            result["individual_predictions"].items()

        ):

            rows.append({

                "Modelo": MODEL_NAMES.get(

                    architecture,

                    architecture

                ),

                "Predicción": CLASS_NAMES.get(

                    prediction["predicted_class"],

                    prediction["predicted_class"]

                ),

                "Probabilidad máxima": round(

                    prediction["confidence"] * 100,

                    2

                )

            })

        st.dataframe(

            pd.DataFrame(rows),

            hide_index=True,

            use_container_width=True

        )


def classification_page():

    st.header(
        "Clasificación de hojas de yuca"
    )

    st.write(

        "Carga una fotografía de una hoja de yuca "
        "para obtener una clasificación mediante "
        "modelos de inteligencia artificial."

    )

    approach = st.selectbox(

        "Enfoque de aprendizaje",

        options=list(APPROACH_NAMES),

        format_func=lambda x: APPROACH_NAMES[x]

    )

    model_name = st.selectbox(

        "Modelo de clasificación",

        options=list(MODEL_NAMES),

        format_func=lambda x: MODEL_NAMES[x]

    )

    uploaded_file = st.file_uploader(

        "Seleccionar imagen",

        type=["jpg", "jpeg", "png"]

    )

    if uploaded_file is None:

        st.info(
            "Selecciona una imagen para comenzar."
        )

        return

    image_bytes = uploaded_file.getvalue()

    if len(image_bytes) > 10 * 1024**2:

        st.error(
            "La imagen supera el límite de 10 MB."
        )

        return

    try:

        with Image.open(
            BytesIO(image_bytes)
        ) as original:

            if (
                original.width * original.height
                > 25_000_000
            ):

                st.error(
                    "La resolución de la imagen "
                    "es demasiado grande."
                )

                return

            image = ImageOps.exif_transpose(

                original

            ).convert("RGB")

    except Exception:

        st.error(
            "No fue posible abrir la imagen."
        )

        return

    image_col, info_col = st.columns([1, 1])

    with image_col:

        st.image(

            image,

            caption="Imagen seleccionada",

            use_container_width=True

        )

    with info_col:

        st.write(
            "**Archivo:**", uploaded_file.name
        )

        st.write(
            "**Resolución:**",
            f"{image.width} × {image.height}"
        )

        st.write(
            "**Modelo:**", MODEL_NAMES[model_name]
        )

        st.write(
            "**Enfoque:**", APPROACH_NAMES[approach]
        )

    prediction_key = (

        hashlib.sha256(image_bytes).hexdigest(),

        approach,

        model_name

    )

    if st.button(

        "Clasificar imagen",

        type="primary",

        use_container_width=True

    ):

        st.session_state.pop(
            "prediction_result", None
        )

        st.session_state.pop(
            "prediction_key", None
        )

        try:

            with st.spinner(

                "Cargando modelos y "
                "analizando la imagen..."

            ):

                engine = get_engine()

                prediction = engine.predict(

                    image=image,

                    approach=approach,

                    model_name=model_name

                )

            st.session_state[
                "prediction_result"
            ] = prediction

            st.session_state[
                "prediction_key"
            ] = prediction_key

        except Exception as error:

            st.error(
                f"Error de inferencia: {error}"
            )

            return

    if st.session_state.get(
        "prediction_key"
    ) != prediction_key:

        return

    prediction = st.session_state.get(
        "prediction_result"
    )

    if prediction is None:
        return

    st.divider()

    st.header("Resultados")

    results = prediction["results"]

    columns = st.columns(len(results))

    for column, (key, result) in zip(

        columns,

        results.items()

    ):

        with column:

            display_prediction(
                key,
                result
            )

    st.caption(

        "Tiempo de inferencia: "

        f"{prediction['inference_seconds']:.2f} s"

    )

    st.download_button(

        "Descargar resultado JSON",

        data=json.dumps(

            prediction,

            indent=4,

            ensure_ascii=False

        ),

        file_name="cassava_prediction.json",

        mime="application/json"

    )

    st.warning(

        "La predicción es orientativa. "
        "Las probabilidades no están calibradas "
        "y el sistema no sustituye una evaluación "
        "fitosanitaria profesional."

    )


def experimental_results_page():

    st.header(
        "Resultados experimentales"
    )

    try:

        df = load_experimental_results()

    except FileNotFoundError:

        st.error(
            "No se encontraron los resultados "
            "experimentales de la Fase 4."
        )

        return

    st.caption(

        "Evaluación sobre 3.210 imágenes "
        "del test independiente."

    )

    ensembles = df[

        df["model"] == "heterogeneous_ensemble"

    ]

    columns = st.columns(2)

    for column, approach in zip(

        columns,

        ["centralized", "federated"]

    ):

        row = ensembles[

            ensembles["approach"] == approach

        ].iloc[0]

        with column:

            st.metric(

                f"Ensemble {APPROACH_NAMES[approach]}",

                f"{row['f1_macro']:.2%}",

                help="F1 Macro sobre el test"

            )

    metrics = [

        "f1_macro",

        "accuracy",

        "precision_macro",

        "recall_macro",

        "f1_weighted",

        "balanced_accuracy"

    ]

    metric = st.selectbox(

        "Métrica de comparación",

        metrics

    )

    chart_data = df.pivot(

        index="model",

        columns="approach",

        values=metric

    )

    st.bar_chart(chart_data)

    st.dataframe(

        df.sort_values(

            metric,

            ascending=False

        ).round(4),

        hide_index=True,

        use_container_width=True

    )

    for approach in [
        "centralized",
        "federated"
    ]:

        path = (

            REPORTS_DIR

            / f"{approach}_ensemble_confusion.png"

        )

        if path.exists():

            st.subheader(

                "Matriz de confusión - "

                + APPROACH_NAMES[approach]

            )

            st.image(
                str(path),
                use_container_width=True
            )


def about_page():

    st.header("Acerca del proyecto")

    st.markdown("""

    **Cassava AI** es un prototipo de clasificación
    de enfermedades de hojas de yuca mediante
    aprendizaje profundo.

    ### Metodología

    - Partición estratificada de desarrollo y test.
    - Validación cruzada estratificada de tres folds.
    - Transfer Learning.
    - Optimización de hiperparámetros con Hyperband.
    - Aprendizaje federado mediante FedAvg.
    - Ensembles heterogéneos mediante Soft Voting.

    ### Arquitecturas

    - MobileNetV3-Large.
    - EfficientNet-B0.
    - ResNet50.
    - Vision Transformer B/16.
    - Swin Transformer Tiny.

    ### Limitaciones

    El aprendizaje federado se desarrolla en un
    entorno simulado.

    El sistema reconoce cinco clases conocidas y
    no dispone de un detector de imágenes ajenas
    al dominio.

    Las probabilidades no representan certeza
    diagnóstica y no han sido calibradas.

    """)


st.title("🌿 Cassava AI")

st.caption(

    "Sistema inteligente de clasificación "
    "de enfermedades de hojas de yuca"

)

st.sidebar.title("Navegación")

page = st.sidebar.radio(

    "Seleccionar sección",

    [

        "Clasificación",

        "Resultados experimentales",

        "Acerca del proyecto"

    ]

)

st.sidebar.divider()

st.sidebar.write(

    "Dispositivo disponible:",

    "CUDA" if torch.cuda.is_available() else "CPU"

)

if page == "Clasificación":

    classification_page()

elif page == "Resultados experimentales":

    experimental_results_page()

else:

    about_page()

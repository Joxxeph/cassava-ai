from pathlib import Path
from copy import deepcopy
from src.deployment import load_deployment_bundle
import json
import time

import numpy as np
import torch

from torch import nn
from PIL import ImageOps

from .final_training import load_final_model

from .feature_cache import (
    CLASSIFIER_ATTRIBUTES,
    FEATURE_DIMS,
    release_memory
)

from .preprocessing import build_transforms

from .ensemble import (
    soft_voting,
    validate_probabilities
)


def read_json(filepath):

    with open(

        filepath,

        "r",

        encoding="utf-8"

    ) as file:

        return json.load(file)


def load_shared_architecture(
    project_dir,
    model_name,
    centralized_registry,
    federated_registry,
    device
):

    project_dir = Path(project_dir)

    centralized_path = (

        project_dir

        / centralized_registry["models"][
            model_name
        ]["model_path"]

    )

    federated_path = (

        project_dir

        / federated_registry["models"][
            model_name
        ]["model_path"]

    )

    # Cargar inicialmente en CPU para
    # evitar duplicar backbones en GPU.

    centralized_model, central_metadata = (

        load_final_model(

            centralized_path,

            torch.device("cpu")

        )

    )

    federated_model, fed_metadata = (

        load_final_model(

            federated_path,

            torch.device("cpu")

        )

    )

    assert central_metadata["class_mapping"] == (

        fed_metadata["class_mapping"]

    )

    assert central_metadata["architecture"] == (

        model_name

    )

    assert fed_metadata["architecture"] == (

        model_name

    )

    attribute = CLASSIFIER_ATTRIBUTES[
        model_name
    ]

    centralized_classifier = getattr(

        centralized_model,

        attribute

    )

    federated_classifier = getattr(

        federated_model,

        attribute

    )

    setattr(

        centralized_model,

        attribute,

        nn.Identity()

    )

    setattr(

        federated_model,

        attribute,

        nn.Identity()

    )

    central_state = (

        centralized_model.state_dict()

    )

    fed_state = (

        federated_model.state_dict()

    )

    assert set(central_state) == set(fed_state)

    # Verificar parámetros y buffers
    # de ambos extractores.

    for key in central_state:

        if not torch.equal(

            central_state[key],

            fed_state[key]

        ):

            raise ValueError(

                f"Backbones diferentes: "

                f"{model_name} | {key}"

            )

    del central_state
    del fed_state
    del federated_model

    extractor = centralized_model.to(device)

    extractor.eval()

    centralized_classifier = (

        centralized_classifier.to(device)

    )

    federated_classifier = (

        federated_classifier.to(device)

    )

    centralized_classifier.eval()

    federated_classifier.eval()

    for module in [

        extractor,

        centralized_classifier,

        federated_classifier

    ]:

        for parameter in module.parameters():

            parameter.requires_grad_(False)

    release_memory()

    return {

        "architecture": model_name,

        "extractor": extractor,

        "centralized": centralized_classifier,

        "federated": federated_classifier,

        "class_mapping":
            central_metadata["class_mapping"],

        "feature_dim":
            FEATURE_DIMS[model_name]

    }


@torch.inference_mode()
def extract_batch_features(
    extractor,
    images,
    device,
    mixed_precision=False
):

    extractor.eval()

    images = images.to(

        device,

        non_blocking=True

    )

    use_amp = (

        mixed_precision

        and device.type == "cuda"

    )

    with torch.autocast(

        device_type=device.type,

        dtype=torch.float16,

        enabled=use_amp

    ):

        features = extractor(images)

    features = features.float()

    assert features.ndim == 2

    assert torch.isfinite(features).all()

    return features


@torch.inference_mode()
def predict_shared_batch(
    bundle,
    images,
    device,
    mixed_precision=False
):

    features = extract_batch_features(

        extractor=bundle["extractor"],

        images=images,

        device=device,

        mixed_precision=mixed_precision

    )

    assert features.shape[1] == (

        bundle["feature_dim"]

    )

    results = {}

    for approach in [

        "centralized",

        "federated"

    ]:

        classifier = bundle[approach]

        classifier.eval()

        logits = classifier(features)

        probabilities = torch.softmax(

            logits.float(),

            dim=1

        ).cpu().numpy()

        results[approach] = (

            validate_probabilities(

                probabilities,

                len(bundle["class_mapping"])

            )

        )

    return results


class CassavaInferenceEngine:

    def __init__(

        self,

        project_dir,

        device=None

    ):

        self.project_dir = Path(project_dir)

        self.device = (

            device

            if device is not None

            else torch.device(

                "cuda"

                if torch.cuda.is_available()

                else "cpu"

            )

        )

        self.config = read_json(

            self.project_dir

            / "configs"

            / "inference_config.json"

        )

        self.centralized_registry = read_json(

            self.project_dir

            / self.config["centralized_registry"]

        )

        self.federated_registry = read_json(

            self.project_dir

            / self.config["federated_registry"]

        )

        self.class_mapping = (

            self.config["class_mapping"]

        )

        self.class_names = sorted(

            self.class_mapping,

            key=self.class_mapping.get

        )

        self.architectures = (

            self.config["architectures"]

        )

        self.weights = self.config["weights"]

        _, self.transform = build_transforms(

            self.project_dir

            / self.config["preprocessing_config"]

        )

        # Mantener como máximo una arquitectura
        # cargada en el dispositivo.

        self.active_architecture = None

        self.active_bundle = None


    def unload(self):

        self.active_bundle = None

        self.active_architecture = None

        release_memory()


    def get_bundle(self, model_name):

        if self.active_architecture == model_name:
            if self.active_bundle is not None:
                return self.active_bundle

        self.unload()

        self.active_bundle = load_deployment_bundle(
            project_dir=self.project_dir,
            model_name=model_name,
            centralized_registry=self.centralized_registry,
            federated_registry=self.federated_registry,
            device=self.device,
        )

        self.active_architecture = model_name

        return self.active_bundle

    def preprocess(self, image):

        image = ImageOps.exif_transpose(

            image

        ).convert("RGB")

        return self.transform(

            image

        ).unsqueeze(0)


    def format_prediction(self, probabilities):

        probabilities = np.asarray(

            probabilities,

            dtype=np.float32

        ).reshape(1, -1)

        probabilities = validate_probabilities(

            probabilities,

            len(self.class_names)

        )[0]

        predicted_index = int(

            probabilities.argmax()

        )

        return {

            "predicted_index": predicted_index,

            "predicted_class":
                self.class_names[predicted_index],

            "confidence": float(

                probabilities[predicted_index]

            ),

            "probabilities": {

                class_name: float(probabilities[i])

                for i, class_name

                in enumerate(self.class_names)

            }

        }


    def predict(

        self,

        image,

        approach="both",

        model_name="ensemble"

    ):

        start = time.perf_counter()

        if approach not in [

            "centralized",

            "federated",

            "both"

        ]:

            raise ValueError("Enfoque no válido.")

        if (

            model_name != "ensemble"

            and model_name not in self.architectures

        ):

            raise ValueError(

                "Arquitectura no válida."

            )

        images = self.preprocess(image)

        selected_models = (

            self.architectures

            if model_name == "ensemble"

            else [model_name]

        )

        predictions = {

            "centralized": {},

            "federated": {}

        }

        for architecture in selected_models:

            bundle = self.get_bundle(

                architecture

            )

            batch_results = predict_shared_batch(

                bundle=bundle,

                images=images,

                device=self.device,

                mixed_precision=self.config[

                    "mixed_precision_by_model"

                ][architecture]

            )

            for key in predictions:

                predictions[key][architecture] = (

                    batch_results[key]

                )

            # Eliminar referencia local para
            # permitir liberar la arquitectura.

            del bundle

        results = {}

        selected_approaches = (

            ["centralized", "federated"]

            if approach == "both"

            else [approach]

        )

        for key in selected_approaches:

            if model_name == "ensemble":

                probabilities = soft_voting(

                    predictions[key],

                    weights=self.weights,

                    num_classes=len(self.class_names)

                )

            else:

                probabilities = (

                    predictions[key][model_name]

                )

            results[key] = {

                **self.format_prediction(

                    probabilities[0]

                ),

                "individual_predictions": {

                    architecture:

                        self.format_prediction(

                            values[0]

                        )

                    for architecture, values

                    in predictions[key].items()

                }

            }

        return {

            "model": model_name,

            "approach": approach,

            "results": results,

            "inference_seconds": (

                time.perf_counter() - start

            )

        }

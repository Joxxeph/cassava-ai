
from pathlib import Path

import torch

from src.feature_cache import (
    create_feature_extractor,
    create_classifier,
)


def load_classifier(
    checkpoint_path,
    model_name,
    num_classes,
    dropout=0.2,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    classifier = create_classifier(
        model_name=model_name,
        num_classes=num_classes,
        dropout=dropout,
    )

    classifier.load_state_dict(
        checkpoint["classifier_state_dict"],
        strict=True,
    )

    classifier.requires_grad_(False)
    classifier.eval()

    return classifier


def load_deployment_bundle(
    project_dir,
    model_name,
    centralized_registry,
    federated_registry,
    device,
):
    project_dir = Path(project_dir)
    device = torch.device(device)

    num_classes = centralized_registry["num_classes"]
    class_mapping = centralized_registry["class_mapping"]

    if class_mapping != federated_registry["class_mapping"]:
        raise ValueError("Mapeo de clases incompatible")

    central_info = centralized_registry["models"][model_name]
    federal_info = federated_registry["models"][model_name]

    extractor = create_feature_extractor(
        model_name=model_name,
        num_classes=num_classes,
        device=device,
    )

    centralized = load_classifier(
        checkpoint_path=(
            project_dir / central_info["classifier_path"]
        ),
        model_name=model_name,
        num_classes=num_classes,
        dropout=central_info.get("dropout", 0.2),
    )

    federated = load_classifier(
        checkpoint_path=(
            project_dir / federal_info["classifier_path"]
        ),
        model_name=model_name,
        num_classes=num_classes,
        dropout=federal_info.get("dropout", 0.2),
    )

    return {
        "architecture": model_name,
        "extractor": extractor.to(device).eval(),
        "centralized": centralized.to(device).eval(),
        "federated": federated.to(device).eval(),
        "class_mapping": class_mapping,
        "feature_dim": central_info["feature_dim"],
    }

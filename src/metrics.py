import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report
)

def calculate_class_weights(
    dataframe,
    class_mapping
):

    num_classes = len(class_mapping)

    labels = dataframe["class"].map(
        class_mapping
    )

    if labels.isna().any():

        raise ValueError(
            "Existen clases no reconocidas."
        )

    labels = labels.to_numpy(dtype=np.int64)

    counts = np.bincount(
        labels,
        minlength=num_classes
    )

    if np.any(counts == 0):

        raise ValueError(
            "Existen clases sin imágenes "
            "en el conjunto de entrenamiento."
        )

    total_samples = len(labels)

    weights = total_samples / (
        num_classes * counts
    )

    return torch.tensor(
        weights,
        dtype=torch.float32
    )

def calculate_metrics(
    y_true,
    y_pred,
    num_classes
):

    labels = list(range(num_classes))

    return {

        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),

        "precision_macro": float(
            precision_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0
            )
        ),

        "recall_macro": float(
            recall_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0
            )
        ),

        "f1_macro": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0
            )
        ),

        "f1_weighted": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="weighted",
                zero_division=0
            )
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred
            )
        )
    }

def calculate_detailed_metrics(
    y_true,
    y_pred,
    class_mapping
):

    ordered_classes = sorted(
        class_mapping,
        key=class_mapping.get
    )

    labels = list(
        range(len(ordered_classes))
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=ordered_classes,
        output_dict=True,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    return {
        "class_names": ordered_classes,
        "classification_report": report,
        "confusion_matrix": matrix.tolist()
    }
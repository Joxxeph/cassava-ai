from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .metrics import (
    calculate_metrics,
    calculate_detailed_metrics
)

from .experiments import (
    save_json,
    save_predictions
)


def validate_probabilities(
    probabilities,
    num_classes=5
):

    probabilities = np.asarray(
        probabilities,
        dtype=np.float32
    )

    if probabilities.ndim != 2:

        raise ValueError(
            "Las probabilidades deben tener "
            "forma [N, num_classes]."
        )

    assert probabilities.shape[0] > 0

    assert probabilities.shape[1] == (
        num_classes
    )

    assert np.isfinite(
        probabilities
    ).all()

    assert (probabilities >= 0).all()

    assert (probabilities <= 1).all()

    assert np.allclose(

        probabilities.sum(axis=1),

        1.0,

        atol=1e-5

    )

    return probabilities


def soft_voting(
    probabilities_by_model,
    weights=None,
    num_classes=5
):

    model_names = list(
        probabilities_by_model
    )

    if not model_names:

        raise ValueError(
            "No hay modelos en el ensemble."
        )

    arrays = [

        validate_probabilities(

            probabilities_by_model[name],

            num_classes

        )

        for name in model_names

    ]

    reference_shape = arrays[0].shape

    assert all(

        array.shape == reference_shape

        for array in arrays

    )

    if weights is None:

        weights = {

            name: 1.0 / len(model_names)

            for name in model_names

        }

    assert set(weights) == set(model_names)

    weight_array = np.asarray(

        [weights[name] for name in model_names],

        dtype=np.float64

    )

    assert np.isfinite(weight_array).all()

    assert (weight_array >= 0).all()

    assert np.isclose(

        weight_array.sum(),

        1.0

    )

    stacked = np.stack(
        arrays,
        axis=0
    )

    ensemble_probs = np.average(

        stacked,

        axis=0,

        weights=weight_array

    ).astype(np.float32)

    return validate_probabilities(

        ensemble_probs,

        num_classes

    )


@torch.inference_mode()
def predict_cached_probabilities(
    classifier,
    features,
    device,
    batch_size=256
):

    classifier.eval()

    all_probabilities = []

    for start in range(

        0,

        len(features),

        batch_size

    ):

        batch = features[

            start:start + batch_size

        ].to(

            device=device,

            dtype=torch.float32

        )

        logits = classifier(batch)

        assert torch.isfinite(logits).all()

        probabilities = torch.softmax(

            logits.float(),

            dim=1

        )

        all_probabilities.append(

            probabilities.cpu()

        )

    return validate_probabilities(

        torch.cat(
            all_probabilities,
            dim=0
        ).numpy()

    )


def evaluate_probabilities(
    model_name,
    approach,
    dataframe,
    probabilities,
    class_mapping,
    report_dir,
    inference_seconds=None
):

    report_dir = Path(report_dir)

    report_dir.mkdir(

        parents=True,

        exist_ok=True

    )

    num_classes = len(class_mapping)

    probabilities = validate_probabilities(

        probabilities,

        num_classes

    )

    y_true = dataframe["class"].map(

        class_mapping

    ).to_numpy(dtype=np.int64)

    assert len(y_true) == len(probabilities)

    y_pred = probabilities.argmax(axis=1)

    metrics = calculate_metrics(

        y_true,

        y_pred,

        num_classes

    )

    detailed = calculate_detailed_metrics(

        y_true,

        y_pred,

        class_mapping

    )

    predictions = {

        "y_true": y_true.tolist(),

        "y_pred": y_pred.tolist(),

        "probabilities": probabilities.tolist()

    }

    save_predictions(

        dataframe=dataframe,

        predictions=predictions,

        class_mapping=class_mapping,

        filepath=(
            report_dir / "predictions.csv"
        )

    )

    save_json(

        detailed,

        report_dir / "detailed_metrics.json"

    )

    pd.DataFrame(

        detailed["classification_report"]

    ).transpose().to_csv(

        report_dir / "classification_report.csv",

        encoding="utf-8-sig"

    )

    pd.DataFrame(

        detailed["confusion_matrix"],

        index=detailed["class_names"],

        columns=detailed["class_names"]

    ).to_csv(

        report_dir / "confusion_matrix.csv",

        encoding="utf-8-sig"

    )

    result = {

        "model": model_name,

        "approach": approach,

        "evaluation_split": "test",

        "num_samples": len(y_true),

        "inference_seconds": inference_seconds,

        **metrics

    }

    save_json(

        result,

        report_dir / "metrics.json"

    )

    return result


def load_saved_probabilities(
    filepath,
    dataframe,
    class_mapping
):

    predictions = pd.read_csv(filepath)

    assert predictions["filepath"].tolist() == (

        dataframe["filepath"].tolist()

    )

    expected_labels = dataframe["class"].map(

        class_mapping

    ).to_numpy(dtype=np.int64)

    assert np.array_equal(

        predictions["y_true"].to_numpy(),

        expected_labels

    )

    ordered_classes = sorted(

        class_mapping,

        key=class_mapping.get

    )

    columns = [

        f"prob_{class_name}"

        for class_name in ordered_classes

    ]

    probabilities = validate_probabilities(

        predictions[columns].to_numpy(),

        len(class_mapping)

    )

    assert np.array_equal(

        probabilities.argmax(axis=1),

        predictions["y_pred"].to_numpy()

    )

    return probabilities

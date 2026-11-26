from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from .utils import set_seed

from .experiments import save_json

from .metrics import calculate_class_weights

from .data_generator import create_dataloader

from .model_factory import create_model

from .feature_cache import (
    FEATURE_DIMS,
    CLASSIFIER_ATTRIBUTES,
    create_classifier,
    create_feature_extractor,
    create_cached_dataset,
    dataframe_fingerprint,
    calculate_sha256,
    release_memory
)

from .cached_training import (
    create_cached_optimizer,
    train_cached_epoch,
    save_classifier_checkpoint
)


def reconstruct_full_model(
    model_name,
    classifier,
    num_classes,
    device
):

    model = create_feature_extractor(
        model_name=model_name,
        num_classes=num_classes,
        device=device
    )

    attribute = CLASSIFIER_ATTRIBUTES[
        model_name
    ]

    setattr(
        model,
        attribute,
        classifier
    )

    model.to(device)

    model.eval()

    return model


def save_full_model(
    model,
    filepath,
    metadata
):

    filepath = Path(filepath)

    filepath.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint = {

        "model_state_dict": {

            key: value.detach().cpu()

            for key, value
            in model.state_dict().items()

        },

        "metadata": metadata

    }

    temporary_path = filepath.with_suffix(
        ".tmp"
    )

    torch.save(
        checkpoint,
        temporary_path
    )

    temporary_path.replace(filepath)


def load_final_model(
    checkpoint_path,
    device
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    metadata = checkpoint["metadata"]

    model_name = metadata["architecture"]

    num_classes = metadata["num_classes"]

    dropout = metadata["dropout"]

    model = create_model(

        model_name=model_name,

        num_classes=num_classes,

        pretrained=False,

        dropout=dropout,

        mode="full_finetuning",

        device=None

    )

    classifier = create_classifier(

        model_name=model_name,

        num_classes=num_classes,

        dropout=dropout

    )

    setattr(

        model,

        CLASSIFIER_ATTRIBUTES[model_name],

        classifier

    )

    model.load_state_dict(

        checkpoint["model_state_dict"],

        strict=True

    )

    for parameter in model.parameters():

        parameter.requires_grad = False

    model.to(device)

    model.eval()

    return model, metadata


def run_final_training(
    model_name,
    cache,
    df_development,
    class_mapping,
    final_config,
    checkpoint_dir,
    report_dir,
    device
):

    start = time.perf_counter()

    checkpoint_dir = Path(checkpoint_dir)

    report_dir = Path(report_dir)

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    result_path = report_dir / "result.json"

    classifier_path = (
        checkpoint_dir / "final_classifier.pth"
    )

    model_path = (
        checkpoint_dir / "final_model.pth"
    )

    history_path = (
        report_dir / "history.csv"
    )

    if result_path.exists():

        import json

        with open(
            result_path,
            "r",
            encoding="utf-8"
        ) as file:

            result = json.load(file)

        if result["status"] == "success":

            required_files = [

                classifier_path,

                model_path,

                history_path

            ]

            assert all(
                path.exists()
                for path in required_files
            )

            for path in required_files:

                assert calculate_sha256(path) == (
                    result["artifact_hashes"][path.name]
                )

            print(
                f"Modelo recuperado: {model_name}"
            )

            return result

    seed = final_config["seed"]

    set_seed(seed)

    release_memory()

    num_classes = len(class_mapping)

    params = final_config[
        "hyperparameters"
    ][model_name]

    assert (
        df_development["split"]
        == "development"
    ).all()

    assert cache["metadata"]["architecture"] == (
        model_name
    )

    assert cache["metadata"]["class_mapping"] == (
        class_mapping
    )

    assert cache["metadata"]["dataframe_hash"] == (
        dataframe_fingerprint(df_development)
    )

    assert cache["filepaths"] == (
        df_development["filepath"].tolist()
    )

    assert cache["features"].shape == (

        len(df_development),

        FEATURE_DIMS[model_name]

    )

    assert torch.isfinite(
        cache["features"]
    ).all()

    expected_labels = torch.tensor(

        df_development["class"]
        .map(class_mapping)
        .tolist(),

        dtype=torch.long

    )

    assert torch.equal(

        cache["labels"],

        expected_labels

    )

    dataset = create_cached_dataset(

        cache,

        np.arange(len(df_development))

    )

    dataloader = create_dataloader(

        dataset=dataset,

        batch_size=final_config["batch_size"],

        shuffle=True,

        num_workers=0,

        seed=seed

    )

    class_weights = calculate_class_weights(

        df_development,

        class_mapping

    ).to(device)

    criterion = nn.CrossEntropyLoss(

        weight=class_weights,

        reduction="sum"

    )

    classifier = create_classifier(

        model_name=model_name,

        num_classes=num_classes,

        dropout=params["dropout"]

    ).to(device)

    optimizer = create_cached_optimizer(

        classifier=classifier,

        learning_rate=params["classifier_lr"],

        weight_decay=params["weight_decay"]

    )

    history = []

    training_start = time.perf_counter()

    for epoch in range(

        1,

        final_config["epochs"] + 1

    ):

        metrics = train_cached_epoch(

            classifier=classifier,

            dataloader=dataloader,

            optimizer=optimizer,

            criterion=criterion,

            device=device,

            num_classes=num_classes

        )

        assert metrics["optimizer_steps"] == (
            len(dataloader)
        )

        history.append({

            "epoch": epoch,

            **{
                f"train_{key}": value
                for key, value in metrics.items()
            }

        })

        print(

            f"{model_name} | "

            f"Epoch {epoch} | "

            f"Loss: {metrics['loss']:.4f} | "

            f"Train F1: {metrics['f1_macro']:.4f} | "

            f"{metrics['seconds']:.2f} s",

            flush=True

        )

    training_seconds = (
        time.perf_counter() - training_start
    )

    classifier.eval()

    metadata = {

        "architecture": model_name,

        "num_classes": num_classes,

        "class_mapping": class_mapping,

        "dropout": params["dropout"],

        "hyperparameters": params,

        "seed": seed,

        "epochs": final_config["epochs"],

        "training_mode": "cached_feature_extraction",

        "training_split": "development",

        "training_images": len(df_development),

        "feature_dim": FEATURE_DIMS[model_name],

        "class_weights": class_weights.cpu().tolist(),

        "cache_metadata": cache["metadata"],

        "checkpoint_selection": "last_epoch",

        "test_used": False

    }

    save_classifier_checkpoint(

        classifier=classifier,

        filepath=classifier_path,

        metadata=metadata

    )

    pd.DataFrame(history).to_csv(

        history_path,

        index=False,

        encoding="utf-8-sig"

    )

    del optimizer

    del criterion

    del dataloader

    del dataset

    release_memory()

    full_model = reconstruct_full_model(

        model_name=model_name,

        classifier=classifier,

        num_classes=num_classes,

        device=device

    )

    total_parameters = sum(

        parameter.numel()

        for parameter in full_model.parameters()

    )

    classifier_parameters = sum(

        parameter.numel()

        for parameter in classifier.parameters()

    )

    save_full_model(

        model=full_model,

        filepath=model_path,

        metadata=metadata

    )

    del full_model

    del classifier

    release_memory()

    # Verificar que el modelo guardado
    # pueda reconstruirse correctamente.

    verified_model, verified_metadata = (
        load_final_model(
            model_path,
            device
        )
    )

    assert verified_metadata == metadata

    with torch.inference_mode():

        dummy_input = torch.zeros(

            1, 3, 224, 224,

            device=device

        )

        outputs = verified_model(
            dummy_input
        )

        assert outputs.shape == (
            1, num_classes
        )

        assert torch.isfinite(outputs).all()

    del verified_model

    release_memory()

    total_seconds = (
        time.perf_counter() - start
    )

    result = {

        "architecture": model_name,

        "epochs_completed": len(history),

        "training_images": len(df_development),

        "hyperparameter_source": params["source"],

        "final_train_loss": history[-1]["train_loss"],

        "final_train_f1_macro":
            history[-1]["train_f1_macro"],

        "optimizer_steps": sum(

            row["train_optimizer_steps"]

            for row in history

        ),

        "total_parameters": total_parameters,

        "classifier_parameters": classifier_parameters,

        "training_seconds": training_seconds,

        "total_seconds": total_seconds,

        "total_minutes": total_seconds / 60,

        "classifier_path": str(classifier_path),

        "model_path": str(model_path),

        "model_size_mib": (

            model_path.stat().st_size

            / (1024 ** 2)

        ),

        "artifact_hashes": {

            path.name: calculate_sha256(path)

            for path in [

                classifier_path,

                model_path,

                history_path

            ]

        },

        "checkpoint_verified": True,

        "test_used": False,

        "status": "success"

    }

    save_json(

        result,

        result_path

    )

    print(

        f"Completado: {model_name} | "

        f"{total_seconds / 60:.2f} min"

    )

    return result

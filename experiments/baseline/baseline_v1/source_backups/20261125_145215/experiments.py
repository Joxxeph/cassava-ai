from pathlib import Path

import gc
import json
import time
import shutil

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from .utils import set_seed

from .data_generator import (
    create_fold_loaders
)

from .model_factory import (
    create_model,
    configure_trainable_layers,
    count_parameters
)

from .metrics import (
    calculate_class_weights,
    calculate_detailed_metrics
)

from .trainer import (
    create_optimizer,
    create_scheduler,
    validate_one_epoch,
    fit
)


def save_json(data, filepath):

    filepath = Path(filepath)

    filepath.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        filepath,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False
        )


def release_memory():

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


def save_predictions(
    dataframe,
    predictions,
    class_mapping,
    filepath
):

    df_predictions = dataframe[
        ["filepath", "class", "fold"]
    ].reset_index(drop=True).copy()

    probabilities = np.asarray(
        predictions["probabilities"],
        dtype=np.float32
    )

    expected_shape = (
        len(df_predictions),
        len(class_mapping)
    )

    assert probabilities.shape == expected_shape

    assert np.isfinite(probabilities).all()

    assert np.allclose(
        probabilities.sum(axis=1),
        1.0,
        atol=1e-5
    )

    df_predictions["y_true"] = (
        predictions["y_true"]
    )

    df_predictions["y_pred"] = (
        predictions["y_pred"]
    )

    expected_labels = (
        df_predictions["class"]
        .map(class_mapping)
        .to_numpy()
    )

    assert np.array_equal(
        df_predictions["y_true"].to_numpy(),
        expected_labels
    )

    assert np.array_equal(
        probabilities.argmax(axis=1),
        df_predictions["y_pred"].to_numpy()
    )

    for class_name, class_id in class_mapping.items():

        df_predictions[
            f"prob_{class_name}"
        ] = probabilities[:, class_id]

    df_predictions.to_csv(
        filepath,
        index=False,
        encoding="utf-8-sig"
    )

    return df_predictions


def summarize_optimizer_steps(
    history,
    stage
):

    expected = int(
        history[
            "train_expected_optimizer_steps"
        ].sum()
    )

    actual = int(
        history[
            "train_optimizer_steps"
        ].sum()
    )

    skipped = expected - actual

    return {

        "stage": stage,

        "expected_steps": expected,

        "actual_steps": actual,

        "skipped_steps": skipped,

        "skipped_pct": (
            skipped / expected * 100
            if expected > 0
            else 0.0
        )
    }


def run_training_experiment(
    model_name,
    df_splits,
    dataset_dir,
    class_mapping,
    train_transform,
    eval_transform,
    training_config,
    experiment_config,
    checkpoint_dir,
    log_dir,
    report_dir,
    device,
    fold=1
):

    experiment_start = time.perf_counter()

    checkpoint_dir = Path(checkpoint_dir)

    log_dir = Path(log_dir)

    report_dir = Path(report_dir)

    for directory in [
        checkpoint_dir,
        log_dir,
        report_dir
    ]:

        directory.mkdir(
            parents=True,
            exist_ok=True
        )

    seed = experiment_config["seed"]

    num_classes = len(class_mapping)

    dropout = experiment_config["dropout"]

    model_settings = training_config[
        "model_settings"
    ][model_name]

    batch_size = model_settings["batch_size"]

    accumulation_steps = model_settings[
        "gradient_accumulation_steps"
    ]

    num_workers = experiment_config[
        "num_workers"
    ]

    use_amp = (
        experiment_config["mixed_precision"]
        and device.type == "cuda"
    )

    amp_config = experiment_config["amp"]

    set_seed(seed)

    release_memory()

    if device.type == "cuda":

        torch.cuda.reset_peak_memory_stats()

    print("=" * 70)

    print("Arquitectura:", model_name)

    print("Fold:", fold)

    print("Batch físico:", batch_size)

    print(
        "Batch efectivo:",
        batch_size * accumulation_steps
    )

    print("Mixed Precision:", use_amp)

    print("=" * 70)


    train_loader, val_loader = create_fold_loaders(

        df_splits=df_splits,

        dataset_dir=dataset_dir,

        class_mapping=class_mapping,

        train_transform=train_transform,

        eval_transform=eval_transform,

        fold=fold,

        batch_size=batch_size,

        num_workers=num_workers,

        seed=seed

    )

    df_train = train_loader.dataset.dataframe

    df_val = val_loader.dataset.dataframe

    assert set(
        df_train["sha256"]
    ).isdisjoint(
        set(df_val["sha256"])
    )

    assert (
        df_train["split"] == "development"
    ).all()

    assert (
        df_val["split"] == "development"
    ).all()

    assert (
        df_train["fold"] != fold
    ).all()

    assert (
        df_val["fold"] == fold
    ).all()

    class_weights = calculate_class_weights(
        dataframe=df_train,
        class_mapping=class_mapping
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        reduction="sum"
    )


    model = create_model(

        model_name=model_name,

        num_classes=num_classes,

        pretrained=True,

        dropout=dropout,

        mode="feature_extraction",

        device=device

    )

    parameters_fe = count_parameters(model)

    metadata = {

        "experiment": experiment_config[
            "experiment_name"
        ],

        "architecture": model_name,

        "fold": fold,

        "seed": seed,

        "batch_size": batch_size,

        "dropout": dropout,

        "class_mapping": class_mapping,

        "class_weights": class_weights.tolist(),

        "train_images": len(train_loader.dataset),

        "validation_images": len(val_loader.dataset),

        "amp": amp_config,

        "pretrained": True
    }

    save_json(
        metadata,
        report_dir / "experiment_metadata.json"
    )


    print("\nETAPA 1: FEATURE EXTRACTION")

    optimizer_fe = create_optimizer(

        model=model,

        model_name=model_name,

        classifier_lr=experiment_config[
            "feature_extraction"
        ]["classifier_lr"],

        weight_decay=experiment_config[
            "weight_decay"
        ]
    )

    scheduler_fe = create_scheduler(

        optimizer=optimizer_fe,

        factor=experiment_config[
            "scheduler"
        ]["factor"],

        patience=experiment_config[
            "scheduler"
        ]["patience"],

        min_lr=experiment_config[
            "scheduler"
        ]["min_lr"]
    )

    history_fe, summary_fe = fit(

        model=model,

        model_name=model_name,

        train_loader=train_loader,

        val_loader=val_loader,

        optimizer=optimizer_fe,

        criterion=criterion,

        device=device,

        num_classes=num_classes,

        epochs=experiment_config[
            "feature_extraction"
        ]["epochs"],

        checkpoint_dir=(
            checkpoint_dir / "feature_extraction"
        ),

        log_dir=(
            log_dir / "feature_extraction"
        ),

        report_dir=(
            report_dir / "feature_extraction"
        ),

        accumulation_steps=accumulation_steps,

        use_amp=use_amp,

        scheduler=scheduler_fe,

        patience=experiment_config[
            "early_stopping"
        ]["patience"],

        min_delta=experiment_config[
            "early_stopping"
        ]["min_delta"],

        metadata=metadata,

        amp_config=amp_config

    )

    del optimizer_fe
    del scheduler_fe

    release_memory()


    print("\nETAPA 2: FINE-TUNING PARCIAL")

    configure_trainable_layers(

        model=model,

        model_name=model_name,

        mode="partial_finetuning"

    )

    parameters_ft = count_parameters(model)

    optimizer_ft = create_optimizer(

        model=model,

        model_name=model_name,

        classifier_lr=experiment_config[
            "partial_finetuning"
        ]["classifier_lr"],

        backbone_lr=experiment_config[
            "partial_finetuning"
        ]["backbone_lr"],

        weight_decay=experiment_config[
            "weight_decay"
        ]

    )

    scheduler_ft = create_scheduler(

        optimizer=optimizer_ft,

        factor=experiment_config[
            "scheduler"
        ]["factor"],

        patience=experiment_config[
            "scheduler"
        ]["patience"],

        min_lr=experiment_config[
            "scheduler"
        ]["min_lr"]

    )

    history_ft, summary_ft = fit(

        model=model,

        model_name=model_name,

        train_loader=train_loader,

        val_loader=val_loader,

        optimizer=optimizer_ft,

        criterion=criterion,

        device=device,

        num_classes=num_classes,

        epochs=experiment_config[
            "partial_finetuning"
        ]["epochs"],

        checkpoint_dir=(
            checkpoint_dir / "partial_finetuning"
        ),

        log_dir=(
            log_dir / "partial_finetuning"
        ),

        report_dir=(
            report_dir / "partial_finetuning"
        ),

        accumulation_steps=accumulation_steps,

        use_amp=use_amp,

        scheduler=scheduler_ft,

        patience=experiment_config[
            "early_stopping"
        ]["patience"],

        min_delta=experiment_config[
            "early_stopping"
        ]["min_delta"],

        metadata=metadata,

        amp_config=amp_config

    )

    del optimizer_ft
    del scheduler_ft

    release_memory()


    if (
        summary_ft["best_f1_macro"]
        > summary_fe["best_f1_macro"]
    ):

        best_stage = "partial_finetuning"

        best_summary = summary_ft

    else:

        best_stage = "feature_extraction"

        best_summary = summary_fe

    best_checkpoint_path = (
        checkpoint_dir
        / best_stage
        / "best.pth"
    )

    checkpoint = torch.load(

        best_checkpoint_path,

        map_location="cpu",

        weights_only=True

    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    configure_trainable_layers(
        model=model,
        model_name=model_name,
        mode=best_stage
    )

    model.eval()

    del checkpoint

    print("\nMejor etapa:", best_stage)

    print(
        "Mejor F1 Macro:",
        best_summary["best_f1_macro"]
    )


    final_metrics, predictions = validate_one_epoch(

        model=model,

        dataloader=val_loader,

        criterion=criterion,

        device=device,

        num_classes=num_classes,

        use_amp=use_amp,

        return_predictions=True

    )

    assert np.isclose(

        final_metrics["f1_macro"],

        best_summary["best_f1_macro"],

        atol=1e-6

    )

    detailed_metrics = calculate_detailed_metrics(

        y_true=predictions["y_true"],

        y_pred=predictions["y_pred"],

        class_mapping=class_mapping

    )

    pd.DataFrame(

        detailed_metrics["classification_report"]

    ).transpose().to_csv(

        report_dir / "classification_report.csv",

        encoding="utf-8-sig"

    )

    save_json(

        detailed_metrics,

        report_dir / "detailed_metrics.json"

    )

    save_predictions(

        dataframe=df_val,

        predictions=predictions,

        class_mapping=class_mapping,

        filepath=(
            report_dir / "validation_predictions.csv"
        )

    )


    history_fe = history_fe.copy()

    history_ft = history_ft.copy()

    history_fe["stage"] = "feature_extraction"

    history_ft["stage"] = "partial_finetuning"

    history_fe["global_epoch"] = (
        history_fe["epoch"]
    )

    history_ft["global_epoch"] = (
        history_ft["epoch"] + len(history_fe)
    )

    history_combined = pd.concat(

        [history_fe, history_ft],

        ignore_index=True

    )

    history_combined.to_csv(

        report_dir / "combined_history.csv",

        index=False,

        encoding="utf-8-sig"

    )

    optimizer_summary = [

        summarize_optimizer_steps(
            history_fe,
            "feature_extraction"
        ),

        summarize_optimizer_steps(
            history_ft,
            "partial_finetuning"
        )

    ]

    pd.DataFrame(
        optimizer_summary
    ).to_csv(

        report_dir / "optimizer_steps.csv",

        index=False,

        encoding="utf-8-sig"

    )


    final_model_path = (
        checkpoint_dir / "baseline_model.pth"
    )

    final_checkpoint = {

        "model_state_dict": {
            key: value.detach().cpu()
            for key, value
            in model.state_dict().items()
        },

        "architecture": model_name,

        "num_classes": num_classes,

        "class_mapping": class_mapping,

        "dropout": dropout,

        "fold": fold,

        "best_stage": best_stage,

        "best_epoch": best_summary["best_epoch"],

        "best_f1_macro": final_metrics["f1_macro"],

        "metadata": metadata

    }

    temporary_path = final_model_path.with_suffix(
        ".tmp"
    )

    torch.save(
        final_checkpoint,
        temporary_path
    )

    temporary_path.replace(
        final_model_path
    )

    del final_checkpoint

    # Conservar el mejor checkpoint de cada etapa.
    # Eliminar los checkpoints de la última época
    # una vez guardado el modelo final.

    if experiment_config.get(
        "remove_last_checkpoints", False
    ):

        for stage in [
            "feature_extraction",
            "partial_finetuning"
        ]:

            last_path = (
                checkpoint_dir
                / stage
                / "last.pth"
            )

            last_path.unlink(
                missing_ok=True
            )


            peak_memory_mib = (

        torch.cuda.max_memory_allocated()
        / 1024**2

        if device.type == "cuda"

        else None

    )

    expected_steps = sum(

        row["expected_steps"]

        for row in optimizer_summary

    )

    skipped_steps = sum(

        row["skipped_steps"]

        for row in optimizer_summary

    )

    total_seconds = (
        time.perf_counter()
        - experiment_start
    )

    result = {

        "architecture": model_name,

        "fold": fold,

        "best_stage": best_stage,

        "best_epoch": best_summary["best_epoch"],

        "f1_macro_fe": summary_fe["best_f1_macro"],

        "f1_macro_ft": summary_ft["best_f1_macro"],

        **final_metrics,

        "epochs_fe": len(history_fe),

        "epochs_ft": len(history_ft),

        "batch_size": batch_size,

        "accumulation_steps": accumulation_steps,

        "effective_batch_size": (
            batch_size * accumulation_steps
        ),

        "total_parameters": (
            parameters_fe["total_parameters"]
        ),

        "trainable_parameters_fe": (
            parameters_fe["trainable_parameters"]
        ),

        "trainable_parameters_ft": (
            parameters_ft["trainable_parameters"]
        ),

        "mean_epoch_minutes_fe": float(
            history_fe["epoch_seconds"].mean() / 60
        ),

        "mean_epoch_minutes_ft": float(
            history_ft["epoch_seconds"].mean() / 60
        ),

        "expected_optimizer_steps": expected_steps,

        "skipped_optimizer_steps": skipped_steps,

        "skipped_steps_pct": (
            skipped_steps / expected_steps * 100
        ),

        "peak_memory_mib": peak_memory_mib,

        "total_seconds": total_seconds,

        "total_minutes": total_seconds / 60,

        "model_path": str(final_model_path),

        "status": "success"

    }

    del model
    del criterion
    del train_loader
    del val_loader

    release_memory()

    print("\nExperimento completado.")

    print(
        "F1 Macro:",
        round(result["f1_macro"], 4)
    )

    print(
        "Accuracy:",
        round(result["accuracy"], 4)
    )

    print(
        "Tiempo total:",
        round(result["total_minutes"], 2),
        "min"
    )

    print(
        "Actualizaciones omitidas:",
        result["skipped_optimizer_steps"]
    )

    # Este archivo se escribe al final y sirve
    # como indicador de experimento completado.

    save_json(
        result,
        report_dir / "result.json"
    )

    return result
from pathlib import Path
import json
import time
import math

import numpy as np
import pandas as pd
import torch
from torch import nn

from .utils import set_seed
from .experiments import save_json, save_predictions
from .metrics import calculate_class_weights
from .data_generator import create_dataloader

from .feature_cache import (
    FEATURE_DIMS,
    create_classifier,
    create_cached_dataset,
    dataframe_fingerprint,
    calculate_sha256,
    release_memory
)

from .cached_training import (
    create_cached_optimizer,
    train_cached_epoch,
    evaluate_cached,
    save_classifier_checkpoint
)

from .final_training import (
    reconstruct_full_model,
    save_full_model,
    load_final_model
)


def clone_state_dict(model):

    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def federated_average(
    client_states,
    client_sizes
):

    if len(client_states) != len(client_sizes):
        raise ValueError(
            "Cantidad de estados y tamaños diferente."
        )

    if not client_states:
        raise ValueError("No hay clientes.")

    if any(size <= 0 for size in client_sizes):
        raise ValueError("Cliente sin muestras.")

    total_samples = sum(client_sizes)

    reference = client_states[0]

    aggregated = {}

    for state in client_states:

        assert set(state) == set(reference)

    for key, reference_tensor in reference.items():

        if not reference_tensor.is_floating_point():

            assert all(
                torch.equal(
                    state[key],
                    reference_tensor
                )
                for state in client_states
            )

            aggregated[key] = (
                reference_tensor.clone()
            )

            continue

        result = torch.zeros_like(
            reference_tensor,
            dtype=torch.float64,
            device="cpu"
        )

        for state, size in zip(
            client_states,
            client_sizes
        ):

            tensor = state[key]

            assert tensor.shape == (
                reference_tensor.shape
            )

            assert tensor.dtype == (
                reference_tensor.dtype
            )

            assert torch.isfinite(tensor).all()

            result.add_(
                tensor.to(
                    device="cpu",
                    dtype=torch.float64
                ),
                alpha=size / total_samples
            )

        aggregated[key] = result.to(
            reference_tensor.dtype
        )

        assert torch.isfinite(
            aggregated[key]
        ).all()

    return aggregated


def train_local_client(
    model_name,
    global_state,
    dataset,
    class_weights,
    params,
    config,
    client_id,
    round_number,
    device
):

    start = time.perf_counter()

    local_seed = (
        config["seed"]
        + round_number * 100
        + client_id
    )

    set_seed(local_seed)

    num_classes = len(class_weights)

    classifier = create_classifier(
        model_name=model_name,
        num_classes=num_classes,
        dropout=params["dropout"]
    ).to(device)

    classifier.load_state_dict(
        global_state,
        strict=True
    )

    # Todos los clientes deben comenzar
    # exactamente con los parámetros globales.

    initial_state = clone_state_dict(
        classifier
    )

    assert all(
        torch.equal(
            initial_state[key],
            global_state[key]
        )
        for key in global_state
    )

    dataloader = create_dataloader(
        dataset=dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        seed=local_seed
    )

    optimizer = create_cached_optimizer(
        classifier=classifier,
        learning_rate=params["classifier_lr"],
        weight_decay=params["weight_decay"]
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        reduction="sum"
    )

    history = []

    for epoch in range(
        1,
        config["local_epochs"] + 1
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

        assert metrics["num_samples"] == (
            len(dataset)
        )

        history.append({
            "architecture": model_name,
            "round": round_number,
            "client_id": client_id,
            "local_epoch": epoch,
            "client_samples": len(dataset),
            **metrics
        })

    local_state = clone_state_dict(
        classifier
    )

    update_norm = math.sqrt(
        sum(
            (
                local_state[key].double()
                - global_state[key].double()
            ).square().sum().item()
            for key in global_state
            if global_state[key].is_floating_point()
        )
    )

    assert math.isfinite(update_norm)

    result = {
        "client_id": client_id,
        "num_samples": len(dataset),
        "state_dict": local_state,
        "history": history,
        "update_norm": update_norm,
        "seconds": time.perf_counter() - start
    }

    del classifier
    del optimizer
    del criterion
    del dataloader
    del initial_state

    release_memory()

    return result


def run_federated_experiment(
    model_name,
    cache,
    df_development,
    df_federated,
    class_mapping,
    config,
    checkpoint_dir,
    report_dir,
    device,
    validation_fold=None
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
        checkpoint_dir / "global_classifier.pth"
    )

    model_path = (
        checkpoint_dir / "federated_model.pth"
    )

    local_history_path = (
        report_dir / "local_history.csv"
    )

    round_history_path = (
        report_dir / "round_history.csv"
    )

    if result_path.exists():

        with open(
            result_path,
            "r",
            encoding="utf-8"
        ) as file:

            result = json.load(file)

        assert result["status"] == "success"

        assert result["validation_fold"] == (
            validation_fold
        )

        for filename, expected_hash in (
            result["artifact_hashes"].items()
        ):

            path = (
                checkpoint_dir / filename
                if filename.endswith(".pth")
                else report_dir / filename
            )

            assert path.exists()

            assert calculate_sha256(path) == (
                expected_hash
            )

        print(
            f"Recuperado: {model_name} | "
            f"Fold: {validation_fold}"
        )

        return result

    seed = config["seed"]

    set_seed(seed)

    num_classes = len(class_mapping)

    params = config["hyperparameters"][
        model_name
    ]

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

    # Alinear clientes con el orden de la caché.
    # Nunca depender del orden del CSV federado.

    assignments = (
        df_federated
        .set_index("filepath")["client_id"]
        .reindex(df_development["filepath"])
    )

    assert assignments.notna().all()

    client_ids = assignments.to_numpy(
        dtype=int
    )

    folds = df_development["fold"].to_numpy()

    if validation_fold is None:

        train_mask = np.ones(
            len(df_development),
            dtype=bool
        )

        val_indices = np.array(
            [],
            dtype=int
        )

        stage = "final"

    else:

        train_mask = folds != validation_fold

        val_indices = np.flatnonzero(
            folds == validation_fold
        )

        stage = "validation"

    train_indices = np.flatnonzero(
        train_mask
    )

    df_train = df_development.iloc[
        train_indices
    ].reset_index(drop=True)

    # Pesos comunes calculados exclusivamente
    # sobre los datos de entrenamiento.

    class_weights = calculate_class_weights(
        df_train,
        class_mapping
    )

    client_datasets = {}

    client_indices = {}

    for client_id in config["client_ids"]:

        indices = np.flatnonzero(
            train_mask
            & (client_ids == client_id)
        )

        assert len(indices) > 0

        client_indices[client_id] = indices

        client_datasets[client_id] = (
            create_cached_dataset(
                cache,
                indices
            )
        )

    combined_indices = np.concatenate(
        list(client_indices.values())
    )

    assert len(np.unique(combined_indices)) == (
        len(combined_indices)
    )

    assert np.array_equal(
        np.sort(combined_indices),
        train_indices
    )

    assert len(
        np.intersect1d(
            combined_indices,
            val_indices
        )
    ) == 0

    val_loader = None

    val_criterion = None

    if validation_fold is not None:

        val_dataset = create_cached_dataset(
            cache,
            val_indices
        )

        val_loader = create_dataloader(
            dataset=val_dataset,
            batch_size=256,
            shuffle=False,
            num_workers=0,
            seed=seed
        )

        val_criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device),
            reduction="sum"
        )

    # Inicialización independiente.
    # No cargar el clasificador centralizado.

    set_seed(seed)

    global_classifier = create_classifier(
        model_name=model_name,
        num_classes=num_classes,
        dropout=params["dropout"]
    ).to(device)

    global_state = clone_state_dict(
        global_classifier
    )

    initial_global_state = {
        key: value.clone()
        for key, value in global_state.items()
    }

    local_history = []

    round_history = []

    # Entrenamiento federado.

    for round_number in range(
        1,
        config["rounds"] + 1
    ):

        round_start = time.perf_counter()

        client_results = []

        for client_id in config["client_ids"]:

            result = train_local_client(
                model_name=model_name,
                global_state=global_state,
                dataset=client_datasets[
                    client_id
                ],
                class_weights=class_weights,
                params=params,
                config=config,
                client_id=client_id,
                round_number=round_number,
                device=device
            )

            client_results.append(result)

            local_history.extend(
                result["history"]
            )

            print(
                f"{model_name} | "
                f"{stage} | "
                f"Round {round_number} | "
                f"Client {client_id} | "
                f"Samples {result['num_samples']} | "
                f"F1 local "
                f"{result['history'][-1]['f1_macro']:.4f}",
                flush=True
            )

        client_sizes = [
            result["num_samples"]
            for result in client_results
        ]

        assert sum(client_sizes) == (
            len(train_indices)
        )

        aggregated_state = federated_average(
            client_states=[
                result["state_dict"]
                for result in client_results
            ],
            client_sizes=client_sizes
        )

        global_update_norm = math.sqrt(
            sum(
                (
                    aggregated_state[key].double()
                    - global_state[key].double()
                ).square().sum().item()
                for key in global_state
                if global_state[key].is_floating_point()
            )
        )

        assert math.isfinite(
            global_update_norm
        )

        global_state = aggregated_state

        global_classifier.load_state_dict(
            global_state,
            strict=True
        )

        row = {
            "architecture": model_name,
            "stage": stage,
            "round": round_number,
            "participating_clients": len(
                client_results
            ),
            "training_samples": sum(
                client_sizes
            ),
            "global_update_norm":
                global_update_norm
        }

        for result in client_results:

            client_id = result["client_id"]

            row[
                f"client_{client_id}_samples"
            ] = result["num_samples"]

            row[
                f"client_{client_id}_weight"
            ] = (
                result["num_samples"]
                / sum(client_sizes)
            )

            row[
                f"client_{client_id}_update_norm"
            ] = result["update_norm"]

        if val_loader is not None:

            val_metrics = evaluate_cached(
                classifier=global_classifier,
                dataloader=val_loader,
                criterion=val_criterion,
                device=device,
                num_classes=num_classes
            )

            row.update({
                f"val_{key}": value
                for key, value
                in val_metrics.items()
            })

            print(
                f"Global validation F1: "
                f"{val_metrics['f1_macro']:.4f}"
            )

        row["round_seconds"] = (
            time.perf_counter() - round_start
        )

        round_history.append(row)

        pd.DataFrame(local_history).to_csv(
            local_history_path,
            index=False,
            encoding="utf-8-sig"
        )

        pd.DataFrame(round_history).to_csv(
            round_history_path,
            index=False,
            encoding="utf-8-sig"
        )

        del client_results

    # Verificar que FedAvg modificó el modelo.

    assert any(
        not torch.equal(
            global_state[key],
            initial_global_state[key]
        )
        for key in global_state
    )

    metadata = {
        "architecture": model_name,
        "num_classes": num_classes,
        "class_mapping": class_mapping,
        "dropout": params["dropout"],
        "hyperparameters": params,
        "seed": seed,
        "algorithm": "FedAvg",
        "simulation": True,
        "training_mode":
            "cached_federated_feature_extraction",
        "stage": stage,
        "validation_fold": validation_fold,
        "rounds": config["rounds"],
        "local_epochs": config["local_epochs"],
        "num_clients": config["num_clients"],
        "training_images": len(train_indices),
        "class_weights": class_weights.tolist(),
        "feature_dim": FEATURE_DIMS[model_name],
        "cache_metadata": cache["metadata"],
        "checkpoint_selection": "last_round",
        "test_used": False
    }

    save_classifier_checkpoint(
        classifier=global_classifier,
        filepath=classifier_path,
        metadata=metadata
    )

    artifacts = [
        classifier_path,
        local_history_path,
        round_history_path
    ]

    final_metrics = {}

    if val_loader is not None:

        final_metrics, predictions = (
            evaluate_cached(
                classifier=global_classifier,
                dataloader=val_loader,
                criterion=val_criterion,
                device=device,
                num_classes=num_classes,
                return_predictions=True
            )
        )

        predictions_path = (
            report_dir
            / "validation_predictions.csv"
        )

        save_predictions(
            dataframe=df_development.iloc[
                val_indices
            ].reset_index(drop=True),
            predictions=predictions,
            class_mapping=class_mapping,
            filepath=predictions_path
        )

        artifacts.append(predictions_path)

    # Reconstruir la red completa únicamente
    # durante el entrenamiento definitivo.

    if stage == "final":

        full_model = reconstruct_full_model(
            model_name=model_name,
            classifier=global_classifier,
            num_classes=num_classes,
            device=device
        )

        save_full_model(
            model=full_model,
            filepath=model_path,
            metadata=metadata
        )

        del full_model

        release_memory()

        verified_model, verified_metadata = (
            load_final_model(
                model_path,
                device
            )
        )

        assert verified_metadata == metadata

        with torch.inference_mode():

            outputs = verified_model(
                torch.zeros(
                    1, 3, 224, 224,
                    device=device
                )
            )

            assert outputs.shape == (
                1,
                num_classes
            )

            assert torch.isfinite(
                outputs
            ).all()

        del verified_model

        artifacts.append(model_path)

    total_seconds = (
        time.perf_counter() - start
    )

    result = {
        "architecture": model_name,
        "stage": stage,
        "validation_fold": validation_fold,
        "rounds_completed": len(round_history),
        "local_epochs_completed": len(
            local_history
        ),
        "training_images": len(train_indices),
        "validation_images": len(val_indices),
        "optimizer_steps": sum(
            row["optimizer_steps"]
            for row in local_history
        ),
        "classifier_path": str(classifier_path),
        "model_path": (
            str(model_path)
            if stage == "final"
            else None
        ),
        "total_seconds": total_seconds,
        "total_minutes": total_seconds / 60,
        "test_used": False,
        "artifact_hashes": {
            path.name: calculate_sha256(path)
            for path in artifacts
        },
        **final_metrics,
        "status": "success"
    }

    save_json(
        result,
        result_path
    )

    del global_classifier
    del client_datasets
    del global_state
    del initial_global_state

    release_memory()

    return result

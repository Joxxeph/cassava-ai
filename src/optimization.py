from pathlib import Path
import gc
import time

import numpy as np
import torch
import optuna

from torch import nn

from .utils import set_seed

from .preprocessing import build_transforms

from .data_generator import create_fold_loaders

from .model_factory import (
    create_model,
    configure_trainable_layers
)

from .metrics import calculate_class_weights

from .trainer import (
    create_optimizer,
    create_scheduler,
    train_one_epoch,
    validate_one_epoch
)


def release_trial_memory():

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


def suggest_hyperparameters(
    trial,
    stage,
    search_space,
    default_dropout
):

    space = search_space[stage]

    params = {

        "classifier_lr": trial.suggest_float(

            "classifier_lr",

            space["classifier_lr"]["low"],

            space["classifier_lr"]["high"],

            log=True

        ),

        "weight_decay": trial.suggest_float(

            "weight_decay",

            space["weight_decay"]["low"],

            space["weight_decay"]["high"],

            log=True

        )

    }

    if stage == "feature_extraction":

        params["dropout"] = trial.suggest_float(

            "dropout",

            space["dropout"]["low"],

            space["dropout"]["high"]

        )

    elif stage == "partial_finetuning":

        params["backbone_lr"] = trial.suggest_float(

            "backbone_lr",

            space["backbone_lr"]["low"],

            space["backbone_lr"]["high"],

            log=True

        )

        params["dropout"] = default_dropout

    else:

        raise ValueError(
            f"Etapa no soportada: {stage}"
        )

    return params


def create_trial_model(
    model_name,
    stage,
    num_classes,
    dropout,
    device,
    baseline_models_dir,
    fold,
    class_mapping
):

    if stage == "feature_extraction":

        model = create_model(

            model_name=model_name,

            num_classes=num_classes,

            pretrained=True,

            dropout=dropout,

            mode="feature_extraction",

            device=device

        )

    elif stage == "partial_finetuning":

        checkpoint_path = (

            Path(baseline_models_dir)

            / model_name

            / "feature_extraction"

            / "best.pth"

        )

        if not checkpoint_path.exists():

            raise FileNotFoundError(
                checkpoint_path
            )

        checkpoint = torch.load(

            checkpoint_path,

            map_location="cpu",

            weights_only=True

        )

        metadata = checkpoint["metadata"]

        assert metadata["architecture"] == model_name

        assert metadata["fold"] == fold

        assert metadata["class_mapping"] == class_mapping

        assert np.isclose(
            metadata["dropout"],
            dropout
        )

        model = create_model(

            model_name=model_name,

            num_classes=num_classes,

            pretrained=False,

            dropout=dropout,

            mode="full_finetuning",

            device=None

        )

        model.load_state_dict(

            checkpoint["model_state_dict"],

            strict=True

        )

        configure_trainable_layers(

            model=model,

            model_name=model_name,

            mode="partial_finetuning"

        )

        del checkpoint

        model = model.to(device)

    else:

        raise ValueError(
            f"Etapa desconocida: {stage}"
        )

    return model


def objective(
    trial,
    model_name,
    df_splits,
    dataset_dir,
    class_mapping,
    preprocessing_config_path,
    training_config,
    hyperband_config,
    search_space,
    baseline_models_dir,
    device
):

    start_time = time.perf_counter()

    model = None
    optimizer = None
    scheduler = None
    scaler = None
    criterion = None
    train_loader = None
    val_loader = None

    seed = hyperband_config["seed"]

    fold = hyperband_config["validation_fold"]

    num_classes = len(class_mapping)

    stage = hyperband_config[
        "architecture_stages"
    ][model_name]

    model_settings = training_config[
        "model_settings"
    ][model_name]

    batch_size = model_settings["batch_size"]

    accumulation_steps = model_settings[
        "gradient_accumulation_steps"
    ]

    use_amp = (

        hyperband_config["mixed_precision"]

        and device.type == "cuda"

    )

    params = suggest_hyperparameters(

        trial=trial,

        stage=stage,

        search_space=search_space,

        default_dropout=hyperband_config[
            "default_dropout"
        ]

    )

    trial.set_user_attr("architecture", model_name)
    trial.set_user_attr("stage", stage)
    trial.set_user_attr("fold", fold)
    trial.set_user_attr("seed", seed)
    trial.set_user_attr("resolved_params", params)

    best_score = -1.0
    best_epoch = 0
    epoch_history = []

    try:

        release_trial_memory()

        set_seed(seed)

        if device.type == "cuda":

            torch.cuda.reset_peak_memory_stats()

        train_transform, eval_transform = (
            build_transforms(
                preprocessing_config_path
            )
        )

        train_loader, val_loader = (
            create_fold_loaders(

                df_splits=df_splits,

                dataset_dir=dataset_dir,

                class_mapping=class_mapping,

                train_transform=train_transform,

                eval_transform=eval_transform,

                fold=fold,

                batch_size=batch_size,

                num_workers=hyperband_config[
                    "num_workers"
                ],

                seed=seed

            )
        )

        df_train = df_splits[

            (df_splits["split"] == "development")

            & (df_splits["fold"] != fold)

        ]

        df_val = df_splits[

            (df_splits["split"] == "development")

            & (df_splits["fold"] == fold)

        ]

        assert set(
            df_train["sha256"]
        ).isdisjoint(
            df_val["sha256"]
        )

        class_weights = calculate_class_weights(

            df_train,

            class_mapping

        ).to(device)

        criterion = nn.CrossEntropyLoss(

            weight=class_weights,

            reduction="sum"

        )

        model = create_trial_model(

            model_name=model_name,

            stage=stage,

            num_classes=num_classes,

            dropout=params["dropout"],

            device=device,

            baseline_models_dir=baseline_models_dir,

            fold=fold,

            class_mapping=class_mapping

        )

        optimizer = create_optimizer(

            model=model,

            model_name=model_name,

            classifier_lr=params["classifier_lr"],

            backbone_lr=params.get(
                "backbone_lr", 1e-5
            ),

            weight_decay=params["weight_decay"]

        )

        scheduler_config = hyperband_config[
            "scheduler"
        ]

        scheduler = create_scheduler(

            optimizer=optimizer,

            factor=scheduler_config["factor"],

            patience=scheduler_config["patience"],

            min_lr=scheduler_config["min_lr"]

        )

        amp_config = hyperband_config["amp"]

        scaler = torch.amp.GradScaler(

            "cuda",

            init_scale=amp_config["init_scale"],

            growth_factor=amp_config["growth_factor"],

            backoff_factor=amp_config["backoff_factor"],

            growth_interval=amp_config["growth_interval"],

            enabled=use_amp

        )

        max_epochs = hyperband_config[
            "max_resource"
        ]

        print("=" * 65)

        print(
            f"{model_name} | Trial {trial.number}"
        )

        print(f"Stage: {stage}")

        print(f"Params: {params}")

        for epoch in range(1, max_epochs + 1):

            epoch_start = time.perf_counter()

            train_metrics = train_one_epoch(

                model=model,

                model_name=model_name,

                dataloader=train_loader,

                optimizer=optimizer,

                criterion=criterion,

                scaler=scaler,

                device=device,

                num_classes=num_classes,

                accumulation_steps=accumulation_steps,

                use_amp=use_amp

            )

            val_metrics = validate_one_epoch(

                model=model,

                dataloader=val_loader,

                criterion=criterion,

                device=device,

                num_classes=num_classes,

                use_amp=use_amp

            )

            expected_steps = train_metrics[
                "expected_optimizer_steps"
            ]

            actual_steps = train_metrics[
                "optimizer_steps"
            ]

            assert (
                0 <= actual_steps <= expected_steps
            )

            score = float(
                val_metrics["f1_macro"]
            )

            if not np.isfinite(score):

                raise FloatingPointError(
                    "F1 Macro no finito."
                )

            scheduler.step(score)

            if score > best_score:

                best_score = score

                best_epoch = epoch

            epoch_seconds = (
                time.perf_counter() - epoch_start
            )

            epoch_history.append({

                "epoch": epoch,

                "train_loss": float(
                    train_metrics["loss"]
                ),

                "train_f1_macro": float(
                    train_metrics["f1_macro"]
                ),

                "val_loss": float(
                    val_metrics["loss"]
                ),

                "val_f1_macro": score,

                "val_accuracy": float(
                    val_metrics["accuracy"]
                ),

                "expected_optimizer_steps": int(
                    expected_steps
                ),

                "actual_optimizer_steps": int(
                    actual_steps
                ),

                "skipped_optimizer_steps": int(
                    expected_steps - actual_steps
                ),

                "amp_scale": float(
                    scaler.get_scale()
                ),

                "epoch_seconds": epoch_seconds

            })

            trial.set_user_attr(
                "epoch_history",
                epoch_history
            )

            trial.set_user_attr(
                "best_epoch",
                best_epoch
            )

            trial.set_user_attr(
                "best_f1_macro",
                best_score
            )

            trial.set_user_attr(
                "epochs_completed",
                epoch
            )

            print(

                f"Epoch {epoch}/{max_epochs} | "

                f"Train F1: "
                f"{train_metrics['f1_macro']:.4f} | "

                f"Val F1: {score:.4f} | "

                f"Time: {epoch_seconds / 60:.2f} min"

            )

            # Reportar el recurso consumido
            # y el rendimiento intermedio

            trial.report(

                score,

                step=epoch

            )

            # Hyperband decide si continúa

            if (

                epoch < max_epochs

                and trial.should_prune()

            ):

                print(
                    f"Trial {trial.number} descartado "
                    f"en época {epoch}."
                )

                raise optuna.TrialPruned()

        return best_score

    finally:

        total_seconds = (
            time.perf_counter() - start_time
        )

        trial.set_user_attr(

            "total_seconds",

            total_seconds

        )

        trial.set_user_attr(

            "total_minutes",

            total_seconds / 60

        )

        if device.type == "cuda":

            trial.set_user_attr(

                "peak_memory_mib",

                torch.cuda.max_memory_allocated()

                / (1024 ** 2)

            )

        del model
        del optimizer
        del scheduler
        del scaler
        del criterion
        del train_loader
        del val_loader

        release_trial_memory()


def create_hyperband_study(
    model_name,
    storage_url,
    hyperband_config
):

    sampler = optuna.samplers.RandomSampler(

        seed=hyperband_config["seed"]

    )

    pruner = optuna.pruners.HyperbandPruner(

        min_resource=hyperband_config[
            "min_resource"
        ],

        max_resource=hyperband_config[
            "max_resource"
        ],

        reduction_factor=hyperband_config[
            "reduction_factor"
        ]

    )

    study_name = (

        f"{hyperband_config['run_id']}"

        f"_{model_name}"

    )

    study = optuna.create_study(

        study_name=study_name,

        storage=storage_url,

        direction="maximize",

        sampler=sampler,

        pruner=pruner,

        load_if_exists=True

    )

    return study


def optimize_architecture(
    study,
    objective_function,
    target_trials
):

    from optuna.trial import TrialState

    finished_trials = [

        trial

        for trial in study.trials

        if trial.state in [

            TrialState.COMPLETE,

            TrialState.PRUNED

        ]

    ]

    remaining_trials = max(

        0,

        target_trials - len(finished_trials)

    )

    print(f"Study: {study.study_name}")

    print(
        f"Trials terminados: "
        f"{len(finished_trials)}"
    )

    print(
        f"Trials pendientes: "
        f"{remaining_trials}"
    )

    if remaining_trials > 0:

        study.optimize(

            objective_function,

            n_trials=remaining_trials,

            n_jobs=1,

            gc_after_trial=True

        )

    return study
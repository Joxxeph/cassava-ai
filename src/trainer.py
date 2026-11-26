from pathlib import Path

import time
import json

import pandas as pd

import torch
import torch.nn as nn


from torch.utils.tensorboard import SummaryWriter

from .metrics import calculate_metrics

from .model_factory import (
    get_classifier,
    set_model_train_mode
)

def create_optimizer(
    model,
    model_name,
    classifier_lr=1e-3,
    backbone_lr=1e-5,
    weight_decay=1e-4
):

    classifier = get_classifier(
        model,
        model_name
    )

    classifier_ids = {
        id(p)
        for p in classifier.parameters()
    }

    classifier_params = [
        p
        for p in classifier.parameters()
        if p.requires_grad
    ]

    backbone_params = [
        p
        for p in model.parameters()
        if p.requires_grad
        and id(p) not in classifier_ids
    ]

    param_groups = []

    if classifier_params:

        param_groups.append({
            "params": classifier_params,
            "lr": classifier_lr,
            "name": "classifier"
        })

    if backbone_params:

        param_groups.append({
            "params": backbone_params,
            "lr": backbone_lr,
            "name": "backbone"
        })

    if not param_groups:

        raise ValueError(
            "No existen parámetros entrenables."
        )

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=weight_decay
    )

    return optimizer


def create_scheduler(
    optimizer,
    factor=0.5,
    patience=1,
    min_lr=1e-7
):

    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=factor,
        patience=patience,
        min_lr=min_lr,
        threshold=0.0
    )


class EarlyStopping:

    def __init__(
        self,
        patience=3,
        min_delta=0.0
    ):

        self.patience = patience

        self.min_delta = min_delta

        self.best_score = float("-inf")

        self.counter = 0

        self.should_stop = False

    def step(self, score):

        if score > (
            self.best_score + self.min_delta
        ):

            self.best_score = score

            self.counter = 0

        else:

            self.counter += 1

            if self.counter >= self.patience:

                self.should_stop = True

        return self.should_stop



def train_one_epoch(
model,
model_name,
dataloader,
optimizer,
criterion,
scaler,
device,
num_classes,
accumulation_steps=1,
use_amp=True
):

    if accumulation_steps < 1:

        raise ValueError(
            "accumulation_steps debe ser >= 1."
        )

    if len(dataloader) == 0:

        raise ValueError(
            "DataLoader vacío."
        )

    if criterion.reduction != "sum":

        raise ValueError(
            "La pérdida debe usar reduction='sum'."
        )

    use_amp = (
        use_amp and device.type == "cuda"
    )

    set_model_train_mode(
        model,
        model_name
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    total_loss = 0.0

    total_weight = 0.0

    accumulated_weight = 0.0

    all_labels = []

    all_predictions = []

    optimizer_steps = 0

    start_time = time.perf_counter()

    for batch_idx, (images, labels) in enumerate(
        dataloader
    ):

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        # Forward pass con precisión mixta
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

        if not torch.isfinite(loss):

            raise RuntimeError(
                "Pérdida no finita durante entrenamiento."
            )

        # Denominador de la pérdida ponderada
        if criterion.weight is not None:

            batch_weight = (
                criterion.weight[labels]
                .sum()
                .item()
            )

        else:

            batch_weight = labels.numel()

        # Acumular gradientes
        # Factor constante para estabilizar
        # la magnitud de los gradientes escalados.

        loss_divisor = (
            dataloader.batch_size
            * accumulation_steps
        )

        scaler.scale(
            loss / loss_divisor
        ).backward()

        accumulated_weight += batch_weight

        total_loss += loss.detach().item()

        total_weight += batch_weight

        is_accumulation_step = (
            (batch_idx + 1)
            % accumulation_steps == 0
        )

        is_last_batch = (
            batch_idx + 1 == len(dataloader)
        )

        if is_accumulation_step or is_last_batch:

            # Recuperar gradientes sin escalado
            scaler.unscale_(optimizer)

            # Normalización por pesos reales
            for group in optimizer.param_groups:

                for param in group["params"]:

                    if param.grad is not None:

                        param.grad.mul_(
                            loss_divisor / accumulated_weight
                        )

            # Registrar escala antes de la actualización
            previous_scale = scaler.get_scale()

            # Actualizar los parámetros del modelo
            scaler.step(optimizer)

            # Actualizar la escala de AMP
            scaler.update()

            current_scale = scaler.get_scale()

            # AMP puede omitir una actualización por overflow
            step_success = current_scale >= previous_scale

            if step_success:

                optimizer_steps += 1

            # Reiniciar gradientes
            optimizer.zero_grad(
                set_to_none=True
            )

            # Reiniciar peso acumulado
            accumulated_weight = 0.0

        predictions = outputs.detach().argmax(
            dim=1
        )

        all_labels.extend(
            labels.detach().cpu().tolist()
        )

        all_predictions.extend(
            predictions.cpu().tolist()
        )

    metrics = calculate_metrics(
        y_true=all_labels,
        y_pred=all_predictions,
        num_classes=num_classes
    )

    elapsed = (
        time.perf_counter() - start_time
    )

    metrics["loss"] = (
        total_loss / total_weight
    )

    metrics["seconds"] = elapsed

    metrics["num_samples"] = len(
        all_labels
    )

    metrics["optimizer_steps"] = (
        optimizer_steps
    )

    expected_steps = (
        len(dataloader)
        + accumulation_steps - 1
    ) // accumulation_steps


    metrics["expected_optimizer_steps"] = (
        expected_steps
    )

    assert 0 <= optimizer_steps <= expected_steps, (
    
            "Conteo inválido de actualizaciones. "
    
            f"Esperadas: {expected_steps}. "
    
            f"Registradas: {optimizer_steps}."
    
        )

    metrics["skipped_optimizer_steps"] = (
        expected_steps - optimizer_steps
    )

    metrics["skipped_steps_pct"] = (
        metrics["skipped_optimizer_steps"]
        / expected_steps
        * 100
    )

    metrics["amp_scale"] = float(
        scaler.get_scale()
    )

    return metrics



@torch.inference_mode()
def validate_one_epoch(
    model,
    dataloader,
    criterion,
    device,
    num_classes,
    use_amp=True,
    return_predictions=False
):

    if len(dataloader) == 0:

        raise ValueError(
            "DataLoader de validación vacío."
        )

    if criterion.reduction != "sum":

        raise ValueError(
            "La pérdida debe usar reduction='sum'."
        )

    use_amp = (
        use_amp and device.type == "cuda"
    )

    model.eval()

    total_loss = 0.0

    total_weight = 0.0

    all_labels = []

    all_predictions = []

    all_probabilities = []

    start_time = time.perf_counter()

    for images, labels in dataloader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

        if not torch.isfinite(loss):

            raise RuntimeError(
                "Pérdida no finita en validación."
            )

        if criterion.weight is not None:

            batch_weight = (
                criterion.weight[labels]
                .sum()
                .item()
            )

        else:

            batch_weight = labels.numel()

        total_loss += loss.item()

        total_weight += batch_weight

        predictions = outputs.argmax(
            dim=1
        )

        all_labels.extend(
            labels.cpu().tolist()
        )

        all_predictions.extend(
            predictions.cpu().tolist()
        )

        if return_predictions:

            probabilities = torch.softmax(
                outputs.float(),
                dim=1
            )

            all_probabilities.extend(
                probabilities.cpu().tolist()
            )

    metrics = calculate_metrics(
        y_true=all_labels,
        y_pred=all_predictions,
        num_classes=num_classes
    )

    metrics["loss"] = (
        total_loss / total_weight
    )

    metrics["seconds"] = (
        time.perf_counter() - start_time
    )

    metrics["num_samples"] = len(
        all_labels
    )

    if return_predictions:

        return metrics, {
            "y_true": all_labels,
            "y_pred": all_predictions,
            "probabilities": all_probabilities
        }

    return metrics


def save_checkpoint(
    filepath,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    best_score,
    early_stopping,
    history,
    metadata
):

    filepath = Path(filepath)

    filepath.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint = {

        "epoch": epoch,

        "model_state_dict": model.state_dict(),

        "optimizer_state_dict": (
            optimizer.state_dict()
        ),

        "scheduler_state_dict": (
            scheduler.state_dict()
            if scheduler is not None
            else None
        ),

        "scaler_state_dict": scaler.state_dict(),

        "best_f1_macro": best_score,

        "early_stopping": {
            "counter": early_stopping.counter,
            "best_score": early_stopping.best_score
        },

        "history": history,

        "metadata": metadata
    }

    # Escritura temporal para proteger el checkpoint
    temporary_path = filepath.with_suffix(
        ".tmp"
    )

    torch.save(
        checkpoint,
        temporary_path
    )

    temporary_path.replace(filepath)


def fit(
    model,
    model_name,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    num_classes,
    epochs,
    checkpoint_dir,
    log_dir,
    report_dir,
    accumulation_steps=1,
    use_amp=True,
    scheduler=None,
    patience=3,
    min_delta=0.0,
    metadata=None,
    amp_config=None
):

    if epochs < 1:

        raise ValueError(
            "Debe existir al menos una época."
        )

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

    use_amp = (
        use_amp and device.type == "cuda"
    )

    amp_config = dict(amp_config or {})

    scaler = torch.amp.GradScaler(
        "cuda",
        init_scale=amp_config.get(
            "init_scale", 1024.0
        ),
        growth_factor=amp_config.get(
            "growth_factor", 2.0
        ),
        backoff_factor=amp_config.get(
            "backoff_factor", 0.5
        ),
        growth_interval=amp_config.get(
            "growth_interval", 2000
        ),
        enabled=use_amp
    )

    early_stopping = EarlyStopping(
        patience=patience,
        min_delta=min_delta
    )

    writer = SummaryWriter(
        log_dir=str(log_dir)
    )

    best_score = float("-inf")

    best_epoch = None

    history = []

    metadata = dict(metadata or {})

    metadata.update({
        "architecture": model_name,
        "training_mode": model.training_mode_name,
        "num_classes": num_classes,
        "accumulation_steps": accumulation_steps,
        "mixed_precision": use_amp
    })

    try:

        for epoch in range(1, epochs + 1):

            epoch_start = time.perf_counter()

            learning_rates = {
                group["name"]: group["lr"]
                for group in optimizer.param_groups
            }

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

            score = val_metrics["f1_macro"]

            improved = score > best_score

            if improved:

                best_score = score

                best_epoch = epoch

            if scheduler is not None:

                scheduler.step(score)

            should_stop = early_stopping.step(
                score
            )

            epoch_seconds = (
                time.perf_counter() - epoch_start
            )

            record = {

                "epoch": epoch,

                **{
                    f"train_{key}": value
                    for key, value
                    in train_metrics.items()
                },

                **{
                    f"val_{key}": value
                    for key, value
                    in val_metrics.items()
                },

                **{
                    f"lr_{key}": value
                    for key, value
                    in learning_rates.items()
                },

                "epoch_seconds": epoch_seconds,

                "is_best": improved
            }

            history.append(record)

            # TensorBoard
            for key, value in train_metrics.items():

                writer.add_scalar(
                    f"Train/{key}",
                    value,
                    epoch
                )

            for key, value in val_metrics.items():

                writer.add_scalar(
                    f"Validation/{key}",
                    value,
                    epoch
                )

            for name, lr in learning_rates.items():

                writer.add_scalar(
                    f"LearningRate/{name}",
                    lr,
                    epoch
                )

            writer.flush()

            checkpoint_args = dict(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_score=best_score,
                early_stopping=early_stopping,
                history=history,
                metadata=metadata
            )

            # Guardar el mejor modelo
            if improved:

                save_checkpoint(
                    filepath=checkpoint_dir / "best.pth",
                    **checkpoint_args
                )

            # Guardar la última época
            save_checkpoint(
                filepath=checkpoint_dir / "last.pth",
                **checkpoint_args
            )

            # Guardar historial
            pd.DataFrame(history).to_csv(
                report_dir / "history.csv",
                index=False,
                encoding="utf-8-sig"
            )

            print(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Train F1: {train_metrics['f1_macro']:.4f} | "
                f"Val F1: {score:.4f} | "
                f"Tiempo: {epoch_seconds / 60:.2f} min"
            )

            if improved:

                print(
                    "Nuevo mejor checkpoint."
                )

            if should_stop:

                print(
                    "Early Stopping activado."
                )

                break

    finally:

        writer.close()

    # Recuperar el mejor modelo
    checkpoint = torch.load(
        checkpoint_dir / "best.pth",
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    summary = {

        "architecture": model_name,

        "best_epoch": best_epoch,

        "best_f1_macro": best_score,

        "epochs_completed": len(history),

        "total_seconds": sum(
            row["epoch_seconds"]
            for row in history
        ),

        "early_stopping": (
            early_stopping.should_stop
        ),

        "metadata": metadata
    }

    with open(
        report_dir / "summary.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )

    return pd.DataFrame(history), summary



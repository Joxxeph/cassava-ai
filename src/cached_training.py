from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from .metrics import calculate_metrics


def create_cached_optimizer(
    classifier,
    learning_rate,
    weight_decay
):

    return torch.optim.AdamW(
        classifier.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )


def train_cached_epoch(
    classifier,
    dataloader,
    optimizer,
    criterion,
    device,
    num_classes
):

    classifier.train()

    start = time.perf_counter()

    total_loss = 0.0
    total_weight = 0.0

    y_true = []
    y_pred = []

    optimizer_steps = 0

    for features, labels in dataloader:

        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = classifier(features)

        loss_sum = criterion(
            outputs,
            labels
        )

        if criterion.weight is not None:

            batch_weight = (
                criterion.weight[labels].sum()
            )

        else:

            batch_weight = labels.numel()

        loss = loss_sum / batch_weight

        if not torch.isfinite(loss):

            raise FloatingPointError(
                "Loss no finita."
            )

        loss.backward()

        optimizer.step()

        optimizer_steps += 1

        total_loss += loss_sum.item()

        total_weight += float(batch_weight)

        y_true.extend(
            labels.detach().cpu().tolist()
        )

        y_pred.extend(
            outputs.detach()
            .argmax(dim=1)
            .cpu()
            .tolist()
        )

    assert optimizer_steps == len(dataloader)

    metrics = calculate_metrics(
        y_true,
        y_pred,
        num_classes
    )

    metrics.update({
        "loss": total_loss / total_weight,
        "seconds": time.perf_counter() - start,
        "optimizer_steps": optimizer_steps,
        "num_samples": len(y_true)
    })

    return metrics


@torch.inference_mode()
def evaluate_cached(
    classifier,
    dataloader,
    criterion,
    device,
    num_classes,
    return_predictions=False
):

    classifier.eval()

    start = time.perf_counter()

    total_loss = 0.0
    total_weight = 0.0

    y_true = []
    y_pred = []
    probabilities = []

    for features, labels in dataloader:

        features = features.to(device)
        labels = labels.to(device)

        outputs = classifier(features)

        if not torch.isfinite(outputs).all():

            raise FloatingPointError(
                "Logits no finitos."
            )

        loss_sum = criterion(
            outputs,
            labels
        )

        if criterion.weight is not None:

            batch_weight = (
                criterion.weight[labels].sum()
            )

        else:

            batch_weight = labels.numel()

        total_loss += loss_sum.item()
        total_weight += float(batch_weight)

        probs = torch.softmax(
            outputs.float(),
            dim=1
        )

        y_true.extend(
            labels.cpu().tolist()
        )

        y_pred.extend(
            probs.argmax(dim=1)
            .cpu()
            .tolist()
        )

        if return_predictions:

            probabilities.append(
                probs.cpu()
            )

    metrics = calculate_metrics(
        y_true,
        y_pred,
        num_classes
    )

    metrics.update({
        "loss": total_loss / total_weight,
        "seconds": time.perf_counter() - start,
        "num_samples": len(y_true)
    })

    if return_predictions:

        predictions = {
            "y_true": np.asarray(y_true),
            "y_pred": np.asarray(y_pred),
            "probabilities": torch.cat(
                probabilities,
                dim=0
            ).numpy()
        }

        return metrics, predictions

    return metrics


def save_classifier_checkpoint(
    classifier,
    filepath,
    metadata
):

    filepath = Path(filepath)

    filepath.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint = {
        "classifier_state_dict": {
            key: value.detach().cpu().clone()
            for key, value
            in classifier.state_dict().items()
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

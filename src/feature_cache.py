from pathlib import Path
import gc
import time
import hashlib

import torch
from torch import nn
from torch.utils.data import TensorDataset

from .model_factory import create_model
from .dataset import CassavaDataset
from .data_generator import create_dataloader


FEATURE_DIMS = {
    "mobilenet_v3_large": 960,
    "efficientnet_b0": 1280,
    "resnet50": 2048,
    "vit_b_16": 768,
    "swin_t": 768
}


CLASSIFIER_ATTRIBUTES = {
    "mobilenet_v3_large": "classifier",
    "efficientnet_b0": "classifier",
    "resnet50": "fc",
    "vit_b_16": "heads",
    "swin_t": "head"
}


def calculate_sha256(filepath):

    sha256 = hashlib.sha256()

    with open(filepath, "rb") as file:

        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):

            sha256.update(chunk)

    return sha256.hexdigest()


def dataframe_fingerprint(dataframe):

    columns = [
        "filepath",
        "class",
        "fold",
        "split",
        "sha256"
    ]

    content = dataframe[columns].to_csv(
        index=False
    )

    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()


def release_memory():

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def create_feature_extractor(
    model_name,
    num_classes,
    device
):

    model = create_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=True,
        dropout=0.2,
        mode="feature_extraction",
        device=device
    )

    attribute = CLASSIFIER_ATTRIBUTES[
        model_name
    ]

    setattr(
        model,
        attribute,
        nn.Identity()
    )

    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()

    return model


def create_classifier(
    model_name,
    num_classes,
    dropout=0.2
):

    input_dim = FEATURE_DIMS[model_name]

    if model_name == "mobilenet_v3_large":

        classifier = nn.Sequential(
            nn.Linear(input_dim, 1280),
            nn.Hardswish(),
            nn.Dropout(p=dropout),
            nn.Linear(1280, num_classes)
        )

    else:

        classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(input_dim, num_classes)
        )

    return classifier


@torch.inference_mode()
def extract_features(
    extractor,
    dataloader,
    device,
    feature_dim,
    use_amp=True
):

    extractor.eval()

    use_amp = (
        use_amp and device.type == "cuda"
    )

    features = torch.empty(
        (len(dataloader.dataset), feature_dim),
        dtype=torch.float32
    )

    labels = torch.empty(
        len(dataloader.dataset),
        dtype=torch.long
    )

    offset = 0

    for batch_idx, (images, targets) in enumerate(
        dataloader
    ):

        images = images.to(
            device,
            non_blocking=True
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp
        ):

            outputs = extractor(images)

        outputs = outputs.float().cpu()

        assert outputs.ndim == 2

        assert outputs.shape[1] == feature_dim

        if not torch.isfinite(outputs).all():

            raise FloatingPointError(
                "Características no finitas."
            )

        batch_size = len(targets)

        features[
            offset:offset + batch_size
        ] = outputs

        labels[
            offset:offset + batch_size
        ] = targets

        offset += batch_size

        if (batch_idx + 1) % 100 == 0:

            print(
                f"  Batch {batch_idx + 1}/"
                f"{len(dataloader)}",
                flush=True
            )

    assert offset == len(dataloader.dataset)

    return features, labels


def build_feature_cache(
    model_name,
    dataframe,
    dataset_dir,
    class_mapping,
    eval_transform,
    cache_path,
    device,
    batch_size=32,
    num_workers=0,
    seed=42,
    use_amp=True,
    cache_metadata=None
):

    cache_path = Path(cache_path)

    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    dataframe = dataframe.reset_index(
        drop=True
    ).copy()

    expected_metadata = {
        "architecture": model_name,
        "num_classes": len(class_mapping),
        "class_mapping": class_mapping,
        "feature_dim": FEATURE_DIMS[model_name],
        "num_samples": len(dataframe),
        "dataframe_hash": dataframe_fingerprint(
            dataframe
        ),
        "batch_size": batch_size,
        "mixed_precision": (
            use_amp and device.type == "cuda"
        ),
        "feature_dtype": "float32",
        "seed": seed,
        "extra": cache_metadata or {}
    }

    if cache_path.exists():

        cache = torch.load(
            cache_path,
            map_location="cpu",
            weights_only=True
        )

        assert cache["metadata"] == expected_metadata, (
            "La caché no coincide con la "
            "configuración actual."
        )

        assert cache["filepaths"] == (
            dataframe["filepath"].tolist()
        )

        assert cache["features"].shape == (
            len(dataframe),
            FEATURE_DIMS[model_name]
        )

        assert torch.isfinite(
            cache["features"]
        ).all()

        expected_labels = torch.tensor(
            dataframe["class"]
            .map(class_mapping)
            .tolist(),
            dtype=torch.long
        )

        assert torch.equal(
            cache["labels"],
            expected_labels
        )

        print(
            f"Caché recuperada: {model_name}"
        )

        return cache

    start = time.perf_counter()

    dataset = CassavaDataset(
        dataframe=dataframe,
        dataset_dir=dataset_dir,
        class_mapping=class_mapping,
        transform=eval_transform
    )

    dataloader = create_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        seed=seed
    )

    extractor = create_feature_extractor(
        model_name=model_name,
        num_classes=len(class_mapping),
        device=device
    )

    features, labels = extract_features(
        extractor=extractor,
        dataloader=dataloader,
        device=device,
        feature_dim=FEATURE_DIMS[model_name],
        use_amp=use_amp
    )

    expected_labels = torch.tensor(
        dataframe["class"]
        .map(class_mapping)
        .tolist(),
        dtype=torch.long
    )

    assert torch.equal(
        labels,
        expected_labels
    )

    extraction_seconds = (
        time.perf_counter() - start
    )

    cache = {
        "features": features,
        "labels": labels,
        "filepaths": dataframe["filepath"].tolist(),
        "metadata": expected_metadata,
        "extraction_seconds": extraction_seconds
    }

    temporary_path = cache_path.with_suffix(
        ".tmp"
    )

    torch.save(
        cache,
        temporary_path
    )

    temporary_path.replace(cache_path)

    del extractor
    del dataloader
    del dataset

    release_memory()

    print(
        f"{model_name} | "
        f"{features.shape} | "
        f"{extraction_seconds / 60:.2f} min"
    )

    return cache


def create_cached_dataset(
    cache,
    indices
):

    indices = torch.as_tensor(
        indices,
        dtype=torch.long
    )

    return TensorDataset(
        cache["features"][indices],
        cache["labels"][indices]
    )

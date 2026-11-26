import random

import numpy as np
import torch

from torch.utils.data import DataLoader

from .dataset import CassavaDataset


def seed_worker(worker_id):

    worker_seed = torch.initial_seed() % 2**32

    np.random.seed(worker_seed)

    random.seed(worker_seed)


def create_dataloader(
    dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    seed=42
):

    generator = torch.Generator()

    generator.manual_seed(seed)

    loader = DataLoader(

        dataset,

        batch_size=batch_size,

        shuffle=shuffle,

        num_workers=num_workers,

        pin_memory=torch.cuda.is_available(),

        drop_last=False,

        worker_init_fn=seed_worker,

        generator=generator

    )

    return loader


def create_fold_loaders(
    df_splits,
    dataset_dir,
    class_mapping,
    train_transform,
    eval_transform,
    fold,
    batch_size=32,
    num_workers=0,
    seed=42
):

    df_development = df_splits[
        df_splits["split"] == "development"
    ]

    available_folds = set(
        df_development["fold"].unique()
    )

    if fold not in available_folds:

        raise ValueError(
            f"Fold inválido: {fold}. "
            f"Disponibles: {available_folds}"
        )

    df_train = df_development[
        df_development["fold"] != fold
    ].copy()

    df_val = df_development[
        df_development["fold"] == fold
    ].copy()

    train_dataset = CassavaDataset(

        dataframe=df_train,

        dataset_dir=dataset_dir,

        class_mapping=class_mapping,

        transform=train_transform

    )

    val_dataset = CassavaDataset(

        dataframe=df_val,

        dataset_dir=dataset_dir,

        class_mapping=class_mapping,

        transform=eval_transform

    )

    train_loader = create_dataloader(

        dataset=train_dataset,

        batch_size=batch_size,

        shuffle=True,

        num_workers=num_workers,

        seed=seed

    )

    val_loader = create_dataloader(

        dataset=val_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=num_workers,

        seed=seed

    )

    return train_loader, val_loader


def create_test_loader(
    df_splits,
    dataset_dir,
    class_mapping,
    eval_transform,
    batch_size=32,
    num_workers=0,
    seed=42
):

    df_test = df_splits[
        df_splits["split"] == "test"
    ].copy()

    test_dataset = CassavaDataset(

        dataframe=df_test,

        dataset_dir=dataset_dir,

        class_mapping=class_mapping,

        transform=eval_transform

    )

    test_loader = create_dataloader(

        dataset=test_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=num_workers,

        seed=seed

    )

    return test_loader
from .dataset import CassavaDataset

from .data_generator import create_dataloader


def create_client_loader(

    df_federated,

    dataset_dir,

    class_mapping,

    train_transform,

    client_id,

    fold,

    batch_size=32,

    num_workers=0,

    seed=42

):

    df_development = df_federated[
        df_federated["split"] == "development"
    ]

    if client_id not in set(
        df_development["client_id"]
    ):

        raise ValueError(
            f"Cliente inválido: {client_id}"
        )

    if fold not in set(df_development["fold"]):

        raise ValueError(
            f"Fold inválido: {fold}"
        )

    df_client = df_development[

        (df_development["client_id"] == client_id) &

        (df_development["fold"] != fold)

    ].copy()

    if df_client.empty:

        raise ValueError(
            f"El cliente {client_id} no tiene "
            f"imágenes de entrenamiento para el fold {fold}."
        )

    client_dataset = CassavaDataset(

        dataframe=df_client,

        dataset_dir=dataset_dir,

        class_mapping=class_mapping,

        transform=train_transform

    )

    client_loader = create_dataloader(

        dataset=client_dataset,

        batch_size=batch_size,

        shuffle=True,

        num_workers=num_workers,

        seed=seed + client_id

    )

    return client_loader


def create_global_val_loader(

    df_federated,

    dataset_dir,

    class_mapping,

    eval_transform,

    fold,

    batch_size=32,

    num_workers=0,

    seed=42

):

    df_val = df_federated[

        (df_federated["split"] == "development") &

        (df_federated["fold"] == fold)

    ].copy()

    if df_val.empty:

        raise ValueError(
            f"No existen imágenes de validación "
            f"para el fold {fold}."
        )

    val_dataset = CassavaDataset(

        dataframe=df_val,

        dataset_dir=dataset_dir,

        class_mapping=class_mapping,

        transform=eval_transform

    )

    val_loader = create_dataloader(

        dataset=val_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=num_workers,

        seed=seed

    )

    return val_loader
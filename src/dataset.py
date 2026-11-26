from pathlib import Path

import torch

from torch.utils.data import Dataset

from .preprocessing import load_image


class CassavaDataset(Dataset):

    def __init__(
        self,
        dataframe,
        dataset_dir,
        class_mapping,
        transform=None
    ):

        self.dataframe = dataframe.reset_index(
            drop=True
        ).copy()

        self.dataset_dir = Path(dataset_dir)

        self.class_mapping = class_mapping

        self.transform = transform


    def __len__(self):

        return len(self.dataframe)


    def __getitem__(self, idx):

        row = self.dataframe.iloc[idx]

        image_path = (
            self.dataset_dir / row["filepath"]
        )

        image = load_image(image_path)

        class_name = row["class"]

        label = self.class_mapping[class_name]

        if self.transform is not None:

            image = self.transform(image)

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return image, label
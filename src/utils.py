import os

os.environ.setdefault(
    "CUBLAS_WORKSPACE_CONFIG",
    ":4096:8"
)

import random

import numpy as np

import torch


def set_seed(seed=42, deterministic=True):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = (
        deterministic
    )

    torch.backends.cudnn.benchmark = (
        not deterministic
    )

    torch.use_deterministic_algorithms(
        deterministic,
        warn_only=False
    )


def get_device():

    return torch.device(

        "cuda" if torch.cuda.is_available() else "cpu"

    )
from dataclasses import dataclass

import numpy as np


@dataclass
class Dataset:
    xs: np.ndarray
    ys: np.ndarray

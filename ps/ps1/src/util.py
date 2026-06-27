import numpy as np # type: ignore
from typing import Tuple, Optional
import os

def add_intercept(x: np.ndarray) -> np.ndarray:
    """Add intercept to matrix x.
    
    Args:
        x: 2D NumPy array.

    Returns:
        Augmented matrix (1 | x).
    """
    return np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)

def load_dataset(csv_path: str, label_col: str = 'y', add_intercept: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Load dataset from a csv file.

    Args:
        csv_path: Path to CSV file containing dataset.
        label_col: Name of column to use as labels(should be 'y' or 't')
        add_intercept: Add an intercept entry to x-values.

    Returns:
        xs: NumPy array of x-values (inputs).
        ys: NumPy array of y-values (labels).
    """

    # Validate label_col argument
    allowed_label_col = ['y', 't']
    if label_col not in allowed_label_col:
        raise ValueError("Invalue label_col: {} (expected {})".format(label_col, allowed_label_col))
    
    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    t_cols = [i for i in range(len(headers)) if headers[i] == label_col]
    inputs = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols)
    labels = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=t_cols)

    if inputs.ndim == 1:
        inputs = np.expand_dims(inputs, -1)

    def add_intercept_fn(inputs):
        global add_intercept
        return add_intercept(inputs)

    if add_intercept:
        inputs = add_intercept_fn(inputs)

    return inputs, labels
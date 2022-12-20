"""
"""
from typing import Any, Union, Optional

import numpy as np


def accuracy(preds: np.ndarray, y: np.ndarray):
    """Compute the accuracy of classification, assuming y is a one-hot encoded
    vector.

    :param preds: the raw predictions from the model.
    :param y: the one-hot encoded true targets/labels
    :returns: accuracy score
    """

    class_labels = np.argmax(y, 1)
    class_preds = np.argmax(preds, 1)
    return np.sum(class_preds == class_labels) / y.shape[0]


# penalty functions


def l1_penalty(w: np.ndarray, lam: float) -> float:
    """"""
    return lam * np.sum(np.abs(w))


def l2_penalty(w: np.ndarray, lam: float) -> float:
    """"""

    return (lam / 2) * np.sum(w**2)


def group_l1_penalty(w: np.ndarray, lam: Union[np.ndarray, float]) -> float:
    """Compute the penalty associated with the regularizer.

    :param w: the parameter at which to compute group l1 penalty. Note that 'w'
        must have shape (x, P), where 'P' is the number of groups.
    :param lam: the coefficient(s) controlling the strength of regularization.
        IF 'lam' is a numpy array, it must have shape (P,).
    :returns: penalty value
    """

    return lam * np.sum(np.sqrt(np.sum(w**2, axis=-1)))


def l1_squared_penalty(w: np.ndarray, lam: Union[np.ndarray, float]) -> float:
    """Compute the penalty associated with the regularizer.

    :param w: the parameter at which to compute group l1 penalty. Note that 'w'
        must have shape (x, P), where 'P' is the number of groups.
    :param lam: the coefficient(s) controlling the strength of regularization.
        IF 'lam' is a numpy array, it must have shape (P,).
    :returns: penalty value
    """

    return (lam / 2) * np.sum(np.sum(np.abs(w), axis=0) ** 2)


# "activation" functions


def relu(x: np.ndarray) -> np.ndarray:
    """Compute ReLU activation,

        $max(x, 0)$
    :param x: pre-activations
    """
    # what's the issue here?

    return np.maximum(x, 0)


def logistic_fn(x: np.ndarray) -> np.ndarray:
    """Compute logistic activation,

        $1 / (1 + exp(-x))$
    :param x: pre-activations
    """
    return 1.0 / (1.0 + np.exp(-x))


def safe_divide(x: np.ndarray, y: np.ndarray):
    return np.divide(x, y, out=np.zeros_like(x), where=(y != 0))

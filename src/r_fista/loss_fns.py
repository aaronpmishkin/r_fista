"""
"""
from typing import Tuple, Callable
from functools import partial
from scipy.special import logsumexp  # type: ignore

import numpy as np

from .helper_fns import logistic_fn


def squared_error_obj(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """ """
    residuals = X @ w - y
    return np.sum((residuals) ** 2) / (2 * len(y))


def squared_error_grad(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """ """
    residuals = X @ w - y

    return (X.T @ residuals) / len(y)


def get_squared_error_closures(
    X: np.ndarray, y: np.ndarray
) -> Tuple[Callable, Callable]:
    return partial(squared_error_obj, X=X, y=y), partial(squared_error_grad, X=X, y=y)


def logistic_obj(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """ """
    logits = X @ w
    res = logsumexp(np.stack([np.zeros_like(logits), -y * logits]), axis=0)

    return np.sum(res) / len(y)


def logistic_grad(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """ """
    logits = X @ w
    return -X.T @ np.multiply(y, logistic_fn(-y * logits)) / len(y)


def get_logistic_closures(X: np.ndarray, y: np.ndarray) -> Tuple[Callable, Callable]:
    return partial(logistic_obj, X=X, y=y), partial(logistic_grad, X=X, y=y)

"""Proximal operators. This module provides functions for solving minimization
problems of the form.

    $argmin_x { d(x,w) + beta * g(x) }$,

where d is a metric, g is a "simple" function, and beta is a parameter controlling the trade-off between d and g.
"""

from typing import Optional, Tuple
import math

import numpy as np

from .helper_fns import safe_divide


class ProximalOperator:

    """Base class for proximal operators."""

    def __call__(self, w: np.ndarray, beta: Optional[float] = None) -> np.ndarray:
        """Evaluate the proximal_operator.

        :param w: parameters to which apply the operator will be applied.
        :param beta: the coefficient in the proximal operator. This is usually a step-size.
        :returns: prox(w)
        """

        raise NotImplementedError("A proximal operator must implement '__call__'!")


class Identity(ProximalOperator):
    """The proximal-operator for the zero-function.

    This proximal-operator always returns the input point.
    """

    def __call__(self, w: np.ndarray, beta: Optional[float] = None) -> np.ndarray:
        """Evaluate the identity operator.

        Args:
            w: the point at which the evaluate the operator.
            beta: the step-size for the proximal step. NOT USED.

        Returns:
            w, the original input point.
        """

        return w


class Regularizer(ProximalOperator):

    """Base class for proximal operators based on regularizers.

    Attributes:
        lam: the regularization strength. This must be non-negative.
    """

    def __init__(self, lam: float):
        """Initialize the proximal operator.

        Args:
            lam: a non-negative parameter controlling the regularization strength.
        """
        self.lam = lam


class L2(Regularizer):
    """The proximal operator for the squared l2-norm.

    The proximal operator returns the unique solution to the following optimization problem:

    .. math:: \\min_x \\{\\|x - w\\|_2^2 + \\frac{\\beta * \\lambda}{2} \\|x\\|_2^2\\}.

    The solution is the shrinkage operator :math:`x^* = (1 + \\beta)^{-1} w`.
    """

    def __call__(self, w: np.ndarray, beta: float) -> np.ndarray:
        """Evaluate the proximal operator given a point and a step-size.

        Args:
            w: the point at which the evaluate the operator.
            beta: the step-size for the proximal step.

        Returns:
            prox(w), the result of the proximal operator.
        """

        return w / (1 + beta * self.lam)


class L1(Regularizer):

    """The proximal operator for the l1-norm.

    The l1 proximal operator is sometimes known as the soft-thresholding operator and
    is the unique solution to the following problem:

    .. math:: \\min_x \\{\\|x - w\\|_2^2 + \\beta * \\lambda \\|x\\|_1\\}.
    """

    def __call__(self, w: np.ndarray, beta: float) -> np.ndarray:
        """Evaluate the proximal operator given a point and a step-size.

        Args:
            w: the point at which the evaluate the operator.
            beta: the step-size for the proximal step.

        Returns:
            prox(w), the result of the proximal operator.
        """

        return np.sign(w) * np.maximum(np.abs(w) - beta * self.lam, 0.0)


class GroupL1(Regularizer):

    """The proximal operator for the group-l1 regularizer.

    Given group indices :math:`\\calI`, the group-l1 regularizer is the sum of
    l2-norms of the groups, :math:`r(w) = \\sum_{i \\in \\calI} \\|w_i\\|_2`.
    The proximal operator is thus the unique solution to the following problem,

    .. math:: \\min_x \\{\\|x - w\\|_2^2 + \\beta * \\lambda \\sum_{i=1 \\in \\calI} \\|x_i\\|_2\\}.

    Groups are either defined to be the last axis of the point w.
    """

    def __init__(self, lam: float, group_shape: Tuple[int, int]):
        """Initialize the proximal operator.

        Args:
            lam: a non-negative parameter controlling the regularization strength.
        """
        self.lam = lam
        self.group_shape = group_shape

    def __call__(self, w: np.ndarray, beta: float) -> np.ndarray:
        """Evaluate the proximal operator given a point and a step-size.

        Args:
            w: the point at which the evaluate the operator.
            beta: the step-size for the proximal step.

        Returns:
            prox(w), the result of the proximal operator.
        """
        # compute the squared norms of each group.
        w_shape = w.shape
        w.reshape(self.group_shape)

        norms = np.sqrt(np.sum(w**2, axis=-1, keepdims=True))

        w_plus = np.multiply(
            safe_divide(w, norms), np.maximum(norms - self.lam * beta, 0)
        )

        return w_plus.reshape(w_shape)

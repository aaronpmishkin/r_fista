"""
"""
from typing import Optional, Tuple

import numpy as np

from .helper_fns import l1_penalty, l2_penalty, group_l1_penalty, safe_divide


class Regularizer:

    """Base class for regularizers."""

    lam: float

    def __init__(
        self,
        lam: float,
    ):
        """
        :param lam: the tuning parameter controlling the strength of regularization.
        """

        self.lam = lam

    # abstract methods that must be overridden

    def penalty(self, w: np.ndarray) -> float:
        """Compute the penalty associated with the regularizer.

        :param w: parameter at which to compute the penalty value.
        :returns: penalty value
        """

        raise NotImplementedError("A regularizer must implement 'penalty'!")

    def subgradient(
        self,
        w: np.ndarray,
        base_grad: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute the gradient of the regularizer.

        :param w: parameter at which to compute the penalty gradient.
        :param base_grad: (optional) the gradient of the un-regularized objective. This is
            used to compute the minimum-norm subgradient for "pseudo-gradient" methods.
        :returns: gradient.
        """

        raise NotImplementedError("A regularizer must implement 'grad'!")


class L1Regularizer(Regularizer):
    """The l1-regularizer, which takes the form.

    .. math:: r(w) = \\lambda \\|w\\||_2^2.

    This regularizer is non-smooth at :math:`w = 0`.

    Attributes:
        lam: the regularization strength. Must be non-negative.
    """

    def penalty(self, w: np.ndarray) -> float:
        """Compute the penalty associated with the regularizer.

        Args:
            w: parameter at which to compute the penalty.
        """

        return l1_penalty(w, self.lam)

    def subgradient(
        self,
        w: np.ndarray,
        base_grad: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute the minimum-norm sub-gradient of the l1 regularizer.

        Args:
            w: parameter at which to compute the penalty gradient.
            base_grad: the smooth component of the gradient, coming from the objective function.

        Returns:
            The minimum-norm subgradient.
        """
        indicators = w != 0

        smooth_term = np.sign(w) * self.lam * indicators

        non_smooth_term = (
            np.sign(base_grad)
            * np.minimum(np.abs(base_grad), self.lam)
            * np.logical_not(indicators)
        )

        return smooth_term - non_smooth_term

    def prox(self, w: np.ndarray, beta: float) -> np.ndarray:
        """Evaluate the proximal operator given a point and a step-size.

        Args:
            w: the point at which the evaluate the operator.
            beta: the step-size for the proximal step.

        Returns:
            prox(w), the result of the proximal operator.
        """

        return np.sign(w) * np.maximum(np.abs(w) - beta * self.lam, 0.0)


class L2Regularizer(Regularizer):

    """L2-regularizer of the form.

    $f(w) = (lambda/2) * ||w||_2^2$
    """

    def penalty(self, w: np.ndarray) -> float:
        """Compute the penalty associated with the regularizer.

        :param w: parameter at which to compute the penalty.
        :returns: penalty value
        """

        return l2_penalty(w, self.lam)

    def subgradient(
        self,
        w: np.ndarray,
        base_grad: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute the gradient of the regularizer.

        :param w: parameter at which to compute the penalty gradient.
        :param base_grad: NOT USED. Gradient from the base model.
        :returns: gradient.
        """

        return self.lam * w

    def prox(self, w: np.ndarray, beta: float) -> np.ndarray:
        """Evaluate the proximal operator given a point and a step-size.

        Args:
            w: the point at which the evaluate the operator.
            beta: the step-size for the proximal step.

        Returns:
            prox(w), the result of the proximal operator.
        """

        return w / (1 + beta * self.lam)


class GroupL1Regularizer(Regularizer):

    """The group-sparsity inducing Group L1-regularizer.

    The group-L1 regularizer, sometimes called the L1-L2 regularizer,
    has the mathematical form,

        ::math.. R(w) = \\lambda \\sum_{i \\in \\mathcal{I} ||w_i||_2,

    where :math:`\\mathcal{I}` is collection of disjoint index sets specifying
    the groups for the regularizer. This class expects the final axis of the
    inputs/weights :math:`w` to be the group axis.

    The group-L1 regularizer induces group-sparsity, meaning entire groups
    :math:`w_i` will be set to zero when :math:`\\lambda` is sufficiently
    large.

    Attributes:
        lam: the regularization strength.

    """

    def __init__(self, lam: float, group_shape: Tuple[int, int]):
        """
        lam: a tuning parameter controlling the regularization strength.
        """

        self.lam = lam
        self.group_shape = group_shape

    def penalty(
        self,
        w: np.ndarray,
    ) -> float:
        """Compute the penalty associated with the regularizer.

        Args:
            w: parameter at which to compute the penalty.

        Returns:
            The value of the penalty at w.
        """
        return group_l1_penalty(w.reshape(self.group_shape), self.lam)

    def subgradient(
        self,
        w: np.ndarray,
        base_grad: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute the minimum-norm subgradient (aka, the pseudo-gradient).

        Args:
            w: parameter at which to compute the penalty gradient.
            base_grad: the gradient of the un-regularized objective.
                This is required to compute the minimum-norm subgradient.

        Returns:
            minimum-norm subgradient.
        """
        # requires base_grad to compute minimum-norm subgradient
        assert base_grad is not None
        w_shape = w.shape
        w = w.reshape(self.group_shape)
        base_grad = base_grad.reshape(self.group_shape)

        weight_norms = np.sqrt(np.sum(w**2, axis=-1, keepdims=True))
        grad_norms = np.sqrt(np.sum(base_grad**2, axis=-1, keepdims=True))

        non_smooth_term = (
            base_grad * np.minimum(self.lam / grad_norms, 1) * (weight_norms == 0)
        )
        smooth_term = self.lam * safe_divide(w, weight_norms)

        # match input shape
        subgrad = smooth_term - non_smooth_term

        return subgrad.reshape(w_shape)

    def prox(self, w: np.ndarray, beta: float) -> np.ndarray:
        """Evaluate the proximal operator given a point and a step-size.

        Args:
            w: the point at which the evaluate the operator.
            beta: the step-size for the proximal step.

        Returns:
            prox(w), the result of the proximal operator.
        """
        # compute the squared norms of each group.
        w_shape = w.shape
        w = w.reshape(self.group_shape)

        norms = np.sqrt(np.sum(w**2, axis=-1, keepdims=True))

        w_plus = np.multiply(
            safe_divide(w, norms), np.maximum(norms - self.lam * beta, 0)
        )

        return w_plus.reshape(w_shape)

"""
"""

from typing import Callable, Dict, Tuple, Any
from functools import partial
import timeit

import numpy as np
from tqdm.auto import tqdm  # type: ignore

from .regularizers import Regularizer

MAX_ATTEMPTS: int = 50
FAILURE_STEP: float = 1e-8


def proximal_gradient_step(
    w: np.ndarray,
    descent_dir: np.ndarray,
    step_size: float,
    prox: Callable,
) -> np.ndarray:
    """Take one step of proximal gradient descent.

    :param w: the parameters to be updated.
    :param grad: the gradient of the smooth component of the loss function.
    :param step_size: the step-size to use.
    :param prox: the proximal operator. It must take a step-size parameter.
    :returns: updated parameters
    """

    w_plus = w - step_size * descent_dir
    return prox(w_plus, step_size)


def quadratic_bound(
    f0: float,
    f1: float,
    step: np.ndarray,
    grad: np.ndarray,
    step_size: float,
) -> bool:
    """Check the Armijo or sufficient progress condition, which is.

        $f(x_{k+1} ≤ f(x_k) + <grad, step> + eta/2 ||step||_2^2$.
    :param f0: previous objective value, f(x_k).
    :param f1: new objective value, f(x_{k+1}).
    :param step: the (descent) step to check against the Armijo condition,
        ie. the difference between the new iterate and the previous one,
            $step = x_{k+1} - x_k$.
    :param grad: the grad at which to check the Armijo condition. Must have the same shape as 'step'.
    :param step_size: the step-size used to generate the step.
    :returns: boolean indicating whether or not the Armijo condition holds.
    """

    return 2.0 * step_size * (f1 - f0 - np.dot(step.ravel(), grad.ravel())) < np.dot(
        step, step
    )


# line-search
def fista_ls_iter(
    w: np.ndarray,
    f0: float,
    grad: np.ndarray,
    obj_fn: Callable,
    init_step_size: float,
    beta: float,
    prox: Callable,
    v: np.ndarray,
    t: float,
    mu: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, float, float, float, Dict[str, Any]]:
    """Take one step of proximal gradient descent using a line-search to pick
    the step-size.

    :param w: the parameters to be updated.
    :param f0: the objective function evaluated at w.
    :param grad: the gradient of the smooth component of the loss function.
    :param obj_fn: function that returns the objective when called at w.
    :param init_step_size: the first step-size to use.
    :param beta: multiplicate decrease parameter for the step-size.
    :param prox: the proximal operator. It must take a step-size parameter.
    :param v: the extrapolation sequence.
    :param t: the extrapolation parameter.
    :param mu: (optional) a lower-bound on the strong-convexity parameter of the objective.
        The method defaults to the parameter sequence for non-strongly convex functions when
        mu is not supplied.
    :returns: (w_next, f1, step_size, exit_status): the updated parameters, objective value, new step-size, and exit state.
    """

    success = True
    step_size = init_step_size
    test_fn = partial(proximal_gradient_step, prox=prox)

    # run line-search
    w_plus = test_fn(v, grad, step_size)
    f1 = obj_fn(w_plus)
    attempts = 1

    # line-search
    while not quadratic_bound(f0, f1, w_plus - v, grad, step_size):
        step_size = step_size * beta
        w_plus = test_fn(v, grad, step_size)
        f1 = obj_fn(w_plus)
        attempts += 1

        # exceeded max attempts
        if attempts > MAX_ATTEMPTS:
            step_size = FAILURE_STEP
            w_plus = test_fn(v, grad, step_size)
            success = False
            break

    exit_status = {"attempts": attempts, "success": success, "step_size": step_size}

    # release memory
    t_plus = t

    # acceleration
    if mu == 0.0:
        t_plus = 1 + np.sqrt(1 + 4 * t**2) / 2
        beta = (t - 1) / t_plus
    else:
        sqrt_kappa = np.sqrt(1 / (step_size * mu))
        beta = (sqrt_kappa - 1) / (sqrt_kappa + 1)

    v = w_plus + beta * (w_plus - w)

    return w_plus, v, t_plus, f1, step_size, exit_status


def fista(
    w0: np.ndarray,
    obj_fn: Callable,
    grad_fn: Callable,
    regularizer: Regularizer,
    init_step_size: float,
    beta: float,
    max_iters: int,
    tol: float,
    mu: float = 0.0,
    use_restarts: bool = True,
    log_freq: int = 50,
    verbose: bool = False,
):

    # initialization
    w = v = w0
    f0 = f1 = obj_fn(w)
    grad = grad_fn(w)
    t = 1.0
    step_size = init_step_size
    alpha = 1.0 / beta

    exit_status = {}

    start_time = timeit.default_timer()
    for itr in tqdm(range(max_iters), disable=(not verbose)):
        min_norm_subgrad = grad + regularizer.subgradient(w, grad)
        subgrad_norm = np.dot(min_norm_subgrad, min_norm_subgrad)

        if itr % log_freq == 0 and verbose:
            time_elapsed = timeit.default_timer() - start_time
            penalty = regularizer.penalty(w)
            tqdm.write(
                f"Obj: {f0 + penalty}, Subgrad Norm: {subgrad_norm}, Time: {time_elapsed}"
            )

        # check termination criteria
        if subgrad_norm <= tol:
            exit_status["success"] = True
            exit_status["iterations"] = itr + 1
            exit_status["elapsed_time"] = timeit.default_timer() - start_time

            print(
                f"Termination criterion satisfied at iteration {itr}/{max_iters}. Exiting optimization loop."
            )
            break

        # fv: function value at extrapolation.
        fv = obj_fn(v)
        # f0: function value at previous parameter
        f0 = f1
        # previous iterates and extrapolation steps
        w0 = w
        v0 = v

        # update model
        (w, v, t, f1, step_size, ls_exit_status) = fista_ls_iter(
            w,
            fv,
            grad_fn(v),
            obj_fn,
            step_size,
            beta,
            regularizer.prox,
            v,
            t,
            mu,
        )

        # update gradient
        grad = grad_fn(w)

        if not ls_exit_status["success"]:
            print(f"Warning: Line-search failed on iteration {itr}")

        # compute step displacement
        step = w - v0

        # increase step-size if necessary
        denom = (f1 - (f0 + np.dot(grad.ravel(), step.ravel()))) * 2 * step_size
        gap = np.sum(step**2)

        if gap > 5.0 * denom:
            step_size = step_size * alpha

        # restart if necessary
        if use_restarts and np.sum((w - w0) * step) < 0.0:
            t = 1.0
            v = w

    if itr == max_iters - 1:
        print(
            "Warning: Max iterations reached before termination criterion was satisfied. Exiting optimization loop."
        )
        exit_status["success"] = False
        exit_status["iterations"] = itr + 1
        exit_status["elapsed_time"] = timeit.default_timer() - start_time

    return w, exit_status

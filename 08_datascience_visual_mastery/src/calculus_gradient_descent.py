"""Differential calculus -> gradient descent teaching simulation.

Shows that the derivative of a function at a point is the slope of the
tangent line there, and that gradient descent is nothing more than
repeatedly stepping "downhill" along that slope: x_{t+1} = x_t - lr * f'(x_t).
"""
from __future__ import annotations

import numpy as np

# A non-convex 1-D "loss landscape": two local minima so that learning-rate
# and starting-point choices visibly matter (a purely convex bowl hides that).
def f(x: np.ndarray | float):
    return 0.15 * x ** 4 - 0.9 * x ** 2 + 0.3 * x + 3


def f_prime(x: np.ndarray | float):
    """Analytic derivative of f, i.e. df/dx."""
    return 0.6 * x ** 3 - 1.8 * x + 0.3


def numerical_derivative(x: float, h: float = 1e-4) -> float:
    """Central-difference approximation, shown alongside the analytic
    derivative so students see the two agree: (f(x+h) - f(x-h)) / 2h."""
    return (f(x + h) - f(x - h)) / (2 * h)


def tangent_line(x0: float, xs: np.ndarray) -> np.ndarray:
    """y = f(x0) + f'(x0) * (x - x0), the first-order Taylor approximation."""
    slope = f_prime(x0)
    return f(x0) + slope * (xs - x0)


def gradient_descent(start: float, learning_rate: float, steps: int) -> list[float]:
    """Return the trajectory x_0, x_1, ..., x_steps of gradient descent."""
    path = [start]
    x = start
    for _ in range(steps):
        x = x - learning_rate * f_prime(x)
        path.append(x)
    return path

"""Chain rule -> backpropagation teaching simulation.

A minimal one-hidden-unit network:

    z1 = w1*x + b1
    a1 = tanh(z1)
    yhat = w2*a1 + b2
    L = 0.5 * (yhat - y)^2

Backprop is just the chain rule applied layer by layer, back to front:

    dL/dyhat = (yhat - y)
    dL/dw2   = dL/dyhat * d(yhat)/dw2      = dL/dyhat * a1
    dL/da1   = dL/dyhat * d(yhat)/da1      = dL/dyhat * w2
    dL/dz1   = dL/da1   * d(a1)/dz1        = dL/da1   * (1 - tanh(z1)^2)
    dL/dw1   = dL/dz1   * d(z1)/dw1        = dL/dz1   * x

Every gradient below is one factor in a chain-rule product, kept separate on
purpose so the app can print the chain term-by-term.
"""
from __future__ import annotations

import numpy as np


def forward(x: float, y: float, w1: float, b1: float, w2: float, b2: float) -> dict:
    z1 = w1 * x + b1
    a1 = np.tanh(z1)
    yhat = w2 * a1 + b2
    loss = 0.5 * (yhat - y) ** 2
    return {"x": x, "y": y, "z1": z1, "a1": a1, "yhat": yhat, "loss": loss,
            "w1": w1, "b1": b1, "w2": w2, "b2": b2}


def backward(cache: dict) -> dict:
    """Chain-rule gradients, each labelled with the local derivative it
    multiplies in, matching the docstring derivation above."""
    dL_dyhat = cache["yhat"] - cache["y"]
    dyhat_dw2 = cache["a1"]
    dyhat_db2 = 1.0
    dyhat_da1 = cache["w2"]

    dL_dw2 = dL_dyhat * dyhat_dw2
    dL_db2 = dL_dyhat * dyhat_db2
    dL_da1 = dL_dyhat * dyhat_da1

    da1_dz1 = 1 - np.tanh(cache["z1"]) ** 2  # tanh'(z) = 1 - tanh(z)^2
    dL_dz1 = dL_da1 * da1_dz1

    dz1_dw1 = cache["x"]
    dz1_db1 = 1.0
    dL_dw1 = dL_dz1 * dz1_dw1
    dL_db1 = dL_dz1 * dz1_db1

    return {
        "dL_dyhat": dL_dyhat, "dL_dw2": dL_dw2, "dL_db2": dL_db2,
        "dL_da1": dL_da1, "da1_dz1": da1_dz1, "dL_dz1": dL_dz1,
        "dL_dw1": dL_dw1, "dL_db1": dL_db1,
    }


def train(x_data: np.ndarray, y_data: np.ndarray, learning_rate: float,
          epochs: int, seed: int = 3) -> dict:
    """Full-batch gradient descent using the hand-derived chain-rule
    gradients above. Returns the loss history and final parameters."""
    rng = np.random.default_rng(seed)
    w1, b1, w2, b2 = rng.normal(scale=0.5, size=4)
    loss_history = []
    for _ in range(epochs):
        grads_sum = {"dL_dw1": 0.0, "dL_db1": 0.0, "dL_dw2": 0.0, "dL_db2": 0.0}
        total_loss = 0.0
        for x, y in zip(x_data, y_data):
            cache = forward(x, y, w1, b1, w2, b2)
            grads = backward(cache)
            for key in grads_sum:
                grads_sum[key] += grads[key]
            total_loss += cache["loss"]
        n = len(x_data)
        w1 -= learning_rate * grads_sum["dL_dw1"] / n
        b1 -= learning_rate * grads_sum["dL_db1"] / n
        w2 -= learning_rate * grads_sum["dL_dw2"] / n
        b2 -= learning_rate * grads_sum["dL_db2"] / n
        loss_history.append(total_loss / n)
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2, "loss_history": loss_history}


def toy_dataset(n: int = 40, seed: int = 1):
    """y = sin(x) + small noise, a nonlinear target the tanh unit can fit."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, size=n)
    y = np.sin(x) + rng.normal(scale=0.05, size=n)
    return x, y

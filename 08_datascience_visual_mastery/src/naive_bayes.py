"""Gaussian Naive Bayes teaching simulation.

Two classes, two continuous features. Each class is modelled as an
independent product of 1-D Gaussians (the "naive" conditional-independence
assumption). The module exposes small, pure functions so the Streamlit app
can drive a live simulation and so the math can be unit-checked by hand.
"""
from __future__ import annotations

import numpy as np


def gaussian_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    """1-D Gaussian probability density, elementwise."""
    std = max(std, 1e-6)
    coeff = 1.0 / (std * np.sqrt(2 * np.pi))
    exponent = -0.5 * ((x - mean) / std) ** 2
    return coeff * np.exp(exponent)


def class_conditional_density(x1: float, x2: float, mean: tuple[float, float],
                               std: tuple[float, float]) -> float:
    """p(x1, x2 | class) under the naive independence assumption:
    p(x1, x2 | class) = p(x1 | class) * p(x2 | class)
    """
    return float(gaussian_pdf(np.array([x1]), mean[0], std[0])[0]
                 * gaussian_pdf(np.array([x2]), mean[1], std[1])[0])


def posterior(x1: float, x2: float,
              mean_a: tuple[float, float], std_a: tuple[float, float], prior_a: float,
              mean_b: tuple[float, float], std_b: tuple[float, float], prior_b: float
              ) -> tuple[float, float]:
    """Return (P(A|x), P(B|x)) via Bayes' theorem:

    P(A|x) = P(x|A) P(A) / [P(x|A) P(A) + P(x|B) P(B)]
    """
    likelihood_a = class_conditional_density(x1, x2, mean_a, std_a)
    likelihood_b = class_conditional_density(x1, x2, mean_b, std_b)
    evidence = likelihood_a * prior_a + likelihood_b * prior_b
    if evidence <= 0:
        return 0.5, 0.5
    return (likelihood_a * prior_a / evidence, likelihood_b * prior_b / evidence)


def decision_grid(mean_a, std_a, prior_a, mean_b, std_b, prior_b,
                   xlim=(-6, 6), ylim=(-6, 6), resolution: int = 120):
    """Grid of posterior P(A|x) for contour/heatmap plotting plus the two
    class-conditional density grids used to draw the Gaussian "bumps"."""
    xs = np.linspace(*xlim, resolution)
    ys = np.linspace(*ylim, resolution)
    xx, yy = np.meshgrid(xs, ys)

    dens_a = (gaussian_pdf(xx, mean_a[0], std_a[0])
              * gaussian_pdf(yy, mean_a[1], std_a[1]))
    dens_b = (gaussian_pdf(xx, mean_b[0], std_b[0])
              * gaussian_pdf(yy, mean_b[1], std_b[1]))
    evidence = dens_a * prior_a + dens_b * prior_b
    post_a = np.divide(dens_a * prior_a, evidence,
                        out=np.full_like(evidence, 0.5), where=evidence > 0)
    return xx, yy, dens_a, dens_b, post_a


def sample_dataset(mean_a, std_a, n_a, mean_b, std_b, n_b, seed: int = 7):
    """Small synthetic labelled dataset for the two classes (for scatter plots)."""
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=mean_a, scale=std_a, size=(n_a, 2))
    b = rng.normal(loc=mean_b, scale=std_b, size=(n_b, 2))
    return a, b

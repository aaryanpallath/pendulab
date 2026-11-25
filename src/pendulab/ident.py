"""Equation-error (least-squares) identification.

The equations of motion are linear in (a, b, c, d, e, beta1, beta2). Given measured
angles, velocities and accelerations, every sample gives two linear equations; fixing
c = 1 (motion only determines ratios) turns this into ordinary least squares.

Fast and needs no initial guess, but it uses numerically differentiated accelerations,
which amplify measurement noise, and it biases the small terms. See README: on this
data it returns a physically impossible negative pivot damping.
"""

from __future__ import annotations

import numpy as np

from .data import Trial
from .model import Params


def regressor(q: np.ndarray, w: np.ndarray, al: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stacked rows for unknowns (a, b, d, e, beta1, beta2) with c = 1 moved to the right side."""
    q1, q2 = q[:, 0], q[:, 1]
    w1, w2 = w[:, 0], w[:, 1]
    a1, a2 = al[:, 0], al[:, 1]
    D = q1 - q2
    z = np.zeros_like(q1)
    rel = w2 - w1
    eq1 = np.stack([a1, np.cos(D) * a2 + np.sin(D) * w2**2, np.sin(q1), z, w1, -rel], axis=1)
    eq2 = np.stack([z, np.cos(D) * a1 - np.sin(D) * w1**2, z, np.sin(q2), z, rel], axis=1)
    return np.vstack([eq1, eq2]), np.concatenate([z, -a2])


def least_squares(trials: list[Trial], window: int = 31) -> tuple[Params, dict]:
    rows, rhs, eq = [], [], []
    for tr in trials:
        q, w, al = tr.derivatives(window)
        s = slice(tr.release, None)
        A, y = regressor(q[s], w[s], al[s])
        rows.append(A)
        rhs.append(y)
        eq.append(np.repeat([1, 2], len(y) // 2))
    A, y, eq = np.vstack(rows), np.concatenate(rhs), np.concatenate(eq)
    x, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ x
    a, b, d, e, beta1, beta2 = x
    p = Params(float(a), float(b), 1.0, float(d), float(e), float(beta1), float(beta2))
    info = {"rows": int(len(y)), "resid_rms_eq1": float(resid[eq == 1].std()),
            "resid_rms_eq2": float(resid[eq == 2].std())}
    return p, info

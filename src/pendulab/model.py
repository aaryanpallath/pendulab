"""NumPy reference model. Same equations as cpp/include/pendulab/model.hpp (see there).

Vectorized over any leading batch shape so thousands of windows integrate at once.
"""

from __future__ import annotations

from dataclasses import astuple, dataclass

import numpy as np

PARAM_NAMES = ("a", "b", "c", "d", "e", "beta1", "beta2")


@dataclass(frozen=True)
class Params:
    a: float
    b: float
    c: float
    d: float
    e: float
    beta1: float = 0.0
    beta2: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array(astuple(self), dtype=float)

    @classmethod
    def from_array(cls, v) -> Params:
        return cls(*map(float, v))

    def normalized(self) -> Params:
        """Scale so c = 1. Motion only determines parameter ratios."""
        return Params.from_array(self.as_array() / self.c)

    def with_damping(self, beta1: float, beta2: float) -> Params:
        return Params(self.a, self.b, self.c, self.d, self.e, beta1, beta2)

    def check(self) -> None:
        if self.a * self.c - self.b**2 <= 0 or self.a <= 0 or self.c <= 0:
            raise ValueError(f"mass matrix is not positive definite: {self}")


def accel(p: Params, s: np.ndarray) -> np.ndarray:
    """Angular accelerations for states s[..., 4] = (q1, q2, w1, w2)."""
    q1, q2, w1, w2 = np.moveaxis(s, -1, 0)
    D = q1 - q2
    cD, sD = np.cos(D), np.sin(D)
    rel = w2 - w1
    r1 = -(p.b * sD * w2**2 + p.d * np.sin(q1) + p.beta1 * w1 - p.beta2 * rel)
    r2 = -(-p.b * sD * w1**2 + p.e * np.sin(q2) + p.beta2 * rel)
    m12 = p.b * cD
    det = p.a * p.c - m12**2
    return np.stack([(p.c * r1 - m12 * r2) / det, (p.a * r2 - m12 * r1) / det], axis=-1)


def deriv(p: Params, s: np.ndarray) -> np.ndarray:
    return np.concatenate([s[..., 2:], accel(p, s)], axis=-1)


def energy(p: Params, s: np.ndarray) -> np.ndarray:
    q1, q2, w1, w2 = np.moveaxis(s, -1, 0)
    T = 0.5 * (p.a * w1**2 + 2 * p.b * np.cos(q1 - q2) * w1 * w2 + p.c * w2**2)
    return T - p.d * np.cos(q1) - p.e * np.cos(q2)


def rk4_step(p: Params, s: np.ndarray, h: float) -> np.ndarray:
    k1 = deriv(p, s)
    k2 = deriv(p, s + 0.5 * h * k1)
    k3 = deriv(p, s + 0.5 * h * k2)
    k4 = deriv(p, s + h * k3)
    return s + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def rollout(p: Params, s0: np.ndarray, h: float, steps: int, substeps: int = 1) -> np.ndarray:
    """RK4 trajectory, shape (steps + 1, *batch, 4); each output step is `substeps` RK4 steps."""
    out = np.empty((steps + 1, *s0.shape))
    out[0] = s = np.asarray(s0, dtype=float)
    hs = h / substeps
    for k in range(steps):
        for _ in range(substeps):
            s = rk4_step(p, s, hs)
        out[k + 1] = s
    return out


def wrap(angle: np.ndarray) -> np.ndarray:
    """Map an angle difference to (-pi, pi]."""
    return np.angle(np.exp(1j * angle))

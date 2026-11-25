"""Score models by how well they predict a held-out recording.

From each start point the model is integrated from the measured state (angles plus
smoothed velocities) and compared with what the real pendulum did. Because the motion is
chaotic, every model eventually diverges; the question is how long it stays close.
"""

from __future__ import annotations

import numpy as np

from .data import Trial
from .model import Params, rollout, wrap

HORIZONS_S = (0.05, 0.1, 0.2, 0.5, 1.0)


def start_indices(tr: Trial, stride_s: float, max_horizon_s: float) -> np.ndarray:
    stride = int(round(stride_s / tr.dt))
    last = len(tr.t) - int(round(max_horizon_s / tr.dt)) - 1
    return np.arange(tr.release, last, stride)


def prediction_errors(tr: Trial, p: Params | None, starts: np.ndarray, horizons=HORIZONS_S,
                      smooth: int = 31, substeps: int = 2) -> dict[float, np.ndarray]:
    """{horizon: (n_starts, 2) absolute angle errors in degrees}. p=None: constant velocity."""
    st = tr.states(smooth)
    s0 = np.concatenate([tr.q[starts], st[starts, 2:]], axis=1)
    K = int(round(max(horizons) / tr.dt))
    if p is None:
        steps = np.arange(K + 1)[:, None, None] * tr.dt
        traj_q = s0[None, :, :2] + steps * s0[None, :, 2:]
    else:
        traj_q = rollout(p, s0, tr.dt, K, substeps=substeps)[..., :2]
    out = {}
    for h in horizons:
        k = int(round(h / tr.dt))
        out[h] = np.degrees(np.abs(wrap(traj_q[k] - tr.q[starts + k])))
    return out


def summarize(errs: dict[float, np.ndarray]) -> dict[str, dict[str, float]]:
    return {f"{h:g}s": {"median_deg_link1": float(np.median(e[:, 0])),
                        "median_deg_link2": float(np.median(e[:, 1])),
                        "within_5deg_both": float(np.mean((e < 5).all(axis=1)))}
            for h, e in errs.items()}

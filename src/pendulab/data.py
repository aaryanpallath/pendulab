"""The recorded double-pendulum trials (see scripts/fetch_data.py for source and license)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

DATA = Path(__file__).resolve().parents[2] / "data" / "pendulum.npz"
# Each recording: pendulum hanging still, lifted to upright by hand (~2 s), held there
# (~1.5 s), then let go. Only motion after the hold is free motion.
HOLD_TOL_DEG = 15.0  # both links within this of straight up
HOLD_SPEED = 0.5  # rad/s; both links slower than this while held
MIN_HOLD_S = 0.3
SETTLE_S = 0.05  # small margin so the hand is clear


@dataclass
class Trial:
    name: str
    t: np.ndarray  # s, uniform spacing
    q: np.ndarray  # (n, 2) absolute link angles, rad, unwrapped
    release: int  # index where the free swing starts (after the release transient)

    @property
    def dt(self) -> float:
        return float(self.t[1] - self.t[0])

    def derivatives(self, window: int = 31, order: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Savitzky-Golay smoothed angle, velocity and acceleration, each (n, 2)."""
        out = [savgol_filter(self.q, window, order, deriv=k, delta=self.dt, axis=0) for k in (0, 1, 2)]
        return out[0], out[1], out[2]

    def states(self, window: int = 31) -> np.ndarray:
        """(n, 4) states (q1, q2, w1, w2) using smoothed velocities."""
        q, w, _ = self.derivatives(window)
        return np.concatenate([q, w], axis=1)


def find_release(q: np.ndarray, dt: float) -> int:
    """End of the upright hold, plus a small settling margin.

    The hold is the longest stretch where both links are near upright and nearly still.
    Earlier versions fired on "first movement" (which is the hand lifting the pendulum)
    and on "first time near upright" (still settling into the hold); see README.
    """
    w = savgol_filter(q, 31, 3, deriv=1, delta=dt, axis=0)
    near_up = (np.abs(np.angle(np.exp(1j * (q - np.pi)))) < np.deg2rad(HOLD_TOL_DEG)).all(axis=1)
    still = near_up & (np.abs(w) < HOLD_SPEED).all(axis=1)
    edges = np.diff(np.concatenate([[0], still.astype(np.int8), [0]]))
    starts, ends = np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]
    if len(starts) == 0:
        raise ValueError("pendulum is never held upright")
    longest = int(np.argmax(ends - starts))
    if (ends[longest] - starts[longest]) * dt < MIN_HOLD_S:
        raise ValueError("no upright hold long enough to be a deliberate release")
    return int(ends[longest] + round(SETTLE_S / dt))


def load(path: Path = DATA) -> dict[str, Trial]:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run: uv run python scripts/fetch_data.py")
    z = np.load(path)
    trials = {}
    for i in (1, 2, 3):
        t, q = z[f"trial{i}_t"], z[f"trial{i}_q"]
        trials[f"trial{i}"] = Trial(f"trial{i}", t, q, find_release(q, float(t[1] - t[0])))
    return trials

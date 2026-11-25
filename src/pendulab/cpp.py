"""Run the C++ simulator (cpp/build/dpsim) and parse its CSV output."""

from __future__ import annotations

import io
import os
import subprocess
from pathlib import Path

import numpy as np

from .model import Params

ROOT = Path(__file__).resolve().parents[2]


def dpsim_path() -> Path:
    p = Path(os.environ.get("PENDULAB_DPSIM", ROOT / "cpp" / "build" / "dpsim"))
    if not p.exists():
        raise FileNotFoundError(f"{p} not found; build it with: cmake -S cpp -B cpp/build && cmake --build cpp/build")
    return p


def simulate(p: Params, state, dt: float, steps: int, integrator: str = "rk4", every: int = 1) -> np.ndarray:
    """Returns an array with columns t, q1, q2, w1, w2, energy."""
    cmd = [str(dpsim_path()), "--integrator", integrator, "--dt", repr(dt), "--steps", str(steps),
           "--every", str(every), "--state", ",".join(map(repr, map(float, state))),
           "--params", ",".join(map(repr, p.as_array().tolist()))]
    out = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip())
    return np.loadtxt(io.StringIO(out.stdout), delimiter=",", skiprows=1, ndmin=2)

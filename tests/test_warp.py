"""The Warp simulator must agree with the NumPy model, and its gradients with finite differences."""

import numpy as np
import pytest

from pendulab import model, warp_fit
from pendulab.model import Params

P = Params(4.0, 0.9, 1.0, 254.0, 99.6, 0.01, 0.02)


def synthetic_windows(n=24, K=40, dt=0.002, seed=0):
    """Windows whose targets come from the NumPy model itself, plus a small known error."""
    rng = np.random.default_rng(seed)
    s0 = np.concatenate([rng.uniform(-3, 3, (n, 2)), rng.uniform(-8, 8, (n, 2))], axis=1)
    traj = model.rollout(P, s0, dt, K, substeps=warp_fit.SUBSTEPS)
    target = traj[1:, :, :2] + rng.normal(0, 0.01, (K, n, 2))
    return warp_fit.Windows(s0[:, :2], s0[:, 2:], target, dt), traj


def test_warp_rollout_matches_numpy():
    win, traj = synthetic_windows()
    sim = warp_fit.Simulator(win)
    import warp as wp
    p_w = wp.array(P.as_array(), dtype=wp.float64)
    w_w = wp.array(win.w0, dtype=wp.vec2d)
    states, _ = sim.forward(p_w, w_w)
    assert np.abs(states.numpy() - traj).max() < 1e-10


@pytest.mark.parametrize("i", [0, 1, 3, 4, 5, 6])
def test_gradients_match_finite_differences(i):
    win, _ = synthetic_windows()
    _, g = warp_fit.loss_and_grad(win, P)
    h = 1e-6 * max(1.0, abs(P.as_array()[i]))
    v = P.as_array().copy()
    v[i] += h
    lp, _ = warp_fit.loss_and_grad(win, Params.from_array(v))
    v[i] -= 2 * h
    lm, _ = warp_fit.loss_and_grad(win, Params.from_array(v))
    fd = (lp - lm) / (2 * h)
    assert abs(g[i] - fd) <= 1e-5 * abs(fd) + 1e-12


def test_fit_recovers_known_parameters_from_synthetic_data():
    """Start 10% off on every parameter; the fit should land close to the truth."""
    win, _ = synthetic_windows(n=48, K=50, seed=1)
    start = Params(P.a * 1.1, P.b * 0.9, 1.0, P.d * 0.9, P.e * 1.1, 0.0, 0.0)
    res = warp_fit.fit(win, start, iters=400, lr=3e-3, fit_velocity=False)
    for name in ("a", "b", "d", "e"):
        assert getattr(res.params, name) == pytest.approx(getattr(P, name), rel=0.02), name
    # The targets carry N(0, 0.01 rad) noise on both angles, so the best achievable loss is
    # about 2 * 0.01^2 = 2e-4 (mean of 2 - 2cos(err) ~ err^2, summed over two angles).
    assert res.history[-1] < 1.1 * 2 * 0.01**2

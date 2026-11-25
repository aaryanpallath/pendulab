import numpy as np
import pytest

from pendulab import cpp, data, evaluate, ident, model
from pendulab.data import Trial
from pendulab.model import Params

P = Params(4.0, 0.9, 1.0, 254.0, 99.6, 0.0, 0.0)
HAVE_DPSIM = (cpp.ROOT / "cpp" / "build" / "dpsim").exists()


def test_energy_is_conserved_without_damping():
    s0 = np.array([2.0, -1.0, 0.0, 0.0])
    traj = model.rollout(P, s0, 1e-4, 20000)
    e = model.energy(P, traj)
    assert np.abs(e - e[0]).max() / (P.d + P.e) < 1e-8


def test_damping_only_removes_energy():
    p = P.with_damping(0.05, 0.02)
    e = model.energy(p, model.rollout(p, np.array([2.0, -1.0, 0.0, 0.0]), 1e-4, 20000))
    assert np.all(np.diff(e) <= 1e-9)
    assert e[-1] < e[0]


def test_parameters_are_only_defined_up_to_scale():
    s0 = np.array([1.0, 2.0, 0.5, -0.5])
    scaled = Params.from_array(P.with_damping(0.01, 0.02).as_array() * 7.3)
    a = model.rollout(P.with_damping(0.01, 0.02), s0, 1e-3, 500)
    b = model.rollout(scaled, s0, 1e-3, 500)
    assert np.abs(a - b).max() < 1e-12
    assert scaled.normalized().as_array() == pytest.approx(P.with_damping(0.01, 0.02).as_array())


def test_wrap():
    assert model.wrap(np.array([3 * np.pi, -3 * np.pi, 0.1]))[2] == pytest.approx(0.1)
    assert np.all(np.abs(model.wrap(np.linspace(-20, 20, 101))) <= np.pi)


@pytest.mark.skipif(not HAVE_DPSIM, reason="C++ simulator not built")
@pytest.mark.parametrize("integrator", ["rk4", "euler", "semi_implicit_euler", "implicit_midpoint"])
def test_cpp_cli_runs_every_integrator(integrator):
    out = cpp.simulate(P, [1.0, 0.5, 0.0, 0.0], 1e-3, 100, integrator=integrator, every=10)
    assert out.shape == (11, 6)
    assert np.all(np.isfinite(out))


@pytest.mark.skipif(not HAVE_DPSIM, reason="C++ simulator not built")
def test_cpp_matches_numpy_reference():
    p = P.with_damping(0.02, 0.01)
    s0 = [2.0, -1.0, 0.3, 0.0]
    c = cpp.simulate(p, s0, 1e-3, 2000)[:, 1:5]
    r = model.rollout(p, np.array(s0), 1e-3, 2000)
    assert np.abs(c - r).max() < 1e-9


@pytest.mark.skipif(not HAVE_DPSIM, reason="C++ simulator not built")
def test_cpp_rejects_bad_input():
    with pytest.raises(RuntimeError, match="positive definite"):
        cpp.simulate(Params(1.0, 2.0, 1.0, 1.0, 1.0), [0, 0, 0, 0], 1e-3, 10)
    with pytest.raises(RuntimeError, match="unknown integrator"):
        cpp.simulate(P, [0, 0, 0, 0], 1e-3, 10, integrator="leapfrog")


def synthetic_trial(p: Params, seconds=6.0, dt=0.002, noise=0.0, seed=0) -> Trial:
    n = int(seconds / dt)
    traj = model.rollout(p, np.array([2.5, -2.0, 0.0, 3.0]), dt, n - 1, substeps=4)
    q = traj[:, :2] + np.random.default_rng(seed).normal(0, noise, (n, 2))
    return Trial("synthetic", np.arange(n) * dt, q, release=50)


def test_least_squares_recovers_parameters_from_clean_data():
    truth = P.with_damping(0.02, 0.03)
    est, _ = ident.least_squares([synthetic_trial(truth)], window=11)
    for name in ("a", "b", "d", "e"):
        assert getattr(est, name) == pytest.approx(getattr(truth, name), rel=0.01), name


def test_least_squares_depends_on_smoothing_when_data_is_noisy():
    """The weakness the README describes: with noise, the answer moves with the filter."""
    tr = synthetic_trial(P.with_damping(0.02, 0.03), noise=np.deg2rad(0.2))
    a_small = ident.least_squares([tr], window=15)[0].a
    a_large = ident.least_squares([tr], window=61)[0].a
    assert abs(a_small - a_large) / P.a > 0.05


def test_perfect_model_predicts_its_own_data():
    tr = synthetic_trial(P, seconds=4.0)
    starts = evaluate.start_indices(tr, 0.5, 1.0)
    errs = evaluate.prediction_errors(tr, P, starts, horizons=(0.1, 0.5), smooth=11)
    assert np.median(errs[0.5]) < 0.5  # degrees; only the smoothed starting velocity differs


def test_find_release_waits_for_the_end_of_the_upright_hold():
    # hanging (0-200), lifted by hand (200-1000), held upright (1000-1600), released (1600-)
    q = np.zeros((3000, 2))
    q[200:1000] = np.linspace(0, -np.pi, 800)[:, None]
    q[1000:1600] = -np.pi + np.deg2rad(1.0) * np.sin(np.arange(600) / 20)[:, None]  # small hand wobble
    q[1600:, 1] = -np.pi - np.linspace(0, 6, 1400) ** 2
    q[1600:, 0] = -np.pi
    r = data.find_release(q, 0.002)
    assert 1600 < r < 1600 + 60


@pytest.mark.skipif(not data.DATA.exists(), reason="data not fetched")
def test_recorded_trials_load():
    trials = data.load()
    assert set(trials) == {"trial1", "trial2", "trial3"}
    for tr in trials.values():
        assert tr.q.shape == (38827, 2)
        assert tr.dt == pytest.approx(0.002)
        assert 2000 < tr.release < 3500
        assert abs(np.degrees(model.wrap(tr.q[tr.release - 30, 0] - np.pi))) < 15  # upright at release

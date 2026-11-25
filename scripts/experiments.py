"""Reproduce every number and figure in the README.

    uv run python scripts/experiments.py            # ~25 min on an Apple M4 CPU

Requires data/pendulum.npz (scripts/fetch_data.py) and cpp/build/dpsim (CMake build).
Writes results/*.json and results/*.png.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pendulab import cpp, data, evaluate, ident, model, warp_fit  # noqa: E402
from pendulab.model import Params  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIT_ITERS = 4000
HORIZON_S, STRIDE_S = 0.2, 0.1  # training windows
TEST_STRIDE_S = 0.25


def dump(name: str, obj) -> None:
    (RES / name).write_text(json.dumps(obj, indent=2) + "\n")


def integrators() -> None:
    """Convergence and energy behaviour of the C++ integrators on the fitted-scale pendulum."""
    p = Params(4.0, 0.9, 1.0, 254.0, 99.6)  # frictionless, so energy should be conserved
    s0 = [2.0, -1.0, 0.0, 0.0]
    T = 0.5
    ref = cpp.simulate(p, s0, T / 65536, 65536, every=65536)[-1, 1:5]
    conv = {}
    for name in ("euler", "semi_implicit_euler", "implicit_midpoint", "rk4"):
        rows = []
        for n in (128, 256, 512, 1024, 2048, 4096, 8192):
            end = cpp.simulate(p, s0, T / n, n, integrator=name, every=n)[-1, 1:5]
            rows.append({"steps": n, "dt": T / n, "max_state_error": float(np.abs(end - ref).max())})
        conv[name] = rows
    drift = {}
    for name in ("euler", "semi_implicit_euler", "implicit_midpoint", "rk4"):
        out = cpp.simulate(p, s0, 1e-3, 60_000, integrator=name, every=10)
        e = out[:, 5]
        drift[name] = {"dt": 1e-3, "seconds": 60, "max_rel_energy_error": float(np.abs(e - e[0]).max() / (p.d + p.e))}
    # Differential check: the C++ and NumPy implementations of the same equations.
    pd = Params(4.0, 0.9, 1.0, 254.0, 99.6, 0.02, 0.01)
    c = cpp.simulate(pd, s0, 1e-3, 2000)[:, 1:5]
    r = model.rollout(pd, np.array(s0), 1e-3, 2000)
    dump("integrators.json", {"convergence": conv, "energy_drift": drift,
                              "cpp_vs_numpy_max_abs_diff_2s": float(np.abs(c - r).max())})


def fit_all(trials: dict) -> dict:
    out = {}
    train = [trials["trial1"], trials["trial2"]]
    ls_sens = {}
    for w in (11, 21, 31, 51, 81):
        p, _ = ident.least_squares(train, window=w)
        ls_sens[w] = p.as_array().tolist()
    p_ls, info = ident.least_squares(train)
    out["least_squares"] = {"params": p_ls.as_array().tolist(), "info": info, "window_sensitivity": ls_sens}

    per_trial = {}
    for name in ("trial1", "trial2", "trial3"):
        pl, _ = ident.least_squares([trials[name]])
        win = warp_fit.make_windows([trials[name]], HORIZON_S, STRIDE_S)
        res = warp_fit.fit(win, pl, iters=FIT_ITERS)
        per_trial[name] = {"least_squares": pl.as_array().tolist(), "warp": res.params.as_array().tolist(),
                           "final_loss": res.history[-1]}
        print(name, "done", flush=True)
    out["per_trial"] = per_trial

    win = warp_fit.make_windows(train, HORIZON_S, STRIDE_S)
    t0 = time.time()
    res = warp_fit.fit(win, p_ls, iters=FIT_ITERS, snapshot_every=500)
    out["warp"] = {"params": res.params.as_array().tolist(), "windows": int(win.target.shape[1]),
                   "samples_per_window": int(win.target.shape[0]), "iters": FIT_ITERS,
                   "seconds": time.time() - t0, "device": warp_fit.DEVICE,
                   "loss_history": res.history[::10] + [res.history[-1]],
                   "snapshots": {k: v.as_array().tolist() for k, v in res.snapshots.items()}}
    out["param_names"] = list(model.PARAM_NAMES)
    dump("fit.json", out)
    return out


def evaluate_all(trials: dict, fits: dict) -> dict:
    test = trials["trial3"]
    starts = evaluate.start_indices(test, TEST_STRIDE_S, max(evaluate.HORIZONS_S))
    p_ls = Params.from_array(fits["least_squares"]["params"])
    p_w = Params.from_array(fits["warp"]["params"])
    models = {"least_squares": p_ls, "warp": p_w, "warp_no_damping": p_w.with_damping(0.0, 0.0),
              "constant_velocity": None}
    errs = {k: evaluate.prediction_errors(test, p, starts) for k, p in models.items()}
    snaps = {int(it): evaluate.summarize(evaluate.prediction_errors(test, Params.from_array(v), starts))
             for it, v in fits["warp"]["snapshots"].items()}
    out = {"test_trial": "trial3", "start_points": int(len(starts)),
           "models": {k: evaluate.summarize(e) for k, e in errs.items()}, "warp_snapshots": snaps}
    dump("evaluation.json", out)

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    hs = list(evaluate.HORIZONS_S)
    for k, label in (("least_squares", "least squares"), ("warp", "Warp output-error fit"),
                     ("constant_velocity", "constant velocity")):
        for j in (0, 1):
            ax[j].plot(hs, [np.median(errs[k][h][:, j]) for h in hs], marker="o", label=label)
    for j in (0, 1):
        ax[j].set_xscale("log")
        ax[j].set_yscale("log")
        ax[j].set_xlabel("prediction horizon (s)")
        ax[j].set_title(f"link {j + 1}")
        ax[j].grid(alpha=0.3, which="both")
    ax[0].set_ylabel("median angle error (deg)")
    ax[1].legend(fontsize=8)
    fig.suptitle("Held-out trial 3: prediction error from measured starting states", fontsize=10)
    fig.tight_layout()
    fig.savefig(RES / "prediction_error.png", dpi=130)
    plt.close(fig)

    # One example: 1.5 s from a start point, both models against the recording.
    tr = test
    j = int(starts[len(starts) // 3])
    K = int(round(1.5 / tr.dt))
    st = tr.states()
    s0 = np.concatenate([tr.q[j], st[j, 2:]])
    tt = np.arange(K + 1) * tr.dt
    fig, ax = plt.subplots(2, 1, figsize=(8, 4.6), sharex=True)
    for k, p, style in (("least squares", p_ls, "--"), ("Warp fit", p_w, "-")):
        traj = model.rollout(p, s0, tr.dt, K, substeps=2)
        for i in (0, 1):
            ax[i].plot(tt, np.degrees(traj[:, i]), style, label=k)
    for i in (0, 1):
        ax[i].plot(tt, np.degrees(tr.q[j: j + K + 1, i]), "k", lw=1, alpha=0.7, label="recorded")
        ax[i].set_ylabel(f"link {i + 1} angle (deg)")
    ax[1].set_xlabel(f"seconds after t = {tr.t[j]:.2f} s in trial 3")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / "example_rollout.png", dpi=130)
    plt.close(fig)
    return out


def energy_check(trials: dict, fits: dict) -> None:
    """Does the fitted damping explain how fast the real pendulum loses energy?

    Energy of the recording (computed from smoothed states with the fitted parameters)
    vs a long simulation from the same release state. Chaos makes the two trajectories
    differ after a second or so, but the energy envelope is a slow, averaged quantity.
    """
    p = Params.from_array(fits["warp"]["params"])
    tr = trials["trial3"]
    st = tr.states(51)
    e_meas = model.energy(p, st)
    r = tr.release
    n = len(tr.t) - r
    sim = model.rollout(p, st[r], tr.dt, n - 1, substeps=2)
    e_sim = model.energy(p, sim)
    e_min = -(p.d + p.e)  # energy at rest hanging down
    marks = [0, 5, 10, 20, 40, 60]
    rows = []
    for s in marks:
        k = int(s / tr.dt)
        if k >= n:
            continue
        sl = slice(max(0, k - 250), k + 250)  # 1 s average, energy oscillates with damping work
        rows.append({"t_after_release_s": s,
                     "measured_energy_above_rest": float(np.mean(e_meas[r:][sl]) - e_min),
                     "simulated_energy_above_rest": float(np.mean(e_sim[sl]) - e_min)})
    dump("energy.json", {"params": "warp fit (c = 1 units)", "rows": rows})

    fig, ax = plt.subplots(figsize=(7, 3.2))
    t = tr.t[r:] - tr.t[r]
    ax.plot(t, e_meas[r:] - e_min, lw=0.6, label="recording")
    ax.plot(t, e_sim - e_min, lw=0.6, label="simulation with fitted damping")
    ax.set_xlabel("seconds after release (trial 3)")
    ax.set_ylabel("energy above rest (c = 1 units)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / "energy.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    RES.mkdir(exist_ok=True)
    trials = data.load()
    integrators()
    print("integrators done", flush=True)
    fits = fit_all(trials)
    evaluate_all(trials, fits)
    energy_check(trials, fits)
    print("all done", flush=True)

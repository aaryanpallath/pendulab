"""Output-error identification with a differentiable simulator written in NVIDIA Warp.

Equation-error least squares (ident.py) fits the equations of motion to numerically
differentiated accelerations. This module instead fits *simulated trajectories* to the
measured angles: many short windows are cut from the recordings, each window is
integrated with RK4 from its measured starting state, and the mean squared angle error
is minimized by gradient descent. Gradients come from Warp's reverse-mode autodiff
(`wp.Tape`) through every integration step.

Because the double pendulum is chaotic, long windows are useless for fitting (any tiny
error grows exponentially), so windows are short and each one's starting velocity is
also a free variable — the smoothed numerical derivative is only a first guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import warp as wp

from .data import Trial
from .model import Params

wp.config.quiet = True
wp.init()

DEVICE = "cuda" if wp.is_cuda_available() else "cpu"

# RK4 steps per data sample (data at 2 ms -> 1 ms integration steps). This must be a
# compile-time constant: with a runtime loop bound (`for _ in range(substeps)`), Warp's
# reverse pass did not correctly handle the state reassigned inside the loop, and
# gradients were off by 2-7% versus finite differences. `wp.static` unrolls the loop at
# compile time; tests/test_warp.py checks gradients against finite differences.
SUBSTEPS = 2


@wp.func
def _accel(s: wp.vec4d, p: wp.array(dtype=wp.float64)):
    a = p[0]
    b = p[1]
    c = p[2]
    d = p[3]
    e = p[4]
    beta1 = p[5]
    beta2 = p[6]
    D = s[0] - s[1]
    cD = wp.cos(D)
    sD = wp.sin(D)
    rel = s[3] - s[2]
    r1 = -(b * sD * s[3] * s[3] + d * wp.sin(s[0]) + beta1 * s[2] - beta2 * rel)
    r2 = -(-b * sD * s[2] * s[2] + e * wp.sin(s[1]) + beta2 * rel)
    m12 = b * cD
    det = a * c - m12 * m12
    return wp.vec4d(s[2], s[3], (c * r1 - m12 * r2) / det, (a * r2 - m12 * r1) / det)


@wp.func
def _rk4(s: wp.vec4d, p: wp.array(dtype=wp.float64), h: wp.float64):
    half = wp.float64(0.5) * h
    k1 = _accel(s, p)
    k2 = _accel(s + half * k1, p)
    k3 = _accel(s + half * k2, p)
    k4 = _accel(s + h * k3, p)
    return s + (h / wp.float64(6.0)) * (k1 + wp.float64(2.0) * k2 + wp.float64(2.0) * k3 + k4)


@wp.kernel
def _make_params(u: wp.array(dtype=wp.float64), offset: wp.array(dtype=wp.float64),
                 scale: wp.array(dtype=wp.float64), p: wp.array(dtype=wp.float64)):
    i = wp.tid()
    p[i] = offset[i] + scale[i] * u[i]


@wp.kernel
def _init(q0: wp.array(dtype=wp.vec2d), w0: wp.array(dtype=wp.vec2d), states: wp.array2d(dtype=wp.vec4d)):
    i = wp.tid()
    states[0, i] = wp.vec4d(q0[i][0], q0[i][1], w0[i][0], w0[i][1])


@wp.kernel
def _step(states: wp.array2d(dtype=wp.vec4d), p: wp.array(dtype=wp.float64), k: int, h: wp.float64):
    i = wp.tid()
    s = states[k, i]
    for _ in range(wp.static(SUBSTEPS)):
        s = _rk4(s, p, h)
    states[k + 1, i] = s


@wp.kernel
def _loss(states: wp.array2d(dtype=wp.vec4d), target: wp.array2d(dtype=wp.vec2d), norm: wp.float64,
          loss: wp.array(dtype=wp.float64)):
    k, i = wp.tid()
    s = states[k + 1, i]
    tq = target[k, i]
    # 2 - 2cos(err) ~ err^2 for small errors and is wrap-safe for angles.
    e1 = wp.float64(2.0) - wp.float64(2.0) * wp.cos(s[0] - tq[0])
    e2 = wp.float64(2.0) - wp.float64(2.0) * wp.cos(s[1] - tq[1])
    wp.atomic_add(loss, 0, (e1 + e2) * norm)


@dataclass
class Windows:
    q0: np.ndarray  # (N, 2)
    w0: np.ndarray  # (N, 2) smoothed-derivative initial velocity
    target: np.ndarray  # (K, N, 2) measured angles at samples 1..K
    dt: float


def make_windows(trials: list[Trial], horizon_s: float, stride_s: float, smooth: int = 31) -> Windows:
    q0, w0, tgt = [], [], []
    dt = trials[0].dt
    K = int(round(horizon_s / dt))
    stride = max(1, int(round(stride_s / dt)))
    for tr in trials:
        if abs(tr.dt - dt) > 1e-12:
            raise ValueError("all trials must share one sample rate")
        st = tr.states(smooth)
        for j in range(tr.release, len(tr.t) - K - 1, stride):
            q0.append(tr.q[j])
            w0.append(st[j, 2:])
            tgt.append(tr.q[j + 1: j + K + 1])
    return Windows(np.array(q0), np.array(w0), np.stack(tgt, axis=1), dt)


class Simulator:
    """Batched differentiable rollout of many windows with shared parameters."""

    def __init__(self, win: Windows, device: str = DEVICE):
        self.win, self.device = win, device
        self.K, self.N = win.target.shape[0], win.target.shape[1]
        self.h = wp.float64(win.dt / SUBSTEPS)
        self.q0 = wp.array(win.q0, dtype=wp.vec2d, device=device)
        self.target = wp.array(win.target, dtype=wp.vec2d, device=device)
        self.norm = wp.float64(1.0 / (self.N * self.K))

    def forward(self, p: wp.array, w0: wp.array) -> tuple[wp.array, wp.array]:
        states = wp.zeros((self.K + 1, self.N), dtype=wp.vec4d, device=self.device, requires_grad=True)
        loss = wp.zeros(1, dtype=wp.float64, device=self.device, requires_grad=True)
        wp.launch(_init, dim=self.N, inputs=[self.q0, w0, states], device=self.device)
        for k in range(self.K):
            wp.launch(_step, dim=self.N, inputs=[states, p, k, self.h], device=self.device)
        wp.launch(_loss, dim=(self.K, self.N), inputs=[states, self.target, self.norm, loss], device=self.device)
        return states, loss


@dataclass
class FitResult:
    params: Params
    w0: np.ndarray
    history: list[float] = field(default_factory=list)
    snapshots: dict[int, Params] = field(default_factory=dict)  # iteration -> parameters


def _adam(x, g, m, v, t, lr, b1=0.9, b2=0.999, eps=1e-12):
    m[:] = b1 * m + (1 - b1) * g
    v[:] = b2 * v + (1 - b2) * g * g
    mh, vh = m / (1 - b1**t), v / (1 - b2**t)
    return x - lr * mh / (np.sqrt(vh) + eps)


def fit(win: Windows, init: Params, *, iters: int = 300, lr: float = 3e-3, lr_w: float = 3e-2,
        damping_scale: float = 0.05, fit_velocity: bool = True, snapshot_every: int = 0,
        log=None) -> FitResult:
    """Adam on relative parameters: p_i = init_i * (1 + u_i) for a, b, d, e; damping
    beta = damping_scale * u (starting from max(init, 0), projected to >= 0); c stays 1."""
    sim = Simulator(win)
    dev = sim.device
    p0 = init.normalized().as_array()
    offset = np.array([p0[0], p0[1], 1.0, p0[3], p0[4], 0.0, 0.0])
    scale = np.array([p0[0], p0[1], 0.0, p0[3], p0[4], damping_scale, damping_scale])
    u = np.zeros(7)
    u[5:] = np.maximum(p0[5:], 0.0) / damping_scale
    w0 = win.w0.copy()
    off_w, sc_w = wp.array(offset, dtype=wp.float64, device=dev), wp.array(scale, dtype=wp.float64, device=dev)
    mu, vu, mw, vw = np.zeros(7), np.zeros(7), np.zeros_like(w0), np.zeros_like(w0)
    history, snapshots = [], {}
    for it in range(1, iters + 1):
        u_w = wp.array(u, dtype=wp.float64, device=dev, requires_grad=True)
        w_w = wp.array(w0, dtype=wp.vec2d, device=dev, requires_grad=True)
        p_w = wp.zeros(7, dtype=wp.float64, device=dev, requires_grad=True)
        tape = wp.Tape()
        with tape:
            wp.launch(_make_params, dim=7, inputs=[u_w, off_w, sc_w, p_w], device=dev)
            _, loss = sim.forward(p_w, w_w)
        tape.backward(loss)
        L = float(loss.numpy()[0])
        history.append(L)
        gu = u_w.grad.numpy()
        gu[2] = 0.0
        u = _adam(u, gu, mu, vu, it, lr)
        u[5:] = np.maximum(u[5:], 0.0)  # damping can't be negative
        if fit_velocity:
            w0 = _adam(w0, w_w.grad.numpy(), mw, vw, it, lr_w)
        tape.zero()
        if snapshot_every and it % snapshot_every == 0:
            snapshots[it] = Params.from_array(offset + scale * u)
        if log and (it == 1 or it % 25 == 0 or it == iters):
            log(f"iter {it:4d} loss {L:.4e}")
    return FitResult(Params.from_array(offset + scale * u), w0, history, snapshots)


def loss_and_grad(win: Windows, p: Params, w0: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    """Loss and d(loss)/d(params) at fixed starting velocities; used by the gradient test."""
    sim = Simulator(win)
    p_w = wp.array(p.as_array(), dtype=wp.float64, device=sim.device, requires_grad=True)
    w_w = wp.array(win.w0 if w0 is None else w0, dtype=wp.vec2d, device=sim.device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        _, loss = sim.forward(p_w, w_w)
    tape.backward(loss)
    return float(loss.numpy()[0]), p_w.grad.numpy().copy()

// Planar double pendulum with rigid (compound) links, written in lumped parameters.
//
// Angles q1, q2 are absolute, measured from hanging straight down. With
// D = q1 - q2 the equations of motion are
//
//   a*q1'' + b*cos(D)*q2'' + b*sin(D)*q2'^2 + d*sin(q1) + beta1*q1' - beta2*(q2'-q1') = 0
//   b*cos(D)*q1'' + c*q2''  - b*sin(D)*q1'^2 + e*sin(q2) + beta2*(q2'-q1')            = 0
//
// a = I1 + m2*l1^2 (link 1 about the pivot), b = m2*l1*c2 (coupling), c = I2 (link 2
// about the elbow), d = (m1*c1 + m2*l1)*g, e = m2*c2*g, and beta1, beta2 are viscous
// damping at the pivot and the elbow. Point masses on massless rods are the special case
// a = (m1+m2)l1^2, b = m2 l1 l2, c = m2 l2^2. Multiplying every parameter by the same
// constant leaves the motion unchanged, so only ratios can be identified from motion data.
#pragma once

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

namespace pendulab {

struct Params {
    double a = 1.0, b = 0.5, c = 0.5, d = 10.0, e = 5.0, beta1 = 0.0, beta2 = 0.0;

    // Point masses m1, m2 on massless rods of length l1, l2.
    static Params point_mass(double m1, double m2, double l1, double l2, double g = 9.81) {
        return {(m1 + m2) * l1 * l1, m2 * l1 * l2, m2 * l2 * l2, (m1 + m2) * l1 * g, m2 * l2 * g, 0, 0};
    }
};

// State: q1, q2, w1, w2 (angles in rad, angular velocities in rad/s).
using State = std::array<double, 4>;

inline std::array<double, 2> accel(const Params& p, const State& s) {
    const double D = s[0] - s[1];
    const double cD = std::cos(D), sD = std::sin(D);
    const double rel = s[3] - s[2];
    const double r1 = -(p.b * sD * s[3] * s[3] + p.d * std::sin(s[0]) + p.beta1 * s[2] - p.beta2 * rel);
    const double r2 = -(-p.b * sD * s[2] * s[2] + p.e * std::sin(s[1]) + p.beta2 * rel);
    const double m12 = p.b * cD;
    const double det = p.a * p.c - m12 * m12;  // > 0 whenever the mass matrix is valid
    return {(p.c * r1 - m12 * r2) / det, (p.a * r2 - m12 * r1) / det};
}

inline State deriv(const Params& p, const State& s) {
    const auto acc = accel(p, s);
    return {s[2], s[3], acc[0], acc[1]};
}

// Total mechanical energy, up to an additive constant. Conserved when beta1 = beta2 = 0.
inline double energy(const Params& p, const State& s) {
    const double D = s[0] - s[1];
    const double T = 0.5 * (p.a * s[2] * s[2] + 2.0 * p.b * std::cos(D) * s[2] * s[3] + p.c * s[3] * s[3]);
    const double V = -p.d * std::cos(s[0]) - p.e * std::cos(s[1]);
    return T + V;
}

inline State axpy(const State& x, double h, const State& k) {
    return {x[0] + h * k[0], x[1] + h * k[1], x[2] + h * k[2], x[3] + h * k[3]};
}

enum class Integrator { ExplicitEuler, SemiImplicitEuler, RK4, ImplicitMidpoint };

inline Integrator parse_integrator(const std::string& name) {
    if (name == "euler") return Integrator::ExplicitEuler;
    if (name == "semi_implicit_euler") return Integrator::SemiImplicitEuler;
    if (name == "rk4") return Integrator::RK4;
    if (name == "implicit_midpoint") return Integrator::ImplicitMidpoint;
    throw std::invalid_argument("unknown integrator: " + name);
}

inline const char* name_of(Integrator m) {
    switch (m) {
        case Integrator::ExplicitEuler: return "euler";
        case Integrator::SemiImplicitEuler: return "semi_implicit_euler";
        case Integrator::RK4: return "rk4";
        case Integrator::ImplicitMidpoint: return "implicit_midpoint";
    }
    return "?";
}

// Implicit midpoint: y1 = y0 + h f((y0 + y1) / 2). Symplectic and time-reversible, so
// energy error stays bounded instead of drifting. Solved by fixed-point iteration, which
// converges for h well below the fastest time scale; it throws rather than return a
// silently unconverged step.
inline State implicit_midpoint_step(const Params& p, const State& y0, double h) {
    State y1 = axpy(y0, h, deriv(p, y0));
    for (int it = 0; it < 100; ++it) {
        State mid;
        for (int i = 0; i < 4; ++i) mid[i] = 0.5 * (y0[i] + y1[i]);
        const State next = axpy(y0, h, deriv(p, mid));
        double diff = 0.0;
        for (int i = 0; i < 4; ++i) diff = std::fmax(diff, std::fabs(next[i] - y1[i]));
        y1 = next;
        if (diff < 1e-14 * (1.0 + std::fabs(y1[2]) + std::fabs(y1[3]))) return y1;
    }
    throw std::runtime_error("implicit midpoint: fixed-point iteration did not converge; reduce dt");
}

inline State step(const Params& p, const State& s, double h, Integrator m) {
    switch (m) {
        case Integrator::ExplicitEuler:
            return axpy(s, h, deriv(p, s));
        case Integrator::SemiImplicitEuler: {
            // Velocities first, then positions with the new velocities.
            const auto acc = accel(p, s);
            const double w1 = s[2] + h * acc[0], w2 = s[3] + h * acc[1];
            return {s[0] + h * w1, s[1] + h * w2, w1, w2};
        }
        case Integrator::RK4: {
            const State k1 = deriv(p, s);
            const State k2 = deriv(p, axpy(s, 0.5 * h, k1));
            const State k3 = deriv(p, axpy(s, 0.5 * h, k2));
            const State k4 = deriv(p, axpy(s, h, k3));
            State out;
            for (int i = 0; i < 4; ++i) out[i] = s[i] + h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
            return out;
        }
        case Integrator::ImplicitMidpoint:
            return implicit_midpoint_step(p, s, h);
    }
    return s;
}

// Integrate for `steps` steps of size h, calling visit(k, state) for k = 0..steps.
template <class Visit>
State simulate(const Params& p, State s, double h, long steps, Integrator m, Visit&& visit) {
    visit(0L, s);
    for (long k = 1; k <= steps; ++k) {
        s = step(p, s, h, m);
        visit(k, s);
    }
    return s;
}

inline State simulate(const Params& p, State s, double h, long steps, Integrator m) {
    return simulate(p, s, h, steps, m, [](long, const State&) {});
}

// Small-oscillation angular frequencies about the hanging equilibrium:
// omega^2 are the eigenvalues of M0^{-1} K with M0 = [[a, b], [b, c]], K = diag(d, e).
inline std::array<double, 2> normal_mode_frequencies(const Params& p) {
    const double det = p.a * p.c - p.b * p.b;
    // M0^{-1} K = (1/det) [[c d, -b e], [-b d, a e]]
    const double tr = (p.c * p.d + p.a * p.e) / det;
    const double dt = (p.d * p.e) / det;  // det(M0^{-1}K) = d e / det(M0)
    const double disc = std::sqrt(tr * tr - 4.0 * dt);
    return {std::sqrt(0.5 * (tr - disc)), std::sqrt(0.5 * (tr + disc))};
}

}  // namespace pendulab

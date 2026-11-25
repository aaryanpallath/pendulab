// Self-contained tests (no framework): each check prints PASS/FAIL; exit code = failures.
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "pendulab/model.hpp"

using namespace pendulab;

static int failures = 0;
static constexpr double kPi = 3.14159265358979323846;
#define CHECK(cond, ...)                                    \
    do {                                                    \
        if (cond) {                                         \
            std::printf("PASS ");                           \
        } else {                                            \
            std::printf("FAIL ");                           \
            ++failures;                                     \
        }                                                   \
        std::printf(__VA_ARGS__);                           \
        std::printf("\n");                                  \
    } while (0)

static const Params kRigid{4.0, 0.9, 1.0, 254.0, 99.6, 0.0, 0.0};  // close to the fitted pendulum

static double state_err(const State& x, const State& y) {
    double e = 0;
    for (int i = 0; i < 4; ++i) e = std::fmax(e, std::fabs(x[i] - y[i]));
    return e;
}

// Textbook point-mass double pendulum (e.g. Goldstein-style derivation), solved independently.
static std::array<double, 2> textbook_accel(double m1, double m2, double l1, double l2, double g, const State& s) {
    const double D = s[0] - s[1];
    const double den = 2 * m1 + m2 - m2 * std::cos(2 * D);
    const double a1 = (-g * (2 * m1 + m2) * std::sin(s[0]) - m2 * g * std::sin(s[0] - 2 * s[1]) -
                       2 * std::sin(D) * m2 * (s[3] * s[3] * l2 + s[2] * s[2] * l1 * std::cos(D))) /
                      (l1 * den);
    const double a2 = (2 * std::sin(D) *
                       (s[2] * s[2] * l1 * (m1 + m2) + g * (m1 + m2) * std::cos(s[0]) + s[3] * s[3] * l2 * m2 * std::cos(D))) /
                      (l2 * den);
    return {a1, a2};
}

static void test_point_mass_matches_textbook() {
    std::mt19937 rng(1);
    std::uniform_real_distribution<double> u(-3, 3);
    double worst = 0;
    for (int i = 0; i < 1000; ++i) {
        const double m1 = 0.5 + std::fabs(u(rng)), m2 = 0.5 + std::fabs(u(rng));
        const double l1 = 0.2 + std::fabs(u(rng)) / 3, l2 = 0.2 + std::fabs(u(rng)) / 3;
        const State s{u(rng), u(rng), 2 * u(rng), 2 * u(rng)};
        const auto mine = accel(Params::point_mass(m1, m2, l1, l2), s);
        const auto ref = textbook_accel(m1, m2, l1, l2, 9.81, s);
        worst = std::fmax(worst, std::fmax(std::fabs(mine[0] - ref[0]), std::fabs(mine[1] - ref[1])) /
                                     (1 + std::fabs(ref[0]) + std::fabs(ref[1])));
    }
    CHECK(worst < 1e-12, "point-mass case matches the textbook equations (max rel diff %.2e)", worst);
}

static void test_convergence_orders() {
    const State s0{2.0, -1.0, 0.0, 0.0};  // large swing: strongly nonlinear
    const double T = 0.5;
    const State ref = simulate(kRigid, s0, T / 65536, 65536, Integrator::RK4);
    // Step counts are chosen per method so errors sit well above round-off (~1e-13):
    // at 4096+ steps RK4 is already at round-off and its measured order collapses.
    struct Case { Integrator m; double expected; long n; };
    for (const Case c : {Case{Integrator::ExplicitEuler, 1, 2048}, Case{Integrator::SemiImplicitEuler, 1, 2048},
                         Case{Integrator::RK4, 4, 512}, Case{Integrator::ImplicitMidpoint, 2, 2048}}) {
        std::vector<double> errs;
        for (long n : {c.n, 2 * c.n, 4 * c.n}) errs.push_back(state_err(simulate(kRigid, s0, T / n, n, c.m), ref));
        const double order = std::log2(errs[1] / errs[2]);
        CHECK(std::fabs(order - c.expected) < 0.2, "%s converges at order %.2f (expected %.0f; errors %.1e -> %.1e)",
              name_of(c.m), order, c.expected, errs[1], errs[2]);
    }
}

static void test_energy_behaviour() {
    const State s0{2.0, -1.0, 0.0, 0.0};
    const double E0 = energy(kRigid, s0), scale = kRigid.d + kRigid.e;
    auto drift = [&](Integrator m, double h, double T) {
        double worst = 0;
        simulate(kRigid, s0, h, static_cast<long>(T / h), m, [&](long, const State& x) {
            worst = std::fmax(worst, std::fabs(energy(kRigid, x) - E0) / scale);
        });
        return worst;
    };
    const double mid = drift(Integrator::ImplicitMidpoint, 1e-3, 20), eul = drift(Integrator::ExplicitEuler, 1e-3, 20);
    CHECK(mid < 1e-3, "implicit midpoint keeps energy error small over 20 s (%.2e)", mid);
    CHECK(eul > 100 * mid, "explicit Euler drifts far more (%.2e)", eul);

    Params damped = kRigid;
    damped.beta1 = 0.05;
    damped.beta2 = 0.02;
    double prev = energy(damped, s0), worst_rise = 0;
    simulate(damped, s0, 1e-4, 50000, Integrator::RK4, [&](long, const State& x) {
        const double e = energy(damped, x);
        worst_rise = std::fmax(worst_rise, e - prev);
        prev = e;
    });
    CHECK(worst_rise < 1e-9 * scale, "with damping, energy never increases (max step rise %.2e)", worst_rise);
}

static void test_small_oscillation_frequencies() {
    // Excite each normal mode alone and compare the measured period with the eigenvalue.
    const auto w = normal_mode_frequencies(kRigid);
    const double det = kRigid.a * kRigid.c - kRigid.b * kRigid.b;
    for (int mode = 0; mode < 2; ++mode) {
        // Eigenvector of M0^{-1}K for omega^2: from its first row, (c d/det - w^2) v1 - (b e/det) v2 = 0.
        const double lam = w[mode] * w[mode];
        const double v1 = kRigid.b * kRigid.e / det, v2 = kRigid.c * kRigid.d / det - lam;
        const double amp = 1e-4 / std::fmax(std::fabs(v1), std::fabs(v2));
        const State s0{amp * v1, amp * v2, 0.0, 0.0};
        const double h = 1e-5;
        std::vector<double> crossings;
        double prev = s0[0];
        simulate(kRigid, s0, h, static_cast<long>(6 * 2 * kPi / w[mode] / h), Integrator::RK4,
                 [&](long k, const State& x) {
                     if (k > 0 && prev > 0 && x[0] <= 0) crossings.push_back((k - prev / (prev - x[0])) * h);
                     prev = x[0];
                 });
        const double period = (crossings.back() - crossings.front()) / (crossings.size() - 1);
        const double rel = std::fabs(period * w[mode] / (2 * kPi) - 1);
        CHECK(rel < 1e-5, "mode %d period matches eigenvalue prediction (rel err %.1e)", mode + 1, rel);
    }
}

static void test_implicit_midpoint_is_reversible() {
    const State s0{1.0, 2.5, 3.0, -4.0};
    State x = simulate(kRigid, s0, 1e-3, 2000, Integrator::ImplicitMidpoint);
    x = simulate(kRigid, x, -1e-3, 2000, Integrator::ImplicitMidpoint);
    const double e = state_err(x, s0);
    CHECK(e < 1e-9, "implicit midpoint retraces its path when run backwards (err %.1e)", e);
}

int main() {
    test_point_mass_matches_textbook();
    test_convergence_orders();
    test_energy_behaviour();
    test_small_oscillation_frequencies();
    test_implicit_midpoint_is_reversible();
    std::printf("%d failure(s)\n", failures);
    return failures;
}

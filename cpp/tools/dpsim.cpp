// dpsim: integrate the double pendulum and print a CSV trajectory.
//
//   dpsim --integrator rk4 --dt 0.001 --steps 1000 --state q1,q2,w1,w2
//         --params a,b,c,d,e,beta1,beta2 [--every N]
//
// Output columns: t,q1,q2,w1,w2,energy. Values are printed with 17 significant digits
// so the Python side can compare trajectories bit-for-bit-ish (to ~1e-15).
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include "pendulab/model.hpp"

using namespace pendulab;

static std::vector<double> parse_list(const std::string& s, size_t n, const char* what) {
    std::vector<double> v;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) v.push_back(std::strtod(tok.c_str(), nullptr));
    if (v.size() != n) {
        std::fprintf(stderr, "%s needs %zu comma-separated values, got %zu\n", what, n, v.size());
        std::exit(2);
    }
    return v;
}

int main(int argc, char** argv) {
    Integrator m = Integrator::RK4;
    double h = 1e-3;
    long steps = 1000, every = 1;
    State s{0.5, 0.0, 0.0, 0.0};
    Params p;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + arg);
            const std::string val = argv[++i];
            if (arg == "--integrator") m = parse_integrator(val);
            else if (arg == "--dt") h = std::stod(val);
            else if (arg == "--steps") steps = std::stol(val);
            else if (arg == "--every") every = std::stol(val);
            else if (arg == "--state") {
                const auto v = parse_list(val, 4, "--state");
                s = {v[0], v[1], v[2], v[3]};
            } else if (arg == "--params") {
                const auto v = parse_list(val, 7, "--params");
                p = {v[0], v[1], v[2], v[3], v[4], v[5], v[6]};
            } else throw std::invalid_argument("unknown option " + arg);
        }
        if (h <= 0 || steps < 0 || every < 1) throw std::invalid_argument("dt > 0, steps >= 0, every >= 1");
        if (p.a * p.c - p.b * p.b <= 0) throw std::invalid_argument("mass matrix must be positive definite");
        std::printf("t,q1,q2,w1,w2,energy\n");
        simulate(p, s, h, steps, m, [&](long k, const State& x) {
            if (k % every == 0)
                std::printf("%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n", k * h, x[0], x[1], x[2], x[3], energy(p, x));
        });
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "dpsim: %s\n", ex.what());
        return 2;
    }
    return 0;
}

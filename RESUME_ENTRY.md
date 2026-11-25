# pendulab — Double-Pendulum Simulation & System Identification with NVIDIA Warp
Tech: C++17, Python, NVIDIA Warp, NumPy, SciPy, CMake, pytest, GitHub Actions
- Built a double-pendulum simulator in C++17 and Python with 4 integrators, validated against convergence orders, energy conservation and small-oscillation eigenfrequencies.
- Fit physical parameters to real pendulum recordings with a differentiable Warp simulator; 86% of 0.5 s held-out predictions stayed within 5°, vs 26% for least squares.

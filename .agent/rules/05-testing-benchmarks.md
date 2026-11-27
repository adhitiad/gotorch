# Testing and Benchmark Rules

- Correctness precedes performance.
- Compare optimized kernels against Pure Go reference results using operation-specific tolerances.
- Use both absolute and relative tolerance where appropriate; never apply one universal epsilon blindly.
- Benchmarks must state hardware, OS, Go version, CPU features, dtype, tensor shape, and thread settings.
- Separate cold-start, compilation, transfer, and steady-state execution costs.
- Report allocations and memory as well as latency/throughput.
- Avoid flaky wall-clock assertions in unit tests.

---
name: gotorch-add-cpu-kernel
description: Add a portable, tested CPU kernel with a Pure Go reference implementation and optional measured optimizations.
---

# Skill: Add a CPU Kernel

1. Define mathematical semantics, dtype, shape, layout, broadcasting, and error behavior.
2. Implement the Pure Go reference kernel first.
3. Add numerical tests, boundary tests, and fuzz/property tests where useful.
4. Benchmark representative small, medium, and large workloads.
5. Add SIMD/BLAS implementation only when a measurable workload justifies it.
6. Gate optimized dispatch on runtime CPU feature detection and supported layout/dtype.
7. Differential-test optimized output against reference using justified `atol` and `rtol`.
8. Test concurrent invocation and aliasing behavior.
9. Document fallback behavior and benchmark methodology.

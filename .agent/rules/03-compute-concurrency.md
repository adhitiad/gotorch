# Compute and Concurrency Rules

- Every optimized kernel must have a correct reference implementation.
- Kernel dispatch must validate CPU capabilities and dtype/layout support.
- Use chunked work, bounded queues, and sequential thresholds.
- Avoid nested worker pools and uncontrolled BLAS threading.
- Do not mutate shared outputs concurrently unless partitioning is provably disjoint.
- Use local partial reductions followed by a defined aggregation strategy.
- Honor context cancellation at safe boundaries.
- Test determinism where the operation promises deterministic behavior.

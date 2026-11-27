# Architecture Decision Index

This file is an index only. The full decision record is stored as an individual ADR under `docs/adr/`. ADR status is authoritative in those files. These decisions are architectural constraints, not proof that all related runtime capabilities exist.

| ADR | Topic | Decision | Status |
|---|---|---|---|
| [ADR-0001](adr/0001-full-framework-cpu-hybrid-distributed-first.md) | Full framework, hybrid CPU, distributed-first architecture | Full framework scope; Pure Go correctness baseline; optional optimized CPU paths; distributed contracts from the foundation | Accepted |
| [ADR-0002](adr/0002-pure-go-reference-and-optional-optimized-backends.md) | CPU backend strategy | Pure Go required reference; SIMD and BLAS/LAPACK optional and differentially tested | Accepted |
| [ADR-0003](adr/0003-distributed-first-class-but-local-fast-path.md) | Distributed execution | Distributed identity/placement contracts early; local execution remains a low-overhead in-process fast path | Accepted |

## ADR requirements

A new ADR should include: ID, title, status, date, context, constraints, decision, alternatives, consequences, security impact, compatibility/migration and validation evidence.

## Status vocabulary

Proposed, Accepted, Rejected, Deprecated, Superseded. Change a fundamental accepted decision through a superseding ADR rather than silently rewriting its rationale.

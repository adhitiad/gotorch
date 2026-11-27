# Architecture Decision Log

This index records decisions that must be formalized as individual ADRs as the implementation matures. These entries are design constraints, not proof that the corresponding runtime exists.

| ID | Topic | Current direction | Status |
|---|---|---|---|
| ADR-0001 | Pure Go reference | Portable correctness baseline | Proposed |
| ADR-0002 | Hybrid CPU compute | Pure Go + optional SIMD + optional BLAS/LAPACK | Proposed |
| ADR-0003 | Tensor memory model | Explicit shape, stride, ownership, and aliasing contracts | Proposed |
| ADR-0004 | Autograd | Explicit graph lifecycle and gradient accumulation semantics | Proposed |
| ADR-0005 | Parallel scheduler | Bounded workers and explicit nested-parallelism policy | Proposed |
| ADR-0006 | Runtime dispatch | Runtime CPU feature detection with safe fallback | Proposed |
| ADR-0007 | Distributed execution | First-class worker and collective contracts | Proposed |
| ADR-0008 | GPU extension | Optional backend behind stable interfaces; future scope | Proposed |
| ADR-0009 | Platform support | Windows AMD64 primary; Linux AMD64/ARM64 compatibility | Proposed |
| ADR-0010 | API stability | Explicit compatibility policy before 1.0 | Proposed |

## ADR template
Each ADR should include: title, status, date, owners, context, constraints, decision, alternatives, consequences, security impact, compatibility/migration, and validation evidence.

## Status vocabulary
Proposed, Accepted, Rejected, Deprecated, Superseded. An accepted decision should be changed through a superseding ADR rather than silently rewritten.

# Decision Making

## Lightweight changes
Typos, isolated documentation clarifications, and internal maintenance can proceed through a focused pull request.

## Architecture changes
Changes affecting public APIs, tensor layout, numerical semantics, memory ownership, concurrency, dispatch, platform support, distributed protocols, or dependency policy require:
1. Problem statement and constraints.
2. At least two viable alternatives where practical.
3. Trade-off analysis: correctness, performance, portability, complexity, and maintenance.
4. Compatibility and migration plan.
5. Test and benchmark evidence or a validation plan.
6. An Architecture Decision Record (ADR).

## Review
Reviewers should distinguish blocking correctness/security concerns from non-blocking preferences. The author should respond to each blocking comment. The project lead resolves deadlocks and records the rationale.

## ADR lifecycle
Proposed → Accepted / Rejected → Superseded. Accepted ADRs are not silently edited; create a superseding ADR when the decision changes.

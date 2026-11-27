---
name: gotorch-review-change
description: Review an implementation for correctness, architecture, memory, concurrency, security, and portability.
---

# Skill: Review a GoTorch Change

Review in this order:

1. Architecture alignment and phase scope.
2. Correctness and numerical semantics.
3. Memory ownership, aliasing, and resource lifetime.
4. Concurrency, cancellation, deadlocks, and races.
5. Distributed failure and security boundaries.
6. Platform portability and optional dependency behavior.
7. API compatibility and documentation.
8. Test quality and benchmark validity.

Report findings by severity with file/line references, impact, and a concrete remediation. Do not approve based solely on compilation.

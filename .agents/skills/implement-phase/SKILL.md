---
name: gotorch-implement-phase
description: Execute one roadmap task using dependency, implementation, verification, and reporting gates.
---

# Skill: Implement a Roadmap Phase

## Inputs
Phase ID, task scope, repository state, and acceptance criteria.

## Procedure
1. Read root `AGENTS.md`, phase roadmap, architecture, and relevant ADRs.
2. Inspect current packages, tests, git status, and public API.
3. Map dependencies and identify the smallest vertical slice.
4. Write a plan listing files, contracts, tests, and risks.
5. Implement behavior; avoid empty scaffolding and speculative abstractions.
6. Add unit, boundary, differential, and concurrency tests as applicable.
7. Update docs and examples.
8. Run gofmt, go test, go vet, race tests, and benchmarks relevant to the change.
9. Review exported API, platform compatibility, and regressions.
10. Report completed work, exact verification, limitations, and remaining tasks.

## Stop conditions
Stop and ask for a decision if requirements conflict with fundamental architecture, public API compatibility, security, or data-loss safety. Do not silently weaken acceptance criteria.

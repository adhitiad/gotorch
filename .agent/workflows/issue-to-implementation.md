# Workflow: Issue to Implementation

1. Read root `AGENTS.md` and `docs/TASK.md`.
2. Select the issue's Task ID (P0–P13) and verify dependencies are complete.
3. Check existing issue/PR ownership so another agent is not implementing the same task.
4. Confirm scope, acceptance criteria, public contracts and affected `src/` packages.
5. Create a task/agent branch from `adhitiad_gpt`; do not push to `main`.
6. Write an implementation plan covering files, API changes, tests and risks.
7. Implement the smallest end-to-end slice.
8. Test correctness, boundary conditions, concurrency, portability and security as applicable.
9. Update docs/ADR and the matching `docs/TASK.md` checklist only after verification.
10. Open a PR to `adhitiad_gpt`; report exact commit, tests, unverified items and next unblocked task.

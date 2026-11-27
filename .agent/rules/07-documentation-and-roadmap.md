# Documentation and Roadmap Rules

- `docs/TASK.md` is the canonical source for P0–P13 task IDs, dependencies, status and checklists.
- Update the task checklist in the same change that implements a task; only check items after verification.
- Update architecture docs when a contract or subsystem boundary changes.
- Mark planned, implemented, experimental and deprecated capabilities distinctly.
- Never describe a roadmap item as implemented until tests and acceptance criteria pass.
- Keep `README.md` concise; put detailed design in `docs/`.
- Every phase must have deliverables, dependencies, tests and exit criteria.
- Record material architecture decisions as ADRs under `docs/adr/`.

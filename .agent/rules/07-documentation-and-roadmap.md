# Documentation and Roadmap Rules

- `docs/TASK.md` is the canonical source for P0–P13 task IDs, dependencies, status and checklists.
- Every completed task must have its matching checkbox ticked in `docs/TASK.md` in the same change that implements it, and only after verification. An in-progress task remains unchecked.
- Work that has no task ID in `docs/TASK.md` must be recorded in `docs/off-task.md` with date, scope, files, verification and status. That file is an event log, not a second roadmap; never invent task IDs or duplicate phase statuses there.
- Update architecture docs when a contract or subsystem boundary changes.
- Mark planned, implemented, experimental and deprecated capabilities distinctly.
- Never describe a roadmap item as implemented until tests and acceptance criteria pass.
- Keep `README.md` concise; put detailed design in `docs/`.
- Every phase must have deliverables, dependencies, tests and exit criteria.
- Record material architecture decisions as ADRs under `docs/adr/`.

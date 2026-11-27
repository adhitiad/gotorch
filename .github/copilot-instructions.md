# GitHub Copilot instructions

Follow the same repository-wide contract as every other coding agent:

1. Read the root [`AGENTS.md`](../AGENTS.md).
2. Read [`docs/TASK.md`](../docs/TASK.md) and select one unchecked task ID whose dependencies are complete.
3. Read the relevant rules in [`.agent/rules/`](../.agent/rules/) and skill in [`.agents/skills/`](../.agents/skills/).
4. Put reusable Go code under `src/`, executables under `cmd/`, and application composition under `app/`. Preserve the root `gotorch.go` facade.
5. Update tests, documentation and the matching checklist. Do not mark work complete before verification.
6. Do not push to `main` unless explicitly authorized by the repository owner.

`AGENTS.md` and `docs/TASK.md` are the authoritative instruction and task sources; do not create a separate Copilot roadmap.

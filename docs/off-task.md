# Off-task work log

This file records completed work that does **not** map to a task ID in [`TASK.md`](TASK.md). It is an event log, not a second roadmap: it never carries phase status and never invents task IDs. An item that later becomes a canonical task should be marked superseded by that task ID in `TASK.md`.

## Entry format

```markdown
### OFF-YYYYMMDD-NNN — Short title
- Date: YYYY-MM-DD
- Reason off-task: why no task ID in TASK.md applies
- Scope: what changed
- Files: changed paths
- Verification: exact commands and results
- Status: completed | superseded by <task ID>
```

## Entries

### OFF-20261128-001 — Mandatory checklist and off-task logging rule
- Date: 2026-11-28
- Reason off-task: documentation and agent-rule maintenance; no P0–P13 task ID covers this change.
- Scope: every completed task must have its matching checkbox ticked in `TASK.md` after verification; work without a task ID must be logged here. The rule is added to `AGENTS.md`, `.agent/rules/07-documentation-and-roadmap.md`, `.agents/skills/implement-phase/SKILL.md` and `TASK.md`.
- Files: `AGENTS.md`, `.agent/rules/07-documentation-and-roadmap.md`, `.agents/skills/implement-phase/SKILL.md`, `docs/TASK.md`, `docs/off-task.md`, `docs/README.md`.
- Verification: documentation-only change; repository Go checks were run to confirm no source regression, and relative documentation links were reviewed.
- Status: completed

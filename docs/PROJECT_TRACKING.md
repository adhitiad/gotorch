# Project Tracking System

## Source of truth

- `docs/TASK.md` is the canonical P0–P13 phase plan: Task IDs, dependencies, deliverables, checklists and phase exit gates.
- GitHub Issues track assignment, ownership, discussion, blockers and review for individual tasks.
- GitHub Projects provides planning/kanban views; it must not become a conflicting copy of phase acceptance criteria.
- `docs/ROADMAP.md` is a compact overview only. Detailed status belongs in `docs/TASK.md`.

When an issue/PR changes a phase deliverable, update `docs/TASK.md` in the same change. A task is checked only after implementation and relevant validation pass.

## Recommended project fields

- Status: Backlog, Ready, In Progress, In Review, Blocked, Done.
- Priority: Critical, High, Medium, Low.
- Phase: P0–P13.
- Task ID: stable identifier from `docs/TASK.md`, such as `P03-006`.
- Area: Tensor, Autograd, CPU, Memory, Runtime, NN, Data, Compiler, Distributed, Interop, Tooling, Security, Docs.
- Estimate: XS, S, M, L, XL.
- Blocked by: linked issue(s).
- Owner/Agent: one accountable implementation owner per task.

## Views

1. Roadmap: grouped by phase and priority.
2. Kanban: grouped by status.
3. Contributor queue: good first issue and help wanted.
4. Quality: bugs, regressions, performance and flaky tests.
5. Security: restricted visibility; never expose sensitive reports in a public project.
6. Release: milestone scope and readiness checklist.

## Issue quality

Every implementation issue should include Task ID, context, expected outcome, acceptance criteria, dependencies, files/areas likely affected, and validation. Break work larger than one reviewable PR into a parent issue and child issues.

## Definition of Ready

- Problem and user impact are clear.
- Task ID and phase are identified.
- Dependencies and active ownership are checked.
- Acceptance criteria are testable.
- Risks and design questions are known.
- Design review is complete when required.

## Definition of Done

- Implementation or documentation is complete.
- Tests and required checks pass.
- Docs and ADRs are updated when relevant.
- Matching checklist in `docs/TASK.md` is updated.
- Review feedback is resolved.
- Platform and benchmark evidence is recorded.
- Issue is linked to the PR merged into the development branch.

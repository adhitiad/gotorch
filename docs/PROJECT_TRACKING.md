# Project Tracking System

## Source of truth
GitHub Issues track actionable work. GitHub Projects provides planning views. The roadmap in docs/ROADMAP.md describes phase scope and dependencies; it does not replace issue-level acceptance criteria.

## Recommended project fields
- Status: Backlog, Ready, In Progress, In Review, Blocked, Done.
- Priority: Critical, High, Medium, Low.
- Phase: P0–P13.
- Area: Tensor, Autograd, CPU, Memory, Runtime, NN, Data, Compiler, Distributed, Interop, Tooling, Security, Docs.
- Estimate: XS, S, M, L, XL.
- Target release: milestone or version.
- Blocked by: linked issue(s).

## Views
1. Roadmap: grouped by phase and priority.
2. Kanban: grouped by status.
3. Contributor queue: good first issue and help wanted.
4. Quality: bugs, regressions, performance, and flaky tests.
5. Security: restricted visibility; never expose sensitive reports in a public project.
6. Release: milestone scope and readiness checklist.

## Issue quality
Every implementation issue should include context, expected outcome, acceptance criteria, dependencies, and validation. Break work larger than one reviewable PR into a parent issue and child issues.

## Definition of Ready
- Problem and user impact are clear.
- Scope and phase are identified.
- Acceptance criteria are testable.
- Dependencies and risks are known.
- Design review is complete when required.

## Definition of Done
- Implementation or documentation is complete.
- Tests and required checks pass.
- Docs and changelog are updated when relevant.
- Review feedback is resolved.
- Platform and benchmark evidence is recorded.
- Issue is linked to the merged PR.

## Labels
Suggested labels: priority: critical/high/medium/low; type: bug/feature/docs/performance/security/research; area: tensor/autograd/cpu/runtime/distributed; status: triage/needs-design/blocked; help wanted; good first issue.

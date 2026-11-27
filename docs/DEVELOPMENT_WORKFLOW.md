# Development Workflow

## Branch model

- `main` is the stable integration/release branch. Do not push directly to it unless the repository owner explicitly authorizes that exact change.
- `adhitiad_gpt` is the active development/integration branch.
- For parallel work, create a short-lived task branch from `adhitiad_gpt`: `feat/P01-003-tensor-constructor`, `fix/P03-006-simd-dispatch`, `docs/P0-agent-contract`, and so on.
- Open task pull requests against `adhitiad_gpt`. Integrate to `main` only through an explicitly approved promotion/release step.
- Never force-push shared branches. Never commit secrets or generated build outputs.

## Commit convention

Use Conventional Commits: `feat`, `fix`, `docs`, `chore`, `test`, `perf`, `refactor`, `build`, `ci` and `security`. Include the phase/task ID where appropriate, for example `feat(P01-003): add tensor constructors`. Keep each commit focused and explain why the change is needed.

## Task lifecycle

1. Select a single unchecked Task ID from [`docs/TASK.md`](TASK.md).
2. Confirm the task's dependencies and check for an existing issue/PR or active owner.
3. Define scope, API impact, files, tests, risks and acceptance criteria.
4. Implement the smallest reviewable vertical slice.
5. Add/update tests and documentation. Update the canonical task checklist only after verification.
6. Run `gofmt`, `go test ./...`, `go vet ./...` and relevant race/fuzz/benchmark/platform checks.
7. Open a PR to `adhitiad_gpt`, link the Task ID and issue, and report exact evidence.
8. Promote to `main` only after the repository owner/authorized maintainer approves.

## Review priorities

Correctness, numerical semantics, memory safety, concurrency, security, API compatibility, portability, maintainability and performance evidence—in that order.

## Change size

Prefer small, reviewable pull requests. Split broad work into independently verifiable changes. Avoid mixing architecture redesign with unrelated formatting or cleanup.

# Development Workflow

## Branch model
- main is the stable integration and release branch.
- Use short-lived branches: feat/<topic>, fix/<topic>, docs/<topic>, chore/<topic>, perf/<topic>, security/<topic>.
- Open pull requests against main unless a release branch policy is explicitly introduced.
- Never force-push shared branches. Never commit secrets or generated build outputs.

## Commit convention
Use Conventional Commits: feat, fix, docs, chore, test, perf, refactor, build, ci, and security. Keep each commit focused and explain why the change is needed.

## Pull request lifecycle
1. Link an issue or explain why no issue is needed.
2. Describe scope, design, risks, and compatibility impact.
3. Add tests and docs for behavior changes.
4. Run gofmt, go test ./..., go vet ./..., and relevant race/fuzz/benchmark checks.
5. Request review; resolve all blocking feedback.
6. Merge only after required checks pass and the change is approved by the project lead or authorized maintainer.

## Review priorities
Correctness, numerical semantics, memory safety, concurrency, security, API compatibility, portability, maintainability, and performance evidence—in that order.

## Change size
Prefer small, reviewable pull requests. Split broad work into independently verifiable changes. Avoid mixing architecture redesign with unrelated formatting or cleanup.

# AGENTS.md — GoTorch Project Instructions

These instructions are the repository-wide contract for every human or coding agent working in GoTorch. Read this file first, then read `docs/TASK.md` before selecting or implementing a task. More-specific nested `AGENTS.md` files may add rules but must not weaken these instructions.

## 1. Project identity and architecture

- GoTorch is a Go-native **full deep-learning and numerical-computing framework**: tensors, autograd, neural networks, optimizers, data, CPU compute, compiler, training, inference, quantization, model interoperability and distributed compute.
- Module: `github.com/adhitiad/gotorch`; Go version is declared in `go.mod` (currently `go 1.27.4`).
- **Hybrid CPU:** Pure Go is the correctness/portability reference. SIMD and BLAS/LAPACK are optional optimized paths.
- **Distributed by design:** distributed contracts are first-class; ordinary local operations must not incur network overhead.
- Platforms: **Windows AMD64 primary**; Linux AMD64 and Linux ARM64 compatibility.
- No mandatory GPU/CUDA/DirectML/CGo/vendor SDK in core.
- Do not claim a capability is implemented merely because an interface, package, config entry or document exists. For P0 status, use `docs/TASK.md` as the canonical checklist.

## 2. Canonical task and instruction sources

For agent-specific discovery paths and first-run setup, see [docs/AGENT_SETUP.md](docs/AGENT_SETUP.md).

Read in this order:
1. `AGENTS.md` (this file).
2. `docs/TASK.md` — canonical phase checklist, task IDs, dependencies and acceptance criteria.
3. The relevant documents under `docs/` and `docs/adr/`.
4. Relevant `.agent/rules/*.md`.
5. The matching portable skill under `.agents/skills/<skill>/SKILL.md`.
6. Related workflows/templates under `.agent/`.

Do not create a competing roadmap or maintain duplicate phase statuses in other files. When scope/status changes, update `docs/TASK.md` in the same PR/commit.

## 3. Repository layout — mandatory

- `src/<package>/`: all reusable Go framework implementation packages and their `*_test.go` files. New core packages must live under `src/`.
- Package import paths therefore start with `github.com/adhitiad/gotorch/src/...`.
- `cmd/<name>/main.go`: executable entry points and `package main` programs.
- `app/<name>/`: application composition roots, service wiring, external adapters and app-specific orchestration.
- Root `gotorch.go` and `gotorch_test.go`: the deliberately small public root-package facade retained for compatibility with imports of `github.com/adhitiad/gotorch`. Do not put framework internals here.
- `go.mod`, `AGENTS.md`, documentation, community policies, CI and repository metadata stay at their current repository-level locations.
- Do not create empty future-phase directories. Create a package when a real implementation task owns it.
- Go test fixtures belong in the relevant package's `testdata/` directory.

See [Repository Layout and Phase Plan](docs/TASK.md#2-repository-layout-and-source-code-placement).

## 4. Task execution protocol

- Select one unchecked task ID from `docs/TASK.md`; verify its dependencies first.
- Check existing issues/PRs and branch state before changing files. Do not duplicate another agent's active task.
- Work on `adhitiad_gpt` or a dedicated task/agent branch based on it. Integrate by PR when work is parallel.
- Do not push directly to `main`, force-push, rewrite history or overwrite other agent work unless the repository owner explicitly authorizes it.
- Keep changes scoped to the selected task. State public API, numerical, memory, concurrency, portability and security impacts.
- Update tests, docs/ADR and the checklist with implementation changes.
- Mark checklist items complete only after the relevant implementation and verification pass.
- Report exact paths, commit/PR, commands, results, limitations and next unblocked task.

## 5. Engineering constraints

- Keep public APIs small, typed, explicit and documented. Avoid unnecessary exported internals and premature abstractions.
- Validate dimensions, strides, offsets, allocation sizes and all multiplication/addition overflow before allocation or slicing.
- Define dtype, layout, shape, aliasing, mutability, ownership, lifetime and concurrency semantics explicitly.
- Every supported core operation must have a Pure Go reference implementation. Optimized kernels must be differentially tested against it.
- Detect CPU features safely at runtime; never assume a host's SIMD support.
- Bound goroutines, queues and memory. No goroutine-per-element or unbounded task fan-out.
- Coordinate Go worker parallelism with BLAS threads. Never silently mutate process-wide `GOMAXPROCS`.
- Honor `context.Context` cancellation at safe boundaries.
- Distributed operations must define run/rank/world identity, collective ordering, deadlines, bounded payloads, retry/idempotency and failure/recovery semantics.
- Authenticate remote worker control paths. Treat model files, checkpoints, datasets, graphs and remote messages as untrusted data.
- Errors must preserve context and support `errors.Is`/`errors.As`. Do not panic for ordinary invalid user input.
- Keep source files under 395 lines where practical; exceed only with a clear reason recorded in review.

## 6. Verification commands

Run focused tests while iterating; run all relevant checks before reporting completion.

```bash
gofmt -w .
go test ./...
go test -race ./...           # concurrency-sensitive; CI runs race on Linux only
go vet ./...
go test -bench=. -benchmem ./...
GOOS=windows GOARCH=amd64 go build ./...
GOOS=linux GOARCH=amd64 go build ./...
GOOS=linux GOARCH=arm64 go build ./...
```

- `Makefile` wrappers: `make fmt|test|race|vet|check|bench` (`check=test+vet`).
- No `golangci-lint`, typecheck, or codegen config is present; do not invent them.
- CI (`.github/workflows/ci.yml`) is the native verification source of truth: Windows AMD64, Linux AMD64, Linux ARM64 (`ubuntu-24.04-arm`), Go `1.27.x`; runs `go test ./...`, `go vet ./...`, and `go test -race ./...` on Linux. Push branches: `main`, `adhitiad_gpt`. Local cross-builds do not replace native CI.

## 7. Definition of done

A task is not done because code compiles once. Implementation, tests, docs, API impact, platform compatibility and exact verification evidence must be accounted for. Report checks not run. A phase is complete only when all its task checkboxes and exit gates in `docs/TASK.md` have been reviewed and satisfied.

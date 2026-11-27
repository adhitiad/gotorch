# Agent Instructions — GoTorch

Root instructions for all coding agents. More-specific nested `AGENTS.md` files may refine these; do not weaken root constraints. Read relevant `.agent/rules/*.md` before editing.

## Project / P0 context

- Go-native **full deep-learning and computing framework** (tensor, autograd, NN, optim, training, inference, compiler, quantization, interoperability, distributed). Module: `github.com/adhitiad/gotorch` (`go.mod`: `go 1.27.4`).
- **Hybrid CPU**: Pure Go is the correctness/portability reference; SIMD and BLAS/LAPACK are optional optimized paths selected by safe runtime capability detection.
- **Distributed is first-class in contracts**, but local execution must not pay unnecessary distributed/network overhead.
- Platforms: **Windows AMD64 primary**; Linux AMD64 and Linux ARM64 compatibility.
- **No mandatory GPU/CUDA/DirectML/CGo/vendor SDK** in core. Do not claim performance wins without reproducible, fair benchmarks.
- Current state is a **P0 documentation/foundation scaffold** (`gotorch.go`, `gotorch_test.go`). Do not create empty packages just to match a target tree; implement phases behind tested contracts. See `docs/ROADMAP.md`, `docs/CORE_CONTRACTS.md`, and `docs/adr/`.

## Commands (verified)

```bash
gofmt -w .                    # format
go test ./...                 # tests
go test -race ./...           # race (Linux-native in CI; run when concurrency-sensitive)
go vet ./...                  # vet
go test -bench=. -benchmem ./...  # benchmarks
make fmt|test|race|vet|check|bench # Makefile wrappers; check=test+vet
```

- No `golangci-lint`/typecheck/codegen config is present; do not invent them. Use `gofmt` + `go test` + `go vet` (+ race/bench as relevant).
- Run focused package tests while iterating; run full relevant checks before reporting done.
- CI (`.github/workflows/ci.yml`): Windows AMD64, Linux AMD64, Linux ARM64 (`ubuntu-24.04-arm`), Go `1.27.x`, runs `go test ./...`, `go vet ./...`, and `go test -race ./...` on Linux only. Push branches: `main`, `adhitiad_gpt`.

## Repo map / sources of truth

- `README.md` - concise project overview; detailed design in `docs/`.
- `docs/` - architecture/contracts/roadmap/testing/platform/security; ADRs in `docs/adr/`.
- `.agent/rules/` - mandatory domain rules (style, tensor/memory, compute/concurrency, distributed, testing/bench, security, docs/roadmap). `.agent/skills/`, `workflows/`, `templates/` are task playbooks.
- `CONTRIBUTING.md` - short contributor flow; license selection pending, do not assume a license.

## Engineering constraints

- Keep public APIs small, explicit, documented. No unnecessary exported internals, no `interface{}` hot paths where typed dispatch fits, no premature generic abstractions that obscure kernel dispatch.
- Validate shape/dimensions and `numel`/byte-size overflow before allocation. Define dtype/layout/shape/aliasing/ownership/lifetime/mutability/concurrency semantics explicitly.
- Pure Go reference kernels required for supported core ops; optimized kernels must be differentially tested against them. Never assume CPU features; detect at runtime.
- Bound goroutines/queues; no goroutine-per-element, no nested parallelism, coordinate Go workers with BLAS threads. Never silently change `GOMAXPROCS` or process-global config.
- Never concurrently mutate shared gradient/output buffers without a defined reduction/partition strategy. Honor `context` cancellation at safe boundaries.
- Distributed ops must define rank/world/run identity, collective semantics, deadlines/timeouts, bounded payloads, retry/idempotency, and failure/recovery behavior. No unauthenticated worker control endpoints.
- Treat model files/checkpoints/datasets/graph defs/remote messages as untrusted: enforce size/rank/dimension/string/nesting/allocation limits; reject overflow/malformed offsets before allocation/slicing.
- Errors preserve context and support `errors.Is`/`errors.As`; no panic for ordinary invalid input. Keep secrets out of logs/errors.
- Code should be <395 lines per file; exceed only with a clear reason (from `.agent/rules/00-project-principles.md`).

## Testing / docs / git

- Tests: unit, boundary, overflow/aliasing/non-contiguous/zero-size, differential (optimized vs Pure Go), fuzz/property for shape/parsers/serialization/graph, race for concurrency-sensitive changes, distributed failure/timeout/cancellation/recovery when applicable. Benchmarks use `-benchmem`; report hardware/OS/Go/CPU features/dtype/shape/thread settings.
- Update docs/ADRs when contracts or subsystem boundaries change. Mark capabilities planned/implemented/experimental/deprecated honestly; do not mark roadmap items implemented until acceptance criteria pass.
- No force-push/history rewrite; do not modify `main` without explicit authorization. Keep commits focused. Do not claim commit/push/PR/merge unless verified.

## Definition of done

Implementation, tests, docs, compatibility impact, and exact verification commands/results are accounted for. A stub is not a completed feature. Report checks not run.
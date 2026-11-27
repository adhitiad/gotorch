# Agent Instructions — GoTorch

These instructions apply to every automated coding agent and contributor working in this repository. More specific instructions may exist in nested `AGENTS.md` files; follow the most specific applicable instructions without weakening these root constraints.

## 1. Project identity and non-negotiable decisions

GoTorch is a Go-native **full deep-learning and computing framework**.

- Scope: tensor engine, autograd, neural networks, optimizers, training, inference, compiler, quantization, interoperability, and distributed computing.
- Compute strategy: **hybrid CPU**. Pure Go is the portable reference path; SIMD and BLAS/LAPACK are optional optimized paths.
- Distributed computing is first-class in architecture, but local execution must not pay unnecessary network/distributed overhead.
- Windows AMD64 is the primary production platform. Linux AMD64 and Linux ARM64 are compatibility targets.
- GPU is not a core dependency or an early milestone. Do not introduce mandatory CUDA, DirectML, GPU drivers, CGo, or native vendor SDKs.
- Module path: `github.com/adhitiad/gotorch`.
- Do not claim performance superiority over PyTorch, TensorFlow, or another framework without reproducible, fair benchmarks.

## 2. Required workflow

1. Read `README.md`, relevant `docs/` pages, and applicable `.agent/rules/` before editing.
2. Identify the roadmap phase and acceptance criteria for the task.
3. Inspect existing code and tests before proposing replacements.
4. Write a short implementation plan for changes spanning multiple packages.
5. Make the smallest coherent change; avoid unrelated refactors.
6. Add or update tests and documentation in the same change.
7. Run formatting, tests, race tests where applicable, vet, and relevant benchmarks.
8. Report exact commands and results. Distinguish executed checks from checks not run.

## 3. Engineering rules

- Use idiomatic Go and keep public APIs small, explicit, and documented.
- Prefer standard library dependencies. Every external dependency requires a clear justification.
- Keep Pure Go reference implementations available for supported core operations.
- Never assume CPU instruction support. Detect features at runtime and dispatch safely.
- Never silently change `GOMAXPROCS` or global process configuration from a library.
- Bound goroutines and queues. Do not spawn one goroutine per tensor element.
- Avoid nested parallelism and coordinate Go workers with BLAS thread counts.
- Define tensor storage ownership, aliasing, mutability, and lifetime explicitly.
- Do not return pooled or arena-backed memory after its lifetime has ended.
- Never concurrently mutate a shared gradient buffer without a defined reduction strategy.
- Make cancellation, shutdown, and resource ownership explicit.
- Distributed operations must define rank identity, collective semantics, timeout, retry, and failure behavior.
- Never deserialize untrusted model or checkpoint data without size, shape, depth, and resource limits.
- Errors must preserve context and support `errors.Is`/`errors.As` where appropriate.

## 4. Platform rules

- CI must include Windows AMD64 and Linux AMD64.
- Linux ARM64 must be tested natively when a runner is available; otherwise cross-compile and clearly label that limitation.
- Keep platform-specific code behind build constraints and narrow interfaces.
- Core CPU functionality must build without CGo or vendor SDKs.
- Use `filepath` and platform-safe filesystem operations.

## 5. API and compatibility

- Do not expose internal implementation types unnecessarily.
- Do not change public API semantics without updating docs and tests.
- Avoid premature generic abstractions that make kernel dispatch harder to reason about.
- Keep backend interfaces narrow and avoid `interface{}`-based hot paths where typed dispatch is feasible.
- Document dtype, layout, shape, overflow, numerical tolerance, and concurrency semantics.

## 6. Testing requirements

- Unit tests for correctness and boundary conditions.
- Differential tests against the Pure Go reference kernel for optimized kernels.
- Property/fuzz tests for shape arithmetic, parsers, serialization, and graph validation.
- `go test -race ./...` for concurrency-sensitive changes.
- Benchmarks with `-benchmem`; do not assert speedups without repeatable measurements.
- Distributed tests must include worker failure, timeout, cancellation, and partial-result behavior.

## 7. Git discipline

- Do not force-push, rewrite history, or modify `main` without explicit authorization.
- Use the task's designated development branch when one is provided.
- Do not claim a commit, push, PR, or merge unless the operation succeeded and was verified.
- Keep commits focused and use descriptive messages.

## 8. Definition of done

A task is complete only when its implementation, tests, documentation, compatibility implications, and verification results are accounted for. A directory or interface stub alone is not a completed feature.

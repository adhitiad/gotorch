# GoTorch Engineering Task Plan — P0 to P13

> **This is the canonical phase checklist for humans and coding agents.**
> Project instructions: [`/AGENTS.md`](../AGENTS.md) · Architecture: [`ARCHITECTURE.md`](ARCHITECTURE.md) · Contracts: [`CORE_CONTRACTS.md`](CORE_CONTRACTS.md) · Detailed phase overview: [`ROADMAP.md`](ROADMAP.md)

- **Repository:** `github.com/adhitiad/gotorch`
- **Development branch:** `adhitiad_gpt`
- **Stable/release target:** `main`
- **Current phase:** P2 — Automatic Differentiation
- **Current status:** P0 and P1 are closed. P1-001 through P1-011 are implemented, reviewed, and pass the native Go 1.27 CI matrix on Windows AMD64, Linux AMD64, and Linux ARM64. See [P1 API Review](P1_API_REVIEW.md) and [Tensor API](TENSOR_API.md).

---
## 1. Project definition

GoTorch is a Go-native, full deep-learning and numerical-computing framework. It is not limited to model inference. The intended system covers tensor storage and operations, automatic differentiation, neural-network modules, optimizers, data pipelines, CPU execution, graph compilation, training, inference, quantization, model interoperability, and distributed execution.

### Design constraints

- **CPU-first, hybrid implementation:** Pure Go provides the portable correctness reference. SIMD and optional BLAS/LAPACK are optimized implementations, never mandatory dependencies.
- **Distributed by design:** distributed identity, placement, membership, serialization boundaries, cancellation, and failure semantics are considered early. Local execution must not pay network overhead.
- **Platform commitments:** Windows AMD64 is the primary production target. Linux AMD64 and Linux ARM64 are compatibility targets.
- **Accelerators:** GPU and vendor-specific runtimes are optional future backends. Core must build without CUDA, DirectML, CGo, or vendor SDKs.
- **Numerical correctness:** optimizations must be compared against the reference path using documented tolerances. Performance claims require reproducible benchmarks.
- **Security:** model files, checkpoints, datasets, graph definitions, and messages received from remote workers are untrusted input.

### Why GoTorch is organized in phases

Deep-learning frameworks have tightly coupled subsystems. Tensor layout affects kernels; kernels affect scheduling; scheduling affects memory lifetime; autograd depends on tensor semantics; training depends on autograd and data input; distributed training depends on all of those. The P0–P13 sequence makes those dependencies explicit so agents do not build isolated modules that later conflict.

---

## 2. Repository layout and source-code placement

All reusable Go framework packages belong under `src/`. This is the project's explicit directory convention. Go module import paths therefore include `/src/`.

```text
gotorch/
├── AGENTS.md                    # universal project instructions for coding agents
├── go.mod                       # module root; never move into src/
├── gotorch.go                   # public root-package facade / compatibility entry point
├── gotorch_test.go              # tests for the root facade
├── src/                         # reusable Go framework packages
│   ├── dtype/
│   ├── errs/
│   ├── shape/
│   ├── tensor/
│   ├── storage/
│   ├── compute/
│   ├── runtime/
│   ├── platform/
│   ├── distributed/
│   ├── autograd/                # introduced when P2 begins
│   ├── nn/                      # introduced when P5 begins
│   ├── optim/                   # introduced when P5 begins
│   ├── data/                    # introduced when P6 begins
│   ├── compiler/                # introduced when P7 begins
│   ├── training/                # introduced when P8 begins
│   ├── inference/               # introduced when P8 begins
│   ├── quantization/            # introduced when P9 begins
│   └── interop/                 # introduced when P11 begins
├── cmd/                         # executable entry points; package main lives here
│   └── <tool-or-server>/main.go
├── app/                         # app composition, adapters, dependency wiring
├── docs/                        # architecture, ADRs, TASK.md, specifications
├── .agents/skills/              # portable agent skill definitions (SKILL.md)
├── .agent/rules/                # domain-specific engineering rules
├── .agent/workflows/            # repeatable repository workflows
├── .agent/templates/            # implementation and review templates
├── .github/                     # CI, issue forms, PR templates, CODEOWNERS
└── community/                   # governance and contributor policies
```

The folders shown for future phases are **planned locations**, not a requirement to create empty directories now.

### Placement rules

| Code or asset | Location | Rule |
|---|---|---|
| Reusable Go libraries and framework packages | `src/<package>/` | All framework implementation packages go here |
| Go package tests | Beside the package in `src/<package>/*_test.go` | Keep unit tests close to implementation |
| Executable CLI/server/worker | `cmd/<name>/main.go` | `package main` is allowed here |
| Application composition root, app-specific adapters and wiring | `app/<name>/` | Keep app-specific code out of reusable framework packages |
| Root public package facade | `gotorch.go` | The current root facade stays to preserve `github.com/adhitiad/gotorch` imports; do not place framework internals here |
| Root facade tests | `gotorch_test.go` | Test only the root public facade |
| Go fixtures used only by tests | `src/<package>/testdata/` | Use Go's standard testdata convention |
| Go code generation templates/tools | `src/` for reusable generator libraries; `cmd/` for executable generators | Generated output must have a clear owner and regeneration command |
| Docs, YAML, TOML, JSON, Makefile, workflows | Their existing root/docs/config directories | These are not Go source and do not belong in `src/` |
| Web/UI assets and frontend apps, if ever added | `app/` or a dedicated frontend workspace | Do not mix frontend code into core Go packages |

**Import rule:** packages in `src/` import each other using paths such as `github.com/adhitiad/gotorch/src/dtype`, `github.com/adhitiad/gotorch/src/tensor`, and `github.com/adhitiad/gotorch/src/storage`.

**Do not:**
- Add new reusable framework packages at the repository root.
- Put `package main` in core library packages.
- Put CLI flags, HTTP handlers, or service wiring inside tensor/autograd/kernel packages.
- Create empty folders just to reserve a future phase.
- Move `go.mod`, `AGENTS.md`, docs, CI, governance, or repository metadata into `src/`.
- Change the root module path to include `/src`. Only package import paths include `/src`.

---

## 3. Phase map and dependency order

| Phase | Name | Main result | Depends on |
|---|---|---|---|
| P0 | Foundation | Stable module conventions, contracts, error model, CI and repository rules | — |
| P1 | Tensor Engine | Owned tensor values, storage, views, shape and indexing operations | P0 |
| P2 | Automatic Differentiation | Forward/reverse AD and gradient lifecycle | P1 |
| P3 | CPU Compute | Pure Go reference kernels, safe dispatch, SIMD and optional BLAS | P1 |
| P4 | Scheduler & Memory | Bounded parallel execution and explicit memory reuse/lifetime | P1, P3 |
| P5 | Neural Networks & Optimizers | Modules, layers, losses, optimizers and parameter management | P2, P3, P4 |
| P6 | Data Engine | Dataset, sampler, batching, DataLoader and transforms | P0, P4 |
| P7 | Compiler & Graph Optimization | Typed graph IR, validation, optimization and execution plans | P1, P3, P4 |
| P8 | Training & Inference Runtime | Trainer, evaluation, inference sessions, checkpoint and resume | P5, P6, P7 |
| P9 | Quantization | Mixed precision, calibration, INT8/INT4 and quantized execution | P3, P5, P8 |
| P10 | Distributed Compute | Coordinator, workers, collectives, DDP and recovery | P8; identity contracts start in P0 |
| P11 | Model Interoperability | ONNX, Safetensors and versioned native model format | P1, P5, P8 |
| P12 | Ecosystem & Tooling | CLI, profiler, registry, extension APIs, examples and tutorials | P8–P11 |
| P13 | Production Readiness | Security, stability, performance, release engineering and supportability | P0–P12 |

**Dependency rule:** Do not implement a phase by bypassing an unfinished dependency. A narrow prototype may be proposed as an experiment, but it must be labelled experimental and must not silently become a public contract.

---

## 4. Detailed phase checklists

Checkboxes are the implementation record. Only mark an item complete when its code/documentation exists and its relevant tests pass. A passing test on an earlier commit does not automatically validate a later refactor.

### P0 — Foundation

**Purpose:** create a stable engineering base so independent contributors and agents can implement later phases without inventing incompatible tensor, memory, compute, runtime, or distributed contracts.

**Deliverables:** module/package conventions, public error taxonomy, dtype and shape definitions, tensor metadata, storage ownership contract, compute backend contract, runtime lifecycle, platform matrix, distributed identity, tests, CI, architecture documents, contributor and agent instructions.

**Current status:** maintainer review is complete, edge-case fixes are covered by regression tests, and the reviewed implementation passed the native three-platform CI matrix. See [P0 API Review](P0_API_REVIEW.md) for scope, findings, accepted constraints, and compatibility notes.

- [x] `P00-001` Define module path, supported Go toolchain, architecture constraints and package naming.
- [x] `P00-002` Define stable sentinel errors and wrapping conventions.
- [x] `P00-003` Implement dtype identifiers, validity checks, byte widths and numeric classification.
- [x] `P00-004` Implement immutable shape metadata, scalar/empty semantics and checked element-count arithmetic.
- [x] `P00-005` Implement tensor dtype/shape/stride/layout metadata and checked byte-span calculation.
- [x] `P00-006` Define storage handle, retain/release, allocator and overflow-safe range contracts.
- [x] `P00-007` Define kernel specification, invocation, backend interface and pre-dispatch validation.
- [x] `P00-008` Implement runtime component lifecycle, rollback and reverse shutdown.
- [x] `P00-009` Define distributed run/worker identity and membership validation.
- [x] `P00-010` Define primary and compatibility platform classification.
- [x] `P00-011` Add boundary tests for dtype, shape, metadata, storage range, lifecycle and distributed membership.
- [x] `P00-012` Document architecture, contracts, platform and security boundaries.
- [x] `P00-013` Establish cross-platform CI for Windows AMD64, Linux AMD64 and Linux ARM64.
- [x] `P00-014` Move reusable Go packages into `src/`, update internal imports, and move portable skills into `.agents/skills/`.
- [x] `P00-015` Verify the post-migration and maintainer-review fixes with native Windows AMD64, Linux AMD64 and Linux ARM64 CI ([GitHub Actions run](https://github.com/adhitiad/gotorch/actions/runs/45931933483) passed on reviewed implementation commit `7fa620879b36b16d618c994d1d5f94d925c95532`).
- [x] `P00-016` Complete maintainer review of exported contracts, edge cases and API stability ([review record](P0_API_REVIEW.md)).
- [x] `P00-017` Close P0 after all exit gates were verified, the phase checklist reviewed, and the maintainer decision recorded in the P0 review/implementation documents.

**P0 is not a tensor engine.** No tensor arithmetic, concrete production allocator, CPU kernel library, SIMD implementation, BLAS binding, graph executor, trainer or distributed transport is claimed complete by P0.

**P0 exit gates:** core packages compile; tests and vet pass; native CI passes on the declared platform matrix; no mandatory CGo/vendor SDK; public contracts are reviewed; documentation matches actual behavior.

### P1 — Tensor Engine

**Purpose:** turn metadata and storage contracts into usable tensor values with explicit ownership, indexing, view, stride and data-layout semantics.

**Deliverables:** concrete CPU storage/allocator, owning Tensor API, constructors, indexing, views, strides, slicing, reshape/transpose, broadcasting metadata, copy/contiguous operations and safe serialization primitives.

- [x] `P01-001` Design `Tensor` ownership, close/release semantics and zero value behavior ([review](P1_API_REVIEW.md)).
- [x] `P01-002` Implement a Pure Go heap-backed storage allocator and independently retained handles ([review and CI](P1_API_REVIEW.md)).
- [x] `P01-003` Implement tensor constructors and typed data access without unsafe aliasing ([review and CI](P1_API_REVIEW.md)).
- [x] `P01-004` Implement row-major indexing and checked multidimensional offset calculation ([metadata API](TENSOR_API.md#typed-access-and-indexing)).
- [x] `P01-005` Implement views and define aliasing/overlapping-write rules ([view semantics](TENSOR_API.md#views-and-aliasing)).
- [x] `P01-006` Implement slice, narrow/select, reshape and transpose metadata operations ([view API](TENSOR_API.md#views-and-aliasing)).
- [x] `P01-007` Implement broadcast shape inference and validation ([broadcast API](TENSOR_API.md#views-and-aliasing)).
- [x] `P01-008` Implement contiguous conversion and explicit copy semantics ([copy API](TENSOR_API.md#copy-and-contiguous-conversion)).
- [x] `P01-009` Add property tests for indexing, stride, view aliasing and round-trip reshape ([property tests](../src/tensor/metadata_ops_test.go)).
- [x] `P01-010` Add benchmarks for allocation, view creation, indexing and copy ([benchmarks](../src/tensor/tensor_bench_test.go)).
- [x] `P01-011` Document public Tensor API and ownership examples ([Tensor API](TENSOR_API.md), [ownership review](P1_TENSOR_OWNERSHIP.md)).

**P1 exit gates:** overflow and bounds checks are covered; views retain storage; aliasing behavior is deterministic and documented; scalar, empty, contiguous and non-contiguous cases pass; Linux race tests pass; the native Windows AMD64/Linux AMD64/Linux ARM64 CI matrix passes. The final verification run is [Go CI 45937237974](https://github.com/adhitiad/gotorch/actions/runs/45937237974); benchmark smoke ran on Linux AMD64.

**Accepted limits:** negative strides are not supported; `Reshape` is a view and requires contiguous input; bulk writes to overlapping broadcast layouts return `errs.ErrOverlappingWrite`; custom-storage strided writes are not transactional; the versioned binary snapshot is pre-1.0 and is not a permanent checkpoint compatibility promise. Tensor arithmetic and CPU kernels are later-phase work.

### P2 — Automatic Differentiation

**Purpose:** provide mathematically correct differentiation that can power neural-network training and custom differentiable operations.

**Deliverables:** graph/tape model, gradient storage, reverse mode, forward-mode foundations, gradient accumulation, backward scheduling and gradient checks.

- [ ] `P02-001` Specify graph node, operation context, saved tensor and graph lifetime contracts.
- [ ] `P02-002` Implement gradient state and parameter requires-grad semantics.
- [ ] `P02-003` Implement reverse-mode graph recording and topological backward traversal.
- [ ] `P02-004` Implement gradient accumulation, zeroing, retention and release behavior.
- [ ] `P02-005` Implement backward rules for core differentiable tensor operations.
- [ ] `P02-006` Add custom operation/gradient registration with validation.
- [ ] `P02-007` Add forward-mode differentiation foundations where API semantics are clear.
- [ ] `P02-008` Implement finite-difference gradient checker with dtype-aware tolerances.
- [ ] `P02-009` Test repeated backward, graph release, shared subgraphs and cancellation.
- [ ] `P02-010` Document gradient lifecycle and unsupported differentiability cases.

**Exit gates:** analytical gradients match finite differences within documented tolerance; no graph/tensor lifetime leaks; repeated backward semantics are explicit; race tests cover supported concurrent use.

### P3 — CPU Compute

**Purpose:** execute supported tensor operations efficiently while keeping a trusted, portable reference implementation.

**Deliverables:** kernel registry, Pure Go scalar kernels, capability detection, Go SIMD implementations, optional BLAS/LAPACK adapters, dispatch and differential tests. Go 1.27 experimental SIMD is an implementation option, not a stable public API; the scalar fallback remains mandatory. See [CPU Compute Strategy](CPU_COMPUTE.md) and the [Go AI Ecosystem Guide](AI_ECOSYSTEM.md).

- [ ] `P03-001` Define operation registry and kernel selection metadata.
- [ ] `P03-002` Implement Pure Go reference kernels for core unary and binary operations.
- [ ] `P03-003` Implement reductions and matrix multiplication reference kernels.
- [ ] `P03-004` Define numerical tolerance and determinism policy per operation.
- [ ] `P03-005` Implement runtime CPU feature detection without assuming host capabilities.
- [ ] `P03-006` Add architecture-isolated SIMD kernels with scalar fallback.
- [ ] `P03-007` Add optional BLAS/LAPACK adapter behind build/runtime configuration.
- [ ] `P03-008` Coordinate BLAS thread count with GoTorch execution policy to prevent oversubscription.
- [ ] `P03-009` Add differential tests against Pure Go reference for every optimized kernel.
- [ ] `P03-010` Benchmark representative shapes, layouts, dtypes and thread counts with `-benchmem`.
- [ ] `P03-011` Verify safe dispatch on Windows AMD64, Linux AMD64 and Linux ARM64.

**Exit gates:** every supported operation has a reference path; unsupported combinations fail clearly; optimized paths pass differential tests; benchmark reports include environment and reproducible commands.

**Current engineering priority after P1:** Pure Go CPU reference kernels first, SIMD dispatch and validated architecture-specific kernels second, optional BLAS/LAPACK adapters third. GPU and distributed acceleration are not prerequisites for P3.

### P4 — Scheduler & Memory Engine

**Purpose:** control parallel work, allocation pressure and memory lifetime without unbounded goroutines or hidden global tuning.

**Deliverables:** bounded worker pool, chunk scheduler, task cancellation, allocator reuse, workspace/arena facilities, memory accounting and backpressure.

- [ ] `P04-001` Define scheduler ownership, task model, queue limits and shutdown lifecycle.
- [ ] `P04-002` Implement bounded worker pool and chunked parallel-for primitives.
- [ ] `P04-003` Implement cancellation, panic containment policy and task error aggregation.
- [ ] `P04-004` Implement allocation reuse/pool with explicit ownership rules.
- [ ] `P04-005` Implement workspace and arena lifetime/reset validation.
- [ ] `P04-006` Add memory statistics: active, allocated, cached and peak bytes.
- [ ] `P04-007` Define nested-parallelism policy across Go workers and BLAS.
- [ ] `P04-008` Add race, leak, shutdown, cancellation and stress tests.
- [ ] `P04-009` Benchmark scheduler overhead and scaling efficiency.

**Exit gates:** queue and goroutine counts are bounded; cancellation and shutdown are reliable; no reuse before last reference; no race/deadlock/leak regression; memory accounting is meaningful.

### P5 — Neural Networks & Optimizers

**Purpose:** deliver composable model-building and parameter-update APIs on top of Tensor, Autograd and CPU execution.

**Deliverables:** module abstraction, parameters, layers, initializers, losses, optimizers and learning-rate schedules.

- [ ] `P05-001` Define Module, Parameter, buffer and state-dictionary contracts.
- [ ] `P05-002` Implement module composition, parameter enumeration and train/eval mode.
- [ ] `P05-003` Implement Linear and core activations.
- [ ] `P05-004` Implement normalization and dropout foundations.
- [ ] `P05-005` Implement convolution/pooling foundations with documented supported layouts.
- [ ] `P05-006` Implement initialization functions and deterministic RNG injection.
- [ ] `P05-007` Implement core loss functions with reduction semantics.
- [ ] `P05-008` Implement SGD, momentum and weight decay.
- [ ] `P05-009` Implement Adam and AdamW with state serialization.
- [ ] `P05-010` Implement learning-rate scheduler interfaces and core schedules.
- [ ] `P05-011` Add gradient and optimizer regression tests, including XOR/simple regression.
- [ ] `P05-012` Document layer, loss and optimizer semantics.

**Exit gates:** forward/backward results are correct; optimizer state is serializable; seeded tests are deterministic; training examples converge within stated tolerances.

### P6 — Data Engine

**Purpose:** load, transform, batch and stream datasets with deterministic behavior and bounded concurrency.

**Deliverables:** dataset interfaces, samplers, DataLoader, transforms, batching, prefetch, caching and streaming adapters. Start with streaming JSONL/NDJSON and CSV adapters using the Go standard library; add Parquet and Hugging Face Hub support as optional adapters. See [Go AI Ecosystem Guide](AI_ECOSYSTEM.md).

- [ ] `P06-001` Define Dataset, Sample, Sampler and Batch contracts.
- [ ] `P06-002` Implement sequential and random samplers with explicit seeds.
- [ ] `P06-003` Implement batching, collate functions and last-batch policies.
- [ ] `P06-004` Implement transform composition and validation.
- [ ] `P06-005` Implement bounded prefetch workers and cancellation-safe shutdown.
- [ ] `P06-006` Implement optional cache and streaming source interfaces.
- [ ] `P06-007` Add determinism, worker failure, cancellation and resource-bound tests.
- [ ] `P06-008` Benchmark throughput, latency and memory footprint.

**Exit gates:** same seed produces documented ordering; errors do not strand workers; memory and worker counts are bounded; throughput benchmarks are reproducible.

### P7 — Compiler & Graph Optimization

**Purpose:** transform validated high-level tensor operations into optimized execution plans without changing numerical semantics beyond documented tolerances.

**Deliverables:** graph IR, type/shape inference, validation, optimization passes, fusion, execution plan and memory plan.

- [ ] `P07-001` Define typed graph IR and stable node/value identity.
- [ ] `P07-002` Implement graph construction and validation.
- [ ] `P07-003` Implement dtype/shape inference and error diagnostics.
- [ ] `P07-004` Implement constant folding and dead-code elimination.
- [ ] `P07-005` Implement operation fusion with explicit legality rules.
- [ ] `P07-006` Implement execution planning and kernel selection.
- [ ] `P07-007` Implement memory planning and temporary lifetime reuse.
- [ ] `P07-008` Add graph fuzz/property tests and malformed-graph tests.
- [ ] `P07-009` Differential-test compiled graph output against eager reference output.
- [ ] `P07-010` Benchmark graph compile cost separately from execution cost.

**Exit gates:** invalid graphs are rejected safely; optimized graphs match reference within tolerance; compile/runtime overhead is measured; optimizer passes are individually testable.

### P8 — Training & Inference Runtime

**Purpose:** provide end-to-end model training, evaluation and production inference APIs with reliable checkpoints and sessions.

**Deliverables:** Trainer, evaluation loop, inference sessions, batching, checkpoint/resume, metrics and profiling hooks.

- [ ] `P08-001` Define training configuration, callbacks and loop lifecycle.
- [ ] `P08-002` Implement train/evaluate loops with loss and metric aggregation.
- [ ] `P08-003` Implement optimizer/scheduler integration and gradient clipping.
- [ ] `P08-004` Define checkpoint schema and versioning strategy.
- [ ] `P08-005` Save/load model, optimizer, scheduler, RNG and progress state atomically.
- [ ] `P08-006` Implement resume and interrupted-training recovery.
- [ ] `P08-007` Implement inference session lifecycle and safe concurrent request policy.
- [ ] `P08-008` Implement batching and request cancellation.
- [ ] `P08-009` Add train-save-load-resume-infer integration tests.
- [ ] `P08-010` Document serving and training resource requirements.

**Exit gates:** train/save/load/resume/infer workflow succeeds; corrupted checkpoints fail safely; session concurrency is tested; resource cleanup and cancellation are reliable.

### P9 — Quantization & Numerical Optimization

**Purpose:** reduce memory and improve execution speed while measuring accuracy loss and keeping conversion behavior explicit.

**Deliverables:** FP16/BF16 where practical, INT8/INT4 pathways, calibration, quantized kernels and mixed-precision policies.

- [ ] `P09-001` Define quantization schemes, scale/zero-point metadata and rounding policy.
- [ ] `P09-002` Implement FP16/BF16 conversion and operation support where practical.
- [ ] `P09-003` Implement calibration data collection and observers.
- [ ] `P09-004` Implement INT8 quantize/dequantize and supported kernels.
- [ ] `P09-005` Implement INT4 packing and kernel path only where benchmarks justify it.
- [ ] `P09-006` Implement mixed-precision execution and accumulation policies.
- [ ] `P09-007` Add accuracy, saturation, overflow and numerical stability tests.
- [ ] `P09-008` Benchmark memory, latency, throughput and accuracy trade-offs.

**Exit gates:** quantization is explicit and versioned; accuracy impacts are measured; no silent dtype conversion; fallback behavior is defined.

### P10 — Distributed Computing

**Purpose:** scale training and compute across processes/nodes while preserving collective correctness, resource bounds and recoverability.

**Deliverables:** coordinator, authenticated workers, transport, collectives, DDP, parameter synchronization, distributed checkpoint and recovery.

- [ ] `P10-001` Define coordinator/worker state machine, run IDs, epochs and membership lifecycle.
- [ ] `P10-002` Define secure transport, peer authentication and message schema.
- [ ] `P10-003` Enforce message, tensor, rank, timeout and resource limits.
- [ ] `P10-004` Implement worker registration, heartbeat and membership changes.
- [ ] `P10-005` Implement Broadcast and AllReduce with ordering and timeout semantics.
- [ ] `P10-006` Implement AllGather and ReduceScatter where required.
- [ ] `P10-007` Implement initial data-parallel/DDP training.
- [ ] `P10-008` Implement parameter/optimizer synchronization policy.
- [ ] `P10-009` Implement consistent distributed checkpoint and restart.
- [ ] `P10-010` Test worker crash, partial collective, timeout, duplicate message and network partition.
- [ ] `P10-011` Benchmark communication overhead and scaling efficiency separately.

**Exit gates:** collectives are correct and ordered; authentication is mandatory; retries are bounded and idempotent where applicable; worker failure and recovery are tested; local operations remain network-free.

### P11 — Model Interoperability

**Purpose:** exchange models and tensors with established ecosystems using versioned formats and safe parsers.

**Deliverables:** ONNX import/export for declared operator coverage, Safetensors support and a versioned native format.

- [ ] `P11-001` Define supported operator and dtype compatibility matrix.
- [ ] `P11-002` Implement bounded, validated Safetensors reader/writer.
- [ ] `P11-003` Implement native model/checkpoint format versioning and integrity metadata.
- [ ] `P11-004` Implement ONNX importer for explicitly listed operators.
- [ ] `P11-005` Implement ONNX export for explicitly listed operators.
- [ ] `P11-006` Add parity, round-trip, malformed-file and resource-limit tests.
- [ ] `P11-007` Document unsupported operators, conversions and compatibility.

**Exit gates:** format versions are explicit; hostile files cannot trigger unbounded allocation; supported models pass parity tests; unsupported features fail with actionable errors.

### P12 — Ecosystem & Tooling

**Purpose:** make GoTorch discoverable, inspectable, extensible and straightforward to adopt and contribute to.

**Deliverables:** CLI, profiler, registry, extension interfaces, examples, documentation and release tooling.

- [ ] `P12-001` Define stable CLI command groups and machine-readable output.
- [ ] `P12-002` Implement model inspection and environment diagnostics.
- [ ] `P12-003` Implement profiler output for op time, memory and execution plan.
- [ ] `P12-004` Define extension/plugin API and versioning/compatibility policy.
- [ ] `P12-005` Implement local model registry metadata and artifact verification.
- [ ] `P12-006` Publish examples and progressive tutorials for inference, training and distributed use.
- [ ] `P12-007` Add contributor onboarding and API documentation checks.
- [ ] `P12-008` Add CLI and extension integration tests.

**Exit gates:** developer workflows are reproducible; examples execute in CI; extensions cannot bypass core validation/security policies; documentation matches shipped APIs.

### P13 — Production Readiness

**Purpose:** make supported capabilities safe, stable, measurable and releasable across the declared platform matrix.

**Deliverables:** API stability policy, security review, fuzzing, stress testing, benchmark baselines, observability and release automation.

- [ ] `P13-001` Define semantic versioning and public API compatibility policy.
- [ ] `P13-002` Add fuzz targets for parsers, shapes, graph validation and serialization.
- [ ] `P13-003` Complete threat model and security review for model and distributed boundaries.
- [ ] `P13-004` Run long-duration stress tests and resource leak checks.
- [ ] `P13-005` Publish reproducible performance baselines for supported hardware.
- [ ] `P13-006` Verify Windows AMD64, Linux AMD64 and Linux ARM64 release builds/tests.
- [ ] `P13-007` Add release automation, changelog generation and artifact checksums.
- [ ] `P13-008` Publish known limitations, support policy and upgrade/migration notes.
- [ ] `P13-009` Complete final architecture, dependency-license and supply-chain review.

**Exit gates:** no unresolved critical/high severity release blockers; required CI passes; supported APIs and limitations are documented; releases are reproducible and artifacts verifiable.

---

## 5. Agent handoff protocol

This process is intentionally tool-agnostic. Agents should not maintain separate copies of the roadmap or invent their own phase status.

### Required reading order

For runtime-specific instruction/skill discovery, consult [`docs/AGENT_SETUP.md`](AGENT_SETUP.md). This does not replace the shared repository rules below.

1. Root `AGENTS.md`.
2. This file: `docs/TASK.md`.
3. Relevant package-level `AGENTS.md`, if any.
4. Relevant `.agent/rules/*.md`.
5. Relevant `.agents/skills/<skill>/SKILL.md`.
6. Architecture/contracts and related ADRs in `docs/`.

### How an agent starts work

1. Identify one unchecked task ID (for example `P01-003`).
2. Check that all dependency phases and required predecessor tasks are complete.
3. Inspect the current branch, worktree/repository state, package API and existing tests before editing.
4. Check GitHub Issues/PRs for another agent already claiming the same task. Do not duplicate active work.
5. Produce a short implementation plan: files, public API impact, tests, risks and acceptance criteria.
6. Work only within the selected task's scope. Do not silently expand into another phase.
7. Add/update tests and documentation with code changes.
8. Run the relevant checks and record exact commands/results.
9. Update this checklist in the same change. Only tick items that are actually verified.
10. Report task ID, files changed, commit/PR, tests, remaining limitations and next unblocked task.

### Multi-agent coordination rules

- One task ID has one active implementation owner at a time.
- Agents may work in parallel only on non-overlapping tasks with non-overlapping file ownership or agreed interfaces.
- Public API, tensor semantics, storage lifetime, numerical behavior, compiler IR and distributed protocol changes require explicit review/ADR before parallel implementation depends on them.
- Prefer a dedicated branch per task/agent, based on `adhitiad_gpt`, then PR back to `adhitiad_gpt`.
- Do not push directly to `main` unless the repository owner explicitly authorizes it for that task.
- Do not force-push, rewrite history or overwrite another agent's uncommitted work.
- Never mark a phase complete because a scaffold, interface or placeholder directory exists.
- If instructions disagree, root `AGENTS.md` and explicit user authorization take precedence; document the conflict and do not guess at destructive changes.

### Checklist status meanings

- `[ ]` Not complete or not verified.
- `[x]` Implemented and verified against its acceptance criteria.
- A task in progress remains `[ ]`; record its owner and branch in the associated GitHub issue/PR or implementation note.
- A failed or skipped check is not complete. State it explicitly in the report.
- Phase completion requires every task and phase exit gate to be checked, plus maintainer review for public contracts.

---

## 6. Definition of Done

A task is done only when all applicable conditions are true:

- [ ] Implementation is real behavior, not an empty package, TODO-only stub or unsupported claim.
- [ ] Public API and errors are documented; ownership, concurrency, cancellation and numerical behavior are explicit where applicable.
- [ ] Unit and boundary tests cover normal, invalid, zero-size, overflow and failure paths as relevant.
- [ ] Race, fuzz, differential and benchmark checks are added where required by the task.
- [ ] `gofmt`, `go test ./...` and `go vet ./...` pass.
- [ ] Windows AMD64, Linux AMD64 and Linux ARM64 CI checks pass; platform-specific exclusions are explained.
- [ ] Relevant docs, ADRs, examples and this checklist are updated.
- [ ] Review comments are resolved and no secrets/generated artifacts/untracked unrelated changes are included.
- [ ] Commit/PR and verification evidence are recorded.

**Important:** this section is a template for each task, not a global completion stamp. Check these items in a task-specific issue/PR or implementation record; do not check this generic list just because one phase passes.

## 7. Sources of truth

| Concern | Canonical location |
|---|---|
| Project instructions and repository-wide agent rules | `AGENTS.md` |
| Phase scope, task IDs, dependency order and checklists | `docs/TASK.md` (this file) |
| High-level phase summary | `docs/ROADMAP.md` |
| Public subsystem contracts | `docs/CORE_CONTRACTS.md` |
| Architecture decisions | `docs/ARCHITECTURE.md` and `docs/adr/` |
| Domain engineering rules | `.agent/rules/` |
| Portable reusable agent skills | `.agents/skills/` |
| Repeatable workflows and templates | `.agent/workflows/` and `.agent/templates/` |
| Contributor process | `CONTRIBUTING.md` |
| Automated platform verification | `.github/workflows/ci.yml` |

If any other document conflicts with this task checklist, update the other document or record an ADR; do not maintain competing roadmaps.

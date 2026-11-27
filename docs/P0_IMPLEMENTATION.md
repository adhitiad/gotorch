# P0 — Foundation Implementation

**Status: source-layout migration and native three-platform CI passed; maintainer API review remains open.** This phase defines the stable contracts on which later work depends. It does not complete the full framework.

## Scope delivered

| Package | Location | Implemented contract | Not included yet |
|---|---|---|---|
| `errs` | `src/errs` | Stable sentinel errors for validation, overflow, state, storage and distributed identity | Domain-specific operator errors |
| `dtype` | `src/dtype` | Typed element IDs, byte widths and numeric classification | Numeric conversion and arithmetic |
| `shape` | `src/shape` | Immutable dimensions, scalar/zero-size semantics and checked element counts | Tensor allocation and shape transforms |
| `tensor` | `src/tensor` | Dtype/shape/stride/layout metadata and checked strided byte-span calculation | Owning Tensor object, views, slicing, operators |
| `storage` | `src/storage` | Storage/allocator interfaces, retain/release contract and bounded byte-range validation | Concrete allocator, buffer pool, arena |
| `compute` | `src/compute` | Kernel specification, invocation metadata, backend interface and pre-dispatch validation | Reference kernels, SIMD, BLAS, dispatch registry |
| `runtime` | `src/runtime` | Ordered component startup, rollback, reverse shutdown and lifecycle states | Execution scheduler and graph executor |
| `distributed` | `src/distributed` | Run/worker identity and complete membership validation | Network transport, coordinator, collectives, DDP |
| `platform` | `src/platform` | Primary/compatibility target classification | CPU feature detection and SIMD dispatch |

The root `gotorch.go` remains a deliberately small public facade for the module root import. Reusable framework code belongs under `src/`; executable entry points belong under `cmd/`; application composition belongs under `app/`.

## Contract decisions

### Dtype and shape

`dtype.DType` is an explicit enum; `Size` returns bytes per element. `shape.Shape` owns a copy of its dimensions, exposes defensive copies, treats rank zero as one scalar element, permits zero-sized dimensions, and rejects negative dimensions and element-count overflow.

### Tensor metadata

`tensor.Metadata` contains dtype, immutable shape metadata, element strides and a derived contiguous/strided layout. Strides are expressed in elements, not bytes. `RequiredBytes` calculates the reachable byte span for strided data, excluding the base-storage offset. Negative strides are rejected in this initial contract; any future change requires an ADR and boundary tests.

### Storage ownership

Each `storage.Storage` value is one independently releasable handle to a shared allocation. `Retain` returns a new handle. `Release` releases only that handle; repeated release must return `errs.ErrStorageReleased`. Views or asynchronous work must retain storage for their entire use period. `storage.ValidateRange` checks offsets without overflowing addition.

### Compute boundary

`compute.KernelSpec` describes an operation and its primary compute dtype/layout. `compute.Invocation` carries metadata plus borrowed storage handles and offsets. A backend inspects the invocation for operation-specific operand compatibility. `compute.Run` validates the context, kernel spec, every metadata/storage span, and backend support before execution. A backend must not retain invocation slices or handles after `Execute` returns.

### Runtime lifecycle

`runtime.Manager` starts components in declaration order and stops them in reverse. A failed start triggers best-effort rollback of already-started components. A component must clean up its own partial startup; the manager rolls back components that had already started. Shutdown attempts all started components and returns joined errors; it is terminal even when one or more stops fail.

### Distributed identity

`distributed.Identity` binds a worker to a run, epoch, rank and world size. `ValidateMembership` requires a complete membership snapshot with one worker per rank and consistent run metadata. IDs are opaque strings; format and generation policy remain with the cluster coordinator.

## Explicit non-goals for P0

P0 does not add tensor arithmetic, autograd, parameter storage, a concrete memory allocator, CPU kernels, SIMD assembly, BLAS bindings, a scheduler, model training, network RPC or collectives. These belong to later roadmap phases. No package should exist solely as an empty directory.

## Validation commands

```bash
gofmt -w .
go test ./...
go test -race ./...
go vet ./...
GOOS=windows GOARCH=amd64 go build ./...
GOOS=linux GOARCH=amd64 go build ./...
GOOS=linux GOARCH=arm64 go build ./...
```

The repository's declared Go toolchain in `go.mod` and native CI are authoritative. A local cross-build is not a substitute for native platform tests.

## Prior CI evidence

The P0 foundation-contract commit `d5f0af9a039635320fd6b1622c2b33b4ce2640a7` passed the native GitHub Actions matrix on Windows AMD64, Linux AMD64 and Linux ARM64. The subsequent `src/` relocation changes package import paths and must pass a fresh CI run before P0 can close.

## P0 exit gates

- [x] Foundation contract packages and boundary tests exist.
- [x] Documentation describes the contracts and explicitly states P0 non-goals.
- [x] Native Windows AMD64, Linux AMD64 and Linux ARM64 CI passed for the pre-migration P0 commit.
- [x] Root task/agent workflow and portable skill directory are defined.
- [x] The new `src/` package layout passes fresh native Windows AMD64, Linux AMD64 and Linux ARM64 CI ([run `45930296453`](https://github.com/adhitiad/gotorch/actions/runs/45930296453), commit `9132cee59991908c08691287689f51f753aa4626)).
- [ ] Maintainer review of exported contracts, edge cases and API stability is complete.
- [ ] P0 checklist in [`TASK.md`](TASK.md) is reviewed and phase closure is approved.

Do not mark P0 complete until every remaining exit gate is satisfied.

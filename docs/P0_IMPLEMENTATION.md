# P0 — Foundation Implementation

**Status: implementation in progress.** This checkpoint establishes public contracts; it does not complete the full framework or the P0 release gate.

## Scope delivered

| Package | Implemented contract | Not included yet |
|---|---|---|
| `errs` | Stable sentinel errors for validation, overflow, state, storage, and distributed identity | Domain-specific operator errors |
| `dtype` | Typed element IDs, byte widths, numeric classification | Numeric conversion and arithmetic |
| `shape` | Immutable dimensions, scalar/zero-size semantics, checked element counts | Tensor allocation and shape transforms |
| `tensor` | Dtype/shape/stride/layout metadata and checked strided byte-span calculation | Owning Tensor object, views, slicing, operators |
| `storage` | Storage/allocator interfaces, retain/release contract, bounded byte-range validation | Concrete allocator, buffer pool, arena |
| `compute` | Kernel specification, invocation metadata, backend interface, pre-dispatch validation | Reference kernels, SIMD, BLAS, dispatch registry |
| `runtime` | Ordered component startup, rollback, reverse shutdown, lifecycle states | Execution scheduler and graph executor |
| `distributed` | Run/worker identity and complete membership validation | Network transport, coordinator, collectives, DDP |
| `platform` | Primary/compatibility target classification | CPU feature detection and SIMD dispatch |

## Contract decisions

### Dtype and shape

`dtype.DType` is an explicit enum; `Size` returns bytes per element. `shape.Shape` owns a copy of its dimensions, exposes defensive copies, treats rank zero as one scalar element, permits zero-sized dimensions, and rejects negative dimensions and element-count overflow.

### Tensor metadata

`tensor.Metadata` contains dtype, immutable shape metadata, element strides, and a derived contiguous/strided layout. Strides are expressed in elements, not bytes. `RequiredBytes` calculates the reachable byte span for strided data, excluding the base-storage offset. Negative strides are rejected in this initial contract; any future change requires an ADR and boundary tests.

### Storage ownership

Each `storage.Storage` value is one independently releasable handle to a shared allocation. `Retain` returns a new handle. `Release` releases only that handle; repeated release must return `errs.ErrStorageReleased`. Views or asynchronous work must retain storage for their entire use period. `storage.ValidateRange` checks offsets without overflowing addition.

### Compute boundary

`compute.KernelSpec` describes an operation and its primary compute dtype/layout. `compute.Invocation` carries metadata plus borrowed storage handles and offsets. A backend inspects the invocation for operation-specific operand compatibility. `compute.Run` validates the context, kernel spec, every metadata/storage span, and backend support before execution. A backend must not retain invocation slices or handles after `Execute` returns.

### Runtime lifecycle

`runtime.Manager` starts components in declaration order and stops them in reverse. A failed start triggers best-effort rollback of already-started components. A component must clean up its own partial startup; the manager rolls back components that had already started. Shutdown attempts all started components and returns joined errors; it is terminal even when one or more stops fail.

### Distributed identity

`distributed.Identity` binds a worker to a run, epoch, rank, and world size. `ValidateMembership` requires a complete membership snapshot with one worker per rank and consistent run metadata. IDs are opaque strings; format and generation policy remain with the cluster coordinator.

## Explicit non-goals for P0

P0 does not add tensor arithmetic, autograd, parameter storage, a concrete memory allocator, CPU kernels, SIMD assembly, BLAS bindings, a scheduler, model training, network RPC, or collectives. These belong to later roadmap phases. No package exists solely as an empty directory.

## Local validation

```bash
gofmt -w .
go test ./...
go test -race ./...
go vet ./...
GOOS=windows GOARCH=amd64 go build ./...
GOOS=linux GOARCH=arm64 go build ./...
```

Local smoke tests were run with a compatible Go 1.23.2 toolchain after temporarily adjusting only the local validation copy of `go.mod`; the repository's declared Go 1.27.4 baseline was not changed in this commit. A local cross-build is not a substitute for native CI tests.

## P0 exit gates

- [x] Foundation contract packages and boundary tests exist.
- [x] Local tests, race tests, and `go vet` pass in the development validation environment.
- [x] Windows AMD64 and Linux ARM64 cross-builds succeed in the development validation environment.
- [ ] Windows AMD64 native CI passes using the repository's declared Go toolchain.
- [ ] Linux AMD64 native CI passes using the repository's declared Go toolchain.
- [ ] Linux ARM64 native CI passes using the repository's declared Go toolchain.
- [ ] Public contracts receive maintainer review and limitations are reflected in the API docs.

Do not mark P0 complete until all exit gates are satisfied.

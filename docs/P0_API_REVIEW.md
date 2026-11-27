# P0 Maintainer API Review

**Review date:** 2026-11-27  
**Reviewed branch:** `adhitiad_gpt`  
**Review scope:** exported P0 contracts, invalid and boundary inputs, ownership/lifecycle semantics, platform behavior, documentation alignment, and pre-1.0 API stability.

## Decision

**Approved for P0 closure after the listed edge-case fixes and CI verification. No unresolved critical or high-severity API blocker remains.**

This approval accepts the current API as an early-stage foundation contract. It is **not** a v1 compatibility promise. The project has not reached a stable public API release; P13-001 remains responsible for the formal semantic-versioning and compatibility policy. Exported names, enum values, and import paths must not be treated as a persistent wire or file format unless a later versioned format explicitly says so.

## Exported contract review

| Area | Review result | Contract accepted for P0 |
|---|---|---|
| `errs` | Pass | Sentinel errors are shared identities; callers should use `errors.Is` for wrapped errors. |
| `dtype` | Pass | `DType` is a typed identifier; `Size` rejects invalid values; Bool is not classified as numeric. Enum ordinals are not a serialization format. |
| `shape` | Pass | Construction copies dimensions; dimensions are non-negative; rank zero is scalar; zero dimensions yield zero elements; product overflow is checked for non-empty shapes. |
| `tensor.Metadata` | Pass after fix | Shape and stride accessors do not expose mutable internal slices; strides are element-based and non-negative; byte spans include strided gaps and are overflow-checked. Empty tensors use canonical zero strides and require zero bytes. |
| `storage` | Pass | Each handle has independent retain/release semantics; range validation avoids addition overflow; concrete allocation and synchronization guarantees must be implemented and tested in P1. |
| `compute` | Pass after fix | Metadata/storage spans are validated before dispatch; context and interface dependencies reject both nil and typed-nil values; backends borrow invocation data only for the duration of execution. |
| `runtime.Manager` | Pass after fix | Ordered startup, reverse shutdown, rollback, idempotent running start, terminal close/failure states, and duplicate-name validation are defined; typed-nil components are rejected. |
| `distributed.Identity` | Pass | Identity validation and complete rank/worker membership checks are deterministic; IDs remain opaque coordinator-supplied strings. |
| `platform` | Pass | Support levels describe project validation commitments, not runtime CPU-feature or performance guarantees. |
| Root facade | Pass | The root module import remains a small compatibility facade; `Version` is explicitly a development marker, not a release version. |

## Edge-case fixes made during review

1. **Empty tensor with an unrepresentable contiguous suffix.** A valid zero-element shape such as `[0, MaxInt64, 2]` previously failed contiguous-stride construction because a suffix product overflowed even though no element could be addressed. `NewMetadata` now canonicalizes empty-shape strides to zero and reports a zero-byte span. A regression test covers this case.
2. **Typed-nil interface inputs.** A typed nil `storage.Storage`, `compute.Backend`, `context.Context`, or runtime `Component` can pass a plain `value == nil` check and panic when a method is called. The affected public boundaries now reject typed-nil inputs with `errs.ErrInvalidArgument`; regression tests cover the affected paths.

## Accepted constraints and follow-up boundaries

- `runtime.Manager.Close` passes its supplied context to every component's `Stop`. A cancelled or expired context may cause a component to abort cleanup. Callers must provide a live cleanup context when teardown must continue; each component remains responsible for bounded, best-effort shutdown. Startup rollback uses the startup context and returns cleanup errors joined with the startup cause.
- P0 defines storage ownership and kernel invocation contracts, not a concrete allocator or execution engine. P1/P3 must test actual buffer lifetime, concurrent access, kernel operand compatibility, determinism, and differential numerical behavior.
- P0 permits non-negative strides but does not define mutable overlapping-view semantics. P1 must specify aliasing, mutation, and view lifetime before exposing writable views.
- The framework is pre-1.0. Public API changes before a stable release require updated tests, documentation, and an ADR when they alter cross-package contracts. No backward-compatibility guarantee is implied by P0 completion.
- The root facade remains the stable import entry point for consumers who want the module root. Package APIs under `src/` are still evolving with their owning phases.

## Verification evidence

The implementation commit for this review must pass the native GitHub Actions matrix:

- Windows AMD64: formatting, package discovery, tests, vet.
- Linux AMD64: formatting, package discovery, tests, vet, race tests.
- Linux ARM64: formatting, package discovery, tests, vet, race tests.

The P0 closure record in [`TASK.md`](TASK.md) and [`P0_IMPLEMENTATION.md`](P0_IMPLEMENTATION.md) must link to the successful run for the reviewed implementation commit. A documentation-only closure commit may follow that verified implementation commit.

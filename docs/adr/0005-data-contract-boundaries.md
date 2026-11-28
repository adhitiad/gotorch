# ADR-0005: Data contract boundaries

- Status: Accepted
- Date: 2026-11-28

## Decision

`data.Dataset` is a finite random-access collection with explicit context and index validation; streaming sources belong to a later adapter contract rather than forcing every dataset to support `At`. `data.Sampler` exposes `Index(ctx, position)` with no hidden cursor, so seeded samplers are deterministic and usable by bounded concurrent prefetch workers. `data.Sample` copies its field map but borrows arbitrary values, and `data.Batch` records source indices alongside samples and copies both slices at construction. Batching and transforms are separate stateless values: `Batcher` resolves positions into batches under an explicit last-batch policy with a caller-supplied `CollateFunc`, while `TransformChain` composes immutable per-sample steps and `CollateWithTransform` bridges them without owning workers or a queue.

Invalid input wraps existing `errs` sentinels, cancellation is returned as `ctx.Err()`, and public constructors never panic. When cancellation happens during a read, the cancellation error takes precedence over the failing step error.

## Consequences

- Sequential and random samplers can be implemented later without changing the loader-facing interface.
- Workers can request positions concurrently without shared cursor locks or duplicated state.
- Partial batches still follow an explicit last-batch policy; an empty batch is never emitted.
- Stateless batchers and transform chains can be shared by bounded workers without locks; a later DataLoader owns worker count, prefetch depth and shutdown.
- Callers must keep borrowed sample values alive until the batch is consumed; adapters returning tensors must define their own close ownership separately.
- Streaming JSONL/CSV readers need a separate source interface in a later task; the indexed contract does not claim to support them.

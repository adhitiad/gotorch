# ADR-0006: Prefetch, cache and streaming source boundaries

- Status: Accepted
- Date: 2026-11-28

## Decision

`data.DataLoader` is the only P6 value that owns goroutines. It starts a fixed number of workers plus one dispatcher, bounds the jobs, result and ordered channels, and caps in-flight batches at `workers + prefetch`. The dispatcher reorders completed batches by batch index, so output is deterministic even when workers finish out of order. `Close` cancels the derived context and joins every goroutine; it is idempotent, and cancellation is reported consistently by later `Next` calls.

`data.Cache[V]` and `data.Source` are optional contracts, not prerequisites for the core pipeline. `MemoryCache[V]` bounds memory by entry count with LRU eviction. `Source` is sequential and stateful; `NewSourceDataset` materializes an explicit maximum into a bounded random-access `Dataset` so existing batching code can consume it without unbounded buffering. It does not turn a stream into an infinite dataset, and it does not close the caller-owned source.

## Consequences

- Worker count, prefetch depth and queue capacities are explicit constructor inputs with hard maxima; no goroutine-per-element fan-out exists.
- Ordering does not depend on scheduling, so seeded runs are reproducible across worker counts within one Go toolchain.
- A failing worker cancels the pipeline; buffered batches produced before the failure remain valid, and the first error is then returned on every subsequent `Next`.
- Callers must close the iterator to release workers. The iterator is single-consumer; `Next` and `Close` must not run concurrently.
- Cache values are stored as given. Callers are responsible for cloning mutable values before `Put` and for bounding value size.
- Streaming parsers, a true pull-based loader for `Source`, and throughput benchmarks remain later work.

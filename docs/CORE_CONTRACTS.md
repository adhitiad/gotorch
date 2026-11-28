# Core Contracts

These contracts define subsystem boundaries before implementation expands. Their initial Go APIs live in the packages listed below; they do not imply that the corresponding execution engines already exist.

## Dtype and shape

- `dtype.DType` is an explicit, validated element-type identifier with a defined byte width.
- `shape.Shape` is immutable through its public API. Dimensions are non-negative; rank zero is a scalar with one element; any zero-sized dimension produces zero elements.
- Element-count and byte-span arithmetic must detect signed integer overflow before allocation or slicing.

## Tensor metadata

A tensor descriptor defines dtype, shape, element strides, layout, storage, ownership, mutability, and placement. The P0 `tensor.Metadata` value covers dtype, shape, strides, and derived contiguous/strided layout only. It owns no storage; P1 adds the owning `tensor.Tensor` lifecycle described below.

Strides are measured in elements. Negative strides are not accepted by the initial P0 contract. A view must keep its backing storage alive for its entire lifetime; aliasing and overlapping-write rules must be made explicit in P1 before mutable views are exposed.

### P1 owning Tensor and storage

- `tensor.Tensor` owns exactly one `storage.Storage` handle. Do not copy a Tensor after first use; use `Retain` to obtain a separate handle with an independent `Close` lifecycle.
- `Close` is idempotent. The zero value owns no resources; its methods other than `Close` return `errs.ErrInvalidState`. Closing the zero value is safe.
- Constructors allocate contiguous storage. Rank zero has one element; any zero dimension has zero elements and owns a valid zero-byte storage handle.
- The Pure Go `storage.HeapAllocator` returns zero-initialized storage. Every retained storage handle is released independently; the allocation is reclaimed after its final handle is released.
- Typed construction and access encode/decode fixed-width values in little-endian form. `From` and `Write` copy input values; `Read` returns a new packed row-major copy. No typed slice aliases internal storage.
- `Metadata.ElementOffset` calculates a checked physical element offset from strides; `LinearIndex` calculates the row-major logical index. `At` and `Set` address one logical element with dtype and bounds validation.
- Tensor views (`Slice`, `Narrow`, `Select`, `Reshape`, `Transpose`, `BroadcastTo`) retain an independent storage handle. View handles have independent `Close` lifetimes but share mutations with the same allocation.
- Slices use positive steps and end-exclusive bounds. Negative strides are not supported. Reshape is a metadata-only view and therefore requires contiguous input.
- Broadcast expansion uses zero strides. Reads are supported; bulk `Write` rejects layouts whose uniqueness cannot be proven with `errs.ErrOverlappingWrite`. A single indexed `Set` is deterministic and visible through aliases.
- `Copy` and `Contiguous` always return independent row-major allocations, even when the source is already contiguous.
- `MarshalBinary` emits a versioned, little-endian logical snapshot. `UnmarshalBinary` validates magic, format version, dtype, dimensions, payload length and canonical Bool bytes before installing a new allocation; it requires a zero or closed receiver.
- No raw backing slice or mutable storage handle is exposed by Tensor.

## Automatic differentiation (P2)

- Each autograd.Tape owns an append-only graph, private snapshots of node values, and committed gradient buffers. Variables are immutable handles identified by Tape and node ID.
- NewVariable and Apply deep-copy their input/output Tensor values. Public Value/Grad accessors return independent copies. The graph remains valid until Tape.Close; Close releases every graph value and gradient.
- Graph parents must exist before a node is appended and must belong to the same Tape. This preserves topological order and prevents cycles through the public API.
- Differentiable dtypes in P2 are float32 and float64 only. Integer, Bool, Float16/BFloat16 bit payloads and complex tensors are not differentiable.
- Reverse mode applies one vector-Jacobian product per operation in reverse topological order. Each rule returns one shape/dtype-compatible gradient per input; the engine validates and owns these returned tensors.
- Gradients accumulate across repeated Backward calls. ZeroGrad clears all or selected Variables. A Backward failure or cancellation must not partially replace previously committed gradients.
- JVP rules are optional per operation. A node reached by a tangent without a forward rule must fail explicitly rather than silently return zero.
- Built-in Add/Mul/Sum are equal-shape, float32/float64 correctness primitives. Broadcasting, matrix operations, general CPU kernels and parallel backward scheduling remain outside P2.
- A Tape serializes graph mutation, Backward, JVP, ZeroGrad and Close. Callbacks receive independent snapshots, must honor context cancellation, and must not retain or close borrowed snapshots.

See [Autograd API](AUTOGRAD_API.md) for usage, ownership, callback contracts and numerical limits.

## Storage ownership

`storage.Storage` is an independently releasable handle to a shared allocation:

- `Retain` returns a distinct handle that must be released independently.
- `Release` relinquishes only the calling handle. Repeated release returns `errs.ErrStorageReleased`.
- Reads and writes after release fail. Implementations follow `io.ReaderAt`/`io.WriterAt` range semantics.
- Asynchronous work and views retain storage until their final use is complete.
- Allocation size, offsets, and lengths are validated before use; range checks must not overflow.

P0 defined the interface and range validator; P1 provides a concrete Pure Go heap allocator.

## Kernel and backend

A kernel declares its operation and primary compute dtype/layout. A backend's support decision may inspect the full invocation because some operations have different input and output signatures. Every invocation argument binds validated tensor metadata to a storage handle and base offset.

The reference implementation will define correctness. SIMD and BLAS/LAPACK implementations are optional optimized paths and must be differentially tested against the reference implementation. A backend may not retain invocation slices or borrowed storage handles after its execution call returns.


### Optional BLAS backend (P3)

- The optional OpenBLAS adapter is available only with the `gotorch_blas` build tag on Linux/Windows and an explicit `GOTORCH_OPENBLAS_LIBRARY` path.
- The initial native operation is contiguous rank-2 float32/float64 `matmul` through CBLAS SGEMM/DGEMM. No LAPACK routine is claimed until a matching GoTorch kernel contract exists.
- Native execution stages storage into Go-owned typed slices, checks cancellation before/after the non-interruptible vendor call, and validates dimensions against the CBLAS integer ABI.
- OpenBLAS thread count is process-global. The adapter serializes thread-count save/set/call/restore and computes a per-call ceiling from `ExecutionPolicy{GoWorkers, BLASThreads}`; it never mutates `GOMAXPROCS`.
- Callers must stop starting new work before closing the registry. `compute.Registry.Close` prevents new registration and dispatch, closes registered closable backends in reverse registration order, waits for each backend's own `Close` synchronization to drain in-flight work, releases the registry's backend references, and aggregates the resulting errors. The scalar reference backend remains mandatory and is always registered last.

## Execution and lifecycle

Long-running operations accept `context.Context`, use bounded resources, and honor cancellation at safe boundaries. Runtime components start in dependency order and stop in reverse order. A component whose `Start` fails is responsible for cleaning up its own partial startup; the manager rolls back components that had already started. Shutdown is best-effort across all components and reports joined errors.

### Scheduler (P4)

- `scheduler.Scheduler` owns a fixed number of workers and a bounded queue. `Config.Validate` rejects zero workers and queue or chunk sizes above the published limits; construction never creates unbounded goroutines or buffers.
- `Submit` waits for queue space while honoring the caller's context; `TrySubmit` never waits and returns `ErrQueueFull` when the queue is full. `Future.Wait` returns the task result or the caller's cancellation error.
- Every task receives a context derived from both its caller and the scheduler. Cancellation of either propagates to the task at safe boundaries; tasks must honor that context.
- A panicking task is contained as `*PanicError` with the panic value and stack, counted as a failure, and cannot terminate a worker. `OnError`, when configured, observes task errors and is itself protected against panics.
- `ParallelFor` splits work into bounded chunks and joins chunk errors. A call made with a worker context runs chunks inline so nested parallelism cannot consume the pool or deadlock on a full queue.
- `Close` stops admission, cancels in-flight work, waits for submissions and workers to drain, and is idempotent. A bounded `Close` context can time out while a non-cooperative task is still running; the shutdown continues in the background.
- `scheduler.Policy` mirrors `compute.ExecutionPolicy`, so the worker pool and optional BLAS adapter derive their CPU budget from the same formula and never modify `GOMAXPROCS`.

## Memory

- Parameter storage, caller-owned inputs/outputs, temporary activations, workspace, and distributed staging have distinct lifetimes.
- Views retain backing storage.
- A buffer cannot be reused while any view or asynchronous operation references it.
- Pooling is an optimization, not an ownership model.
- `sync.Pool` is opportunistic and must not be used for deterministic resource accounting.
- Arena reset is legal only after all allocations from the arena are dead.
- `memory.Pool` is a deterministic `storage.Allocator`: cached buffers are keyed by exact size, zeroed before reuse, and returned only after the final retained handle is released.
- `memory.Arena` is a fixed-capacity, 8-byte-aligned bump allocator; `memory.Workspace` validates all arenas before reset and closes them in reverse registration order.
- Pool, arena and workspace statistics expose active, allocated, cached and peak bytes plus reuse and rejection counters; limits fail with explicit errors instead of silent growth.

## Distributed identity

A distributed identity includes run ID, worker ID, rank, world size, and membership epoch. A membership snapshot must use one run/epoch/world-size tuple, have unique worker IDs, and contain every rank exactly once. IDs are opaque values supplied by the coordinator.

Collectives must define rank membership, ordering, shape compatibility, timeout, cancellation, bounded payloads, retries/idempotency, and failure behavior. Distributed transport must not leak into ordinary local tensor operations.

## Public API

Public APIs document error behavior, concurrency safety, mutation, ownership, dtype conversion, and compatibility guarantees. Errors preserve context and support `errors.Is`/`errors.As`; ordinary invalid input must not panic.

## P0 implementation map

- `errs`: stable sentinel errors.
- `dtype`, `shape`, `tensor`: metadata and checked arithmetic, plus the initial P1 owning Tensor API.
- `storage`: ownership-aware storage interface and Pure Go heap allocator.
- `compute`: kernel specification, invocation, backend interface, and pre-dispatch validation.
- `runtime`: component lifecycle manager.
- `distributed`: identity and membership validation.
- `platform`: primary and compatibility target classification.

See [`P0_IMPLEMENTATION.md`](P0_IMPLEMENTATION.md) for P0 limits and exit gates, and [`TASK.md`](TASK.md) for P1 task status.


## CPU compute (P3 initial slice)

- `compute.Registry` dispatches to the first registered backend that supports the operation signature. Registration order is stable and is the precedence rule.
- `compute.ReferenceCPUBackend` is the mandatory Pure Go scalar baseline for float32/float64 elementwise arithmetic, unary math, full sum/mean reductions and rank-2 matrix multiplication.
- Reference kernels accept strided input metadata and require contiguous caller-owned outputs. They do not broadcast shapes or allocate storage.
- Reductions and matrix products accumulate in float64 and cast once to the output dtype; traversal order is logical row-major and deterministic.
- IEEE-754 division/log behavior is retained; ReLU propagates NaN. Cancellation may leave a caller-provided output partially written, so failed outputs must be discarded or overwritten.
- SIMD and BLAS backends are optional and must be registered ahead of the scalar backend only after capability checks, numerical differential tests and reproducible benchmarks.

## Neural networks, transformers and PEFT (P5)

- `nn.Context` binds one `autograd.Tape` to cached parameter leaves for a single training step. Layer forwards obtain parameters through `Context.Variable` so a parameter is not duplicated as a second leaf.
- `nn.Module` owns parameters and persistent buffers, propagates train/eval mode through its children, and reports fully qualified, lexicographically sorted names. `nn.StateDict` stores copies in a versioned binary format with sorted names; loading matches by exact name.
- P5 differential operations are registered on a tape through `nn.NewTape` or the idempotent `nn.RegisterOperations`. Backward hooks return gradients in input order; non-differentiable int64 control inputs return empty gradients.
- `Linear` uses weight `[out, in]` and computes `input @ weightᵀ` for `[batch, in]` input. Conv2D and pooling support NCHW layout only. Attention uses `[batch, heads, time, dim]` layouts, an additive mask, and max-subtracted softmax; key/value heads may be grouped for GQA/MQA.
- Optimizer state is keyed by full parameter name and serialized in sorted order. `SGD` uses coupled L2 decay; Adam and AdamW differ by coupled pre-gradient decay versus decoupled post-correction decay. Schedulers mutate the optimizer learning rate explicitly and never mutate process-wide runtime settings.
- PEFT targeting matches glob patterns on full parameter names. LoRA wraps a `Projector`, keeps the base frozen by default, and merge/unmerge operates through `LinearLike` base weights with rollback. IA3, prompt tuning and prefix tuning are prototypes with explicit contract limits; prefix tuning does not own masks or multi-layer state.
- See [`P5_NEURAL_NETWORKS.md`](P5_NEURAL_NETWORKS.md) for layer, loss, optimizer, transformer and adapter semantics, and [`TASK.md`](TASK.md) for status.

## Data engine contracts (P6 initial slice)

- `data.Sample` owns a copied field map but borrows field values. Empty names and nil values are rejected; names are reported in sorted order.
- `data.Dataset` is finite and random-access: `Len` is stable, `At` validates context and index, returns an independent field map, and supports bounded concurrent readers. `Close` is idempotent and `At` after close fails with `errs.ErrInvalidState`.
- `data.Sampler` has no hidden cursor. It maps an epoch position to a dataset index deterministically, supports concurrent calls, and may emit a different number of positions than the dataset length. Sequential sampling is the identity map; random sampling uses a seeded local RNG to build one immutable permutation, never the global RNG.
- `data.Batch` pairs each sample with its source dataset index. Both slices are copied at construction, must be non-empty and equal in length, and are immutable through the public API.
- `data.Batcher` resolves sampler positions into batches with an explicit `KeepLast` or `DropLast` policy, a caller-supplied `CollateFunc`, and context checks before each read. It holds no cursor. If cancellation occurs mid-read, `ctx.Err()` wins over the step error.
- `data.Transform` maps a sample without mutation. `TransformChain` composes transforms in order with context checks and defensive clones; `RequireFields` validates required names; `CollateWithTransform` applies a chain per sample before collation. Transforms themselves own no workers or queues.
- `data.DataLoader` adds bounded prefetch: a fixed worker count and prefetch depth, one dispatcher that preserves batch-index order, and `Close` that cancels and joins every goroutine. `Next` returns `io.EOF` at the end, `ctx.Err()` on cancellation, or the first worker error.
- `data.Cache[V]` is an optional context-aware key/value contract; `MemoryCache[V]` is a bounded concurrent LRU. `data.Source` is a sequential reader and `NewSourceDataset` materializes at most a caller-specified number of samples into a bounded `Dataset`. No JSONL, CSV or remote adapter is claimed.
- `data.Tokenizer` maps text to token IDs and back deterministically. `Encode`/`Decode` take `context.Context`, support cancellation, and are safe for concurrent calls. `Encoding` holds IDs, tokens, character offsets, and a special-token mask, all defensively copied. `ByteTokenizer` is a deterministic reference implementation (one ID per byte; BOS/EOS wrapped). `NewCharLevelBPETokenizer` and `NewByteLevelBPETokenizer` implement BPE with merge-rank training; `Train` builds vocabulary and merge rules deterministically, and `Encode` applies merges in rank order. `NewWordPieceTokenizer` implements WordPiece with longest-match-first segmentation and `##` continuation prefix. `NewUnigramTokenizer` implements SentencePiece/Unigram with unigram language model probabilities and dynamic-programming segmentation. `data.TokenizerConfig` loads tokenizer assets directly from files: `vocab.json`/`vocab.txt`, `merges.txt`, `added_tokens.json`, `special_tokens_map.json`, and `tokenizer_config.json`; `FromDirectory` and `FromFiles` build configs, `Validate` checks consistency, and lookups (`SpecialTokenID`, `TokenForID`, `IDForToken`, `SortedVocab`) provide bidirectional access. Special token names are constants; `DefaultSpecialIDs` reserves IDs 0..6. Invalid input wraps `errs` sentinels and cancellation is returned as `ctx.Err()`.
- `data.InstructionSample` and `data.PreferenceSample` define SFT and DPO dataset schemas with field validation. `data.DatasetSchema` distinguishes instruction vs preference. `data.TruncationPolicy` (`TruncateEnd`/`TruncateStart`/`TruncateNone`) and `data.Truncate` bound sequence lengths. `data.CausalMask`, `data.BidirectionalMask`, and `data.LossMask` generate attention/loss masks. `data.InstructionCollate` and `data.DPOCollate` implement SFT/DPO collation with truncation, padding, and per-sample loss masks. `data.ValidateSchema` checks required fields per schema.
- Errors wrap `errs` sentinels and support `errors.Is`; cancellation is returned as `ctx.Err()`. Dataset files and remote records remain untrusted input for later adapters.
- See [`P6_DATA_ENGINE.md`](P6_DATA_ENGINE.md) for the implemented contract slice and [`TASK.md`](TASK.md) for status.

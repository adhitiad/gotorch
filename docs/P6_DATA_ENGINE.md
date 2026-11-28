# P6 data engine contracts

This document records the implemented contract slice for phase P6. [`TASK.md`](TASK.md) remains the canonical status checklist. Only `P06-001` through `P06-013` are implemented; everything else in this phase is planned unless its checkbox is marked complete.

## Implemented: `src/data` contracts

### `Sample`

- A `Sample` is one dataset record with named fields. The zero value is a valid empty sample.
- `NewSample` copies the field map and rejects empty names and nil values with an error wrapping `errs.ErrInvalidArgument`.
- Field values are borrowed and read-only through `Sample`. The package does not clone arbitrary values or close them; callers must keep them valid while the sample is in use. `Clone` copies the field map, not the values.
- `Names` returns a fresh, lexicographically sorted slice. `Value` reports whether a field exists.

### `Dataset`

- `Dataset` is a finite, ordered, random-access collection: `Len` is non-negative and stable for the lifetime of the value.
- `At(ctx, index)` accepts `index` in `[0, Len)`, checks cancellation before reading, and returns a sample with an independent field map. Invalid indexes wrap `errs.ErrOutOfBounds`; a nil context wraps `errs.ErrInvalidArgument`; cancellation is returned as `ctx.Err()`.
- Implementations must be safe for concurrent `Len` and `At` calls from bounded loader workers. Cancellation must not leave partial state.
- `Close` is idempotent. After close, `At` wraps `errs.ErrInvalidState`.

### `Sampler`

- `Sampler` maps an epoch position to a dataset index. It has no hidden mutable cursor: `Index(ctx, position)` is deterministic for a given configuration and seed and is safe for concurrent calls.
- `Len` is the non-negative number of positions in one epoch and may differ from `Dataset.Len`. `Index` accepts `position` in `[0, Len)`.
- Invalid positions wrap `errs.ErrOutOfBounds`; cancellation is returned as `ctx.Err()`. Implementations do not mutate the dataset.

### `Batch`

- `Batch` is an ordered group of samples with the source dataset index for each sample. Samples and indices have identical lengths and at least one element.
- `NewBatch` copies both slices and rejects empty batches, length mismatches and negative indices. Their upper bound remains the producing dataset's responsibility.
- `Batch` is immutable through its public API. `At` returns a cloned sample plus its source index; `Samples` and `Indices` return fresh copies. A nil batch behaves as an empty batch and `At` on it wraps `errs.ErrInvalidState`.
- A batch does not own or close field values. Last-batch policies decide whether a partial batch is emitted; an empty batch is never constructed.

## Semantics shared by the contracts

- Ordinary invalid input returns an error that supports `errors.Is` against `errs` sentinels and never panics.
- Long-running public operations receive a `context.Context` as their first argument and honor cancellation at safe boundaries.
- Ownership is explicit: maps and slices crossing a constructor boundary are copied; arbitrary sample values are borrowed.
- Concurrency safety is a property of the implementation, not of global state. Loader worker counts and prefetch queues remain bounded by later tasks.

## Implemented samplers (`P06-002`)

- `NewSequentialSampler(length)` emits dataset indexes in natural order. Length must be non-negative; zero is valid and emits no positions.
- `NewRandomSampler(length, seed)` builds one seeded permutation of `[0, length)` with a local `math/rand` source. The global RNG is never used, and the seed is exposed by `Seed()`.
- Both samplers validate context and position, are immutable after construction, and are safe for concurrent `Index` calls. A nil sampler behaves as an empty sampler.
- Random order is deterministic for the same seed within one Go toolchain. Cross-version reproducibility requires checked-in reference fixtures, which are not part of this task.
- Random sampler construction allocates an `int` slice of length positions; lengths that do not fit `int` are rejected with an error wrapping `errs.ErrShapeOverflow` before allocation.

## Implemented batching (`P06-003`)

- `NewBatcher(batchSize, policy, collate)` requires a positive size, one of `KeepLast` or `DropLast`, and a non-nil `CollateFunc`. The batcher holds no dataset, sampler or cursor, so one value can be reused across epochs and by concurrent workers.
- `BatchCount(samplerLen)` counts full batches and adds one partial batch only under `KeepLast`. `Batch(ctx, dataset, sampler, batchIndex)` resolves sampler positions, validates each dataset index, checks cancellation before every read, and passes the result to collate.
- `DefaultCollate` builds a `Batch` without transformation. A custom collate must not mutate its inputs; it receives ephemeral sample and index slices, and its result is checked for non-nil.
- If cancellation happens during a read, `ctx.Err()` takes precedence over the underlying step error. Batch sizes that cannot fit `int` are rejected before allocation.

## Implemented transforms (`P06-004`)

- `Transform.Apply(ctx, sample)` maps one sample to another without mutating the input. `TransformFunc` adapts a plain function; a nil function fails with `errs.ErrInvalidState`.
- `NewTransformChain(transforms...)` copies the transform slice and rejects nil or typed-nil elements. `Apply` clones before the first step, checks context before each step, clones between steps, and wraps step errors with the position. An empty chain is an identity clone; a nil chain fails with `errs.ErrInvalidState`.
- `RequireFields(names...)` is a validation transform that rejects missing or nil fields with `errs.ErrInvalidArgument`. Empty names are rejected at construction.
- `CollateWithTransform(transform, base)` applies a transform to each sample before the base collate. It passes a copied index slice to the base and leaves caller-owned samples and indices untouched. This is a composition helper, not a DataLoader.

## Implemented prefetch (`P06-005`)

- `NewDataLoader(batcher, dataset, sampler, workers, prefetch)` validates typed-nil inputs, `workers` in `[1, MaxWorkers]` (64), and `prefetch` in `[0, MaxPrefetch]` (1024). It owns no goroutine or channel until `Iterate` is called and never closes the caller's dataset.
- `Iterate(ctx)` starts exactly `workers` worker goroutines plus one dispatcher goroutine. The jobs channel holds `prefetch` batch indexes, the result channel holds `workers` results, and the ordered channel holds `workers + prefetch` batches. At most `workers + prefetch` batches are in flight.
- Workers compute batches through `Batcher.Batch`; the dispatcher reorders completed results by batch index, so output order does not depend on worker completion order. `Next(ctx)` returns batches, `io.EOF` at the end, `ctx.Err()` on cancellation, or the first worker error.
- `Close()` cancels the derived context and waits for every goroutine to exit. It is idempotent, and batch reads stop after cancellation. A cancelled read reports cancellation consistently on later `Next` calls.

## Implemented cache and source interfaces (`P06-006`)

- `Cache[V]` is a context-aware key/value contract with `Get`, `Put`, `Delete`, `Len` and `Close`. `NewMemoryCache[V](capacity)` provides a bounded LRU implementation: positive capacity, empty-key rejection, concurrent-safe access, idempotent close, and `errs.ErrInvalidState` after close.
- `Source` is a sequential reader with `Next(ctx) (Sample, error)` returning `io.EOF` at the end. `SliceSource` is a finite implementation that clones its input and output samples; it keeps a position and must be consumed by one goroutine.
- `NewSourceDataset(ctx, source, maxSamples)` materializes at most `maxSamples` samples into a bounded, concurrency-safe `Dataset` usable by `Batcher` and `DataLoader`. The limit must fit `int`; unread samples remain in the source, which the caller still owns. This is explicit materialization, not an infinite streaming adapter.
- JSONL/NDJSON and CSV readers remain separate planned adapters; no parser is claimed here.

## Verification coverage (`P06-007`)

- Determinism: the same random seed yields the same batch order across runs with different worker counts.
- Failure: a failing dataset or transform stops the pipeline, propagates a sentinel error, and leaves no active worker reads.
- Cancellation: cancellation before `Iterate` and during a blocked read returns `ctx.Err()` without hanging; repeated `Close` is safe.
- Resource bounds: channel capacities, in-flight limit, maximum concurrent reads, and total submitted batches are checked against the configured worker and prefetch limits.

## Implemented benchmarks (`P06-008`)

- `BenchmarkSequentialSamplerIndex` and `BenchmarkRandomSamplerIndex`: per-index lookup latency and allocations. Both report 0 B/op and 0 allocs/op for steady-state lookups.
- `BenchmarkBatchConstruction` and `BenchmarkBatchAt`: batch build and per-position read cost, with bytes and allocations per sample.
- `BenchmarkTransformChainApply`: end-to-end chain apply cost, including defensive clones between steps.
- `BenchmarkCacheGet` and `BenchmarkCachePut`: LRU cache hit path cost.
- `BenchmarkBatcherPipeline`: sampler-to-batch pipeline cost per batch.
- `BenchmarkDataLoaderThroughput`: full prefetch pipeline throughput across repeated epochs, including iterator restart on `io.EOF`.
- `BenchmarkSampleClone` and `BenchmarkNewSample`: per-sample copy and construction cost.
- All benchmarks report `ns/op`, `MB/s`, `B/op`, and `allocs/op` via `-benchmem`. Results are stable across repeated runs on the same machine; no checked-in reference vectors are required because these are performance characterizations, not numerical correctness tests.

## Implemented tokenizers (`P06-009` and `P06-010`)

- `Tokenizer` interface: `Encode(ctx, text)`, `EncodeBatch(ctx, texts)`, `Decode(ctx, ids)`, `VocabSize()`, `SpecialTokens()`, `ID(token)`, and `Token(id)`. All public operations take `context.Context` as the first argument and honor cancellation. Implementations must be deterministic and safe for concurrent calls.
- `Encoding` holds `IDs`, `Tokens`, `Offsets` (character `[start, end)`), and `SpecialTokensMask`, all with equal length. Slices are defensively copied on construction and on read; `Clone` produces an independent encoding. `HasSpecial` reports whether any special token is present.
- Special token names are defined as constants (`[BOS]`, `[EOS]`, `[UNK]`, `[PAD]`, `[CLS]`, `[SEP]`, `[MASK]`). `DefaultSpecialIDs()` returns a stable reserved ID map (0..6); implementations may use different IDs as long as they are documented and consistent between `Encode` and `Decode`.
- `NewByteTokenizer()` is a deterministic reference tokenizer: one ID per byte (7..262), BOS prepended and EOS appended on encode, BOS/EOS/PAD/UNK skipped on decode, and IDs outside `[0, 262]` rejected with `errs.ErrOutOfBounds`.
- `TokenizerFunc` adapts a plain encode/decode function pair into a `Tokenizer`, validating constructor inputs (non-nil functions, positive vocab size, unique in-range special IDs).
- `NewCharLevelBPETokenizer(vocabSize, special)` and `NewByteLevelBPETokenizer(vocabSize, special)` create BPE tokenizers. `Train(texts, vocabSize)` builds the vocabulary and merge rules deterministically. `Encode` applies merges in rank order until no more merges are possible; `Decode` concatenates token strings (validating UTF-8 for byte-level). `ID`/`Token` provide bidirectional token-ID lookup.
- `NewWordPieceTokenizer(byteLevel, special)` implements WordPiece (mirip BERT) dengan pencocokan terpanjang terlebih dahulu dan awalan `##` untuk subkata lanjutan. `Train` membangun kosakata secara deterministik dari frekuensi pasangan. `NewBytePieceTokenizerWithVocab` memungkinkan pemasangan kosakata langsung untuk uji kompatibilitas.
- `NewUnigramTokenizer(byteLevel, special)` implements SentencePiece/Unigram gaya dengan model bahasa unigram (log probabilitas per potongan) dan pemotongan melalui pemrograman dinamis untuk memaksimalkan jumlah log probabilitas. Awalan `_` menandai potongan awal kata (gaya SentencePiece). `NewUnigramTokenizerWithVocab` memungkinkan pemasangan kosakata dan probabilitas langsung untuk uji kompatibilitas.
- Invalid input wraps existing `errs` sentinels and supports `errors.Is`; cancellation is returned as `ctx.Err()`. A nil or untrained tokenizer fails with `errs.ErrInvalidState`.

## Implemented tokenizer asset parsing (`P06-012`)

- `TokenizerConfig` holds the loaded tokenizer assets: `Vocab` (token string -> ID), `IDToVocab` (inverse), `Merges` (BPE merge rules ranked), `MergeIndex` (pair -> merge index), `AddedTokens` (extra tokens with metadata), `SpecialTokens` (name -> token string), and `Source` (directory). All fields optional; nil means the asset was not available.
- `NewTokenizerConfig()` creates an empty configuration. `Len()`/`VocabSize()` report vocabulary size; `HasMerges()`, `HasAddedTokens()`, `HasSpecialTokens()` report asset availability.
- `FromDirectory(dir)` loads all available tokenizer assets from a directory: `vocab.json`, `vocab.txt`, `merges.txt`, `added_tokens.json`, `special_tokens_map.json`, and `tokenizer_config.json`. Missing files are skipped; at least one vocabulary file is expected for a useful config.
- `FromFiles(vocabJSON, vocabTXT, mergesTXT, addedTokensJSON, specialTokensMapJSON)` builds a config from individual file paths; empty paths are skipped, and at least one vocabulary file is required.
- `VocabJSON` supports both a flat map `{"token": id}` and a wrapper `{"vocab": {"token": id}}`. `VocabTXT` reads one token per line with line-number IDs; blank lines and `#` comments are skipped. `MergesTXT` reads `left right` pairs with `#` comments skipped; `#version:` header lines are ignored. `AddedTokensJSON` supports both a list of objects and a token->ID map. `SpecialTokensMapJSON` reads name->token pairs. `TokenizerConfigJSON` extracts `special_tokens_map` from a full tokenizer config.
- `SpecialTokenID(name)`, `TokenForID(id)`, and `IDForToken(token)` provide bidirectional lookups; all return zero values for missing entries on nil configs. `SortedVocab()` returns tokens in lexicographic order for deterministic testing.
- `Validate(ctx)` checks consistency: special tokens and added tokens must exist in the vocabulary, and merge rule IDs must resolve to known tokens. Invalid input wraps existing `errs` sentinels and supports `errors.Is`; cancellation is returned as `ctx.Err()`.

## Implemented instruction/preference schemas and collators (`P06-013`)

- `InstructionSample` represents one SFT training record with `Instruction`, `Input`, and `Output` fields. `NewInstructionSample` validates that instruction and output are non-empty. `SampleText` produces a formatted text ready for tokenization.
- `PreferenceSample` represents one DPO/PPO training record with `Prompt`, `Chosen`, and `Rejected` fields. `NewPreferenceSample` validates that all three fields are non-empty. `ChosenText` and `RejectedText` produce combined prompt+response text.
- `DatasetSchema` defines `SchemaInstruction` and `SchemaPreference` with `IsValid` and `String` methods.
- `TruncationPolicy` defines `TruncateEnd`, `TruncateStart`, and `TruncateNone`. `Truncate(ids, max, policy)` truncates token ID sequences to a maximum length, returning a defensive copy; `TruncateNone` rejects overflowing sequences with an error.
- `MaskType` defines `MaskCausal`, `MaskBidirectional`, and `MaskLoss`. `CausalMask(n)`, `BidirectionalMask(n)`, and `LossMask(n, excludeIndices...)` generate attention/loss masks.
- `InstructionCollate` implements SFT collation: encodes instruction+input+output, truncates to `maxLen`, pads with PAD ID, and produces a loss mask active only on output tokens. Samples must have `instruction`, `input`, and `output` fields.
- `DPOCollate` implements DPO collation: encodes chosen and rejected separately, truncates, pads, and produces loss masks for both. Samples must have `prompt`, `chosen`, and `rejected` fields.
- `SortIndices(indices)` returns a sorted copy for deterministic comparison. `ValidateSchema(schema, sample)` checks required fields per schema.
- Invalid input wraps existing `errs` sentinels and supports `errors.Is`; cancellation is returned as `ctx.Err()`.

## Planned, not yet implemented

The following are roadmap items and must not be assumed available because the contracts exist: JSONL/NDJSON and CSV readers. See [`TASK.md`](TASK.md) for their task IDs and acceptance gates.

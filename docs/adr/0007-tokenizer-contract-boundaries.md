# ADR-0007: Tokenizer contract boundaries

- Status: Accepted
- Date: 2026-11-28

## Decision

`data.Tokenizer` is the public contract for encoding text to token IDs and decoding back. `Encode`/`Decode` take `context.Context` as their first argument, honor cancellation, and must be deterministic for the same tokenizer configuration. `Encoding` is an immutable result holding IDs, tokens, character offsets, and a special-token mask; all slices are defensively copied on construction and on read.

Special token names (`[BOS]`, `[EOS]`, `[UNK]`, `[PAD]`, `[CLS]`, `[SEP]`, `[MASK]`) are package-level constants. `DefaultSpecialIDs()` reserves IDs 0..6 for them, but implementations may assign different IDs as long as they are documented and consistent between `Encode` and `Decode`. A tokenizer's vocabulary and special-to-ID mapping are owned by the implementation; the contract only requires `VocabSize`, `SpecialTokens`, `ID`, and `Token` to be consistent.

`NewByteTokenizer()` is a deterministic reference implementation: one ID per byte (7..262), BOS prepended and EOS appended on encode, and BOS/EOS/PAD/UNK skipped on decode. It is intended for testing and as a correctness reference, not as a production subword tokenizer. `TokenizerFunc` adapts a plain encode/decode function pair into a `Tokenizer` with constructor validation.

## Consequences

- Tokenizer behavior is verified against checked-in reference vectors (byte-level round trips) and reports unsupported token IDs instead of silently degrading.
- No BPE, WordPiece, SentencePiece, or asset parser is claimed here; those are separate planned implementations behind the same `Tokenizer` interface.
- Encodings are immutable and safe to share; callers must not mutate returned slices.
- Cancellation during batch encode stops at the current text and returns `ctx.Err()`.

# ADR-0004: P5 context binding and PEFT boundaries

- Status: Accepted
- Date: 2026-11-28

## Decision

`nn.Context` binds one `autograd.Tape` to the parameter leaves used in one training step, and caches one leaf per `*nn.Parameter`. Layer and functional APIs accept the context explicitly instead of using an ambient tape. Custom differential operations are registered idempotently on the tape.

PEFT adapters depend on small `Projector` and `LinearLike` interfaces instead of changing `nn.Linear` contracts. Merge and unmerge operate only through typed weight access and roll back across all wrapped layers. LoRA is a production contract; IA3, prompt tuning and prefix tuning remain explicitly scoped prototypes with independent acceptance tests. Prefix tuning owns key/value prefixes only and does not own mask or multi-layer state.

Optimizer state is keyed by full parameter name and encoded in sorted order so serialization does not depend on map iteration.

## Consequences

- A parameter used twice in one forward pass receives one gradient contribution path, not duplicate leaves.
- `autograd.CheckGradient` needs operation registration inside its own tape; test helpers follow that rule.
- New layer types can be wrapped by PEFT without adding merge methods to core layers.
- Prototype PEFT methods can be tested without implying production contract stability.
- Serialized optimizer state is reproducible and reject mismatched parameter sets.

# P5 neural networks, transformers and PEFT contracts

This document records the implemented semantics of phase P5. It is a behavioural contract, not a second roadmap: [`TASK.md`](TASK.md) remains the canonical status checklist. All public APIs are under `src/nn`, `src/optim`, `src/transformer` and `src/peft`.

## Common execution model

- A `nn.Context` binds exactly one `autograd.Tape` to the parameter leaves used in one forward/backward cycle. `Context.Variable` caches one leaf per `*nn.Parameter`; `Context.Gradient` reads the gradient recorded on that tape and returns `errs.ErrNoGradient` when none exists.
- `nn.NewTape()` registers every P5 differential operation. `nn.RegisterOperations` is idempotent and tolerates `errs.ErrOperationAlreadyRegistered`; `autograd.CheckGradient` creates its own tape, so tests must register operations inside the scalar function.
- Parameter and buffer tensors remain owned by their modules. `Close` is terminal; callers must not use a module after closing it. State snapshots are copies.
- Supported differentiable element types are `dtype.Float32` and `dtype.Float64`. Integer tensors are control inputs (indices and operation configuration) and are non-differentiable.
- Shape, dtype, rank and overflow checks happen before allocation. Ordinary invalid input returns an error that wraps a sentinel from `errs`; it does not panic.

## Module, parameter and state contracts

- `nn.Module` exposes `Name`, `Parameters`, `Buffers`, `Train`, `IsTraining` and `Close`. `nn.ModuleBase` composes children and propagates train/eval mode to the whole subtree.
- `Parameters()` and `Buffers()` return fully qualified names such as `encoder.layer0.attention.query.weight`, sorted lexicographically. Non-persistent buffers are omitted from `Buffers()` and from state dictionaries.
- `nn.Parameter` owns a trainable tensor with an explicit `requiresGrad` flag. `SetRequiresGrad` is rejected after close. `Update` applies one in-place transformation and preserves dtype and shape.
- `nn.StateDict` stores copied tensors by name. `MarshalBinary`/`UnmarshalBinary` use a versioned binary format with stable sorted ordering. `StateDictOf` snapshots persistent parameters and buffers; `LoadStateDict` copies entries by exact name and rejects unknown names.

## Layers, losses and operations

- `Linear` computes `y = x @ weightᵀ + bias` for `x` shaped `[*, in]` and weight shaped `[out, in]`. Bias is optional and shaped `[out]`. Default weight initialization is Kaiming uniform; default bias is zero.
- Activations are element-wise: ReLU, sigmoid, tanh, GELU and SiLU. ReLU returns the input when it is greater than zero and zero otherwise.
- `LayerNorm` normalizes the last `features` dimension and applies learned gain/bias; its backward includes the covariance term of the upstream gradient with the normalized output. `RMSNorm` normalizes by root mean square and applies a learned scale. Both accept float32/float64.
- `Dropout` applies inverted dropout only in training mode. In eval mode `Forward` returns the same variable. An injected `*rand.Rand` makes masks deterministic; zero probability is supported.
- `Conv2D` and pooling use NCHW layout only. `Conv2D` computes cross-correlation with weight `[out, in, kernelH, kernelW]`, optional bias `[out]`, positive stride and non-negative padding. `MaxPool2D` and `AvgPool2D` accept `[batch, channels, height, width]` with kernel `[kernelH, kernelW]`, positive stride and non-negative padding. Configuration tensors are non-trainable int64 leaves.
- `Embedding` accepts int64 indices and a `[vocabulary, dim]` weight. `RoPE` accepts `[batch, heads, time, dim]` with even `dim` and cosine/sine tables `[time, dim/2]`. Attention accepts query `[batch, queryHeads, queryTime, dim]`, key/value `[batch, keyValueHeads, keyTime, dim]`, where `queryHeads` is divisible by `keyValueHeads` (MHA, GQA, MQA). An optional additive mask is `[batch, queryTime, keyTime]` or `[1, queryTime, keyTime]`; blocked scores use `-Inf` and softmax is stabilized by subtracting the row maximum.
- `Reshape`, `Transpose`, `MatMul`, `BiasAdd`, `Scale`, `ScaleLast`, `ConcatTime` and `ConcatTimePrefix` are differentiable with shape-checked backward functions.
- `MSELoss` supports `none`, `mean` and `sum` reductions. `none` returns the elementwise loss; `mean` and `sum` return scalars. Gradients flow to both prediction and target when the target variable requires gradients.
- `CrossEntropyLoss` accepts logits `[batch, classes]` and int64 targets `[batch]`. It computes a numerically stable softmax log-loss with mean reduction; target values are control inputs, not differentiable parameters.
- Initializers are deterministic when given an injected RNG: `Uniform`, `Normal`, `XavierUniform`, `XavierNormal`, `KaimingUniform` and `KaimingNormal`, plus `Fill`, `Zeros` and `Ones`.

## Optimizers and schedules

- `optim.Param` binds a full parameter name to an `*nn.Parameter`. `Parameters` enumerates all parameters; `SelectParameters` uses glob patterns on full names.
- `SGDConfig` supports learning rate, momentum in `[0,1)`, coupled L2 weight decay and optional Nesterov momentum. `Step` updates only parameters that have a gradient on the supplied `nn.Context`.
- `AdamConfig` supports learning rate, beta1/beta2, epsilon and weight decay. `AdamW: false` applies coupled L2 decay before the moment estimates; `AdamW: true` applies decoupled decay after bias correction. Both keep a per-parameter step count, first moment and second moment.
- Optimizer state is keyed by full parameter name and encoded in sorted order. `MarshalState`/`UnmarshalState` reject mismatched parameter sets, duplicate names and malformed payloads. `Close` is terminal.
- `Scheduler.Step` advances the schedule and sets the optimizer learning rate. Available schedules are constant, exponential, step, linear, cosine and warmup; all explicitly validate their parameters. `Reset` restores the initial rate.

## Transformer contracts

- `TokenEmbedding` maps int64 tokens `[batch, time]` to `[batch, time, dim]`. `RotaryEmbedding` stores persistent cosine/sine buffers and applies rotary embedding to `[batch, heads, time, headDim]`.
- `MultiHeadAttention` requires `ModelDim == QueryHeads * HeadDim`, `KeyValueHeads` divides `QueryHeads`, and `KeyValueHeads <= QueryHeads`. `KeyValueHeads == QueryHeads` is MHA, `1` is MQA, and intermediate values are GQA. Forward accepts query/key/value `[batch, time, modelDim]`, optional padding mask (`1` active, `0` padding) and returns `[batch, queryTime, modelDim]`. Causal masking is configured at construction.
- `FeedForward` is Linear -> GELU -> Linear; `SwiGLU` is `down(silu(gate(x)) * up(x))`. Both flatten `[batch, time, modelDim]` to `[batch*time, modelDim]` internally and restore the sequence shape.
- `EncoderBlock`, `DecoderBlock` and `EncoderStack` compose norms, attention and feed-forward modules. Their forward methods preserve `[batch, time, modelDim]` and validate mask lengths against batch and key time.

## PEFT contracts

- `Freeze`, `Unfreeze` and `TrainableParameters` operate on full parameter names with glob patterns. Invalid patterns are rejected even when a module has no parameters.
- `LoRALinear` wraps a `Projector`; base weights stay frozen by default and low-rank A/B parameters are trainable. Forward adds `(x @ Aᵀ) @ Bᵀ` scaled by `alpha/rank`. `Merge` folds B@A into a `LinearLike` base by a typed interface, and `Unmerge` restores the original base weights; both are idempotent and transactional across adapters.
- `Adapter` groups wrappers by name, exposes trainable parameters, and round-trips A/B values plus the merge flag in a versioned binary state. Loading state in the merged flag restores the flag without silently overwriting base weights.
- `IA3Linear` scales projected outputs by a learned vector; `PromptTuning` prepends a trainable `[promptLength, dim]` prompt to embedded tokens. Both are accepted prototypes with dedicated tests.
- `PrefixTuning` is an explicit prototype: it owns trainable key/value prefixes `[prefixLength, dim]` and prepends them to key/value sequences with `ConcatTimePrefix`. It does not own causal/padding mask semantics, layer-wise prefixes or adapter state serialization. Callers must use non-causal attention with no padding mask until those semantics are specified.
- `LoRA`, adapter state and merge/unmerge are production contracts. IA3, prompt tuning and prefix tuning are prototypes; additional acceptance tests are required before treating them as production features.

## Verification gates

P5 acceptance requires finite-difference gradient checks for every differentiable operation, optimizer step tests against hand-computed values, state round-trip checks, deterministic seeded initialization, convergence tests for simple regression and XOR, and transformer shape/mask/numerical checks. LoRA tests must prove that only declared trainable parameters change and that merged base weights are restored on unmerge. Race tests are executed by Linux CI; the Windows development host has no C compiler, so local `go test -race` is unavailable.

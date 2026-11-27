# GoTorch Roadmap P0–P13

This file is the compact phase overview. **The authoritative task breakdown, dependency graph, Task IDs, status and checklists live in [`TASK.md`](TASK.md).** Do not maintain a separate phase status here.

| Phase | Name | Main deliverable | Dependencies |
|---|---|---|---|
| P0 | Foundation | Module conventions, contracts, errors, platform policy, CI and agent workflow | — |
| P1 | Tensor Engine | Owning tensors, storage, shape/stride, views and tensor data operations | P0 |
| P2 | Automatic Differentiation | Reverse/forward AD, gradients and graph lifecycle | P1 |
| P3 | CPU Compute | Pure Go reference kernels, safe SIMD dispatch and optional BLAS/LAPACK | P1 |
| P4 | Scheduler & Memory | Bounded parallelism, allocator reuse and memory lifetime | P1, P3 |
| P5 | NN, Transformers & PEFT | Modules, transformer architectures, parameters, optimizers and Go-native adapters | P2, P3, P4 |
| P6 | Data & Tokenizer Engine | Dataset, tokenizer, DataLoader, samplers, collators and streaming | P0, P4 |
| P7 | Compiler | Typed graph IR, shape inference, optimization, fusion and execution planning | P1, P3, P4 |
| P8 | Training & Inference Runtime | Generation, SFT, DPO/RL trainers, evaluation, checkpoint and resume | P5, P6, P7 |
| P9 | Quantization & QLoRA | Mixed precision, INT8/INT4, NF4/FP4, optimizer states and quantized kernels | P3, P5, P8 |
| P10 | Distributed & Accelerated Runtime | Process launcher, data parallelism, sharding, collectives and recovery | P8; contracts begin P0 |
| P11 | Model Interoperability | Hugging Face configs/weights, ONNX, Safetensors and native model format | P1, P5, P8 |
| P12 | Ecosystem & Tooling | CLI, profiler, registry, extensions, docs and examples | P8–P11 |
| P13 | Production | Security, stress, performance, stability and release | P0–P12 |

## Go-Native AI Stack

GoTorch will provide first-party Go implementations of the major workflows commonly associated with **Transformers, PEFT, bitsandbytes, TRL and Accelerate**. These names describe capability targets, not a promise to clone Python APIs or to provide binary/API compatibility with those projects.

**Language and runtime constraint:** all GoTorch-owned framework implementation, model loading, tokenization, training and runtime orchestration must be written in Go. The supported CPU path must run without Python, subprocess-based Python tools, CGo, Rust, C or C++ components. Pure Go scalar kernels are the correctness baseline; optional SIMD paths must retain a scalar fallback. No feature is considered GoTorch functionality merely because an external Go project offers it.

| Capability target | GoTorch-native deliverables | Main roadmap phases |
|---|---|---|
| Transformers | Model/config registry; embeddings; positional encoding/RoPE; normalization; attention (MHA/GQA/MQA); masks; feed-forward blocks; encoder/decoder architectures; tokenizers; KV cache; generation and sampling APIs; Hugging Face config/weight loading for explicitly supported models | P5, P6, P8, P11 |
| PEFT | Parameter freezing/selection; adapter injection; LoRA first; adapter save/load and merge/unmerge; later IA3 and prompt/prefix tuning; QLoRA integration | P5, P8, P9, P11 |
| bitsandbytes-equivalent quantization | Blockwise INT8 and 4-bit NF4/FP4 weight quantization; scale metadata and double quantization; quantized matrix kernels; 8-bit optimizer-state support; explicit dequantization and accuracy contracts | P3, P5, P9 |
| TRL-style post-training | Supervised fine-tuning (SFT); preference datasets/collators; reward-model training; DPO; PPO/RLHF as a later trainer; evaluation and reproducible recipes | P6, P8 |
| Accelerate-style execution | Runtime/device abstraction; gradient accumulation and checkpointing; mixed-precision policy; process launcher and worker topology; CPU-first data parallelism; sharded training and distributed checkpoint/recovery | P4, P8, P9, P10 |

### Delivery order and compatibility boundary

1. Build transformer primitives and a small decoder-only reference model on top of the completed Tensor API, Autograd and CPU kernels.
2. Add tokenizer and streaming text/instruction/preference dataset support, then pretrained-weight loading for a declared model compatibility matrix.
3. Deliver stable inference/generation before training workflows; add LoRA before broader adapter methods.
4. Deliver SFT and DPO first, then reward modeling and PPO/RLHF after gradient, optimizer and checkpoint semantics are proven.
5. Implement quantization in layers: INT8, 4-bit NF4/FP4, double quantization, then QLoRA and low-bit optimizer states.
6. Scale from one-process CPU training to multi-process data parallelism and sharded state. Hardware-specific acceleration is optional and must not be required to use the Go CPU path.

**Compatibility is declared per architecture, tokenizer, weight format, dtype and training algorithm.** “Transformers/PEFT/bitsandbytes/TRL/Accelerate support” must never be claimed as complete from one demo or one imported model. Each supported combination needs a test fixture, numerical/parity criteria, documented limitations and reproducible examples.

**Go-native exit gates:** a clean machine can build, test, load a supported model, tokenize data, run inference and execute the supported training recipe using the Go toolchain only; CI must verify that the required path invokes no Python interpreter or foreign-language runtime. Pure Go CPU correctness remains available when SIMD or optional backends are unavailable.

## Current implementation checkpoint

<<<<<<< HEAD
The near-term compute emphasis is CPU-first: Pure Go correctness kernels, SIMD acceleration with safe fallback, then optional BLAS/LAPACK. The [Go AI Ecosystem Guide](AI_ECOSYSTEM.md) lists reusable external Go modules as research/reference options; they are not GoTorch core dependencies or substitutes for the first-party Go implementations planned above.
=======
P0 Foundation, P1 Tensor Engine and P2 Automatic Differentiation are implemented. The next implementation phase is P3 CPU Compute: Pure Go reference kernels first, then safe SIMD acceleration and optional BLAS/LAPACK. The [Go AI Ecosystem Guide](AI_ECOSYSTEM.md) lists reusable external Go modules as research/reference options; they are not GoTorch core dependencies or substitutes for the first-party Go implementations planned above.
>>>>>>> origin/main

P0 contracts exist for errors, dtype, shape, tensor metadata, storage, compute backend, runtime lifecycle, distributed identity and platform policy. The reusable Go packages are being organized under `src/`; see [P0 implementation](P0_IMPLEMENTATION.md).

A phase is not complete because its package or interface exists. It is complete only after its implementation, tests, documentation, platform verification and exit criteria pass. See [`TASK.md`](TASK.md) for the operational checklist.

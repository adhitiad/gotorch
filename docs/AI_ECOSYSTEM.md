# Go AI Ecosystem Guide

**Last reviewed: 2026-11-27**

This guide answers a common contributor question: *which Go package or module should I use to build, train, fine-tune, load data for, optimize, or serve an AI model?*

The important distinction is between:

1. **GoTorch-owned capabilities** — code implemented in this repository.
2. **Existing Go ecosystem projects** — libraries contributors can use today without reimplementing everything.
3. **Python/C/C++ ecosystem components** — tools that do not have a fully compatible Go-native equivalent.

Do not add every library in this guide to GoTorch's `go.mod`. These are ecosystem choices and optional integrations, not mandatory core dependencies.

## 1. Quick decision table

| Goal | Go package / project to investigate | What it provides | Important limitation |
|---|---|---|---|
| Build and train neural networks in Go today | [GoMLX](https://github.com/gomlx/gomlx) | Go-native tensors, differentiable graphs, layers, optimizers, training, Pure Go and OpenXLA backends | Separate framework; not a GoTorch dependency |
| Download Hugging Face model/dataset files | [go-huggingface hub](https://github.com/gomlx/go-huggingface) | Hub metadata, downloads and cache | Does not implement the whole Python Transformers API |
| Tokenize Hugging Face text | [go-huggingface tokenizers](https://github.com/gomlx/go-huggingface) | Go tokenizers for supported model tokenizers | Tokenizer/model coverage is still expanding |
| Read selected Hugging Face transformer models | [go-huggingface/models](https://github.com/gomlx/go-huggingface) | Experimental conversion of selected architectures to GoMLX | Not a universal `AutoModel` replacement |
| Read GGUF / Safetensors weights | [go-huggingface](https://github.com/gomlx/go-huggingface) | GGUF and Safetensors readers, tensor metadata/weights | Model execution still needs a compatible runtime |
| Matrix and numerical algorithms | [Gonum](https://github.com/gonum/gonum) | Matrices, statistics, optimization and BLAS/LAPACK interfaces | Not a complete deep-learning framework |
| Detect CPU SIMD features | [golang.org/x/sys/cpu](https://pkg.go.dev/golang.org/x/sys/cpu) | AMD64 AVX/AVX2 and ARM64 ASIMD feature flags | Detection only; it does not implement tensor kernels |
| Alternative detailed CPU feature detection | [klauspost/cpuid](https://github.com/klauspost/cpuid) | CPU features, topology and architecture information | Optional; do not use two detection libraries without a reason |
| Stream JSONL / CSV datasets | Go standard library | `bufio`, `encoding/json`, `encoding/csv`, `io` | JSONL record/schema validation is application responsibility |
| Read/write Parquet datasets | [parquet-go](https://github.com/parquet-go/parquet-go) | Go Parquet reader/writer | Not a training DataLoader by itself |
| Run an existing ONNX model | [ONNX Runtime Go bindings](https://pkg.go.dev/github.com/microsoft/onnxruntime/go/onnxruntime) | Go bindings to ONNX Runtime C API | Requires the native ONNX Runtime shared library; this is an inference runtime, not a training stack |

## 2. Hugging Face Python stack versus Go

The following Hugging Face projects are Python-first libraries. They do not have a drop-in Go module with full API and feature parity:

| Python project | Go status | Practical Go path |
|---|---|---|
| `transformers` | No general-purpose Go port with full `AutoModel` / `Trainer` parity | GoMLX plus the experimental `go-huggingface/models/transformer` conversion path for supported architectures; ONNX Runtime for supported exported models |
| `peft` (LoRA/QLoRA) | No standalone Go PEFT package with full parity | GoMLX documents PEFT/LoRA adapter hooks. Treat this as a supported feature subset, not a complete PEFT clone |
| `bitsandbytes` | No drop-in Go bitsandbytes module | GoMLX documents selected quantized kernels and QuantNF4/QLoRA-related compute. This is not full bitsandbytes compatibility |
| `trl` (SFT/DPO/RL post-training) | No Go package with TRL's trainer/API coverage | GoMLX has model training and fine-tuning primitives; SFT/DPO recipes still require custom training/loss/data logic or a Python TRL workflow |
| `accelerate` (DDP/FSDP/device orchestration) | No drop-in Go Accelerate package | GoMLX documents experimental distributed execution across multiple GPUs/TPUs through XLA. GoTorch's own distributed plan is P10 |

**Correction to the common assumption that “Go has no AI training ecosystem”:** GoMLX is an actively developed Go-native machine-learning framework. It supports training and fine-tuning, a portable Pure Go backend, and an OpenXLA backend for accelerated hardware. Its documentation also describes LoRA/PEFT capabilities, selected quantization paths, Hugging Face model/dataset tooling, and experimental multi-GPU/TPU distributed execution. These features belong to GoMLX and its ecosystem; they are not implemented by GoTorch merely because this guide references them.

References:
- [GoMLX overview](https://gomlx.github.io/docs/overview/)
- [GoMLX backends](https://gomlx.github.io/docs/backends/)
- [GoMLX repository and PEFT overview](https://github.com/gomlx/gomlx)
- [go-huggingface README](https://github.com/gomlx/go-huggingface)

GoMLX is a useful choice when someone needs to train a Go model today. GoTorch remains an independent framework project with its own tensor, autograd, compute, data, training and distributed roadmap.

## 3. Recommended CPU + SIMD path

**This is the current engineering priority for GoTorch.**

The required execution hierarchy is:

1. **Pure Go scalar reference kernels** — correctness, portability and mandatory fallback.
2. **Go SIMD kernels** — vectorized CPU execution, selected only when supported and beneficial.
3. **Optional BLAS/LAPACK adapters** — dense matrix operations for workloads where benchmarks justify the dependency or native runtime.
4. **Specialized fused kernels** — only after correctness and benchmark evidence exists.

### Go 1.27 SIMD APIs

Go 1.27 provides experimental SIMD APIs:

- `simd` — portable, vector-size-agnostic operations.
- `simd/archsimd` — architecture-specific vector operations.

Both are experimental and require enabling the experiment when building:

```bash
GOEXPERIMENT=simd go test ./...
GOEXPERIMENT=simd go test -bench=. -benchmem ./src/...
```

For PowerShell:

```powershell
$env:GOEXPERIMENT = "simd"
go test ./...
```

Do not make the experimental SIMD API a public GoTorch compatibility promise. Isolate it behind internal kernel interfaces/build constraints, retain the scalar implementation, and verify every optimized result against the reference implementation. Keep a normal build that does not set `GOEXPERIMENT`.

Official references:
- [Go 1.27 release notes](https://go.dev/doc/go1.27)
- [Platform-independent SIMD in Go](https://go.dev/blog/simd-experiment)
- [Architecture-specific SIMD in Go](https://go.dev/blog/archsimd)

### Runtime CPU feature detection

Use runtime capability checks rather than assuming that every AMD64 processor has AVX2 or that every supported platform has the same vector width.

A minimal optional dependency is:

```bash
go get golang.org/x/sys/cpu
```

Relevant feature flags include `cpu.X86.HasAVX2`, `cpu.X86.HasAVX512`, and `cpu.ARM64.HasASIMD`. The package performs architecture-specific feature detection and accounts for OS support where required.

The GoTorch dispatcher must still have a safe scalar fallback. A CPU feature flag is permission to choose a kernel, not proof that a given kernel is numerically correct.

### BLAS / LAPACK

Gonum is a practical Go-native numerical library:

```bash
go get gonum.org/v1/gonum@latest
```

Useful packages include:

- `gonum.org/v1/gonum/mat` — matrix structures and linear algebra.
- `gonum.org/v1/gonum/blas` — BLAS interfaces.
- `gonum.org/v1/gonum/blas/gonum` — Pure Go BLAS implementation.
- `gonum.org/v1/gonum/blas/blas64` and `blas32` — wrappers for real-valued BLAS APIs.

The documented Pure Go BLAS implementation is incomplete and implements the float32/float64 API subset. Check operation coverage before depending on it as a complete BLAS implementation.

OpenBLAS, BLIS or vendor BLAS can be explored later as optional native adapters. They must not become mandatory CGo dependencies in GoTorch core. Keep the Pure Go reference path available and benchmark conversion/copy overhead as part of any adapter evaluation.

References:
- [Gonum](https://github.com/gonum/gonum)
- [Gonum BLAS package](https://pkg.go.dev/gonum.org/v1/gonum/blas)
- [Gonum Pure Go BLAS implementation](https://pkg.go.dev/gonum.org/v1/gonum/blas/gonum)

## 4. Dataset input: JSONL, CSV and Parquet

A separate “AI dataset framework” is not required just to read a dataset.

### JSONL / NDJSON

Use the standard library:

- `bufio` for bounded streaming input.
- `encoding/json` for each record.
- `io` for streaming and EOF handling.
- `context` for cancellable dataset iteration.

Read one line at a time and decode into a typed sample struct. Set an explicit maximum record size when using `bufio.Scanner`; the default token limit is unsuitable for arbitrary large training examples. Do not load an entire multi-gigabyte JSONL file into memory.

### CSV

Use `encoding/csv` and stream records from an `io.Reader`. Validate column names, field counts, types and missing-value policy in the dataset adapter.

### Parquet and Hugging Face datasets

For Parquet files:

```bash
go get github.com/parquet-go/parquet-go@latest
```

For Hugging Face Hub downloads, tokenizers, and the Parquet-based dataset iterators provided by that project:

```bash
go get github.com/gomlx/go-huggingface@latest
```

The `go-huggingface` modules are componentized; downloading Hub files does not require importing all the model-conversion packages. Model conversion and supported dataset/model coverage are still under active development.

The GoTorch P6 Data Engine should eventually provide a stable `Dataset`, `Sampler`, `Batch`, `Transform` and bounded `DataLoader` API over ordinary `io.Reader`-based adapters. It should not force every user to adopt one file format or dataset provider.

## 5. Practical install recipes

Choose only the dependencies for the problem at hand.

### Train a model in Go today

```bash
go get github.com/gomlx/gomlx@latest
```

Start with the [GoMLX getting-started guide](https://gomlx.github.io/docs/get-started/). Use its Pure Go backend for portable CPU execution or its XLA backend when the required native runtime/device setup is available.

### Download Hugging Face model files / tokenize / scan supported datasets

```bash
go get github.com/gomlx/go-huggingface@latest
```

Start with the [go-huggingface README](https://github.com/gomlx/go-huggingface). Confirm that the specific model architecture, tokenizer and dataset format you need are supported; transformer conversion is not universal.

### Numerical algorithms and BLAS reference

```bash
go get gonum.org/v1/gonum@latest
```

### CPU feature detection for a custom kernel dispatcher

```bash
go get golang.org/x/sys/cpu
```

### Read Parquet directly

```bash
go get github.com/parquet-go/parquet-go@latest
```

Use `go mod tidy` after choosing packages. Do not add optional ecosystem dependencies to GoTorch core just because they are listed here.

## 6. What belongs in GoTorch itself?

The purpose of this guide is to help users reuse existing tools while keeping the GoTorch core small, portable and testable.

| Capability | Existing Go option | GoTorch direction |
|---|---|---|
| Tensor storage / views | GoMLX, Gonum have their own tensor/matrix models | GoTorch P1 owns its Tensor and storage contracts |
| Autograd | GoMLX and other projects | GoTorch P2 |
| CPU scalar kernels | Pure Go implementation | GoTorch P3 reference kernels |
| SIMD | Go 1.27 experimental APIs and assembly patterns | GoTorch P3 internal SIMD kernels with scalar fallback |
| BLAS | Gonum BLAS, optional native BLAS providers | GoTorch P3 optional backend adapter |
| Dataset parsing | Standard library, parquet-go, go-huggingface | GoTorch P6 streaming Dataset/DataLoader abstractions |
| Training loop / optimizers | GoMLX | GoTorch P5/P8 |
| Quantization | Selected GoMLX kernels and model formats | GoTorch P9 |
| Distributed training | Experimental GoMLX XLA support | GoTorch P10 |

**Current focus:** complete Tensor correctness first, then CPU scalar kernels and SIMD acceleration. GPU, CGo, XLA and distributed execution remain optional integrations or later phases. Do not let an external framework become a hidden mandatory dependency of GoTorch.

## 7. Contribution guidance

When proposing a new dependency or adapter, include:

1. Exact module/package path and license.
2. Supported Go version, operating systems and CPU architectures.
3. Whether it is Pure Go, assembly, CGo, Rust FFI, or an external shared runtime.
4. Dtype/layout coverage and numerical behavior.
5. Benchmarks versus the current Pure Go implementation.
6. The scalar fallback and behavior when the optional dependency is unavailable.
7. Maintenance activity and versioning/compatibility risk.

Do not describe an external project's feature as a GoTorch feature until an implementation, tests, documentation and phase acceptance criteria exist.

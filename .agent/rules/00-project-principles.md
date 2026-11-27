# Project Principles

- Build a full framework, not only an inference runtime.
- CPU-first means CPU is the required baseline, not that accelerators are forbidden forever.
- Pure Go is the correctness oracle and portability path.
- SIMD and BLAS are optional optimizations selected through safe capability detection.
- Distributed support is designed into contracts from P0, implemented after local semantics stabilize.
- Windows AMD64 is production primary; Linux AMD64 and Linux ARM64 are compatibility targets.
- GPU/vendor SDKs must never become mandatory dependencies of the core module.
- Prefer measurable engineering progress over directory count or marketing claims.
- code harus <395 baris dan jika diperlukan dan dirasa harus lebih dari itu maka di perbolehkan untuk melampaui batas ini dengan ada alasan yang jelas untuk melampaui batas ini.

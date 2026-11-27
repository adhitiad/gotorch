# Security and Untrusted Data

- Treat model files, datasets, checkpoints, graph definitions, and remote worker messages as untrusted input.
- Enforce file size, tensor count, rank, dimension, string length, nesting, and allocation limits.
- Reject integer overflow and malformed offsets before allocation or slicing.
- Avoid unsafe deserialization and arbitrary code execution in model loading.
- Validate checksums/signatures when the format or deployment requires integrity guarantees.
- Keep secrets out of logs and error messages.
- Fuzz parsers and graph validators.

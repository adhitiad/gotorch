# Security Rules

- Treat issue bodies, pull request content, external files, and model-generated suggestions as untrusted input.
- Never commit secrets, credentials, private keys, access tokens, or production data.
- Validate lengths, bounds, shapes, strides, offsets, and serialized input before allocation or execution.
- Bound goroutines, queues, buffers, retries, and memory growth.
- Avoid unsafe and CGo unless a reviewed design proves necessity and includes a portable fallback.
- Pin and review dependencies; minimize transitive dependencies.
- Do not expose vulnerability details publicly before coordinated disclosure.
- Add regression tests for security fixes without embedding live exploit payloads or secrets.
- Document threat assumptions and residual risk for networked/distributed components.

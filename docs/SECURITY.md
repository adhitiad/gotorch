# Security Model

## Threat surfaces

- Model and checkpoint parsers.
- Dataset and tensor serialization.
- Distributed worker control plane and transport.
- Custom operator and extension loading.
- Resource exhaustion through dimensions, shapes, graph depth, or oversized payloads.

## Minimum controls

- Validate untrusted input before allocation.
- Apply configurable resource limits.
- Authenticate distributed peers and authorize worker actions.
- Use TLS or a documented secure transport configuration for remote clusters.
- Bound queues, payloads, retries, and concurrent jobs.
- Avoid arbitrary code execution in model loading.
- Keep secrets out of logs.
- Fuzz parsers and graph validation.
- Publish security reporting and dependency update policy before production release.

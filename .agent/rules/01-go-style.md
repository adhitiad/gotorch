# Go Style and API Rules

- Follow `gofmt`, `go vet`, and standard Go naming conventions.
- Keep packages cohesive and avoid import cycles.
- Put context first in long-running or cancelable public operations.
- Return errors; do not panic for ordinary invalid input.
- Validate dimensions and multiplication overflow before allocation.
- Document exported identifiers and concurrency guarantees.
- Avoid global mutable state and implicit process-wide configuration.
- Prefer explicit options over environment-dependent hidden behavior.
- Add examples for public APIs when semantics are non-obvious.

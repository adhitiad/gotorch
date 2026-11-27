# Skill: Change a Public API

1. Identify callers and compatibility surface.
2. State intended semantics and concurrency/lifetime guarantees.
3. Prefer additive changes; explain why a breaking change is necessary.
4. Add compile-time usage examples and behavioral tests.
5. Update GoDoc, README links, and relevant design documents.
6. Check Windows AMD64, Linux AMD64, and Linux ARM64 builds.
7. Record the decision in an ADR when it changes a fundamental contract.

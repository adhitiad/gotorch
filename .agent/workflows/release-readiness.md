# Workflow: Release Readiness

- Confirm phase acceptance criteria are met.
- Review public API and compatibility notes.
- Run full tests, race tests, vet, fuzz smoke tests, and benchmarks.
- Validate Windows AMD64 primary build and Linux compatibility builds.
- Audit dependencies and model/parser security.
- Verify documentation, changelog, examples, and license status.
- Confirm no secrets, generated artifacts, or temporary files are committed.
- Produce a release report with known limitations.

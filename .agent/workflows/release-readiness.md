# Workflow: Release Readiness

- Confirm all relevant task checklists and phase exit gates in `docs/TASK.md` are satisfied.
- Review public API and compatibility notes/ADRs.
- Run full tests, race tests, vet, fuzz smoke tests and benchmarks as applicable.
- Validate Windows AMD64 primary build and Linux AMD64/Linux ARM64 compatibility tests.
- Audit dependencies and model/parser/distributed security boundaries.
- Verify documentation, changelog, examples and license status.
- Confirm no secrets, generated artifacts or temporary files are committed.
- Ensure development branch CI passes before promoting from `adhitiad_gpt` to `main`.
- Produce a release report with known limitations and reproducible verification evidence.

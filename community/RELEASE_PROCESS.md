# Release Process

## Release principles
Releases are evidence-based. A roadmap phase is not complete merely because its documentation exists. Do not publish a stable release while required checks or platform claims remain unverified.

## Readiness checklist
- [ ] Release scope and supported platforms are explicit.
- [ ] Relevant unit, integration, fuzz, race, and numerical tests pass.
- [ ] Windows AMD64 and Linux AMD64 checks pass; Linux ARM64 status is recorded.
- [ ] gofmt, go vet, static analysis, and dependency review pass.
- [ ] Benchmarks are reproducible and compared against a documented baseline.
- [ ] Security review and known limitations are documented.
- [ ] CHANGELOG and migration notes are complete.
- [ ] Release artifacts are reproducible and checksummed.

## Versioning
Use Semantic Versioning once public APIs are declared stable. Before 1.0.0, breaking changes may occur; document them clearly. Tag releases as vMAJOR.MINOR.PATCH.

## Workflow
1. Open a release issue and define scope.
2. Freeze scope and resolve release-blocking issues.
3. Run the complete validation matrix and record evidence.
4. Update changelog, docs, and version metadata.
5. Obtain project lead approval.
6. Tag and publish release notes with known limitations and checksums.
7. Monitor issues and prepare a patch release if required.

## Rollback
If an artifact is defective, mark the release as withdrawn, publish a clear advisory, and issue a corrected version. Never silently replace immutable release artifacts.

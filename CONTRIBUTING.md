# Contributing to GoTorch

Thank you for helping improve GoTorch. Contributions include code, tests, benchmarks, documentation, architecture review, issue triage, and community support.

## Before opening a pull request

1. Read [AGENTS.md](AGENTS.md) and the relevant [.agent/rules/](.agent/rules/) documents.
2. Review the [project charter](docs/PROJECT_CHARTER.md), [roadmap](docs/ROADMAP.md), and [development workflow](docs/DEVELOPMENT_WORKFLOW.md).
3. Search existing issues and discussions. Open an issue for non-trivial work and identify its P0–P13 phase.
4. For cross-package, public API, numerical contract, concurrency, platform, or distributed architecture changes, submit an architecture proposal and ADR first.

## Development requirements

- Keep changes focused and reviewable.
- Add or update tests and documentation with behavior changes.
- Run `gofmt`, `go test ./...`, and `go vet ./...`; run race, fuzz, and benchmark checks when relevant.
- Record exact commands, environment, and results. State explicitly which checks were not run.
- Report platform coverage honestly. Do not claim performance improvements without reproducible benchmark evidence.
- Do not commit secrets, local environment files, generated binaries, or private data.

## Pull request process

Use the pull request template, link related issues, describe compatibility and security impact, and respond to review comments. Do not merge with failing required checks or unresolved blocking feedback.

## Community standards

Participation is subject to the [Code of Conduct](community/CODE_OF_CONDUCT.md). See [Governance](community/GOVERNANCE.md), [Support](community/SUPPORT.md), and [Security Policy](community/SECURITY.md).

## License

By contributing, you agree that your contributions are provided under the repository's [MIT License](LICENSE), subject to applicable law.

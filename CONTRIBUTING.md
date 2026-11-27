# Contributing to GoTorch

Thank you for helping improve GoTorch. Contributions include code, tests, benchmarks, documentation, architecture review, issue triage and community support.

## Before opening a pull request

1. Read [AGENTS.md](AGENTS.md) and the relevant [.agent/rules/](.agent/rules/) documents.
2. Read the canonical [P0–P13 task plan](docs/TASK.md), [project charter](docs/PROJECT_CHARTER.md), [architecture](docs/ARCHITECTURE.md), [roadmap](docs/ROADMAP.md) and [Go AI Ecosystem Guide](docs/AI_ECOSYSTEM.md).
3. Search existing issues and discussions. Identify the task ID (for example `P03-006`) and confirm that no other contributor is already implementing it.
4. For public API, numerical, concurrency, storage, compiler, platform or distributed protocol changes, submit an architecture proposal and ADR before implementation when the change affects a fundamental contract.

## Repository structure

- Reusable Go packages and package tests belong under `src/<package>/`.
- Executables belong under `cmd/<name>/main.go`.
- Application composition and external adapters belong under `app/<name>/`.
- Do not add reusable framework packages at the repository root. Keep the root package facade small.
- New phase folders should be created only when a real implementation task needs them.

## Development requirements

- Keep changes focused and reviewable.
- Add or update tests and documentation with behavior changes.
- Run `gofmt`, `go test ./...` and `go vet ./...`; run race, fuzz and benchmark checks when relevant.
- Record exact commands, environment and results. State explicitly which checks were not run.
- Report platform coverage honestly. Do not claim performance improvements without reproducible benchmark evidence.
- Do not commit secrets, local environment files, generated binaries or private data.

## Agent collaboration

All coding agents use root `AGENTS.md`, `docs/TASK.md` and the portable skills under `.agents/skills/`. Use one task ID per implementation branch/PR. Parallel work is allowed only when ownership and interfaces do not overlap. Update the checklist in the same change that completes a task.

## Pull request process

Use the pull request template, link related issues, include the task ID, describe compatibility/security impact and list verification results. Do not merge with failing required checks or unresolved blocking feedback.

## Community standards

Participation is subject to the [Code of Conduct](community/CODE_OF_CONDUCT.md). See [Governance](community/GOVERNANCE.md), [Support](community/SUPPORT.md) and [Security Policy](community/SECURITY.md).

## License

By contributing, you agree that your contributions are provided under the repository's [MIT License](LICENSE), subject to applicable law.

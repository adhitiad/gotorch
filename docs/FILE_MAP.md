# File Map

- `AGENTS.md`: root-level agent instructions.
- `.agent/rules/`: project, Go, tensor/memory, compute, distributed, testing, security, documentation rules.
- `.agent/skills/`: phase implementation, CPU kernel, public API, distributed feature, and review playbooks.
- `.agent/workflows/`: issue-to-implementation and release-readiness workflows.
- `.agent/templates/`: implementation plan and review checklist.
- `docs/`: architecture, contracts, CPU, distributed, memory, roadmap, development, tests, platforms, security, glossary, ADRs.
- `.github/`: CI and issue templates.
- Root build files: `go.mod`, `Makefile`, `.gitignore`, `CONTRIBUTING.md`, `CHANGELOG.md`.

This is a foundation/documentation scaffold, not a complete framework implementation.

- `gotorch.go` and `gotorch_test.go`: minimal root package and scaffold smoke test.

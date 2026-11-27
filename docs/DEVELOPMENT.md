# Development Workflow

## Branching

- `main` is the stable integration branch.
- Use the explicitly designated development branch for active work when provided.
- Use short-lived feature branches for isolated changes.
- Do not force-push or rewrite shared history without explicit authorization.

## Change workflow

1. Identify roadmap phase and acceptance criteria.
2. Inspect current code and working tree.
3. Write a scoped plan.
4. Implement a coherent vertical slice.
5. Add tests and docs.
6. Run format, tests, vet, race tests, and benchmarks as relevant.
7. Review platform and compatibility impact.
8. Summarize evidence and limitations.

## Local checks

```bash
gofmt -w .
go test ./...
go test -race ./...
go vet ./...
go test -bench=. -benchmem ./...
```

Run only commands appropriate to the changed packages when iterating, but run the full suite before integration.

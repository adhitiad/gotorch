# Publishing GoTorch on pkg.go.dev

This document is the release procedure for the public Go module
`github.com/adhitiad/gotorch`.

## Release policy

- The Go module root is the repository root; do not create a nested `go.mod`.
- The first public foundation preview is `v0.1.0-alpha.1`.
- GoTorch is pre-1.0. The alpha line makes no backward-compatibility promise.
- Publish only immutable, annotated SemVer tags. Never move, delete, or reuse a tag after publication.
- A release tag must point at a commit whose native CI passed on Windows AMD64, Linux AMD64, and Linux ARM64.
- The module's license, `go.mod`, package comments, README, and release notes must be present before tagging.

## First publication

From a clean checkout of the reviewed development commit:

```bash
git fetch origin
git switch adhitiad_gpt
git pull --ff-only origin adhitiad_gpt

# Confirm the reviewed source and tests.
go test ./...
go test -race ./...
go vet ./...

# Create and push the immutable first-preview tag.
git tag -a v0.1.0-alpha.1 -m "GoTorch v0.1.0-alpha.1 — foundation preview"
git push origin v0.1.0-alpha.1
```

The tag is the publication event. The `Publish Go module` GitHub Actions workflow
runs the platform checks and then requests that exact version from
`proxy.golang.org`.

## Verify the module proxy

After the tag is pushed, the following command requests the version through the
official Go module proxy and verifies that Go can resolve it:

```bash
GOPROXY=https://proxy.golang.org go mod download -json github.com/adhitiad/gotorch@v0.1.0-alpha.1
GOPROXY=https://proxy.golang.org go list -m -versions github.com/adhitiad/gotorch
```

The Go module proxy can take a short time to index a new version. The publishing
workflow explicitly requests the tagged version to trigger the proxy fetch.

## Verify pkg.go.dev

- Module documentation: https://pkg.go.dev/github.com/adhitiad/gotorch
- Root package: https://pkg.go.dev/github.com/adhitiad/gotorch
- Tensor metadata: https://pkg.go.dev/github.com/adhitiad/gotorch/src/tensor
- Dtype definitions: https://pkg.go.dev/github.com/adhitiad/gotorch/src/dtype
- Shape definitions: https://pkg.go.dev/github.com/adhitiad/gotorch/src/shape

The pkg.go.dev service downloads module source from the Go module mirror and
indexes discovered modules. If the page is not visible after the proxy fetch,
open the module URL on pkg.go.dev and use its **Request** action.

## Future releases

1. Update `gotorch.Version` and release notes for the new version.
2. Run formatting, tests, race tests, vet, and the full native CI matrix.
3. Review exported API changes and document any contract change.
4. Create a new immutable SemVer tag; do not reuse a published version.
5. Let the tag workflow request the module version from the Go proxy.
6. Verify the version and package pages before announcing the release.

For a future stable release, use `v1.0.0` only after the public API and
backward-compatibility policy have been stabilized and reviewed.

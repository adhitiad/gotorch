GO ?= go

.PHONY: fmt test race vet check bench
fmt:
	gofmt -w .

test:
	$(GO) test ./...

race:
	$(GO) test -race ./...

vet:
	$(GO) vet ./...

check: test vet

bench:
	$(GO) test -bench=. -benchmem ./...

run:
	$(GO) run ./cmd/gotorch version

build:
	$(GO) build ./cmd/gotorch/

build-cross:
	CGO_ENABLED=0 GOOS=windows GOARCH=amd64 $(GO) build ./cmd/gotorch/
	CGO_ENABLED=0 GOOS=linux GOARCH=amd64 $(GO) build ./cmd/gotorch/
	CGO_ENABLED=0 GOOS=linux GOARCH=arm64 $(GO) build ./cmd/gotorch/

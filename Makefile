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

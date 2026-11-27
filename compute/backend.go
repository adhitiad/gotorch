// Package compute defines backend-independent kernel contracts. Concrete kernels
// are added in the CPU Compute phase (P3).
package compute

import (
	"context"
	"fmt"
	"strings"

	"github.com/adhitiad/gotorch/dtype"
	"github.com/adhitiad/gotorch/errs"
	"github.com/adhitiad/gotorch/storage"
	"github.com/adhitiad/gotorch/tensor"
)

// BackendKind identifies the implementation family, not a hardware guarantee.
type BackendKind uint8

const (
	BackendInvalid BackendKind = iota
	BackendReference
	BackendSIMD
	BackendBLAS
)

func (k BackendKind) Valid() bool { return k >= BackendReference && k <= BackendBLAS }

// KernelSpec describes the operation and its primary compute dtype/layout. The backend
// inspects Invocation for operation-specific operand signatures.
type KernelSpec struct {
	Operation string
	DType     dtype.DType
	Layout    tensor.Layout
}

// Validate checks a kernel specification before dispatch.
func (s KernelSpec) Validate() error {
	if strings.TrimSpace(s.Operation) == "" {
		return fmt.Errorf("%w: operation is required", errs.ErrInvalidArgument)
	}
	if !s.DType.Valid() {
		return fmt.Errorf("%w: %d", errs.ErrUnsupportedDType, s.DType)
	}
	if !s.Layout.Valid() {
		return errs.ErrInvalidLayout
	}
	return nil
}

// Argument binds metadata to a byte-storage handle and its base offset.
type Argument struct {
	Metadata tensor.Metadata
	Storage  storage.Storage
	Offset   int64
}

// Validate checks metadata and verifies its full strided byte span fits storage.
func (a Argument) Validate() error {
	if a.Storage == nil {
		return fmt.Errorf("%w: nil storage", errs.ErrInvalidArgument)
	}
	if err := a.Metadata.Validate(); err != nil {
		return err
	}
	span, err := a.Metadata.RequiredBytes()
	if err != nil {
		return err
	}
	return storage.ValidateRange(a.Storage.Size(), a.Offset, span)
}

// Invocation contains borrowed inputs and caller-provided outputs. A backend
// must not retain these slices or storage handles after Execute returns.
type Invocation struct {
	Inputs  []Argument
	Outputs []Argument
}

// Validate checks all arguments. Output allocation remains the caller's responsibility.
func (i Invocation) Validate() error {
	if len(i.Outputs) == 0 {
		return fmt.Errorf("%w: at least one output is required", errs.ErrInvalidArgument)
	}
	for index, arg := range i.Inputs {
		if err := arg.Validate(); err != nil {
			return fmt.Errorf("input %d: %w", index, err)
		}
	}
	for index, arg := range i.Outputs {
		if err := arg.Validate(); err != nil {
			return fmt.Errorf("output %d: %w", index, err)
		}
	}
	return nil
}

// Backend executes compatible kernels. Implementations must document determinism
// and concurrency guarantees and must honor cancellation at safe boundaries.
type Backend interface {
	Name() string
	Kind() BackendKind
	Supports(KernelSpec, Invocation) bool
	Execute(context.Context, KernelSpec, Invocation) error
}

// Run validates the call before handing borrowed storage to a backend.
func Run(ctx context.Context, backend Backend, spec KernelSpec, invocation Invocation) error {
	if ctx == nil || backend == nil {
		return fmt.Errorf("%w: context and backend are required", errs.ErrInvalidArgument)
	}
	if strings.TrimSpace(backend.Name()) == "" || !backend.Kind().Valid() {
		return fmt.Errorf("%w: backend identity is invalid", errs.ErrInvalidArgument)
	}
	if err := spec.Validate(); err != nil {
		return err
	}
	if err := invocation.Validate(); err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if !backend.Supports(spec, invocation) {
		return fmt.Errorf("%w: backend %q does not support %q", errs.ErrUnsupportedBackend, backend.Name(), spec.Operation)
	}
	return backend.Execute(ctx, spec, invocation)
}

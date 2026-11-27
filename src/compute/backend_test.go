package compute

import (
	"context"
	"errors"
	"io"
	"testing"

	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/shape"
	"github.com/adhitiad/gotorch/src/storage"
	"github.com/adhitiad/gotorch/src/tensor"
)

type testStorage struct{ size int64 }

func (s *testStorage) Size() int64                            { return s.size }
func (s *testStorage) ReadAt([]byte, int64) (int, error)      { return 0, io.EOF }
func (s *testStorage) WriteAt(p []byte, _ int64) (int, error) { return len(p), nil }
func (s *testStorage) Retain() (storage.Storage, error)       { return &testStorage{size: s.size}, nil }
func (s *testStorage) Release() error                         { return nil }

type testBackend struct{}

func (testBackend) Name() string                                                  { return "test-reference" }
func (testBackend) Kind() BackendKind                                             { return BackendReference }
func (testBackend) Supports(KernelSpec, Invocation) bool                          { return true }
func (testBackend) Execute(ctx context.Context, _ KernelSpec, _ Invocation) error { return ctx.Err() }

func TestKernelSpecValidation(t *testing.T) {
	if err := (KernelSpec{Operation: "add", DType: dtype.Float32, Layout: tensor.LayoutContiguous}).Validate(); err != nil {
		t.Fatal(err)
	}
	if err := (KernelSpec{DType: dtype.Float32, Layout: tensor.LayoutContiguous}).Validate(); err == nil {
		t.Fatal("empty operation accepted")
	}
}

func TestInvocationValidatesStorageSpan(t *testing.T) {
	s, _ := shape.New(2, 2)
	m, _ := tensor.NewMetadata(dtype.Float32, s)
	arg := Argument{Metadata: m, Storage: &testStorage{size: 16}}
	if err := (Invocation{Outputs: []Argument{arg}}).Validate(); err != nil {
		t.Fatal(err)
	}
	arg.Offset = 1
	if err := (Invocation{Outputs: []Argument{arg}}).Validate(); !errors.Is(err, errs.ErrOutOfBounds) {
		t.Fatalf("got %v", err)
	}
}

func TestRunValidatesBeforeBackend(t *testing.T) {
	ctx := context.Background()
	s, _ := shape.New(1)
	m, _ := tensor.NewMetadata(dtype.Float32, s)
	arg := Argument{Metadata: m, Storage: &testStorage{size: 4}}
	spec := KernelSpec{Operation: "identity", DType: dtype.Float32, Layout: tensor.LayoutContiguous}
	if err := Run(ctx, testBackend{}, spec, Invocation{Outputs: []Argument{arg}}); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(ctx)
	cancel()
	if err := Run(ctx, testBackend{}, spec, Invocation{Outputs: []Argument{arg}}); err == nil {
		t.Fatal("cancelled execution accepted")
	}
}

func TestRunRejectsUnsupportedBackend(t *testing.T) {
	s, _ := shape.New(1)
	m, _ := tensor.NewMetadata(dtype.Float32, s)
	arg := Argument{Metadata: m, Storage: &testStorage{size: 4}}
	backend := unsupportedBackend{}
	err := Run(context.Background(), backend, KernelSpec{Operation: "identity", DType: dtype.Float32, Layout: tensor.LayoutContiguous}, Invocation{Outputs: []Argument{arg}})
	if err == nil || !errors.Is(err, errs.ErrUnsupportedBackend) {
		t.Fatalf("got %v, want ErrUnsupportedBackend", err)
	}
}

type unsupportedBackend struct{ testBackend }

func (unsupportedBackend) Supports(KernelSpec, Invocation) bool { return false }

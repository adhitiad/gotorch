package compute

import (
	"context"
	"encoding/binary"
	"errors"
	"math"
	"testing"

	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/shape"
	"github.com/adhitiad/gotorch/src/storage"
	"github.com/adhitiad/gotorch/src/tensor"
)

func floatArg(t *testing.T, values []float64, dims ...int64) Argument {
	t.Helper()
	sh, err := shape.New(dims...)
	if err != nil {
		t.Fatal(err)
	}
	meta, err := tensor.NewMetadata(dtype.Float64, sh)
	if err != nil {
		t.Fatal(err)
	}
	size, err := meta.RequiredBytes()
	if err != nil {
		t.Fatal(err)
	}
	s, err := storage.NewHeapAllocator().Allocate(context.Background(), size)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = s.Release() })
	raw := make([]byte, len(values)*8)
	for i, v := range values {
		binary.LittleEndian.PutUint64(raw[i*8:], math.Float64bits(v))
	}
	if len(raw) > 0 {
		if n, e := s.WriteAt(raw, 0); e != nil || n != len(raw) {
			t.Fatalf("write fixture: %d %v", n, e)
		}
	}
	return Argument{Metadata: meta, Storage: s}
}

func outputArg(t *testing.T, dims ...int64) Argument {
	t.Helper()
	sh, err := shape.New(dims...)
	if err != nil {
		t.Fatal(err)
	}
	meta, err := tensor.NewMetadata(dtype.Float64, sh)
	if err != nil {
		t.Fatal(err)
	}
	size, err := meta.RequiredBytes()
	if err != nil {
		t.Fatal(err)
	}
	s, err := storage.NewHeapAllocator().Allocate(context.Background(), size)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = s.Release() })
	return Argument{Metadata: meta, Storage: s}
}

func readFixture(t *testing.T, arg Argument, n int) []float64 {
	t.Helper()
	raw := make([]byte, n*8)
	if n > 0 {
		got, err := arg.Storage.ReadAt(raw, 0)
		if err != nil || got != len(raw) {
			t.Fatalf("read fixture: %d %v", got, err)
		}
	}
	out := make([]float64, n)
	for i := range out {
		out[i] = math.Float64frombits(binary.LittleEndian.Uint64(raw[i*8:]))
	}
	return out
}

func TestReferenceElementwiseAndReduction(t *testing.T) {
	backend := NewReferenceCPUBackend()
	a := floatArg(t, []float64{1, 2, 3}, 3)
	b := floatArg(t, []float64{4, 5, 6}, 3)
	out := outputArg(t, 3)
	spec := KernelSpec{Operation: OpMul, DType: dtype.Float64, Layout: tensor.LayoutContiguous}
	if err := Run(context.Background(), backend, spec, Invocation{Inputs: []Argument{a, b}, Outputs: []Argument{out}}); err != nil {
		t.Fatal(err)
	}
	got := readFixture(t, out, 3)
	want := []float64{4, 10, 18}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("mul[%d]=%v want %v", i, got[i], want[i])
		}
	}

	scalar := outputArg(t)
	if err := Run(context.Background(), backend, KernelSpec{Operation: OpSum, DType: dtype.Float64, Layout: tensor.LayoutContiguous}, Invocation{Inputs: []Argument{a}, Outputs: []Argument{scalar}}); err != nil {
		t.Fatal(err)
	}
	if got := readFixture(t, scalar, 1)[0]; got != 6 {
		t.Fatalf("sum=%v want 6", got)
	}
}

func TestReferenceMatMul(t *testing.T) {
	backend := NewReferenceCPUBackend()
	a := floatArg(t, []float64{1, 2, 3, 4, 5, 6}, 2, 3)
	b := floatArg(t, []float64{7, 8, 9, 10, 11, 12}, 3, 2)
	out := outputArg(t, 2, 2)
	err := Run(context.Background(), backend, KernelSpec{Operation: OpMatMul, DType: dtype.Float64, Layout: tensor.LayoutContiguous}, Invocation{Inputs: []Argument{a, b}, Outputs: []Argument{out}})
	if err != nil {
		t.Fatal(err)
	}
	got := readFixture(t, out, 4)
	want := []float64{58, 64, 139, 154}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("matmul[%d]=%v want %v", i, got[i], want[i])
		}
	}
}

func TestReferenceRejectsInvalidShapesAndEmptyMean(t *testing.T) {
	backend := NewReferenceCPUBackend()
	a := floatArg(t, []float64{}, 0)
	out := outputArg(t)
	err := Run(context.Background(), backend, KernelSpec{Operation: OpMean, DType: dtype.Float64, Layout: tensor.LayoutContiguous}, Invocation{Inputs: []Argument{a}, Outputs: []Argument{out}})
	if !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("mean empty: %v", err)
	}
}

func TestRegistryDispatchAndDuplicateNames(t *testing.T) {
	registry := NewRegistry()
	if err := registry.Register(NewReferenceCPUBackend()); err != nil {
		t.Fatal(err)
	}
	if err := registry.Register(NewReferenceCPUBackend()); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("duplicate: %v", err)
	}
	a := floatArg(t, []float64{2}, 1)
	out := outputArg(t, 1)
	call := Invocation{Inputs: []Argument{a}, Outputs: []Argument{out}}
	spec := KernelSpec{Operation: OpNeg, DType: dtype.Float64, Layout: tensor.LayoutContiguous}
	selected, err := registry.Select(spec, call)
	if err != nil {
		t.Fatal(err)
	}
	if selected.Name() != "purego-reference" {
		t.Fatalf("selected %q", selected.Name())
	}
	if err := registry.Execute(context.Background(), spec, call); err != nil {
		t.Fatal(err)
	}
	if got := readFixture(t, out, 1)[0]; got != -2 {
		t.Fatalf("neg=%v", got)
	}
}

func TestReferenceRejectsUnsupportedDType(t *testing.T) {
	backend := NewReferenceCPUBackend()
	sh, _ := shape.New(1)
	meta, _ := tensor.NewMetadata(dtype.Int32, sh)
	arg := Argument{Metadata: meta, Storage: &testStorage{size: 4}}
	if backend.Supports(KernelSpec{Operation: OpNeg, DType: dtype.Int32, Layout: tensor.LayoutContiguous}, Invocation{Inputs: []Argument{arg}, Outputs: []Argument{arg}}) {
		t.Fatal("int32 unexpectedly supported")
	}
}

type closableTestBackend struct {
	testBackend
	name       string
	closeCalls int
	closeErr   error
}

func (b *closableTestBackend) Name() string {
	if b.name != "" {
		return b.name
	}
	return b.testBackend.Name()
}

func (b *closableTestBackend) Close() error {
	b.closeCalls++
	return b.closeErr
}

func TestRegistryCloseOwnsBackendLifecycle(t *testing.T) {
	registry := NewRegistry()
	backend := &closableTestBackend{}
	if err := registry.Register(backend); err != nil {
		t.Fatal(err)
	}
	if err := registry.Close(); err != nil {
		t.Fatal(err)
	}
	if backend.closeCalls != 1 {
		t.Fatalf("close calls=%d, want 1", backend.closeCalls)
	}
	if err := registry.Close(); err != nil {
		t.Fatalf("second close: %v", err)
	}
	if backend.closeCalls != 1 {
		t.Fatalf("close calls after repeated Close=%d, want 1", backend.closeCalls)
	}
	if err := registry.Register(NewReferenceCPUBackend()); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("register after close: %v", err)
	}
	input := floatArg(t, []float64{1}, 1)
	output := outputArg(t, 1)
	spec := KernelSpec{Operation: OpNeg, DType: dtype.Float64, Layout: tensor.LayoutContiguous}
	call := Invocation{Inputs: []Argument{input}, Outputs: []Argument{output}}
	if _, err := registry.Select(spec, call); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("select after close: %v", err)
	}
	if err := registry.Execute(context.Background(), spec, call); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("execute after close: %v", err)
	}
}

func TestRegistryCloseAggregatesBackendErrors(t *testing.T) {
	firstErr := errors.New("first close failure")
	secondErr := errors.New("second close failure")
	registry := NewRegistry()
	first := &closableTestBackend{name: "first", closeErr: firstErr}
	second := &closableTestBackend{name: "second", closeErr: secondErr}
	if err := registry.Register(first); err != nil {
		t.Fatal(err)
	}
	if err := registry.Register(second); err != nil {
		t.Fatal(err)
	}
	err := registry.Close()
	if !errors.Is(err, firstErr) || !errors.Is(err, secondErr) {
		t.Fatalf("close error=%v, want both backend failures", err)
	}
	if first.closeCalls != 1 || second.closeCalls != 1 {
		t.Fatalf("close calls first=%d second=%d, want 1 each", first.closeCalls, second.closeCalls)
	}
}

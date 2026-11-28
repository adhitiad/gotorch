package tensor

import (
	"context"
	"errors"
	"io"
	"reflect"
	"sync"
	"testing"

	"github.com/adhitiad/gotorch/src/storage"

	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/errs"
)

func TestTensorZeroValueAndClose(t *testing.T) {
	var zero Tensor
	if zero.Valid() {
		t.Fatal("zero Tensor is valid")
	}
	if _, err := zero.Metadata(); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("zero Metadata error = %v", err)
	}
	if err := zero.Close(); err != nil {
		t.Fatalf("zero Close: %v", err)
	}
	if err := zero.Close(); err != nil {
		t.Fatalf("repeated zero Close: %v", err)
	}
	if _, err := zero.Retain(); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("zero Retain error = %v", err)
	}
}

func TestTensorRetainHasIndependentCloseLifecycle(t *testing.T) {
	first, err := From([]float32{1, 2, 3}, 3)
	if err != nil {
		t.Fatal(err)
	}
	defer first.Close()
	second, err := first.Retain()
	if err != nil {
		t.Fatal(err)
	}
	if err := first.Close(); err != nil {
		t.Fatal(err)
	}
	if first.Valid() || !second.Valid() {
		t.Fatalf("validity after closing first: first=%v second=%v", first.Valid(), second.Valid())
	}
	got, err := Read[float32](second)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, []float32{1, 2, 3}) {
		t.Fatalf("retained data = %v", got)
	}
	if err := second.Close(); err != nil {
		t.Fatal(err)
	}
	if err := second.Close(); err != nil {
		t.Fatalf("repeated Tensor.Close: %v", err)
	}
}

func TestTensorFromCopiesInputAndReadCopiesOutput(t *testing.T) {
	input := []float32{1, 2, 3, 4}
	tensor, err := From(input, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer tensor.Close()

	input[0] = 99
	first, err := Read[float32](tensor)
	if err != nil {
		t.Fatal(err)
	}
	if first[0] != 1 {
		t.Fatalf("input mutation changed tensor value: %v", first)
	}
	first[1] = 88
	second, err := Read[float32](tensor)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(second, []float32{1, 2, 3, 4}) {
		t.Fatalf("read result aliases tensor storage: %v", second)
	}
}

func TestTensorConstructorsAndTypedAccess(t *testing.T) {
	tests := []struct {
		name string
		run  func(*testing.T)
	}{
		{"bool", func(t *testing.T) { roundTrip(t, []bool{true, false, true}, dtype.Bool) }},
		{"int8", func(t *testing.T) { roundTrip(t, []int8{-1, 2}, dtype.Int8) }},
		{"int16", func(t *testing.T) { roundTrip(t, []int16{-300, 400}, dtype.Int16) }},
		{"int32", func(t *testing.T) { roundTrip(t, []int32{-30000, 40000}, dtype.Int32) }},
		{"int64", func(t *testing.T) { roundTrip(t, []int64{-1 << 40, 1 << 40}, dtype.Int64) }},
		{"uint8", func(t *testing.T) { roundTrip(t, []uint8{0, 255}, dtype.Uint8) }},
		{"uint16", func(t *testing.T) { roundTrip(t, []uint16{0, 65535}, dtype.Uint16) }},
		{"uint32", func(t *testing.T) { roundTrip(t, []uint32{0, 1 << 30}, dtype.Uint32) }},
		{"uint64", func(t *testing.T) { roundTrip(t, []uint64{0, 1 << 60}, dtype.Uint64) }},
		{"float16-bits", func(t *testing.T) { roundTrip(t, []Float16Bits{0x3c00, 0xc000}, dtype.Float16) }},
		{"bfloat16-bits", func(t *testing.T) { roundTrip(t, []BFloat16Bits{0x3f80, 0xc000}, dtype.BFloat16) }},
		{"float32", func(t *testing.T) { roundTrip(t, []float32{1.25, -2.5}, dtype.Float32) }},
		{"float64", func(t *testing.T) { roundTrip(t, []float64{1.25, -2.5}, dtype.Float64) }},
		{"complex64", func(t *testing.T) { roundTrip(t, []complex64{1 + 2i, -3 - 4i}, dtype.Complex64) }},
		{"complex128", func(t *testing.T) { roundTrip(t, []complex128{1 + 2i, -3 - 4i}, dtype.Complex128) }},
	}
	for _, test := range tests {
		t.Run(test.name, test.run)
	}
}

func roundTrip[T Element](t *testing.T, values []T, want dtype.DType) {
	t.Helper()
	tensor, err := From(values, int64(len(values)))
	if err != nil {
		t.Fatal(err)
	}
	defer tensor.Close()
	metadata, err := tensor.Metadata()
	if err != nil {
		t.Fatal(err)
	}
	if metadata.DType() != want {
		t.Fatalf("dtype = %s, want %s", metadata.DType(), want)
	}
	got, err := Read[T](tensor)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, values) {
		t.Fatalf("round trip = %v, want %v", got, values)
	}
}

func TestTensorRejectsMismatchedTypeAndElementCount(t *testing.T) {
	tensor, err := NewTensor(dtype.Float32, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer tensor.Close()
	if err := Write(tensor, []float64{1, 2}); !errors.Is(err, errs.ErrUnsupportedDType) {
		t.Fatalf("mismatched dtype error = %v", err)
	}
	if err := Write(tensor, []float32{1}); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("mismatched element count error = %v", err)
	}
	if _, err := Read[int32](tensor); !errors.Is(err, errs.ErrUnsupportedDType) {
		t.Fatalf("mismatched read dtype error = %v", err)
	}
}

func TestTensorEmptyAndScalarShapes(t *testing.T) {
	empty, err := From([]float32{}, 0, 3)
	if err != nil {
		t.Fatal(err)
	}
	defer empty.Close()
	if got, err := Read[float32](empty); err != nil || len(got) != 0 {
		t.Fatalf("empty read = %v, %v", got, err)
	}

	scalar, err := From([]int32{42})
	if err != nil {
		t.Fatal(err)
	}
	defer scalar.Close()
	got, err := Read[int32](scalar)
	if err != nil || len(got) != 1 || got[0] != 42 {
		t.Fatalf("scalar read = %v, %v", got, err)
	}
}

func TestClosedTensorRejectsDataAccess(t *testing.T) {
	tensor, err := From([]float32{1}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := tensor.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := Read[float32](tensor); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("Read after Close error = %v", err)
	}
	if err := Write(tensor, []float32{2}); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("Write after Close error = %v", err)
	}
}


// unlockedTestAllocator intentionally supplies storage without internal locking.
// Tensor's own handle must serialize overlapping writes made through that handle.
type unlockedTestAllocator struct{}

type unlockedTestAllocation struct {
	data []byte
	refs int64
}

type unlockedTestStorage struct {
	allocation *unlockedTestAllocation
	released   bool
}

func (unlockedTestAllocator) Allocate(_ context.Context, size int64) (storage.Storage, error) {
	return &unlockedTestStorage{allocation: &unlockedTestAllocation{data: make([]byte, int(size)), refs: 1}}, nil
}

func (s *unlockedTestStorage) Size() int64 {
	if s == nil || s.released || s.allocation == nil {
		return 0
	}
	return int64(len(s.allocation.data))
}

func (s *unlockedTestStorage) Retain() (storage.Storage, error) {
	if s == nil || s.released || s.allocation == nil {
		return nil, errs.ErrStorageReleased
	}
	s.allocation.refs++
	return &unlockedTestStorage{allocation: s.allocation}, nil
}

func (s *unlockedTestStorage) Release() error {
	if s == nil || s.released || s.allocation == nil {
		return errs.ErrStorageReleased
	}
	s.released = true
	s.allocation.refs--
	if s.allocation.refs == 0 {
		clear(s.allocation.data)
		s.allocation.data = nil
	}
	return nil
}

func (s *unlockedTestStorage) ReadAt(p []byte, off int64) (int, error) {
	if s == nil || s.released || s.allocation == nil {
		return 0, errs.ErrStorageReleased
	}
	if off < 0 {
		return 0, errs.ErrInvalidArgument
	}
	if off >= int64(len(s.allocation.data)) && len(p) > 0 {
		return 0, io.EOF
	}
	n := copy(p, s.allocation.data[int(off):])
	if n != len(p) {
		return n, io.EOF
	}
	return n, nil
}

func (s *unlockedTestStorage) WriteAt(p []byte, off int64) (int, error) {
	if s == nil || s.released || s.allocation == nil {
		return 0, errs.ErrStorageReleased
	}
	if off < 0 {
		return 0, errs.ErrInvalidArgument
	}
	if off >= int64(len(s.allocation.data)) && len(p) > 0 {
		return 0, io.ErrShortWrite
	}
	n := copy(s.allocation.data[int(off):], p)
	if n != len(p) {
		return n, io.ErrShortWrite
	}
	return n, nil
}

func TestConcurrentWritesThroughOneTensorHandleAreSerialized(t *testing.T) {
	value, err := NewTensorWithAllocator(unlockedTestAllocator{}, dtype.Float64, 4096)
	if err != nil {
		t.Fatal(err)
	}
	defer value.Close()
	left := make([]float64, 4096)
	right := make([]float64, 4096)
	for i := range left {
		left[i], right[i] = 1, 2
	}
	var workers sync.WaitGroup
	for _, values := range [][]float64{left, right} {
		values := values
		workers.Add(1)
		go func() {
			defer workers.Done()
			for iteration := 0; iteration < 20; iteration++ {
				if err := Write(value, values); err != nil {
					t.Errorf("concurrent Write: %v", err)
					return
				}
			}
		}()
	}
	workers.Wait()
	got, err := Read[float64](value)
	if err != nil {
		t.Fatal(err)
	}
	first := got[0]
	if first != 1 && first != 2 {
		t.Fatalf("unexpected first value %v", first)
	}
	for index, item := range got {
		if item != first {
			t.Fatalf("concurrent writes interleaved at index %d: got %v, first %v", index, item, first)
		}
	}
}

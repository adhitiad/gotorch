package storage

import (
	"context"
	"io"
	"reflect"
	"sync"

	"github.com/adhitiad/gotorch/src/errs"
)

// HeapAllocator allocates zero-initialized byte storage on the Go heap.
// Its zero value is ready to use.
type HeapAllocator struct{}

// NewHeapAllocator returns a Pure Go heap-backed allocator.
func NewHeapAllocator() *HeapAllocator { return &HeapAllocator{} }

// Allocate creates a new allocation with one independently releasable handle.
// A zero-byte allocation is valid. The context is checked before allocation.
func (a *HeapAllocator) Allocate(ctx context.Context, size int64) (Storage, error) {
	if a == nil || nilLike(ctx) {
		return nil, errs.ErrInvalidArgument
	}
	if size < 0 {
		return nil, errs.ErrInvalidArgument
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	maxInt := uint64(^uint(0) >> 1)
	if uint64(size) > maxInt {
		return nil, errs.ErrShapeOverflow
	}

	allocation := &heapAllocation{
		data:       make([]byte, int(size)),
		references: 1,
	}
	return &heapStorage{allocation: allocation}, nil
}

type heapAllocation struct {
	mu         sync.RWMutex
	data       []byte
	references int64
	released   bool
}

type heapStorage struct {
	mu         sync.RWMutex
	allocation *heapAllocation
	released   bool
}

// Size returns the allocation size, or zero after this handle is released.
func (s *heapStorage) Size() int64 {
	if s == nil {
		return 0
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.released || s.allocation == nil {
		return 0
	}
	return int64(len(s.allocation.data))
}

// Retain creates a distinct handle and increments the shared allocation count.
func (s *heapStorage) Retain() (Storage, error) {
	if s == nil {
		return nil, errs.ErrStorageReleased
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.released || s.allocation == nil {
		return nil, errs.ErrStorageReleased
	}

	a := s.allocation
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.released || a.references <= 0 {
		return nil, errs.ErrStorageReleased
	}
	if a.references == int64(^uint64(0)>>1) {
		return nil, errs.ErrShapeOverflow
	}
	a.references++
	return &heapStorage{allocation: a}, nil
}

// Release relinquishes only this handle. Other retained handles stay valid.
func (s *heapStorage) Release() error {
	if s == nil {
		return errs.ErrStorageReleased
	}
	s.mu.Lock()
	if s.released || s.allocation == nil {
		s.mu.Unlock()
		return errs.ErrStorageReleased
	}
	a := s.allocation
	s.released = true
	s.allocation = nil
	s.mu.Unlock()

	a.mu.Lock()
	defer a.mu.Unlock()
	if a.references <= 0 {
		return errs.ErrInvalidState
	}
	a.references--
	if a.references == 0 {
		clear(a.data)
		a.data = nil
		a.released = true
	}
	return nil
}

// ReadAt follows io.ReaderAt semantics and never exposes the backing slice.
func (s *heapStorage) ReadAt(p []byte, off int64) (int, error) {
	if s == nil {
		return 0, errs.ErrStorageReleased
	}
	if off < 0 {
		return 0, errs.ErrInvalidArgument
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.released || s.allocation == nil {
		return 0, errs.ErrStorageReleased
	}
	a := s.allocation
	a.mu.RLock()
	defer a.mu.RUnlock()
	if a.released {
		return 0, errs.ErrStorageReleased
	}
	size := int64(len(a.data))
	if off > size {
		return 0, io.EOF
	}
	if len(p) == 0 {
		return 0, nil
	}
	if off == size {
		return 0, io.EOF
	}
	n := copy(p, a.data[int(off):])
	if n != len(p) {
		return n, io.EOF
	}
	return n, nil
}

// WriteAt follows io.WriterAt semantics and copies bytes into the allocation.
func (s *heapStorage) WriteAt(p []byte, off int64) (int, error) {
	if s == nil {
		return 0, errs.ErrStorageReleased
	}
	if off < 0 {
		return 0, errs.ErrInvalidArgument
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.released || s.allocation == nil {
		return 0, errs.ErrStorageReleased
	}
	a := s.allocation
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.released {
		return 0, errs.ErrStorageReleased
	}
	size := int64(len(a.data))
	if off > size {
		return 0, io.ErrShortWrite
	}
	if len(p) == 0 {
		return 0, nil
	}
	if off == size {
		return 0, io.ErrShortWrite
	}
	n := copy(a.data[int(off):], p)
	if n != len(p) {
		return n, io.ErrShortWrite
	}
	return n, nil
}

func nilLike(value any) bool {
	if value == nil {
		return true
	}
	v := reflect.ValueOf(value)
	switch v.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return v.IsNil()
	default:
		return false
	}
}

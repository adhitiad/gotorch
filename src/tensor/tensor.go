package tensor

import (
	"context"
	"fmt"
	"reflect"
	"sync"

	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/shape"
	"github.com/adhitiad/gotorch/src/storage"
)

// noCopy is recognized by go vet's copylocks analyzer.
type noCopy struct{}

func (*noCopy) Lock()   {}
func (*noCopy) Unlock() {}

// Tensor is an owning handle to tensor metadata and storage.
//
// A Tensor must not be copied after first use. Use Retain to obtain another
// independently closable handle. Close is safe to call more than once.
// The zero value owns no memory: methods other than Close return ErrInvalidState.
type Tensor struct {
	noCopy   noCopy
	mu       sync.RWMutex
	metadata Metadata
	storage  storage.Storage
	offset   int64
	closed   bool
}

// NewTensor allocates a zero-initialized contiguous tensor using the Pure Go
// heap allocator. The rank-zero shape (no dimensions) is a one-element scalar.
func NewTensor(elementType dtype.DType, dimensions ...int64) (*Tensor, error) {
	return NewTensorWithAllocator(storage.NewHeapAllocator(), elementType, dimensions...)
}

// NewTensorWithAllocator allocates a zero-initialized contiguous tensor using
// allocator. The returned Tensor owns the allocator's initial storage handle.
func NewTensorWithAllocator(allocator storage.Allocator, elementType dtype.DType, dimensions ...int64) (*Tensor, error) {
	if nilDependency(allocator) {
		return nil, errs.ErrInvalidArgument
	}
	dimensionsShape, err := shape.New(dimensions...)
	if err != nil {
		return nil, err
	}
	metadata, err := NewMetadata(elementType, dimensionsShape)
	if err != nil {
		return nil, err
	}
	size, err := metadata.RequiredBytes()
	if err != nil {
		return nil, err
	}
	handle, err := allocator.Allocate(context.Background(), size)
	if err != nil {
		return nil, fmt.Errorf("tensor: allocate %d bytes: %w", size, err)
	}
	if nilDependency(handle) {
		return nil, fmt.Errorf("%w: allocator returned a nil storage handle", errs.ErrInvalidState)
	}
	actualSize := handle.Size()
	if actualSize != size {
		_ = handle.Release()
		return nil, fmt.Errorf("%w: allocator returned %d bytes, requested %d", errs.ErrInvalidState, actualSize, size)
	}
	return &Tensor{metadata: metadata, storage: handle}, nil
}

// Metadata returns the immutable descriptor for this tensor.
func (t *Tensor) Metadata() (Metadata, error) {
	if t == nil {
		return Metadata{}, errs.ErrInvalidArgument
	}
	t.mu.RLock()
	defer t.mu.RUnlock()
	if t.closed || nilDependency(t.storage) {
		return Metadata{}, errs.ErrInvalidState
	}
	return t.metadata, nil
}

// Retain returns a new Tensor handle that shares the same allocation but has
// an independent Close lifecycle. Neither handle owns the other's lifetime.
func (t *Tensor) Retain() (*Tensor, error) {
	if t == nil {
		return nil, errs.ErrInvalidArgument
	}
	t.mu.RLock()
	defer t.mu.RUnlock()
	if t.closed || nilDependency(t.storage) {
		return nil, errs.ErrInvalidState
	}
	handle, err := t.storage.Retain()
	if err != nil {
		return nil, fmt.Errorf("tensor: retain storage: %w", err)
	}
	if nilDependency(handle) {
		return nil, fmt.Errorf("%w: storage returned a nil retained handle", errs.ErrInvalidState)
	}
	return &Tensor{
		metadata: t.metadata,
		storage:  handle,
		offset:   t.offset,
	}, nil
}

// Close releases this Tensor's storage handle. It is idempotent. Closing the
// zero value is also safe and makes it permanently closed.
func (t *Tensor) Close() error {
	if t == nil {
		return errs.ErrInvalidArgument
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.closed {
		return nil
	}
	if nilDependency(t.storage) {
		t.closed = true
		return nil
	}
	if err := t.storage.Release(); err != nil {
		return fmt.Errorf("tensor: release storage: %w", err)
	}
	t.storage = nil
	t.closed = true
	return nil
}

// Valid reports whether this Tensor currently owns an active storage handle.
func (t *Tensor) Valid() bool {
	if t == nil {
		return false
	}
	t.mu.RLock()
	defer t.mu.RUnlock()
	return !t.closed && !nilDependency(t.storage)
}

func (t *Tensor) readAll() ([]byte, error) {
	if t == nil {
		return nil, errs.ErrInvalidArgument
	}
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.readLogicalBytesLocked()
}

func (t *Tensor) writeAll(data []byte) error {
	if t == nil {
		return errs.ErrInvalidArgument
	}
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.writeLogicalBytesLocked(data)
}

func nilDependency(value any) bool {
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

// Package storage defines ownership-aware byte storage contracts. Concrete allocators
// and memory implementations are introduced with the Tensor Engine in P1.
package storage

import (
	"context"
	"fmt"
	"io"

	"github.com/adhitiad/gotorch/src/errs"
)

// Storage is one independently releasable handle to a shared allocation.
// Retain must return a new independently releasable handle; Release must release
// only the caller's handle and reject a repeated release with ErrStorageReleased.
// ReadAt and WriteAt must reject use after Release and follow io.ReaderAt/io.WriterAt
// range semantics. Implementations must be safe for concurrent reads and for
// caller-coordinated writes to disjoint byte ranges.
type Storage interface {
	io.ReaderAt
	io.WriterAt
	Size() int64
	Retain() (Storage, error)
	Release() error
}

// Allocator creates storage whose initial handle belongs to the caller.
type Allocator interface {
	Allocate(context.Context, int64) (Storage, error)
}

// ValidateRange checks an offset/length pair without overflowing offset+length.
func ValidateRange(size, offset, length int64) error {
	if size < 0 || offset < 0 || length < 0 {
		return fmt.Errorf("%w: size, offset, and length must be non-negative", errs.ErrInvalidArgument)
	}
	if offset > size || length > size-offset {
		return errs.ErrOutOfBounds
	}
	return nil
}

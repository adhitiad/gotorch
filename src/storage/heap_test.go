package storage

import (
	"bytes"
	"context"
	"errors"
	"io"
	"testing"

	"github.com/adhitiad/gotorch/src/errs"
)

func TestHeapAllocatorRetainedHandlesHaveIndependentLifetime(t *testing.T) {
	allocator := NewHeapAllocator()
	first, err := allocator.Allocate(context.Background(), 5)
	if err != nil {
		t.Fatal(err)
	}
	if n, err := first.WriteAt([]byte("hello"), 0); err != nil || n != 5 {
		t.Fatalf("WriteAt = %d, %v; want 5, nil", n, err)
	}

	second, err := first.Retain()
	if err != nil {
		t.Fatal(err)
	}
	if err := first.Release(); err != nil {
		t.Fatal(err)
	}
	if _, err := first.ReadAt(make([]byte, 1), 0); !errors.Is(err, errs.ErrStorageReleased) {
		t.Fatalf("released first handle ReadAt error = %v", err)
	}

	got := make([]byte, 5)
	if n, err := second.ReadAt(got, 0); err != nil || n != 5 {
		t.Fatalf("second ReadAt = %d, %v; want 5, nil", n, err)
	}
	if !bytes.Equal(got, []byte("hello")) {
		t.Fatalf("second handle read %q, want hello", got)
	}
	if err := second.Release(); err != nil {
		t.Fatal(err)
	}
	if err := second.Release(); !errors.Is(err, errs.ErrStorageReleased) {
		t.Fatalf("second Release error = %v, want ErrStorageReleased", err)
	}
}

func TestHeapStorageRangeAndIOBoundaries(t *testing.T) {
	s, err := NewHeapAllocator().Allocate(context.Background(), 3)
	if err != nil {
		t.Fatal(err)
	}
	if n, err := s.WriteAt([]byte("abcd"), 0); n != 3 || !errors.Is(err, io.ErrShortWrite) {
		t.Fatalf("partial WriteAt = %d, %v; want 3, ErrShortWrite", n, err)
	}
	buf := make([]byte, 4)
	if n, err := s.ReadAt(buf, 0); n != 3 || !errors.Is(err, io.EOF) {
		t.Fatalf("partial ReadAt = %d, %v; want 3, EOF", n, err)
	}
	if n, err := s.ReadAt(nil, 3); n != 0 || err != nil {
		t.Fatalf("zero-length ReadAt at end = %d, %v; want 0, nil", n, err)
	}
	if _, err := s.ReadAt(buf, -1); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("negative ReadAt offset error = %v", err)
	}
	if err := s.Release(); err != nil {
		t.Fatal(err)
	}
}

func TestHeapAllocatorValidatesArgumentsAndContext(t *testing.T) {
	allocator := NewHeapAllocator()
	if _, err := allocator.Allocate(context.Background(), -1); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("negative allocation error = %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := allocator.Allocate(ctx, 1); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled allocation error = %v", err)
	}
	if _, err := allocator.Allocate(context.Background(), 0); err != nil {
		t.Fatalf("zero-byte allocation: %v", err)
	}
}

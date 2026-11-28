package memory

import (
	"context"
	"errors"
	"io"
	"sync"
	"testing"

	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/storage"
)

func mustArena(t *testing.T, capacity int64) *Arena {
	t.Helper()
	arena, err := NewArena(capacity)
	if err != nil {
		t.Fatalf("NewArena: %v", err)
	}
	return arena
}

func TestArenaAlignmentAndCapacity(t *testing.T) {
	arena := mustArena(t, 32)
	first, err := arena.Allocate(context.Background(), 3)
	if err != nil {
		t.Fatalf("first Allocate: %v", err)
	}
	second, err := arena.Allocate(context.Background(), 1)
	if err != nil {
		t.Fatalf("second Allocate: %v", err)
	}
	if stats := arena.Stats(); stats.Used != 9 || stats.Active != 2 || stats.ActiveBytes != 4 {
		t.Fatalf("stats = %+v", stats)
	}
	if _, err := arena.Allocate(context.Background(), 23); !errors.Is(err, ErrArenaFull) {
		t.Fatalf("over-capacity Allocate: got %v", err)
	}
	if err := first.Release(); err != nil {
		t.Fatalf("first Release: %v", err)
	}
	if err := second.Release(); err != nil {
		t.Fatalf("second Release: %v", err)
	}
	if stats := arena.Stats(); stats.Active != 0 || stats.ActiveBytes != 0 {
		t.Fatalf("stats after release = %+v", stats)
	}
	if err := arena.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
}

func TestArenaResetValidation(t *testing.T) {
	arena := mustArena(t, 16)
	handle, err := arena.Allocate(context.Background(), 8)
	if err != nil {
		t.Fatalf("Allocate: %v", err)
	}
	if err := arena.Reset(); !errors.Is(err, ErrArenaLive) {
		t.Fatalf("Reset with live handle: got %v", err)
	}
	if err := arena.Close(); !errors.Is(err, ErrArenaLive) {
		t.Fatalf("Close with live handle: got %v", err)
	}
	if err := handle.Release(); err != nil {
		t.Fatalf("Release: %v", err)
	}
	if err := arena.Reset(); err != nil {
		t.Fatalf("Reset after release: %v", err)
	}
	stats := arena.Stats()
	if stats.Used != 0 || stats.Resets != 1 {
		t.Fatalf("stats after reset = %+v", stats)
	}
	next, err := arena.Allocate(context.Background(), 8)
	if err != nil {
		t.Fatalf("Allocate after reset: %v", err)
	}
	if err := next.Release(); err != nil {
		t.Fatalf("next Release: %v", err)
	}
	if err := arena.Close(); err != nil {
		t.Fatalf("final Close: %v", err)
	}
	if _, err := arena.Allocate(context.Background(), 1); !errors.Is(err, ErrArenaClosed) {
		t.Fatalf("Allocate after close: got %v", err)
	}
	if err := arena.Reset(); !errors.Is(err, ErrArenaClosed) {
		t.Fatalf("Reset after close: got %v", err)
	}
}

func TestArenaRetainAndRangeSemantics(t *testing.T) {
	arena := mustArena(t, 64)
	handle, err := arena.Allocate(context.Background(), 16)
	if err != nil {
		t.Fatalf("Allocate: %v", err)
	}
	if _, err := handle.WriteAt([]byte("hello world"), 0); err != nil {
		t.Fatalf("WriteAt: %v", err)
	}
	buffer := make([]byte, 11)
	if n, err := handle.ReadAt(buffer, 0); err != nil || n != 11 || string(buffer) != "hello world" {
		t.Fatalf("ReadAt = %d, %q, %v", n, buffer, err)
	}
	retained, err := handle.Retain()
	if err != nil {
		t.Fatalf("Retain: %v", err)
	}
	if err := handle.Release(); err != nil {
		t.Fatalf("Release: %v", err)
	}
	clear(buffer)
	if n, err := retained.ReadAt(buffer, 0); err != nil || n != 11 || string(buffer) != "hello world" {
		t.Fatalf("retained ReadAt = %d, %q, %v", n, buffer, err)
	}
	if err := retained.Release(); err != nil {
		t.Fatalf("retained Release: %v", err)
	}
	if _, err := retained.ReadAt(buffer, 0); !errors.Is(err, errs.ErrStorageReleased) {
		t.Fatalf("ReadAt after release: got %v", err)
	}

	small, err := arena.Allocate(context.Background(), 4)
	if err != nil {
		t.Fatalf("small Allocate: %v", err)
	}
	wide := make([]byte, 8)
	if n, err := small.ReadAt(wide, 0); n != 4 || !errors.Is(err, io.EOF) {
		t.Fatalf("partial ReadAt = %d, %v", n, err)
	}
	if n, err := small.WriteAt(wide, 0); n != 4 || !errors.Is(err, io.ErrShortWrite) {
		t.Fatalf("partial WriteAt = %d, %v", n, err)
	}
	if _, err := small.ReadAt(wide, 5); !errors.Is(err, io.EOF) {
		t.Fatalf("out-of-range ReadAt: got %v", err)
	}
	if err := small.Release(); err != nil {
		t.Fatalf("small Release: %v", err)
	}
}

func TestArenaInvalidInputs(t *testing.T) {
	if _, err := NewArena(-1); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("negative capacity: got %v", err)
	}
	arena := mustArena(t, 8)
	var ctx context.Context
	if _, err := arena.Allocate(ctx, 1); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("nil context: got %v", err)
	}
	if _, err := arena.Allocate(context.Background(), -1); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("negative size: got %v", err)
	}
}

func TestArenaConcurrentStress(t *testing.T) {
	arena := mustArena(t, 8192)
	ctx := context.Background()
	var wg sync.WaitGroup
	for worker := 0; worker < 8; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for index := 0; index < 8; index++ {
				handle, err := arena.Allocate(ctx, 128)
				if err != nil {
					t.Errorf("Allocate: %v", err)
					return
				}
				if _, err := handle.WriteAt([]byte{7}, 0); err != nil {
					t.Errorf("WriteAt: %v", err)
					return
				}
				buffer := make([]byte, 1)
				if _, err := handle.ReadAt(buffer, 0); err != nil || buffer[0] != 7 {
					t.Errorf("ReadAt: %v, %v", buffer, err)
					return
				}
				if err := handle.Release(); err != nil {
					t.Errorf("Release: %v", err)
					return
				}
			}
		}()
	}
	wg.Wait()
	if stats := arena.Stats(); stats.Active != 0 || stats.ActiveBytes != 0 || stats.Allocations != 64 {
		t.Fatalf("stats = %+v", stats)
	}
	if err := arena.Reset(); err != nil {
		t.Fatalf("Reset: %v", err)
	}
	if err := arena.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
}

var _ storage.Allocator = (*Arena)(nil)
var _ storage.Storage = (*arenaStorage)(nil)

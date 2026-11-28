package memory

import (
	"bytes"
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/storage"
)

func mustPool(t *testing.T, config PoolConfig) *Pool {
	t.Helper()
	pool, err := NewPool(config)
	if err != nil {
		t.Fatalf("NewPool: %v", err)
	}
	return pool
}

func TestPoolReuseAndStats(t *testing.T) {
	pool := mustPool(t, PoolConfig{MaxCachedBytes: 1024, MaxTotalBytes: 4096})
	ctx := context.Background()

	handle, err := pool.Allocate(ctx, 128)
	if err != nil {
		t.Fatalf("Allocate: %v", err)
	}
	if got := handle.Size(); got != 128 {
		t.Fatalf("Size = %d", got)
	}
	if _, err := handle.WriteAt([]byte("data"), 0); err != nil {
		t.Fatalf("WriteAt: %v", err)
	}
	buffer := make([]byte, 4)
	if _, err := handle.ReadAt(buffer, 0); err != nil {
		t.Fatalf("ReadAt: %v", err)
	}
	if string(buffer) != "data" {
		t.Fatalf("read = %q", buffer)
	}
	if err := handle.Release(); err != nil {
		t.Fatalf("Release: %v", err)
	}

	stats := pool.Stats()
	if stats.ActiveBytes != 0 || stats.CachedBytes != 128 || stats.AllocatedBytes != 128 || stats.Releases != 1 {
		t.Fatalf("stats after release = %+v", stats)
	}

	reused, err := pool.Allocate(ctx, 128)
	if err != nil {
		t.Fatalf("reuse Allocate: %v", err)
	}
	stats = pool.Stats()
	if stats.Allocations != 1 || stats.Reuses != 1 || stats.ActiveBytes != 128 || stats.CachedBytes != 0 {
		t.Fatalf("stats after reuse = %+v", stats)
	}
	clear(buffer)
	if _, err := reused.ReadAt(buffer, 0); err != nil {
		t.Fatalf("reuse ReadAt: %v", err)
	}
	if !bytes.Equal(buffer, make([]byte, 4)) {
		t.Fatalf("reused buffer was not zeroed: %v", buffer)
	}
	if err := reused.Release(); err != nil {
		t.Fatalf("reuse Release: %v", err)
	}
	if err := pool.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	stats = pool.Stats()
	if !stats.Closed || stats.CachedBytes != 0 || stats.AllocatedBytes != 0 {
		t.Fatalf("stats after close = %+v", stats)
	}
	if _, err := pool.Allocate(ctx, 8); !errors.Is(err, ErrPoolClosed) {
		t.Fatalf("Allocate after close: got %v", err)
	}
}

func TestPoolRetainKeepsBufferCachedUntilFinalRelease(t *testing.T) {
	pool := mustPool(t, PoolConfig{MaxCachedBytes: 1024})
	ctx := context.Background()
	first, err := pool.Allocate(ctx, 64)
	if err != nil {
		t.Fatalf("Allocate: %v", err)
	}
	retained, err := first.Retain()
	if err != nil {
		t.Fatalf("Retain: %v", err)
	}
	if _, err := first.WriteAt([]byte("abc"), 0); err != nil {
		t.Fatalf("WriteAt: %v", err)
	}
	if err := first.Release(); err != nil {
		t.Fatalf("first Release: %v", err)
	}
	if stats := pool.Stats(); stats.ActiveBytes != 64 || stats.CachedBytes != 0 {
		t.Fatalf("stats after first release = %+v", stats)
	}

	other, err := pool.Allocate(ctx, 64)
	if err != nil {
		t.Fatalf("second Allocate: %v", err)
	}
	if stats := pool.Stats(); stats.Allocations != 2 {
		t.Fatalf("allocated a second buffer too early: %+v", stats)
	}
	buffer := make([]byte, 3)
	if _, err := retained.ReadAt(buffer, 0); err != nil {
		t.Fatalf("retained ReadAt: %v", err)
	}
	if string(buffer) != "abc" {
		t.Fatalf("retained read = %q", buffer)
	}
	if err := retained.Release(); err != nil {
		t.Fatalf("retained Release: %v", err)
	}
	if err := other.Release(); err != nil {
		t.Fatalf("other Release: %v", err)
	}
	if stats := pool.Stats(); stats.ActiveBytes != 0 || stats.CachedBytes != 128 || stats.AllocatedBytes != 128 {
		t.Fatalf("final stats = %+v", stats)
	}
}

func TestPoolLimits(t *testing.T) {
	pool := mustPool(t, PoolConfig{MaxCachedBytes: 256, MaxTotalBytes: 256})
	ctx := context.Background()
	first, err := pool.Allocate(ctx, 128)
	if err != nil {
		t.Fatalf("first Allocate: %v", err)
	}
	second, err := pool.Allocate(ctx, 128)
	if err != nil {
		t.Fatalf("second Allocate: %v", err)
	}
	if _, err := pool.Allocate(ctx, 1); !errors.Is(err, ErrMemoryLimit) {
		t.Fatalf("over-limit Allocate: got %v", err)
	}
	if stats := pool.Stats(); stats.Rejected != 1 {
		t.Fatalf("rejected = %d", stats.Rejected)
	}
	if err := first.Release(); err != nil {
		t.Fatalf("first Release: %v", err)
	}
	if err := second.Release(); err != nil {
		t.Fatalf("second Release: %v", err)
	}

	unlimited := mustPool(t, PoolConfig{MaxTotalBytes: 1024})
	handle, err := unlimited.Allocate(ctx, 64)
	if err != nil {
		t.Fatalf("unlimited cache Allocate: %v", err)
	}
	if err := handle.Release(); err != nil {
		t.Fatalf("unlimited cache Release: %v", err)
	}
	if stats := unlimited.Stats(); stats.CachedBytes != 64 {
		t.Fatalf("zero MaxCachedBytes should allow caching: %+v", stats)
	}
}

func TestPoolCloseWithActiveHandle(t *testing.T) {
	pool := mustPool(t, DefaultPoolConfig())
	handle, err := pool.Allocate(context.Background(), 16)
	if err != nil {
		t.Fatalf("Allocate: %v", err)
	}
	if err := pool.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if _, err := handle.WriteAt([]byte("still valid"), 0); err != nil {
		t.Fatalf("WriteAt after pool close: %v", err)
	}
	buffer := make([]byte, 11)
	if _, err := handle.ReadAt(buffer, 0); err != nil {
		t.Fatalf("ReadAt after pool close: %v", err)
	}
	if string(buffer) != "still valid" {
		t.Fatalf("read = %q", buffer)
	}
	if err := handle.Release(); err != nil {
		t.Fatalf("Release: %v", err)
	}
	if stats := pool.Stats(); stats.AllocatedBytes != 0 || stats.ActiveBytes != 0 {
		t.Fatalf("stats after late release = %+v", stats)
	}
}

func TestPoolInvalidInputs(t *testing.T) {
	if _, err := NewPool(PoolConfig{MaxCachedBytes: -1}); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("negative cache limit: got %v", err)
	}
	pool := mustPool(t, PoolConfig{})
	var ctx context.Context
	if _, err := pool.Allocate(ctx, 1); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("nil context: got %v", err)
	}
	if _, err := pool.Allocate(context.Background(), -1); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("negative size: got %v", err)
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := pool.Allocate(canceled, 1); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled context: got %v", err)
	}
}

func TestPoolConcurrentStress(t *testing.T) {
	pool := mustPool(t, PoolConfig{MaxCachedBytes: 8192, MaxTotalBytes: 8192})
	ctx := context.Background()
	var wg sync.WaitGroup
	for worker := 0; worker < 8; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for index := 0; index < 128; index++ {
				handle, err := pool.Allocate(ctx, 32)
				if err != nil {
					t.Errorf("Allocate: %v", err)
					return
				}
				retained, err := handle.Retain()
				if err != nil {
					t.Errorf("Retain: %v", err)
					return
				}
				payload := []byte{1, 2, 3, 4}
				if _, err := handle.WriteAt(payload, 0); err != nil {
					t.Errorf("WriteAt: %v", err)
					return
				}
				buffer := make([]byte, 4)
				if _, err := retained.ReadAt(buffer, 0); err != nil {
					t.Errorf("ReadAt: %v", err)
					return
				}
				if !bytes.Equal(buffer, payload) {
					t.Errorf("payload = %v", buffer)
					return
				}
				if err := handle.Release(); err != nil {
					t.Errorf("Release: %v", err)
					return
				}
				if err := retained.Release(); err != nil {
					t.Errorf("retained Release: %v", err)
					return
				}
			}
		}()
	}
	wg.Wait()
	stats := pool.Stats()
	if stats.ActiveBytes != 0 || stats.ActiveBuffers != 0 {
		t.Fatalf("active state leaked: %+v", stats)
	}
	if stats.AllocatedBytes != stats.CachedBytes {
		t.Fatalf("allocated/cached mismatch: %+v", stats)
	}
	if stats.Releases != 8*128 {
		t.Fatalf("final releases = %d", stats.Releases)
	}
}

var _ storage.Allocator = (*Pool)(nil)
var _ storage.Storage = (*poolStorage)(nil)

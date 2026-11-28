package memory

import (
	"context"
	"errors"
	"fmt"
	"testing"
)

func BenchmarkPoolAllocateRelease(b *testing.B) {
	for _, size := range []int64{64, 4096} {
		b.Run(fmt.Sprintf("size=%d", size), func(b *testing.B) {
			pool, err := NewPool(PoolConfig{MaxCachedBytes: 1 << 20})
			if err != nil {
				b.Fatalf("NewPool: %v", err)
			}
			defer func() {
				if err := pool.Close(); err != nil {
					b.Fatalf("Close: %v", err)
				}
			}()
			ctx := context.Background()
			b.ResetTimer()
			for index := 0; index < b.N; index++ {
				handle, err := pool.Allocate(ctx, size)
				if err != nil {
					b.Fatalf("Allocate: %v", err)
				}
				if err := handle.Release(); err != nil {
					b.Fatalf("Release: %v", err)
				}
			}
		})
	}
}

func BenchmarkArenaAllocateRelease(b *testing.B) {
	arena, err := NewArena(1 << 20)
	if err != nil {
		b.Fatalf("NewArena: %v", err)
	}
	defer func() {
		if err := arena.Close(); err != nil {
			b.Fatalf("Close: %v", err)
		}
	}()
	ctx := context.Background()
	b.ResetTimer()
	for index := 0; index < b.N; index++ {
		handle, err := arena.Allocate(ctx, 64)
		if errors.Is(err, ErrArenaFull) {
			if err := arena.Reset(); err != nil {
				b.Fatalf("Reset: %v", err)
			}
			handle, err = arena.Allocate(ctx, 64)
		}
		if err != nil {
			b.Fatalf("Allocate: %v", err)
		}
		if err := handle.Release(); err != nil {
			b.Fatalf("Release: %v", err)
		}
	}
}

func BenchmarkWorkspaceReset(b *testing.B) {
	workspace := NewWorkspace()
	for _, name := range []string{"activations", "gradients", "scratch"} {
		if _, err := workspace.Add(name, 1<<16); err != nil {
			b.Fatalf("Add: %v", err)
		}
	}
	defer func() {
		if err := workspace.Close(); err != nil {
			b.Fatalf("Close: %v", err)
		}
	}()
	b.ResetTimer()
	for index := 0; index < b.N; index++ {
		if err := workspace.Reset(); err != nil {
			b.Fatalf("Reset: %v", err)
		}
	}
}

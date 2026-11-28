package scheduler

import (
	"context"
	"fmt"
	"testing"
)

func BenchmarkSubmitWait(b *testing.B) {
	for _, workers := range []int{1, 4} {
		b.Run(fmt.Sprintf("workers=%d", workers), func(b *testing.B) {
			s, err := New(testConfig(workers, workers*8))
			if err != nil {
				b.Fatalf("New: %v", err)
			}
			defer func() {
				if err := s.Close(context.Background()); err != nil {
					b.Fatalf("Close: %v", err)
				}
			}()
			ctx := context.Background()
			b.ResetTimer()
			for index := 0; index < b.N; index++ {
				if err := s.SubmitWait(ctx, func(context.Context) error { return nil }); err != nil {
					b.Fatalf("SubmitWait: %v", err)
				}
			}
		})
	}
}

func BenchmarkParallelFor(b *testing.B) {
	for _, chunk := range []int{64, 256} {
		b.Run(fmt.Sprintf("chunk=%d", chunk), func(b *testing.B) {
			s, err := New(testConfig(4, 64))
			if err != nil {
				b.Fatalf("New: %v", err)
			}
			defer func() {
				if err := s.Close(context.Background()); err != nil {
					b.Fatalf("Close: %v", err)
				}
			}()
			ctx := context.Background()
			b.ResetTimer()
			for index := 0; index < b.N; index++ {
				if err := s.ParallelFor(ctx, 4096, chunk, func(context.Context, int, int) error {
					return nil
				}); err != nil {
					b.Fatalf("ParallelFor: %v", err)
				}
			}
		})
	}
}

func BenchmarkStats(b *testing.B) {
	s, err := New(testConfig(4, 16))
	if err != nil {
		b.Fatalf("New: %v", err)
	}
	defer func() {
		if err := s.Close(context.Background()); err != nil {
			b.Fatalf("Close: %v", err)
		}
	}()
	b.ResetTimer()
	for index := 0; index < b.N; index++ {
		_ = s.Stats()
	}
}

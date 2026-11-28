package scheduler

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/adhitiad/gotorch/src/compute"
	"github.com/adhitiad/gotorch/src/errs"
)

func TestParallelFor(t *testing.T) {
	s := mustNew(t, testConfig(2, 8))
	defer func() {
		if err := s.Close(context.Background()); err != nil {
			t.Fatalf("Close: %v", err)
		}
	}()

	var mu sync.Mutex
	var ranges []string
	var total atomic.Int64
	err := s.ParallelFor(context.Background(), 10, 3, func(ctx context.Context, start, end int) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		mu.Lock()
		ranges = append(ranges, fmt.Sprintf("%d:%d", start, end))
		mu.Unlock()
		for index := start; index < end; index++ {
			total.Add(1)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("ParallelFor: %v", err)
	}
	if got := total.Load(); got != 10 {
		t.Fatalf("visited = %d, want 10", got)
	}
	if len(ranges) != 4 {
		t.Fatalf("chunks = %d, want 4", len(ranges))
	}
}

func TestParallelForNestedAndErrors(t *testing.T) {
	s := mustNew(t, testConfig(1, 2))
	defer func() {
		if err := s.Close(context.Background()); err != nil {
			t.Fatalf("Close: %v", err)
		}
	}()

	sentinel := errors.New("chunk failed")
	err := s.SubmitWait(context.Background(), func(ctx context.Context) error {
		var total atomic.Int64
		err := s.ParallelFor(ctx, 9, 4, func(ctx context.Context, start, end int) error {
			total.Add(int64(end - start))
			if start == 4 {
				return sentinel
			}
			return nil
		})
		if !errors.Is(err, sentinel) {
			return fmt.Errorf("nested error: %w", err)
		}
		if got := total.Load(); got != 9 {
			return fmt.Errorf("nested total = %d", got)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("SubmitWait: %v", err)
	}

	if err := s.ParallelFor(context.Background(), -1, 1, nil); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("invalid args: got %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := s.ParallelFor(ctx, 10, 2, func(context.Context, int, int) error { return nil }); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled ParallelFor: got %v", err)
	}
}

func TestNestedSubmitDoesNotBlock(t *testing.T) {
	s := mustNew(t, testConfig(1, 1))
	defer func() {
		if err := s.Close(context.Background()); err != nil {
			t.Fatalf("Close: %v", err)
		}
	}()
	err := s.SubmitWait(context.Background(), func(ctx context.Context) error {
		future, err := s.Submit(ctx, func(context.Context) error { return nil })
		if err != nil {
			return err
		}
		return future.Wait(ctx)
	})
	if err != nil {
		t.Fatalf("nested Submit: %v", err)
	}
}

func TestPolicyMatchesCompute(t *testing.T) {
	s := mustNew(t, testConfig(2, 4))
	defer func() {
		if err := s.Close(context.Background()); err != nil {
			t.Fatalf("Close: %v", err)
		}
	}()
	policy := s.Policy(8)
	if err := policy.Validate(); err != nil {
		t.Fatalf("Validate: %v", err)
	}
	reference := compute.ExecutionPolicy{GoWorkers: 2, BLASThreads: 8}
	if got, want := policy.EffectiveBLASThreads(), reference.EffectiveBLASThreads(); got != want {
		t.Fatalf("effective threads = %d, want %d", got, want)
	}
	if got := policy.EffectiveBLASThreads(); got < 1 || got > runtime.GOMAXPROCS(0) {
		t.Fatalf("effective threads out of budget: %d", got)
	}
	if err := (Policy{}).Validate(); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("empty policy: got %v", err)
	}
}

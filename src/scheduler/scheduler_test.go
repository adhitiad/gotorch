package scheduler

import (
	"context"
	"errors"
	"runtime"
	"sync/atomic"
	"testing"
	"time"

	"github.com/adhitiad/gotorch/src/errs"
)

func testConfig(workers, queue int) Config {
	return Config{Workers: workers, Queue: queue}
}

func mustNew(t *testing.T, config Config) *Scheduler {
	t.Helper()
	s, err := New(config)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return s
}

func waitForCondition(t *testing.T, condition func() bool, message string) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if condition() {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("condition not met: %s", message)
}

func TestConfigValidate(t *testing.T) {
	if err := (Config{}).Validate(); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("empty config: got %v", err)
	}
	if err := DefaultConfig().Validate(); err != nil {
		t.Fatalf("default config: %v", err)
	}
	tooMany := DefaultConfig()
	tooMany.Workers = MaxWorkers + 1
	if err := tooMany.Validate(); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("workers above limit: got %v", err)
	}
	tooLong := DefaultConfig()
	tooLong.Queue = MaxQueue + 1
	if err := tooLong.Validate(); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("queue above limit: got %v", err)
	}
	negativeChunk := DefaultConfig()
	negativeChunk.ChunkSize = -1
	if err := negativeChunk.Validate(); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("negative chunk: got %v", err)
	}
	negativeBLAS := DefaultConfig()
	negativeBLAS.BLASThreads = -1
	if err := negativeBLAS.Validate(); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("negative BLAS threads: got %v", err)
	}
}

func TestSubmitAndWait(t *testing.T) {
	s := mustNew(t, testConfig(2, 8))
	defer func() {
		if err := s.Close(context.Background()); err != nil {
			t.Fatalf("Close: %v", err)
		}
	}()

	var total atomic.Int64
	futures := make([]*Future, 16)
	for index := range futures {
		value := int64(index + 1)
		future, err := s.Submit(context.Background(), func(context.Context) error {
			total.Add(value)
			return nil
		})
		if err != nil {
			t.Fatalf("Submit %d: %v", index, err)
		}
		futures[index] = future
	}
	for index, future := range futures {
		if err := future.Wait(context.Background()); err != nil {
			t.Fatalf("Wait %d: %v", index, err)
		}
	}
	if got, want := total.Load(), int64(136); got != want {
		t.Fatalf("total = %d, want %d", got, want)
	}
	stats := s.Stats()
	if stats.Submitted != 16 || stats.Completed != 16 || stats.Failed != 0 {
		t.Fatalf("stats = %+v", stats)
	}
}

func TestTrySubmitQueueBound(t *testing.T) {
	s := mustNew(t, testConfig(1, 1))
	gate := make(chan struct{})
	defer func() {
		select {
		case <-gate:
		default:
			close(gate)
		}
	}()

	first, err := s.Submit(context.Background(), func(context.Context) error {
		<-gate
		return nil
	})
	if err != nil {
		t.Fatalf("first Submit: %v", err)
	}
	waitForCondition(t, func() bool { return s.Stats().Running == 1 }, "worker should be running")

	second, err := s.TrySubmit(context.Background(), func(context.Context) error { return nil })
	if err != nil {
		t.Fatalf("second TrySubmit: %v", err)
	}
	if _, err := s.TrySubmit(context.Background(), func(context.Context) error { return nil }); !errors.Is(err, ErrQueueFull) {
		t.Fatalf("third TrySubmit: got %v, want ErrQueueFull", err)
	}
	close(gate)
	if err := first.Wait(context.Background()); err != nil {
		t.Fatalf("first Wait: %v", err)
	}
	if err := second.Wait(context.Background()); err != nil {
		t.Fatalf("second Wait: %v", err)
	}
	if err := s.Close(context.Background()); err != nil {
		t.Fatalf("Close: %v", err)
	}
}

func TestSubmitHonorsCallerContext(t *testing.T) {
	s := mustNew(t, testConfig(1, 1))
	gate := make(chan struct{})
	defer func() {
		select {
		case <-gate:
		default:
			close(gate)
		}
	}()

	blocker, err := s.Submit(context.Background(), func(context.Context) error {
		<-gate
		return nil
	})
	if err != nil {
		t.Fatalf("blocker Submit: %v", err)
	}
	waitForCondition(t, func() bool { return s.Stats().Running == 1 }, "worker should be running")
	if _, err := s.TrySubmit(context.Background(), func(context.Context) error { return nil }); err != nil {
		t.Fatalf("fill queue: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	if _, err := s.Submit(ctx, func(context.Context) error { return nil }); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("blocked Submit: got %v, want DeadlineExceeded", err)
	}
	close(gate)
	if err := blocker.Wait(context.Background()); err != nil {
		t.Fatalf("blocker Wait: %v", err)
	}
	if err := s.Close(context.Background()); err != nil {
		t.Fatalf("Close: %v", err)
	}
}

func TestTaskCancellation(t *testing.T) {
	s := mustNew(t, testConfig(1, 2))
	defer func() {
		if err := s.Close(context.Background()); err != nil {
			t.Fatalf("Close: %v", err)
		}
	}()
	ctx, cancel := context.WithCancel(context.Background())
	started := make(chan struct{})
	future, err := s.Submit(ctx, func(ctx context.Context) error {
		close(started)
		<-ctx.Done()
		return ctx.Err()
	})
	if err != nil {
		t.Fatalf("Submit: %v", err)
	}
	<-started
	cancel()
	if err := future.Wait(context.Background()); !errors.Is(err, context.Canceled) {
		t.Fatalf("Wait: got %v, want Canceled", err)
	}
	if got := s.Stats().Canceled; got != 1 {
		t.Fatalf("canceled = %d, want 1", got)
	}
}

func TestPanicContainment(t *testing.T) {
	var reported []error
	config := testConfig(1, 4)
	config.OnError = func(err error) { reported = append(reported, err) }
	s := mustNew(t, config)
	defer func() {
		if err := s.Close(context.Background()); err != nil {
			t.Fatalf("Close: %v", err)
		}
	}()

	future, err := s.Submit(context.Background(), func(context.Context) error {
		panic("boom")
	})
	if err != nil {
		t.Fatalf("Submit panic: %v", err)
	}
	waitErr := future.Wait(context.Background())
	var panicErr *PanicError
	if !errors.As(waitErr, &panicErr) {
		t.Fatalf("Wait: got %T, want *PanicError", waitErr)
	}
	if panicErr.Value != "boom" || len(panicErr.Stack) == 0 {
		t.Fatalf("panic payload = %+v", panicErr)
	}
	if err := s.SubmitWait(context.Background(), func(context.Context) error { return nil }); err != nil {
		t.Fatalf("worker did not survive panic: %v", err)
	}
	stats := s.Stats()
	if stats.Panics != 1 || stats.Failed != 1 || stats.Completed != 1 {
		t.Fatalf("stats = %+v", stats)
	}
	if len(reported) != 1 {
		t.Fatalf("OnError calls = %d, want 1", len(reported))
	}
}

func TestCloseLifecycle(t *testing.T) {
	s := mustNew(t, testConfig(1, 1))
	gate := make(chan struct{})
	defer func() {
		select {
		case <-gate:
		default:
			close(gate)
		}
	}()
	if _, err := s.Submit(context.Background(), func(context.Context) error {
		<-gate
		return nil
	}); err != nil {
		t.Fatalf("Submit: %v", err)
	}
	waitForCondition(t, func() bool { return s.Stats().Running == 1 }, "worker should be running")

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	if err := s.Close(ctx); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("timed Close: got %v", err)
	}
	close(gate)
	if err := s.Close(context.Background()); err != nil {
		t.Fatalf("second Close: %v", err)
	}
	if err := s.Close(context.Background()); err != nil {
		t.Fatalf("third Close: %v", err)
	}
	if _, err := s.Submit(context.Background(), func(context.Context) error { return nil }); !errors.Is(err, ErrSchedulerClosed) {
		t.Fatalf("Submit after close: got %v", err)
	}
	if _, err := s.TrySubmit(context.Background(), func(context.Context) error { return nil }); !errors.Is(err, ErrSchedulerClosed) {
		t.Fatalf("TrySubmit after close: got %v", err)
	}
	if !s.Stats().Closed {
		t.Fatal("Stats.Closed should be true")
	}
}

func TestNoGoroutineLeak(t *testing.T) {
	baseline := runtime.NumGoroutine()
	for round := 0; round < 4; round++ {
		s := mustNew(t, testConfig(4, 8))
		for index := 0; index < 32; index++ {
			if err := s.SubmitWait(context.Background(), func(context.Context) error { return nil }); err != nil {
				t.Fatalf("SubmitWait: %v", err)
			}
		}
		if err := s.Close(context.Background()); err != nil {
			t.Fatalf("Close: %v", err)
		}
	}
	waitForCondition(t, func() bool { return runtime.NumGoroutine() <= baseline+2 }, "goroutines should return to baseline")
}

package scheduler

import (
	"context"
	"errors"

	"github.com/adhitiad/gotorch/src/errs"
)

// ChunkFunc memproses rentang indeks [start,end) dari sebuah ParallelFor.
type ChunkFunc func(context.Context, int, int) error

// ParallelFor membagi total indeks menjadi chunk dan menjalankannya dengan worker
// scheduler. Saat dipanggil dari dalam task worker, chunk dijalankan sebaris agar
// tidak terjadi deadlock akibat nested parallelism.
func (s *Scheduler) ParallelFor(ctx context.Context, total, chunk int, fn ChunkFunc) error {
	if s == nil || isNil(ctx) || fn == nil {
		return errs.ErrInvalidArgument
	}
	if total < 0 || chunk < 0 {
		return errs.ErrInvalidArgument
	}
	if total == 0 {
		return nil
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if chunk == 0 {
		chunk = s.config.ChunkSize
	}
	if chunk <= 0 {
		chunk = 1 + (total-1)/s.config.Workers
	}
	if chunk < 1 {
		chunk = 1
	}
	chunks := 1 + (total-1)/chunk

	if isWorker(ctx) {
		var result []error
		for index := 0; index < chunks; index++ {
			if err := ctx.Err(); err != nil {
				result = append(result, err)
				break
			}
			start, end := chunkRange(index, chunk, total)
			err := s.executeInline(ctx, func(ctx context.Context) error {
				return fn(ctx, start, end)
			})
			if err != nil {
				result = append(result, err)
			}
		}
		return errors.Join(result...)
	}

	futures := make([]*Future, 0, chunks)
	var result []error
	for index := 0; index < chunks; index++ {
		start, end := chunkRange(index, chunk, total)
		future, err := s.Submit(ctx, func(ctx context.Context) error {
			return fn(ctx, start, end)
		})
		if err != nil {
			result = append(result, err)
			break
		}
		futures = append(futures, future)
	}
	for _, future := range futures {
		if err := future.Wait(ctx); err != nil {
			result = append(result, err)
		}
	}
	return errors.Join(result...)
}

// executeInline menjalankan task di goroutine pembanggil dengan perlindungan panic
// dan pencatatan statistik yang sama seperti task yang lewat antrean.
func (s *Scheduler) executeInline(ctx context.Context, fn Task) error {
	s.submitMu.Lock()
	closed := s.closed
	s.submitMu.Unlock()
	if closed {
		return ErrSchedulerClosed
	}
	future := &Future{done: make(chan struct{})}
	s.submitted.Add(1)
	s.executeJob(&job{ctx: ctx, fn: fn, future: future})
	return future.err
}

func chunkRange(index, chunk, total int) (int, int) {
	start := index * chunk
	if chunk > total-start {
		return start, total
	}
	return start, start + chunk
}

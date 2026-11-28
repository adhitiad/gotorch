package data

import (
	"context"
	"errors"
	"fmt"
	"io"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/adhitiad/gotorch/src/errs"
)

type countingDataset struct {
	length     int64
	failIndex  int64
	hasFailure bool
	active     atomic.Int64
	maxActive  atomic.Int64
	reads      atomic.Int64
}

func (d *countingDataset) Len() int64 { return d.length }

func (d *countingDataset) At(ctx context.Context, index int64) (Sample, error) {
	if err := checkContext(ctx); err != nil {
		return Sample{}, err
	}
	current := d.active.Add(1)
	for {
		maximum := d.maxActive.Load()
		if current <= maximum || d.maxActive.CompareAndSwap(maximum, current) {
			break
		}
	}
	defer d.active.Add(-1)
	d.reads.Add(1)
	if d.hasFailure && index == d.failIndex {
		return Sample{}, fmt.Errorf("%w: dataset index %d", errs.ErrInvalidState, index)
	}
	return NewSample(map[string]any{"id": index})
}

func (d *countingDataset) Close() error { return nil }

func testBatcher(t *testing.T, size int64) *Batcher {
	t.Helper()
	batcher, err := NewBatcher(size, KeepLast, DefaultCollate)
	if err != nil {
		t.Fatalf("NewBatcher: %v", err)
	}
	return batcher
}

func collectBatches(t *testing.T, iterator *BatchIterator) []*Batch {
	t.Helper()
	var batches []*Batch
	for {
		batch, err := iterator.Next(context.Background())
		if errors.Is(err, io.EOF) {
			return batches
		}
		if err != nil {
			t.Fatalf("Next: %v", err)
		}
		batches = append(batches, batch)
	}
}

func TestNewDataLoaderValidation(t *testing.T) {
	batcher := testBatcher(t, 2)
	dataset := &countingDataset{length: 4}
	sampler := fixedSampler{indices: []int64{0, 1, 2, 3}}
	var nilBatcher *Batcher
	var nilDataset *countingDataset
	var nilSampler *fixedSampler
	cases := []struct {
		name     string
		batcher  *Batcher
		dataset  Dataset
		sampler  Sampler
		workers  int
		prefetch int
	}{
		{name: "nil batcher", batcher: nilBatcher, dataset: dataset, sampler: sampler, workers: 1, prefetch: 1},
		{name: "nil dataset", batcher: batcher, dataset: nil, sampler: sampler, workers: 1, prefetch: 1},
		{name: "nil sampler", batcher: batcher, dataset: dataset, sampler: nil, workers: 1, prefetch: 1},
		{name: "typed nil dataset", batcher: batcher, dataset: nilDataset, sampler: sampler, workers: 1, prefetch: 1},
		{name: "typed nil sampler", batcher: batcher, dataset: dataset, sampler: nilSampler, workers: 1, prefetch: 1},
		{name: "zero workers", batcher: batcher, dataset: dataset, sampler: sampler, workers: 0, prefetch: 1},
		{name: "too many workers", batcher: batcher, dataset: dataset, sampler: sampler, workers: MaxWorkers + 1, prefetch: 1},
		{name: "negative prefetch", batcher: batcher, dataset: dataset, sampler: sampler, workers: 1, prefetch: -1},
		{name: "too much prefetch", batcher: batcher, dataset: dataset, sampler: sampler, workers: 1, prefetch: MaxPrefetch + 1},
	}
	for _, item := range cases {
		t.Run(item.name, func(t *testing.T) {
			loader, err := NewDataLoader(item.batcher, item.dataset, item.sampler, item.workers, item.prefetch)
			if loader != nil || !errors.Is(err, errs.ErrInvalidArgument) {
				t.Fatalf("NewDataLoader = %v, %v", loader, err)
			}
		})
	}
	var nilLoader *DataLoader
	if nilLoader.Len() != 0 || nilLoader.Workers() != 0 || nilLoader.Prefetch() != 0 {
		t.Fatal("loader nil tidak bersifat kosong")
	}
	if _, err := nilLoader.Iterate(context.Background()); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("Iterate pada loader nil = %v", err)
	}
	var nilIterator *BatchIterator
	if _, err := nilIterator.Next(context.Background()); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("Next pada iterator nil = %v", err)
	}
	if err := nilIterator.Close(); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("Close pada iterator nil = %v", err)
	}
	loader, err := NewDataLoader(batcher, dataset, sampler, 2, 1)
	if err != nil {
		t.Fatalf("NewDataLoader: %v", err)
	}
	if _, err := loader.Iterate(nil); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("Iterate dengan context nil = %v", err)
	}
	iterator, err := loader.Iterate(context.Background())
	if err != nil {
		t.Fatalf("Iterate: %v", err)
	}
	defer iterator.Close()
	if _, err := iterator.Next(nil); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("Next dengan context nil = %v", err)
	}
	if loader.Len() != 2 || loader.Workers() != 2 || loader.Prefetch() != 1 {
		t.Fatalf("konfigurasi loader = %d/%d/%d", loader.Len(), loader.Workers(), loader.Prefetch())
	}
}

func TestDataLoaderOrderedBatches(t *testing.T) {
	dataset := &countingDataset{length: 6}
	sampler := fixedSampler{indices: []int64{0, 1, 2, 3, 4, 5}}
	loader, err := NewDataLoader(testBatcher(t, 2), dataset, sampler, 4, 3)
	if err != nil {
		t.Fatalf("NewDataLoader: %v", err)
	}
	iterator, err := loader.Iterate(context.Background())
	if err != nil {
		t.Fatalf("Iterate: %v", err)
	}
	defer iterator.Close()
	if cap(iterator.jobs) != 3 || cap(iterator.results) != 4 || cap(iterator.ordered) != 7 {
		t.Fatalf("kapasitas channel = %d/%d/%d", cap(iterator.jobs), cap(iterator.results), cap(iterator.ordered))
	}
	batches := collectBatches(t, iterator)
	if len(batches) != 3 {
		t.Fatalf("jumlah batch = %d, diharapkan 3", len(batches))
	}
	for batchIndex, batch := range batches {
		want := []int64{int64(batchIndex * 2), int64(batchIndex*2 + 1)}
		if got := batch.Indices(); !equalInt64(got, want) {
			t.Fatalf("batch %d = %v, diharapkan %v", batchIndex, got, want)
		}
	}
	if dataset.maxActive.Load() > 4 {
		t.Fatalf("worker aktif maksimum = %d", dataset.maxActive.Load())
	}
}

func TestDataLoaderEmptyAndZeroPrefetch(t *testing.T) {
	empty, err := NewDataLoader(testBatcher(t, 2), &countingDataset{}, fixedSampler{}, 2, 0)
	if err != nil {
		t.Fatalf("NewDataLoader(empty): %v", err)
	}
	if empty.Len() != 0 {
		t.Fatalf("Len = %d, diharapkan 0", empty.Len())
	}
	emptyIterator, err := empty.Iterate(context.Background())
	if err != nil {
		t.Fatalf("Iterate(empty): %v", err)
	}
	if batch, err := emptyIterator.Next(context.Background()); batch != nil || !errors.Is(err, io.EOF) {
		t.Fatalf("Next(empty) = %v, %v", batch, err)
	}
	if err := emptyIterator.Close(); err != nil {
		t.Fatalf("Close(empty): %v", err)
	}

	dataset := &countingDataset{length: 5}
	sampler := fixedSampler{indices: []int64{4, 3, 2, 1, 0}}
	loader, err := NewDataLoader(testBatcher(t, 1), dataset, sampler, 3, 0)
	if err != nil {
		t.Fatalf("NewDataLoader(prefetch 0): %v", err)
	}
	iterator, err := loader.Iterate(context.Background())
	if err != nil {
		t.Fatalf("Iterate(prefetch 0): %v", err)
	}
	defer iterator.Close()
	batches := collectBatches(t, iterator)
	if len(batches) != 5 {
		t.Fatalf("jumlah batch = %d, diharapkan 5", len(batches))
	}
	for position, want := range sampler.indices {
		if got := batches[position].Indices()[0]; got != want {
			t.Fatalf("batch %d = %d, diharapkan %d", position, got, want)
		}
	}
}

func TestDataLoaderWorkerFailure(t *testing.T) {
	dataset := &countingDataset{length: 8, failIndex: 3, hasFailure: true}
	sampler := fixedSampler{indices: []int64{0, 1, 2, 3, 4, 5, 6, 7}}
	loader, err := NewDataLoader(testBatcher(t, 1), dataset, sampler, 4, 2)
	if err != nil {
		t.Fatalf("NewDataLoader: %v", err)
	}
	iterator, err := loader.Iterate(context.Background())
	if err != nil {
		t.Fatalf("Iterate: %v", err)
	}
	defer iterator.Close()
	var failure error
	for attempt := 0; attempt < 8; attempt++ {
		batch, nextErr := iterator.Next(context.Background())
		if batch != nil {
			if index := batch.Indices()[0]; index >= dataset.failIndex {
				t.Fatalf("batch setelah kegagalan: %v", batch.Indices())
			}
			continue
		}
		if nextErr == nil {
			t.Fatal("Next kembali tanpa batch maupun error")
		}
		failure = nextErr
		break
	}
	if !errors.Is(failure, errs.ErrInvalidState) {
		t.Fatalf("error worker = %v", failure)
	}
	for attempt := 0; attempt < 3; attempt++ {
		batch, nextErr := iterator.Next(context.Background())
		if batch != nil || !errors.Is(nextErr, errs.ErrInvalidState) {
			t.Fatalf("Next(%d) = %v, %v", attempt, batch, nextErr)
		}
	}
	readsAfterFailure := dataset.reads.Load()
	if err := iterator.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if _, err := iterator.Next(context.Background()); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("Next setelah Close = %v", err)
	}
	if dataset.reads.Load() != readsAfterFailure || dataset.active.Load() != 0 {
		t.Fatalf("worker masih membaca setelah gagal: reads=%d active=%d", dataset.reads.Load(), dataset.active.Load())
	}
}

func TestDataLoaderCancellation(t *testing.T) {
	blocking := &blockingDataset{entered: make(chan struct{})}
	loader, err := NewDataLoader(testBatcher(t, 1), blocking, fixedSampler{indices: []int64{0, 1, 2, 3}}, 2, 1)
	if err != nil {
		t.Fatalf("NewDataLoader: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	iterator, err := loader.Iterate(ctx)
	if err != nil {
		t.Fatalf("Iterate: %v", err)
	}
	defer iterator.Close()
	type result struct {
		batch *Batch
		err   error
	}
	results := make(chan result, 1)
	go func() {
		batch, nextErr := iterator.Next(ctx)
		results <- result{batch: batch, err: nextErr}
	}()
	select {
	case <-blocking.entered:
	case <-time.After(2 * time.Second):
		t.Fatal("worker tidak mulai membaca")
	}
	cancel()
	select {
	case got := <-results:
		if got.batch != nil || !errors.Is(got.err, context.Canceled) {
			t.Fatalf("Next = %v, %v; diharapkan context.Canceled", got.batch, got.err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Next tidak kembali setelah pembatalan")
	}
	if err := iterator.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := iterator.Close(); err != nil {
		t.Fatalf("Close kedua: %v", err)
	}
}

type blockingDataset struct {
	entered chan struct{}
	once    sync.Once
}

func (d *blockingDataset) Len() int64 { return 4 }

func (d *blockingDataset) At(ctx context.Context, index int64) (Sample, error) {
	if err := checkContext(ctx); err != nil {
		return Sample{}, err
	}
	d.once.Do(func() { close(d.entered) })
	<-ctx.Done()
	return Sample{}, ctx.Err()
}

func (d *blockingDataset) Close() error { return nil }

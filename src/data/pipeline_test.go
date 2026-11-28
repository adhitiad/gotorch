package data

import (
	"context"
	"errors"
	"fmt"
	"io"
	"testing"

	"github.com/adhitiad/gotorch/src/errs"
)

func collectIndexStrings(t *testing.T, iterator *BatchIterator) []string {
	t.Helper()
	var result []string
	for {
		batch, err := iterator.Next(context.Background())
		if errors.Is(err, io.EOF) {
			return result
		}
		if err != nil {
			t.Fatalf("Next: %v", err)
		}
		result = append(result, fmt.Sprint(batch.Indices()))
	}
}

func TestPipelineDeterministicOrder(t *testing.T) {
	const length = 12
	var baseline []string
	for run, workers := range []int{1, 2, 4, 8, 3} {
		dataset := &countingDataset{length: length}
		sampler, err := NewRandomSampler(length, 42)
		if err != nil {
			t.Fatalf("NewRandomSampler: %v", err)
		}
		loader, err := NewDataLoader(testBatcher(t, 2), dataset, sampler, workers, 2)
		if err != nil {
			t.Fatalf("NewDataLoader: %v", err)
		}
		iterator, err := loader.Iterate(context.Background())
		if err != nil {
			t.Fatalf("Iterate: %v", err)
		}
		got := collectIndexStrings(t, iterator)
		if err := iterator.Close(); err != nil {
			t.Fatalf("Close: %v", err)
		}
		if run == 0 {
			baseline = got
			continue
		}
		if len(got) != len(baseline) {
			t.Fatalf("putaran %d menghasilkan %d batch, diharapkan %d", run, len(got), len(baseline))
		}
		for position := range got {
			if got[position] != baseline[position] {
				t.Fatalf("putaran %d batch %d = %s, diharapkan %s", run, position, got[position], baseline[position])
			}
		}
	}
}

func TestPipelineTransformFailure(t *testing.T) {
	collate, err := CollateWithTransform(failingTransform{err: errs.ErrUnsupportedOperation}, DefaultCollate)
	if err != nil {
		t.Fatalf("CollateWithTransform: %v", err)
	}
	batcher, err := NewBatcher(1, KeepLast, collate)
	if err != nil {
		t.Fatalf("NewBatcher: %v", err)
	}
	dataset := &countingDataset{length: 6}
	sampler := fixedSampler{indices: []int64{0, 1, 2, 3, 4, 5}}
	loader, err := NewDataLoader(batcher, dataset, sampler, 4, 2)
	if err != nil {
		t.Fatalf("NewDataLoader: %v", err)
	}
	iterator, err := loader.Iterate(context.Background())
	if err != nil {
		t.Fatalf("Iterate: %v", err)
	}
	defer iterator.Close()
	batch, err := iterator.Next(context.Background())
	if batch != nil || !errors.Is(err, errs.ErrUnsupportedOperation) {
		t.Fatalf("Next = %v, %v", batch, err)
	}
	readsAtFailure := dataset.reads.Load()
	if err := iterator.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if dataset.reads.Load() != readsAtFailure || dataset.active.Load() != 0 {
		t.Fatalf("worker aktif setelah gagal: reads=%d active=%d", dataset.reads.Load(), dataset.active.Load())
	}
}

func TestPipelineCancelBeforeIterate(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	loader, err := NewDataLoader(testBatcher(t, 1), &countingDataset{length: 2}, fixedSampler{indices: []int64{0, 1}}, 2, 1)
	if err != nil {
		t.Fatalf("NewDataLoader: %v", err)
	}
	iterator, err := loader.Iterate(ctx)
	if iterator != nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("Iterate = %v, %v", iterator, err)
	}
}

func TestPipelineCancelDuringRead(t *testing.T) {
	blocking := &blockingDataset{entered: make(chan struct{})}
	loader, err := NewDataLoader(testBatcher(t, 1), blocking, fixedSampler{indices: []int64{0, 1, 2, 3}}, 4, 2)
	if err != nil {
		t.Fatalf("NewDataLoader: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	iterator, err := loader.Iterate(ctx)
	if err != nil {
		t.Fatalf("Iterate: %v", err)
	}
	cancel()
	batch, err := iterator.Next(context.Background())
	if batch != nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("Next = %v, %v", batch, err)
	}
	if err := iterator.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if _, err := iterator.Next(context.Background()); !errors.Is(err, context.Canceled) {
		t.Fatalf("Next setelah pembatalan = %v", err)
	}
}

func TestPipelineResourceBounds(t *testing.T) {
	const (
		workers  = 4
		prefetch = 3
		length   = 16
	)
	dataset := &countingDataset{length: length}
	sampler := fixedSampler{indices: make([]int64, length)}
	for index := range sampler.indices {
		sampler.indices[index] = int64(index)
	}
	loader, err := NewDataLoader(testBatcher(t, 1), dataset, sampler, workers, prefetch)
	if err != nil {
		t.Fatalf("NewDataLoader: %v", err)
	}
	iterator, err := loader.Iterate(context.Background())
	if err != nil {
		t.Fatalf("Iterate: %v", err)
	}
	if cap(iterator.jobs) != prefetch || cap(iterator.results) != workers || cap(iterator.ordered) != workers+prefetch {
		t.Fatalf("kapasitas = %d/%d/%d", cap(iterator.jobs), cap(iterator.results), cap(iterator.ordered))
	}
	if iterator.limit != workers+prefetch {
		t.Fatalf("limit = %d", iterator.limit)
	}
	batches := collectIndexStrings(t, iterator)
	if len(batches) != length {
		t.Fatalf("jumlah batch = %d, diharapkan %d", len(batches), length)
	}
	if err := iterator.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if got := dataset.maxActive.Load(); got > workers {
		t.Fatalf("worker aktif maksimum = %d, batas %d", got, workers)
	}
	if iterator.submitted != length {
		t.Fatalf("batch dikirim = %d, diharapkan %d", iterator.submitted, length)
	}
	if dataset.active.Load() != 0 {
		t.Fatalf("worker tersisa = %d", dataset.active.Load())
	}
}

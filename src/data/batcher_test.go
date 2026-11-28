package data

import (
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/adhitiad/gotorch/src/errs"
)

func batchSamples(t *testing.T, count int) []Sample {
	t.Helper()
	samples := make([]Sample, count)
	for index := range samples {
		sample, err := NewSample(map[string]any{"id": int64(index)})
		if err != nil {
			t.Fatalf("NewSample(%d): %v", index, err)
		}
		samples[index] = sample
	}
	return samples
}

type errorDataset struct {
	length int64
	err    error
	cancel context.CancelFunc
}

func (d *errorDataset) Len() int64 { return d.length }

func (d *errorDataset) At(ctx context.Context, index int64) (Sample, error) {
	if err := checkContext(ctx); err != nil {
		return Sample{}, err
	}
	if d.cancel != nil {
		d.cancel()
	}
	return Sample{}, d.err
}

func (d *errorDataset) Close() error { return nil }

type errorSampler struct {
	length int64
	err    error
}

func (s *errorSampler) Len() int64 { return s.length }

func (s *errorSampler) Index(ctx context.Context, position int64) (int64, error) {
	if err := checkContext(ctx); err != nil {
		return 0, err
	}
	return 0, s.err
}

func TestBatcherBatchCount(t *testing.T) {
	keep, err := NewBatcher(2, KeepLast, DefaultCollate)
	if err != nil {
		t.Fatalf("NewBatcher(keep): %v", err)
	}
	drop, err := NewBatcher(2, DropLast, DefaultCollate)
	if err != nil {
		t.Fatalf("NewBatcher(drop): %v", err)
	}
	var nilBatcher *Batcher
	cases := []struct {
		name    string
		batcher *Batcher
		length  int64
		want    int64
	}{
		{name: "nil batcher", batcher: nilBatcher, length: 5, want: 0},
		{name: "zero length", batcher: keep, length: 0, want: 0},
		{name: "negative length", batcher: keep, length: -1, want: 0},
		{name: "exact keep", batcher: keep, length: 6, want: 3},
		{name: "partial keep", batcher: keep, length: 5, want: 3},
		{name: "exact drop", batcher: drop, length: 6, want: 3},
		{name: "partial drop", batcher: drop, length: 5, want: 2},
		{name: "smaller than batch", batcher: keep, length: 1, want: 1},
		{name: "smaller than batch dropped", batcher: drop, length: 1, want: 0},
	}
	for _, item := range cases {
		t.Run(item.name, func(t *testing.T) {
			if got := item.batcher.BatchCount(item.length); got != item.want {
				t.Fatalf("BatchCount(%d) = %d, diharapkan %d", item.length, got, item.want)
			}
		})
	}
}

func TestBatcherKeepAndDropLast(t *testing.T) {
	dataset := &sliceDataset{samples: batchSamples(t, 5)}
	sampler := fixedSampler{indices: []int64{0, 1, 2, 3, 4}}
	keep, err := NewBatcher(2, KeepLast, DefaultCollate)
	if err != nil {
		t.Fatalf("NewBatcher(keep): %v", err)
	}
	drop, err := NewBatcher(2, DropLast, DefaultCollate)
	if err != nil {
		t.Fatalf("NewBatcher(drop): %v", err)
	}
	keepBatch, err := keep.Batch(context.Background(), dataset, sampler, 2)
	if err != nil {
		t.Fatalf("Batch(keep, 2): %v", err)
	}
	if keepBatch.Len() != 1 || keepBatch.Indices()[0] != 4 {
		t.Fatalf("batch terakhir keep = %d / %v", keepBatch.Len(), keepBatch.Indices())
	}
	if batch, err := drop.Batch(context.Background(), dataset, sampler, 2); !errors.Is(err, errs.ErrOutOfBounds) || batch != nil {
		t.Fatalf("Batch(drop, 2) = %v, %v", batch, err)
	}
	first, err := drop.Batch(context.Background(), dataset, sampler, 0)
	if err != nil {
		t.Fatalf("Batch(drop, 0): %v", err)
	}
	if want := []int64{0, 1}; !equalInt64(first.Indices(), want) {
		t.Fatalf("indeks = %v, diharapkan %v", first.Indices(), want)
	}
	if got, ok := first.Samples()[0].Value("id"); !ok || got != int64(0) {
		t.Fatalf("sampel pertama = %v", got)
	}
}

func TestBatcherUsesSamplerOrder(t *testing.T) {
	dataset := &sliceDataset{samples: batchSamples(t, 5)}
	sampler := fixedSampler{indices: []int64{4, 2, 0, 3, 1}}
	batcher, err := NewBatcher(3, KeepLast, DefaultCollate)
	if err != nil {
		t.Fatalf("NewBatcher: %v", err)
	}
	batch, err := batcher.Batch(context.Background(), dataset, sampler, 0)
	if err != nil {
		t.Fatalf("Batch: %v", err)
	}
	if want := []int64{4, 2, 0}; !equalInt64(batch.Indices(), want) {
		t.Fatalf("indeks = %v, diharapkan %v", batch.Indices(), want)
	}
	for position, wantIndex := range sampler.indices[:3] {
		sample, index, err := batch.At(position)
		if err != nil {
			t.Fatalf("At(%d): %v", position, err)
		}
		value, _ := sample.Value("id")
		if index != wantIndex || value != wantIndex {
			t.Fatalf("posisi %d: indeks %d, id %v", position, index, value)
		}
	}
}

func TestBatcherCustomCollate(t *testing.T) {
	dataset := &sliceDataset{samples: batchSamples(t, 4)}
	sampler := fixedSampler{indices: []int64{3, 1, 2, 0}}
	var seenSamples []Sample
	var seenIndices []int64
	collate := func(ctx context.Context, samples []Sample, indices []int64) (*Batch, error) {
		if err := checkContext(ctx); err != nil {
			return nil, err
		}
		seenSamples = append([]Sample(nil), samples...)
		seenIndices = append([]int64(nil), indices...)
		return NewBatch(samples, indices)
	}
	batcher, err := NewBatcher(4, KeepLast, collate)
	if err != nil {
		t.Fatalf("NewBatcher: %v", err)
	}
	batch, err := batcher.Batch(context.Background(), dataset, sampler, 0)
	if err != nil {
		t.Fatalf("Batch: %v", err)
	}
	if !equalInt64(seenIndices, sampler.indices) {
		t.Fatalf("collate menerima indeks %v", seenIndices)
	}
	if len(seenSamples) != 4 {
		t.Fatalf("collate menerima %d sampel", len(seenSamples))
	}
	seenSamples[0], _ = NewSample(map[string]any{"id": int64(99)})
	seenIndices[0] = 99
	if got := batch.Indices(); !equalInt64(got, sampler.indices) {
		t.Fatalf("batch berubah setelah collate: %v", got)
	}
	sample, _, err := batch.At(0)
	if err != nil {
		t.Fatalf("At(0): %v", err)
	}
	if value, _ := sample.Value("id"); value != int64(3) {
		t.Fatalf("id pertama = %v", value)
	}
}

func TestBatcherValidation(t *testing.T) {
	_, err := NewBatcher(0, KeepLast, DefaultCollate)
	if !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("ukuran nol diterima: %v", err)
	}
	_, err = NewBatcher(-1, KeepLast, DefaultCollate)
	if !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("ukuran negatif diterima: %v", err)
	}
	_, err = NewBatcher(2, LastBatchPolicy("unknown"), DefaultCollate)
	if !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("kebijakan asing diterima: %v", err)
	}
	_, err = NewBatcher(2, KeepLast, nil)
	if !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("collate nil diterima: %v", err)
	}
	if _, ctxErr := DefaultCollate(nil, nil, nil); !errors.Is(ctxErr, errs.ErrInvalidArgument) {
		t.Fatalf("DefaultCollate context nil: %v", ctxErr)
	}

	batcher, err := NewBatcher(2, KeepLast, DefaultCollate)
	if err != nil {
		t.Fatalf("NewBatcher: %v", err)
	}
	dataset := &sliceDataset{samples: batchSamples(t, 3)}
	sampler := fixedSampler{indices: []int64{0, 1, 2}}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	var nilBatcher *Batcher
	var nilDataset *sliceDataset
	var nilSampler *fixedSampler
	failingCollate, err := NewBatcher(2, KeepLast, func(context.Context, []Sample, []int64) (*Batch, error) {
		return nil, errs.ErrUnsupportedOperation
	})
	if err != nil {
		t.Fatalf("NewBatcher(failing): %v", err)
	}
	nilCollate, err := NewBatcher(2, KeepLast, func(context.Context, []Sample, []int64) (*Batch, error) {
		return nil, nil
	})
	if err != nil {
		t.Fatalf("NewBatcher(nil result): %v", err)
	}
	cases := []struct {
		name    string
		batcher *Batcher
		ctx     context.Context
		dataset Dataset
		sampler Sampler
		index   int64
		wantErr error
	}{
		{name: "nil context", batcher: batcher, ctx: nil, dataset: dataset, sampler: sampler, index: 0, wantErr: errs.ErrInvalidArgument},
		{name: "canceled context", batcher: batcher, ctx: canceled, dataset: dataset, sampler: sampler, index: 0, wantErr: context.Canceled},
		{name: "nil batcher", batcher: nilBatcher, ctx: context.Background(), dataset: dataset, sampler: sampler, index: 0, wantErr: errs.ErrInvalidState},
		{name: "nil dataset", batcher: batcher, ctx: context.Background(), dataset: nil, sampler: sampler, index: 0, wantErr: errs.ErrInvalidArgument},
		{name: "nil sampler", batcher: batcher, ctx: context.Background(), dataset: dataset, sampler: nil, index: 0, wantErr: errs.ErrInvalidArgument},
		{name: "typed nil dataset", batcher: batcher, ctx: context.Background(), dataset: nilDataset, sampler: sampler, index: 0, wantErr: errs.ErrInvalidArgument},
		{name: "typed nil sampler", batcher: batcher, ctx: context.Background(), dataset: dataset, sampler: nilSampler, index: 0, wantErr: errs.ErrInvalidArgument},
		{name: "negative batch index", batcher: batcher, ctx: context.Background(), dataset: dataset, sampler: sampler, index: -1, wantErr: errs.ErrOutOfBounds},
		{name: "batch index at count", batcher: batcher, ctx: context.Background(), dataset: dataset, sampler: sampler, index: 2, wantErr: errs.ErrOutOfBounds},
		{name: "sampler error", batcher: batcher, ctx: context.Background(), dataset: dataset, sampler: &errorSampler{length: 2, err: errs.ErrUnsupportedOperation}, index: 0, wantErr: errs.ErrUnsupportedOperation},
		{name: "dataset error", batcher: batcher, ctx: context.Background(), dataset: &errorDataset{length: 3, err: errs.ErrInvalidState}, sampler: sampler, index: 0, wantErr: errs.ErrInvalidState},
		{name: "collate error", batcher: failingCollate, ctx: context.Background(), dataset: dataset, sampler: sampler, index: 0, wantErr: errs.ErrUnsupportedOperation},
		{name: "nil collate result", batcher: nilCollate, ctx: context.Background(), dataset: dataset, sampler: sampler, index: 0, wantErr: errs.ErrInvalidState},
		{name: "sampler index outside dataset", batcher: batcher, ctx: context.Background(), dataset: dataset, sampler: fixedSampler{indices: []int64{9, 0}}, index: 0, wantErr: errs.ErrOutOfBounds},
	}
	for _, item := range cases {
		t.Run(item.name, func(t *testing.T) {
			batch, batchErr := item.batcher.Batch(item.ctx, item.dataset, item.sampler, item.index)
			if batch != nil || !errors.Is(batchErr, item.wantErr) {
				t.Fatalf("Batch = %v, %v; diharapkan error %v", batch, batchErr, item.wantErr)
			}
		})
	}
}

func TestBatcherCancellationDuringRead(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	batcher, err := NewBatcher(2, KeepLast, DefaultCollate)
	if err != nil {
		t.Fatalf("NewBatcher: %v", err)
	}
	dataset := &errorDataset{length: 3, err: errs.ErrInvalidState, cancel: cancel}
	batch, err := batcher.Batch(ctx, dataset, fixedSampler{indices: []int64{0, 1}}, 0)
	if batch != nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("Batch = %v, %v; diharapkan context.Canceled", batch, err)
	}
}

func TestBatcherConcurrent(t *testing.T) {
	dataset := &sliceDataset{samples: batchSamples(t, 8)}
	sampler := fixedSampler{indices: []int64{7, 0, 5, 1, 6, 2, 4, 3}}
	batcher, err := NewBatcher(2, KeepLast, DefaultCollate)
	if err != nil {
		t.Fatalf("NewBatcher: %v", err)
	}
	var group sync.WaitGroup
	for worker := 0; worker < 8; worker++ {
		group.Add(1)
		go func() {
			defer group.Done()
			for batchIndex := int64(0); batchIndex < batcher.BatchCount(sampler.Len()); batchIndex++ {
				batch, err := batcher.Batch(context.Background(), dataset, sampler, batchIndex)
				if err != nil {
					t.Errorf("Batch(%d): %v", batchIndex, err)
					return
				}
				if batch.Len() != 2 {
					t.Errorf("Batch(%d).Len() = %d", batchIndex, batch.Len())
					return
				}
				for offset, want := range sampler.indices[batchIndex*2 : batchIndex*2+2] {
					if index := batch.Indices()[offset]; index != want {
						t.Errorf("Batch(%d)[%d] = %d, diharapkan %d", batchIndex, offset, index, want)
						return
					}
				}
			}
		}()
	}
	group.Wait()
}

func equalInt64(got, want []int64) bool {
	if len(got) != len(want) {
		return false
	}
	for index := range got {
		if got[index] != want[index] {
			return false
		}
	}
	return true
}

package data

import (
	"context"
	"errors"
	"io"
	"math"
	"testing"

	"github.com/adhitiad/gotorch/src/errs"
)

func sourceSamples(t *testing.T) []Sample {
	t.Helper()
	samples := make([]Sample, 3)
	for index := range samples {
		sample, err := NewSample(map[string]any{"id": int64(index)})
		if err != nil {
			t.Fatalf("NewSample: %v", err)
		}
		samples[index] = sample
	}
	return samples
}

func TestSliceSource(t *testing.T) {
	samples := sourceSamples(t)
	source := NewSliceSource(samples)
	if source.Len() != 3 {
		t.Fatalf("Len = %d, diharapkan 3", source.Len())
	}
	for index := range samples {
		sample, err := source.Next(context.Background())
		if err != nil {
			t.Fatalf("Next(%d): %v", index, err)
		}
		value, _ := sample.Value("id")
		if value != int64(index) {
			t.Fatalf("id = %v, diharapkan %d", value, index)
		}
	}
	if source.Len() != 0 {
		t.Fatalf("Len akhir = %d", source.Len())
	}
	if _, err := source.Next(context.Background()); !errors.Is(err, io.EOF) {
		t.Fatalf("Next setelah akhir = %v", err)
	}
	if err := source.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := source.Close(); err != nil {
		t.Fatalf("Close kedua: %v", err)
	}
	if _, err := source.Next(context.Background()); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("Next setelah Close = %v", err)
	}
}

func TestSliceSourceValidation(t *testing.T) {
	var nilSource *SliceSource
	if nilSource.Len() != 0 {
		t.Fatalf("nil Len = %d", nilSource.Len())
	}
	if _, err := nilSource.Next(context.Background()); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("nil Next = %v", err)
	}
	if err := nilSource.Close(); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("nil Close = %v", err)
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	source := NewSliceSource(sourceSamples(t))
	if _, err := source.Next(nil); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("Next context nil = %v", err)
	}
	if _, err := source.Next(canceled); !errors.Is(err, context.Canceled) {
		t.Fatalf("Next context dibatalkan = %v", err)
	}
}

func TestNewSourceDataset(t *testing.T) {
	source := NewSliceSource(sourceSamples(t))
	dataset, err := NewSourceDataset(context.Background(), source, 10)
	if err != nil {
		t.Fatalf("NewSourceDataset: %v", err)
	}
	if dataset.Len() != 3 {
		t.Fatalf("Len = %d, diharapkan 3", dataset.Len())
	}
	for index := int64(0); index < dataset.Len(); index++ {
		sample, err := dataset.At(context.Background(), index)
		if err != nil {
			t.Fatalf("At(%d): %v", index, err)
		}
		value, _ := sample.Value("id")
		if value != index {
			t.Fatalf("id = %v, diharapkan %d", value, index)
		}
	}
	if _, err := dataset.At(context.Background(), dataset.Len()); !errors.Is(err, errs.ErrOutOfBounds) {
		t.Fatalf("At di luar batas = %v", err)
	}
	if err := dataset.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := dataset.Close(); err != nil {
		t.Fatalf("Close kedua: %v", err)
	}
	if _, err := dataset.At(context.Background(), 0); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("At setelah Close = %v", err)
	}
}

func TestNewSourceDatasetLimit(t *testing.T) {
	dataset, err := NewSourceDataset(context.Background(), NewSliceSource(sourceSamples(t)), 2)
	if err != nil {
		t.Fatalf("NewSourceDataset: %v", err)
	}
	if dataset.Len() != 2 {
		t.Fatalf("Len = %d, diharapkan 2", dataset.Len())
	}
	defer dataset.Close()
	batcher, err := NewBatcher(2, KeepLast, DefaultCollate)
	if err != nil {
		t.Fatalf("NewBatcher: %v", err)
	}
	sampler, err := NewSequentialSampler(dataset.Len())
	if err != nil {
		t.Fatalf("NewSequentialSampler: %v", err)
	}
	batch, err := batcher.Batch(context.Background(), dataset, sampler, 0)
	if err != nil {
		t.Fatalf("Batch: %v", err)
	}
	if want := []int64{0, 1}; !equalInt64(batch.Indices(), want) {
		t.Fatalf("indeks = %v, diharapkan %v", batch.Indices(), want)
	}
}

type failingSource struct {
	err error
}

func (s failingSource) Next(ctx context.Context) (Sample, error) {
	if err := checkContext(ctx); err != nil {
		return Sample{}, err
	}
	return Sample{}, s.err
}

func (s failingSource) Close() error { return nil }

func TestNewSourceDatasetValidation(t *testing.T) {
	var nilSource *SliceSource
	if _, err := NewSourceDataset(nil, NewSliceSource(nil), 1); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("context nil = %v", err)
	}
	if _, err := NewSourceDataset(context.Background(), nilSource, 1); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("source typed nil = %v", err)
	}
	if _, err := NewSourceDataset(context.Background(), NewSliceSource(nil), 0); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("batas nol = %v", err)
	}
	if maxInt := int64(^uint(0) >> 1); maxInt < math.MaxInt64 {
		if _, err := NewSourceDataset(context.Background(), NewSliceSource(nil), maxInt+1); !errors.Is(err, errs.ErrShapeOverflow) {
			t.Fatalf("batas di luar int = %v", err)
		}
	}
	if _, err := NewSourceDataset(context.Background(), failingSource{err: errs.ErrUnsupportedOperation}, 1); !errors.Is(err, errs.ErrUnsupportedOperation) {
		t.Fatalf("error source tidak terbawa: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := NewSourceDataset(ctx, NewSliceSource(sourceSamples(t)), 1); !errors.Is(err, context.Canceled) {
		t.Fatalf("context dibatalkan = %v", err)
	}
}

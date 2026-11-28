package data

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"

	"github.com/adhitiad/gotorch/src/errs"
)

type sliceDataset struct {
	mu      sync.RWMutex
	samples []Sample
	closed  bool
}

func (d *sliceDataset) Len() int64 {
	return int64(len(d.samples))
}

func (d *sliceDataset) At(ctx context.Context, index int64) (Sample, error) {
	if err := checkContext(ctx); err != nil {
		return Sample{}, err
	}
	d.mu.RLock()
	defer d.mu.RUnlock()
	if d.closed {
		return Sample{}, fmt.Errorf("%w: dataset closed", errs.ErrInvalidState)
	}
	if err := checkPosition(int64(len(d.samples)), index, "dataset index"); err != nil {
		return Sample{}, err
	}
	return d.samples[index].Clone(), nil
}

func (d *sliceDataset) Close() error {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.closed = true
	return nil
}

type fixedSampler struct {
	indices []int64
}

func (s fixedSampler) Len() int64 {
	return int64(len(s.indices))
}

func (s fixedSampler) Index(ctx context.Context, position int64) (int64, error) {
	if err := checkContext(ctx); err != nil {
		return 0, err
	}
	if err := checkPosition(int64(len(s.indices)), position, "sampler position"); err != nil {
		return 0, err
	}
	return s.indices[position], nil
}

func labeledSamples(t *testing.T) []Sample {
	t.Helper()
	samples := make([]Sample, 3)
	for index := range samples {
		sample, err := NewSample(map[string]any{"id": int64(index), "text": fmt.Sprintf("record-%d", index)})
		if err != nil {
			t.Fatalf("NewSample(%d): %v", index, err)
		}
		samples[index] = sample
	}
	return samples
}

func TestDatasetContractValidation(t *testing.T) {
	dataset := &sliceDataset{samples: labeledSamples(t)}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	cases := []struct {
		name    string
		ctx     context.Context
		index   int64
		wantErr error
	}{
		{name: "nil context", ctx: nil, index: 0, wantErr: errs.ErrInvalidArgument},
		{name: "canceled context", ctx: canceled, index: 0, wantErr: context.Canceled},
		{name: "negative index", ctx: context.Background(), index: -1, wantErr: errs.ErrOutOfBounds},
		{name: "index at length", ctx: context.Background(), index: dataset.Len(), wantErr: errs.ErrOutOfBounds},
	}
	for _, item := range cases {
		t.Run(item.name, func(t *testing.T) {
			if _, err := dataset.At(item.ctx, item.index); !errors.Is(err, item.wantErr) {
				t.Fatalf("At error = %v, diharapkan %v", err, item.wantErr)
			}
		})
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

func TestSamplerContractValidation(t *testing.T) {
	sampler := fixedSampler{indices: []int64{2, 0, 1, 2}}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if sampler.Len() != 4 {
		t.Fatalf("Len = %d, diharapkan 4", sampler.Len())
	}
	for position, want := range sampler.indices {
		index, err := sampler.Index(context.Background(), int64(position))
		if err != nil {
			t.Fatalf("Index(%d): %v", position, err)
		}
		if index != want {
			t.Fatalf("Index(%d) = %d, diharapkan %d", position, index, want)
		}
	}
	cases := []struct {
		name     string
		ctx      context.Context
		position int64
		wantErr  error
	}{
		{name: "nil context", ctx: nil, position: 0, wantErr: errs.ErrInvalidArgument},
		{name: "canceled context", ctx: canceled, position: 0, wantErr: context.Canceled},
		{name: "negative position", ctx: context.Background(), position: -1, wantErr: errs.ErrOutOfBounds},
		{name: "position at length", ctx: context.Background(), position: sampler.Len(), wantErr: errs.ErrOutOfBounds},
	}
	for _, item := range cases {
		t.Run(item.name, func(t *testing.T) {
			if _, err := sampler.Index(item.ctx, item.position); !errors.Is(err, item.wantErr) {
				t.Fatalf("Index error = %v, diharapkan %v", err, item.wantErr)
			}
		})
	}
}

func TestSamplerIndexIsConcurrent(t *testing.T) {
	sampler := fixedSampler{indices: []int64{3, 1, 2, 0, 2, 3, 1, 0}}
	var group sync.WaitGroup
	for round := 0; round < 8; round++ {
		group.Add(1)
		go func() {
			defer group.Done()
			for position := int64(0); position < sampler.Len(); position++ {
				index, err := sampler.Index(context.Background(), position)
				if err != nil {
					t.Errorf("Index(%d): %v", position, err)
					return
				}
				if index != sampler.indices[position] {
					t.Errorf("Index(%d) = %d, diharapkan %d", position, index, sampler.indices[position])
					return
				}
			}
		}()
	}
	group.Wait()
}

func TestDatasetSamplerBatchRoundTrip(t *testing.T) {
	dataset := &sliceDataset{samples: labeledSamples(t)}
	sampler := fixedSampler{indices: []int64{2, 0, 1}}
	samples := make([]Sample, 0, sampler.Len())
	for position := int64(0); position < sampler.Len(); position++ {
		index, err := sampler.Index(context.Background(), position)
		if err != nil {
			t.Fatalf("Index(%d): %v", position, err)
		}
		sample, err := dataset.At(context.Background(), index)
		if err != nil {
			t.Fatalf("At(%d): %v", index, err)
		}
		samples = append(samples, sample)
	}
	batch, err := NewBatch(samples, sampler.indices)
	if err != nil {
		t.Fatalf("NewBatch: %v", err)
	}
	if batch.Len() != len(samples) {
		t.Fatalf("Len = %d, diharapkan %d", batch.Len(), len(samples))
	}
	for position, wantIndex := range sampler.indices {
		sample, index, err := batch.At(position)
		if err != nil {
			t.Fatalf("At(%d): %v", position, err)
		}
		if index != wantIndex {
			t.Fatalf("source index = %d, diharapkan %d", index, wantIndex)
		}
		value, ok := sample.Value("id")
		if !ok || value != wantIndex {
			t.Fatalf("id = %v, diharapkan %d", value, wantIndex)
		}
	}
}

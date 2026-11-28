package data

import (
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/adhitiad/gotorch/src/errs"
)

func TestSequentialSamplerIndexes(t *testing.T) {
	sampler, err := NewSequentialSampler(4)
	if err != nil {
		t.Fatalf("NewSequentialSampler: %v", err)
	}
	if sampler.Len() != 4 {
		t.Fatalf("Len = %d, diharapkan 4", sampler.Len())
	}
	for position := int64(0); position < sampler.Len(); position++ {
		index, err := sampler.Index(context.Background(), position)
		if err != nil {
			t.Fatalf("Index(%d): %v", position, err)
		}
		if index != position {
			t.Fatalf("Index(%d) = %d", position, index)
		}
	}
}

func TestSequentialSamplerValidation(t *testing.T) {
	if _, err := NewSequentialSampler(-1); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("panjang negatif diterima: %v", err)
	}
	sampler, err := NewSequentialSampler(0)
	if err != nil {
		t.Fatalf("NewSequentialSampler(0): %v", err)
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	cases := []struct {
		name     string
		ctx      context.Context
		position int64
		wantErr  error
	}{
		{name: "nil context", ctx: nil, position: 0, wantErr: errs.ErrInvalidArgument},
		{name: "canceled context", ctx: canceled, position: 0, wantErr: context.Canceled},
		{name: "negative position", ctx: context.Background(), position: -1, wantErr: errs.ErrOutOfBounds},
		{name: "empty sampler", ctx: context.Background(), position: 0, wantErr: errs.ErrOutOfBounds},
	}
	for _, item := range cases {
		t.Run(item.name, func(t *testing.T) {
			if _, err := sampler.Index(item.ctx, item.position); !errors.Is(err, item.wantErr) {
				t.Fatalf("error = %v, diharapkan %v", err, item.wantErr)
			}
		})
	}
	var nilSampler *SequentialSampler
	if nilSampler.Len() != 0 {
		t.Fatalf("nil Len = %d", nilSampler.Len())
	}
	if _, err := nilSampler.Index(context.Background(), 0); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("nil Index = %v", err)
	}
}

func TestSequentialSamplerConcurrent(t *testing.T) {
	sampler, err := NewSequentialSampler(8)
	if err != nil {
		t.Fatalf("NewSequentialSampler: %v", err)
	}
	var group sync.WaitGroup
	for worker := 0; worker < 8; worker++ {
		group.Add(1)
		go func() {
			defer group.Done()
			for position := int64(0); position < sampler.Len(); position++ {
				index, err := sampler.Index(context.Background(), position)
				if err != nil {
					t.Errorf("Index(%d): %v", position, err)
					return
				}
				if index != position {
					t.Errorf("Index(%d) = %d", position, index)
					return
				}
			}
		}()
	}
	group.Wait()
}

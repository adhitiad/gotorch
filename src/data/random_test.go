package data

import (
	"context"
	"errors"
	"math"
	"reflect"
	"sort"
	"sync"
	"testing"

	"github.com/adhitiad/gotorch/src/errs"
)

func samplerIndices(t *testing.T, sampler Sampler) []int64 {
	t.Helper()
	indices := make([]int64, 0, sampler.Len())
	for position := int64(0); position < sampler.Len(); position++ {
		index, err := sampler.Index(context.Background(), position)
		if err != nil {
			t.Fatalf("Index(%d): %v", position, err)
		}
		indices = append(indices, index)
	}
	return indices
}

func TestRandomSamplerDeterministicPermutation(t *testing.T) {
	first, err := NewRandomSampler(16, 7)
	if err != nil {
		t.Fatalf("NewRandomSampler(seed 7): %v", err)
	}
	second, err := NewRandomSampler(16, 7)
	if err != nil {
		t.Fatalf("NewRandomSampler(seed 7 ulang): %v", err)
	}
	if first.Seed() != 7 || second.Seed() != 7 {
		t.Fatalf("Seed = %d/%d, diharapkan 7", first.Seed(), second.Seed())
	}
	got := samplerIndices(t, first)
	want := samplerIndices(t, second)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("seed sama menghasilkan %v dan %v", got, want)
	}
	sorted := append([]int64(nil), got...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	for index, value := range sorted {
		if value != int64(index) {
			t.Fatalf("bukan permutasi: %v", got)
		}
	}
}

func TestRandomSamplerSeedChangesOrder(t *testing.T) {
	first, err := NewRandomSampler(16, 1)
	if err != nil {
		t.Fatalf("NewRandomSampler(seed 1): %v", err)
	}
	second, err := NewRandomSampler(16, 2)
	if err != nil {
		t.Fatalf("NewRandomSampler(seed 2): %v", err)
	}
	if reflect.DeepEqual(samplerIndices(t, first), samplerIndices(t, second)) {
		t.Fatal("seed berbeda menghasilkan urutan sama")
	}
}

func TestRandomSamplerValidation(t *testing.T) {
	if _, err := NewRandomSampler(-1, 1); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("panjang negatif diterima: %v", err)
	}
	if maxInt := int64(^uint(0) >> 1); maxInt < math.MaxInt64 {
		if _, err := NewRandomSampler(maxInt+1, 1); !errors.Is(err, errs.ErrShapeOverflow) {
			t.Fatalf("panjang di luar jangkauan int diterima: %v", err)
		}
	}
	sampler, err := NewRandomSampler(0, 3)
	if err != nil {
		t.Fatalf("NewRandomSampler(0, 3): %v", err)
	}
	if sampler.Len() != 0 || sampler.Seed() != 3 {
		t.Fatalf("Len/Seed = %d/%d", sampler.Len(), sampler.Seed())
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
	var nilSampler *RandomSampler
	if nilSampler.Len() != 0 || nilSampler.Seed() != 0 {
		t.Fatal("nil sampler tidak bersifat kosong")
	}
	if _, err := nilSampler.Index(context.Background(), 0); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("nil Index = %v", err)
	}
}

func TestRandomSamplerConcurrent(t *testing.T) {
	sampler, err := NewRandomSampler(12, 42)
	if err != nil {
		t.Fatalf("NewRandomSampler: %v", err)
	}
	want := samplerIndices(t, sampler)
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
				if index != want[position] {
					t.Errorf("Index(%d) = %d, diharapkan %d", position, index, want[position])
					return
				}
			}
		}()
	}
	group.Wait()
}

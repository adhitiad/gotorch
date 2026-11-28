package data

import (
	"errors"
	"testing"

	"github.com/adhitiad/gotorch/src/errs"
)

func testBatch(t *testing.T) (*Batch, []Sample, []int64) {
	t.Helper()
	all := labeledSamples(t)
	samples := []Sample{all[2], all[0], all[1]}
	indices := []int64{2, 0, 1}
	batch, err := NewBatch(samples, indices)
	if err != nil {
		t.Fatalf("NewBatch: %v", err)
	}
	return batch, samples, indices
}

func TestBatchCopiesInputsAndIsImmutable(t *testing.T) {
	batch, samples, indices := testBatch(t)
	samples[0] = Sample{}
	indices[0] = 99
	if got := batch.Indices(); len(got) != 3 || got[0] != 2 {
		t.Fatalf("Indices = %v", got)
	}
	returned := batch.Samples()
	if returned[0].Len() == 0 {
		t.Fatal("Samples mengikuti slice masukan")
	}
	returned[0] = Sample{}
	if batch.Samples()[0].Len() == 0 {
		t.Fatal("Samples mengubah batch")
	}
	got := batch.Indices()
	got[0] = 42
	if batch.Indices()[0] != 2 {
		t.Fatal("Indices mengubah batch")
	}
}

func TestBatchValidation(t *testing.T) {
	samples := labeledSamples(t)
	cases := []struct {
		name    string
		samples []Sample
		indices []int64
		wantErr error
	}{
		{name: "empty", samples: nil, indices: nil, wantErr: errs.ErrInvalidArgument},
		{name: "length mismatch", samples: samples, indices: []int64{1}, wantErr: errs.ErrInvalidArgument},
		{name: "negative index", samples: samples[:1], indices: []int64{-1}, wantErr: errs.ErrOutOfBounds},
	}
	for _, item := range cases {
		t.Run(item.name, func(t *testing.T) {
			if _, err := NewBatch(item.samples, item.indices); !errors.Is(err, item.wantErr) {
				t.Fatalf("error = %v, diharapkan %v", err, item.wantErr)
			}
		})
	}
}

func TestBatchAtBoundsAndNil(t *testing.T) {
	batch, _, _ := testBatch(t)
	if _, _, err := batch.At(-1); !errors.Is(err, errs.ErrOutOfBounds) {
		t.Fatalf("At(-1) = %v", err)
	}
	if _, _, err := batch.At(batch.Len()); !errors.Is(err, errs.ErrOutOfBounds) {
		t.Fatalf("At(Len) = %v", err)
	}
	var nilBatch *Batch
	if nilBatch.Len() != 0 || len(nilBatch.Samples()) != 0 || len(nilBatch.Indices()) != 0 {
		t.Fatal("nil batch tidak bersifat kosong")
	}
	if _, _, err := nilBatch.At(0); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("nil At = %v", err)
	}

	sample, index, err := batch.At(0)
	if err != nil {
		t.Fatalf("At(0): %v", err)
	}
	value, ok := sample.Value("id")
	if !ok || value != int64(2) {
		t.Fatalf("id = %v, diharapkan 2", value)
	}
	if index != 2 {
		t.Fatalf("index = %d, diharapkan 2", index)
	}
}

package data

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"

	"github.com/adhitiad/gotorch/src/errs"
)

type appendTransform struct {
	token string
}

func (t appendTransform) Apply(ctx context.Context, sample Sample) (Sample, error) {
	if err := checkContext(ctx); err != nil {
		return Sample{}, err
	}
	value, _ := sample.Value("trace")
	fields := map[string]any{"trace": fmt.Sprintf("%v%s", value, t.token)}
	for _, name := range sample.Names() {
		if name == "trace" {
			continue
		}
		fieldValue, _ := sample.Value(name)
		fields[name] = fieldValue
	}
	return NewSample(fields)
}

type failingTransform struct {
	err error
}

func (t failingTransform) Apply(ctx context.Context, sample Sample) (Sample, error) {
	if err := checkContext(ctx); err != nil {
		return Sample{}, err
	}
	return Sample{}, t.err
}

func traceSample(t *testing.T) Sample {
	t.Helper()
	sample, err := NewSample(map[string]any{"trace": "", "id": int64(1)})
	if err != nil {
		t.Fatalf("NewSample: %v", err)
	}
	return sample
}

func TestTransformChainOrder(t *testing.T) {
	chain, err := NewTransformChain(appendTransform{token: "a"}, appendTransform{token: "b"}, appendTransform{token: "c"})
	if err != nil {
		t.Fatalf("NewTransformChain: %v", err)
	}
	if chain.Len() != 3 {
		t.Fatalf("Len = %d, diharapkan 3", chain.Len())
	}
	sample := traceSample(t)
	result, err := chain.Apply(context.Background(), sample)
	if err != nil {
		t.Fatalf("Apply: %v", err)
	}
	if value, _ := result.Value("trace"); value != "abc" {
		t.Fatalf("trace = %v, diharapkan abc", value)
	}
	if value, _ := sample.Value("trace"); value != "" {
		t.Fatalf("sample masukan berubah: %v", value)
	}
	if value, _ := result.Value("id"); value != int64(1) {
		t.Fatalf("field lain hilang: %v", value)
	}
}

func TestTransformChainValidation(t *testing.T) {
	if _, err := NewTransformChain(nil); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("transform nil diterima: %v", err)
	}
	var typedNil *appendTransform
	if _, err := NewTransformChain(typedNil); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("transform typed nil diterima: %v", err)
	}
	var nilChain *TransformChain
	if nilChain.Len() != 0 {
		t.Fatalf("nil Len = %d", nilChain.Len())
	}
	if _, err := nilChain.Apply(context.Background(), traceSample(t)); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("nil Apply = %v", err)
	}

	empty, err := NewTransformChain()
	if err != nil {
		t.Fatalf("NewTransformChain(): %v", err)
	}
	result, err := empty.Apply(context.Background(), traceSample(t))
	if err != nil {
		t.Fatalf("empty Apply: %v", err)
	}
	if value, _ := result.Value("id"); value != int64(1) {
		t.Fatalf("rantai kosong mengubah sample: %v", value)
	}

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := empty.Apply(nil, traceSample(t)); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("context nil = %v", err)
	}
	if _, err := empty.Apply(canceled, traceSample(t)); !errors.Is(err, context.Canceled) {
		t.Fatalf("context dibatalkan = %v", err)
	}

	failing, err := NewTransformChain(appendTransform{token: "a"}, failingTransform{err: errs.ErrUnsupportedOperation}, appendTransform{token: "b"})
	if err != nil {
		t.Fatalf("NewTransformChain(failing): %v", err)
	}
	if _, err := failing.Apply(context.Background(), traceSample(t)); !errors.Is(err, errs.ErrUnsupportedOperation) {
		t.Fatalf("error transform tidak terbawa: %v", err)
	}

	ctx, stop := context.WithCancel(context.Background())
	defer stop()
	cancelMidway := TransformFunc(func(ctx context.Context, sample Sample) (Sample, error) {
		stop()
		return sample.Clone(), nil
	})
	chain, err := NewTransformChain(cancelMidway, appendTransform{token: "b"})
	if err != nil {
		t.Fatalf("NewTransformChain(cancel): %v", err)
	}
	if _, err := chain.Apply(ctx, traceSample(t)); !errors.Is(err, context.Canceled) {
		t.Fatalf("pembatalan di tengah = %v", err)
	}
}

func TestTransformFuncAndRequireFields(t *testing.T) {
	var nilFunc TransformFunc
	if _, err := nilFunc.Apply(context.Background(), Sample{}); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("TransformFunc nil = %v", err)
	}
	if _, err := RequireFields(); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("RequireFields tanpa nama diterima: %v", err)
	}
	if _, err := RequireFields("id", ""); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("nama kosong diterima: %v", err)
	}
	required, err := RequireFields("id", "trace")
	if err != nil {
		t.Fatalf("RequireFields: %v", err)
	}
	valid, err := required.Apply(context.Background(), traceSample(t))
	if err != nil {
		t.Fatalf("Apply(valid): %v", err)
	}
	if value, _ := valid.Value("id"); value != int64(1) {
		t.Fatalf("id = %v", value)
	}
	if _, err := required.Apply(context.Background(), Sample{}); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("field hilang diterima: %v", err)
	}
	if _, err := required.Apply(nil, traceSample(t)); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("context nil = %v", err)
	}
}

func TestTransformChainConcurrent(t *testing.T) {
	chain, err := NewTransformChain(appendTransform{token: "x"}, appendTransform{token: "y"}, appendTransform{token: "z"})
	if err != nil {
		t.Fatalf("NewTransformChain: %v", err)
	}
	want, err := chain.Apply(context.Background(), traceSample(t))
	if err != nil {
		t.Fatalf("Apply awal: %v", err)
	}
	var group sync.WaitGroup
	for worker := 0; worker < 8; worker++ {
		group.Add(1)
		go func() {
			defer group.Done()
			for round := 0; round < 16; round++ {
				got, err := chain.Apply(context.Background(), traceSample(t))
				if err != nil {
					t.Errorf("Apply: %v", err)
					return
				}
				value, _ := got.Value("trace")
				wantValue, _ := want.Value("trace")
				if value != wantValue {
					t.Errorf("trace = %v, diharapkan %v", value, wantValue)
					return
				}
			}
		}()
	}
	group.Wait()
}

func TestCollateWithTransform(t *testing.T) {
	tag := TransformFunc(func(ctx context.Context, sample Sample) (Sample, error) {
		if err := checkContext(ctx); err != nil {
			return Sample{}, err
		}
		fields := map[string]any{"tag": "yes"}
		for _, name := range sample.Names() {
			value, _ := sample.Value(name)
			fields[name] = value
		}
		return NewSample(fields)
	})
	var seenIndices []int64
	base := func(ctx context.Context, samples []Sample, indices []int64) (*Batch, error) {
		if err := checkContext(ctx); err != nil {
			return nil, err
		}
		seenIndices = append([]int64(nil), indices...)
		indices[0] = 99
		return NewBatch(samples, indices)
	}
	collate, err := CollateWithTransform(tag, base)
	if err != nil {
		t.Fatalf("CollateWithTransform: %v", err)
	}
	samples := []Sample{traceSample(t), traceSample(t)}
	indices := []int64{2, 3}
	batch, err := collate(context.Background(), samples, indices)
	if err != nil {
		t.Fatalf("collate: %v", err)
	}
	if !equalInt64(seenIndices, []int64{2, 3}) || !equalInt64(indices, []int64{2, 3}) {
		t.Fatalf("indeks collate = %v, asal = %v", seenIndices, indices)
	}
	sample, index, err := batch.At(0)
	if err != nil {
		t.Fatalf("At(0): %v", err)
	}
	if index != 99 {
		t.Fatalf("indeks batch = %d", index)
	}
	if value, ok := sample.Value("tag"); !ok || value != "yes" {
		t.Fatalf("tag = %v, %v", value, ok)
	}
	if _, ok := samples[0].Value("tag"); ok {
		t.Fatal("sample masukan berubah")
	}
}

func TestCollateWithTransformValidation(t *testing.T) {
	base := DefaultCollate
	var typedNil *appendTransform
	if _, err := CollateWithTransform(nil, base); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("transform nil diterima: %v", err)
	}
	if _, err := CollateWithTransform(typedNil, base); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("transform typed nil diterima: %v", err)
	}
	if _, err := CollateWithTransform(appendTransform{token: "a"}, nil); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("collate nil diterima: %v", err)
	}
	collate, err := CollateWithTransform(failingTransform{err: errs.ErrUnsupportedOperation}, base)
	if err != nil {
		t.Fatalf("CollateWithTransform: %v", err)
	}
	samples := []Sample{traceSample(t)}
	if _, err := collate(context.Background(), samples, []int64{0}); !errors.Is(err, errs.ErrUnsupportedOperation) {
		t.Fatalf("error transform tidak terbawa: %v", err)
	}
	if _, err := collate(nil, samples, []int64{0}); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("context nil = %v", err)
	}
}

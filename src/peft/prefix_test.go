package peft

import (
	"context"
	"errors"
	"math/rand"
	"testing"

	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/errs"
)

func newPrefixTuning(t *testing.T, seed int64) *PrefixTuning {
	t.Helper()
	prefix, err := NewPrefixTuning("prefix", 3, PrefixTuningConfig{
		PrefixLength: 2,
		DType:        dtype.Float64,
		RNG:          rand.New(rand.NewSource(seed)),
	})
	if err != nil {
		t.Fatalf("NewPrefixTuning: %v", err)
	}
	t.Cleanup(func() {
		if err := prefix.Close(); err != nil {
			t.Errorf("prefix.Close: %v", err)
		}
	})
	return prefix
}

func TestPrefixTuningParametersAndForward(t *testing.T) {
	prefix := newPrefixTuning(t, 5)
	setParameterValues(t, prefix.KeyPrefix, []float64{1, 2, 3, 4, 5, 6})
	setParameterValues(t, prefix.ValuePrefix, []float64{7, 8, 9, 10, 11, 12})
	named := prefix.Parameters()
	if len(named) != 2 || named[0].Name != "prefix.key" || named[1].Name != "prefix.value" {
		t.Fatalf("Parameters: %+v", named)
	}
	for _, item := range named {
		if !item.Parameter.RequiresGrad() {
			t.Fatalf("%s tidak trainable", item.Name)
		}
		metadata, err := item.Parameter.Value().Metadata()
		if err != nil {
			t.Fatalf("Metadata: %v", err)
		}
		if dims := metadata.Shape().Dims(); len(dims) != 2 || dims[0] != 2 || dims[1] != 3 {
			t.Fatalf("%s dims = %v", item.Name, dims)
		}
	}

	tape, ctx := newTestTape(t)
	key := newVariable(t, tape, newFloatTensor(t, []int64{1, 2, 3}, []float64{10, 11, 12, 13, 14, 15}), false)
	value := newVariable(t, tape, newFloatTensor(t, []int64{1, 2, 3}, []float64{20, 21, 22, 23, 24, 25}), false)
	extendedKey, extendedValue, err := prefix.Forward(ctx, key, value)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	keyTensor, err := extendedKey.Value()
	if err != nil {
		t.Fatalf("key Value: %v", err)
	}
	assertClose(t, "prefix key", floatValues(t, keyTensor), []float64{1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15}, 1e-12)
	if err := keyTensor.Close(); err != nil {
		t.Fatalf("keyTensor.Close: %v", err)
	}
	valueTensor, err := extendedValue.Value()
	if err != nil {
		t.Fatalf("value Value: %v", err)
	}
	assertClose(t, "prefix value", floatValues(t, valueTensor), []float64{7, 8, 9, 10, 11, 12, 20, 21, 22, 23, 24, 25}, 1e-12)
	if err := valueTensor.Close(); err != nil {
		t.Fatalf("valueTensor.Close: %v", err)
	}

	keyTotal, err := tape.Sum(extendedKey)
	if err != nil {
		t.Fatalf("Sum(key): %v", err)
	}
	valueTotal, err := tape.Sum(extendedValue)
	if err != nil {
		t.Fatalf("Sum(value): %v", err)
	}
	total, err := tape.Add(keyTotal, valueTotal)
	if err != nil {
		t.Fatalf("Add: %v", err)
	}
	if err := tape.Backward(context.Background(), total); err != nil {
		t.Fatalf("Backward: %v", err)
	}
	want := make([]float64, 6)
	for index := range want {
		want[index] = 1
	}
	keyGradient, err := gradientValues(t, ctx, prefix.KeyPrefix)
	if err != nil {
		t.Fatalf("key gradient: %v", err)
	}
	assertClose(t, "key gradient", keyGradient, want, 1e-12)
	valueGradient, err := gradientValues(t, ctx, prefix.ValuePrefix)
	if err != nil {
		t.Fatalf("value gradient: %v", err)
	}
	assertClose(t, "value gradient", valueGradient, want, 1e-12)
}

func TestPrefixTuningDeterministicInitialization(t *testing.T) {
	first := newPrefixTuning(t, 17)
	second := newPrefixTuning(t, 17)
	assertClose(t, "key init", parameterValues(t, first.KeyPrefix), parameterValues(t, second.KeyPrefix), 0)
	assertClose(t, "value init", parameterValues(t, first.ValuePrefix), parameterValues(t, second.ValuePrefix), 0)
}

func TestPrefixTuningValidation(t *testing.T) {
	if _, err := NewPrefixTuning("prefix", 0, PrefixTuningConfig{PrefixLength: 1, RNG: rand.New(rand.NewSource(1))}); err == nil {
		t.Fatal("dimensi nol diterima")
	}
	if _, err := NewPrefixTuning("prefix", 3, PrefixTuningConfig{PrefixLength: 0, RNG: rand.New(rand.NewSource(1))}); err == nil {
		t.Fatal("panjang prefix nol diterima")
	}
	if _, err := NewPrefixTuning("prefix", 3, PrefixTuningConfig{PrefixLength: 1}); err == nil {
		t.Fatal("RNG nil diterima")
	}
	if _, err := NewPrefixTuning("prefix", 3, PrefixTuningConfig{PrefixLength: 1, DType: dtype.Int64, RNG: rand.New(rand.NewSource(1))}); !errors.Is(err, errs.ErrUnsupportedDType) {
		t.Fatalf("dtype bukan float diterima: %v", err)
	}

	prefix := newPrefixTuning(t, 3)
	tape, ctx := newTestTape(t)
	key := newVariable(t, tape, newFloatTensor(t, []int64{2, 3}, make([]float64, 6)), false)
	value := newVariable(t, tape, newFloatTensor(t, []int64{1, 2, 3}, make([]float64, 6)), false)
	if _, _, err := prefix.Forward(ctx, key, value); !errors.Is(err, errs.ErrInvalidShape) {
		t.Fatalf("key rank dua diterima: %v", err)
	}
	wrong := newVariable(t, tape, newFloatTensor(t, []int64{1, 2, 4}, make([]float64, 8)), false)
	if _, _, err := prefix.Forward(ctx, wrong, value); !errors.Is(err, errs.ErrInvalidShape) {
		t.Fatalf("dimensi key salah diterima: %v", err)
	}
	if _, _, err := prefix.Forward(ctx, key, value); err == nil {
		t.Fatal("key dan value dengan bentuk berbeda diterima")
	}
	if _, _, err := prefix.Forward(ctx, nil, value); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("key nil diterima: %v", err)
	}
}

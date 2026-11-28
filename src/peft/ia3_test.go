package peft

import (
	"math/rand"
	"testing"

	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/nn"
)

func newIA3(t *testing.T, name string, base Projector, config IA3Config) *IA3Linear {
	t.Helper()
	adapter, err := NewIA3Linear(name, base, config)
	if err != nil {
		t.Fatalf("NewIA3Linear(%q): %v", name, err)
	}
	t.Cleanup(func() {
		if err := adapter.Close(); err != nil {
			t.Errorf("adapter.Close(%q): %v", name, err)
		}
	})
	return adapter
}

func TestIA3ForwardAndScaleGradient(t *testing.T) {
	base := newLinearWithWeight(t, "base", 2, 2, []float64{0.5, -0.25, 1.5, 0.75})
	if err := Freeze(base); err != nil {
		t.Fatalf("Freeze(base): %v", err)
	}
	adapter := newIA3(t, "ia3", base, IA3Config{DType: dtype.Float64, RNG: rand.New(rand.NewSource(9))})
	assertClose(t, "skala awal", parameterValues(t, adapter.Scale), []float64{1, 1}, 0)

	_, ctx := newTestTape(t)
	tape := ctx.Tape
	input := newVariable(t, tape, newFloatTensor(t, []int64{2, 2}, []float64{1, 2, 3, 4}), false)
	output, err := adapter.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	value, err := output.Value()
	if err != nil {
		t.Fatalf("output.Value: %v", err)
	}
	assertClose(t, "forward awal IA3", floatValues(t, value), []float64{0, 3, 0.5, 7.5}, 1e-12)
	_ = value.Close()
	backwardSum(t, tape, output)
	gradient, err := gradientValues(t, ctx, adapter.Scale)
	if err != nil {
		t.Fatalf("gradien scale: %v", err)
	}
	assertClose(t, "gradien scale", gradient, []float64{0.5, 10.5}, 1e-12)
	assertNoGradient(t, ctx, base.Weight)

	setParameterValues(t, adapter.Scale, []float64{0.5, -2})
	_, ctx2 := newTestTape(t)
	tape2 := ctx2.Tape
	input2 := newVariable(t, tape2, newFloatTensor(t, []int64{2, 2}, []float64{1, 2, 3, 4}), false)
	output2, err := adapter.Forward(ctx2, input2)
	if err != nil {
		t.Fatalf("Forward kedua: %v", err)
	}
	value2, err := output2.Value()
	if err != nil {
		t.Fatalf("output.Value kedua: %v", err)
	}
	assertClose(t, "forward IA3 dengan skala kustom", floatValues(t, value2), []float64{0, -6, 0.25, -15}, 1e-12)
	_ = value2.Close()
}

func TestIA3ConstructionValidation(t *testing.T) {
	base := newLinearWithWeight(t, "base", 2, 3, []float64{1, 0, 0, 1, 0.5, -0.5})
	adapter := newIA3(t, "ia3", base, IA3Config{DType: dtype.Float64, Init: 0.25, RNG: rand.New(rand.NewSource(9))})
	assertClose(t, "skala inisial", parameterValues(t, adapter.Scale), []float64{0.25, 0.25, 0.25}, 0)
	if adapter.InputDim() != 2 || adapter.OutputDim() != 3 {
		t.Fatalf("dimensi = (%d, %d), diharapkan (2, 3)", adapter.InputDim(), adapter.OutputDim())
	}
	if !adapter.Scale.RequiresGrad() {
		t.Fatal("skala IA3 seharusnya trainable")
	}

	config := IA3Config{DType: dtype.Float64, RNG: rand.New(rand.NewSource(9))}
	_, err := NewIA3Linear("ia3_nil", nil, config)
	assertErrorIs(t, err, errs.ErrInvalidArgument)
	_, err = NewIA3Linear("ia3_dtype", base, IA3Config{DType: dtype.Int64, RNG: rand.New(rand.NewSource(9))})
	assertErrorIs(t, err, errs.ErrUnsupportedDType)
	opaque := newOpaqueProjector(t, "kosong", 0, 0)
	_, err = NewIA3Linear("ia3_kosong", opaque, config)
	assertErrorIs(t, err, errs.ErrInvalidArgument)
	_, err = NewIA3Linear("", base, config)
	assertErrorIs(t, err, errs.ErrInvalidArgument)
}

var _ nn.Module = (*IA3Linear)(nil)

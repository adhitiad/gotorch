package optim

import (
	"errors"
	"testing"

	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/nn"
	"github.com/adhitiad/gotorch/src/tensor"
)

func TestSGDPlainUpdate(t *testing.T) {
	parameter := testParam(t, "weight", []float64{1, -2}, 2)
	defer parameter.Parameter.Close()
	optimizer, err := NewSGD(SGDConfig{LearningRate: 0.1})
	if err != nil {
		t.Fatalf("NewSGD: %v", err)
	}
	defer optimizer.Close()

	ctx, release := newGradientContext(t)
	applyGradientUsingMul(t, ctx, &parameter, []float64{0.5, -0.25})
	if err := optimizer.Step(ctx, []Param{parameter}); err != nil {
		t.Fatalf("Step: %v", err)
	}
	// θ = θ - lr·g, tanpa momentum dan tanpa weight decay.
	assertClose(t, "langkah pertama", valuesOf(t, &parameter), []float64{0.95, -1.975}, 1e-12)
	release()

	ctx, release = newGradientContext(t)
	applyGradientUsingMul(t, ctx, &parameter, []float64{0.2, 0.4})
	if err := optimizer.Step(ctx, []Param{parameter}); err != nil {
		t.Fatalf("Step kedua: %v", err)
	}
	assertClose(t, "langkah kedua", valuesOf(t, &parameter), []float64{0.93, -2.015}, 1e-12)
	release()
}

func TestSGDMomentumAccumulation(t *testing.T) {
	parameter := testParam(t, "weight", []float64{1, 2}, 2)
	defer parameter.Parameter.Close()
	optimizer, err := NewSGD(SGDConfig{LearningRate: 0.1, Momentum: 0.8})
	if err != nil {
		t.Fatalf("NewSGD: %v", err)
	}
	defer optimizer.Close()

	for index, gradient := range [][]float64{{1, -0.5}, {0.5, 0.25}} {
		ctx, release := newGradientContext(t)
		applyGradientUsingMul(t, ctx, &parameter, gradient)
		if err := optimizer.Step(ctx, []Param{parameter}); err != nil {
			t.Fatalf("Step %d: %v", index+1, err)
		}
		release()
	}
	// v1 = g1 = [1, -0,5]; v2 = 0,8·v1 + g2 = [1,3, -0,15].
	assertClose(t, "parameter setelah momentum", valuesOf(t, &parameter), []float64{0.77, 2.065}, 1e-12)

	state, err := optimizer.MarshalState([]Param{parameter})
	if err != nil {
		t.Fatalf("MarshalState: %v", err)
	}
	decoded, err := decodeState(state)
	if err != nil {
		t.Fatalf("decodeState: %v", err)
	}
	assertClose(t, "velocity tersimpan", decoded["weight"].vectors[0], []float64{1.3, -0.15}, 1e-12)
}

func TestSGDWeightDecayIsCoupled(t *testing.T) {
	parameter := testParam(t, "weight", []float64{2, -3}, 2)
	defer parameter.Parameter.Close()
	optimizer, err := NewSGD(SGDConfig{LearningRate: 0.1, WeightDecay: 0.1})
	if err != nil {
		t.Fatalf("NewSGD: %v", err)
	}
	defer optimizer.Close()
	ctx, release := newGradientContext(t)
	defer release()
	applyGradientUsingMul(t, ctx, &parameter, []float64{0.5, 0.25})

	if err := optimizer.Step(ctx, []Param{parameter}); err != nil {
		t.Fatalf("Step: %v", err)
	}
	// g' = g + wd·θ = [0,7, -0,05].
	assertClose(t, "weight decay terkopel", valuesOf(t, &parameter), []float64{1.93, -2.995}, 1e-12)
}

func TestSGDNesterovMomentum(t *testing.T) {
	parameter := testParam(t, "weight", []float64{1, 1}, 2)
	defer parameter.Parameter.Close()
	optimizer, err := NewSGD(SGDConfig{LearningRate: 0.1, Momentum: 0.9, Nesterov: true})
	if err != nil {
		t.Fatalf("NewSGD: %v", err)
	}
	defer optimizer.Close()

	ctx, release := newGradientContext(t)
	applyGradientUsingMul(t, ctx, &parameter, []float64{1, 1})
	if err := optimizer.Step(ctx, []Param{parameter}); err != nil {
		t.Fatalf("Step pertama: %v", err)
	}
	release()
	// v = g = 1; Δ = (1 + 0,9)·v = 1,9.
	assertClose(t, "Nesterov langkah pertama", valuesOf(t, &parameter), []float64{0.81, 0.81}, 1e-12)

	ctx, release = newGradientContext(t)
	applyGradientUsingMul(t, ctx, &parameter, []float64{0, 0})
	if err := optimizer.Step(ctx, []Param{parameter}); err != nil {
		t.Fatalf("Step kedua: %v", err)
	}
	release()
	// v = 0,9·1 + 0 = 0,9; Δ = 0,9 + 0,9·0,9 = 1,71.
	assertClose(t, "Nesterov langkah kedua", valuesOf(t, &parameter), []float64{0.639, 0.639}, 1e-12)
}

func TestSGDWithoutGradientLeavesParameterUntouched(t *testing.T) {
	withGradient := testParam(t, "a.weight", []float64{1}, 1)
	withoutGradient := testParam(t, "b.weight", []float64{5}, 1)
	defer withGradient.Parameter.Close()
	defer withoutGradient.Parameter.Close()
	optimizer, err := NewSGD(SGDConfig{LearningRate: 0.1, Momentum: 0.9})
	if err != nil {
		t.Fatalf("NewSGD: %v", err)
	}
	defer optimizer.Close()
	ctx, release := newGradientContext(t)
	defer release()
	applyGradientUsingMul(t, ctx, &withGradient, []float64{1})

	parameters := []Param{withoutGradient, withGradient}
	if err := optimizer.Step(ctx, parameters); err != nil {
		t.Fatalf("Step: %v", err)
	}
	assertClose(t, "tanpa gradien", valuesOf(t, &withoutGradient), []float64{5}, 0)
	assertClose(t, "dengan gradien", valuesOf(t, &withGradient), []float64{0.9}, 1e-12)

	state, err := optimizer.MarshalState(parameters)
	if err != nil {
		t.Fatalf("MarshalState: %v", err)
	}
	decoded, err := decodeState(state)
	if err != nil {
		t.Fatalf("decodeState: %v", err)
	}
	assertClose(t, "velocity parameter tanpa gradien", decoded["b.weight"].vectors[0], []float64{0}, 0)
}

func TestSGDStateRoundTripContinuesTrajectory(t *testing.T) {
	left := testParam(t, "weight", []float64{1, -2, 3}, 3)
	right := testParam(t, "weight", []float64{1, -2, 3}, 3)
	defer left.Parameter.Close()
	defer right.Parameter.Close()

	original, err := NewSGD(SGDConfig{LearningRate: 0.05, Momentum: 0.7, WeightDecay: 0.01})
	if err != nil {
		t.Fatalf("NewSGD: %v", err)
	}
	defer original.Close()
	continued, err := NewSGD(SGDConfig{LearningRate: 0.05, Momentum: 0.7, WeightDecay: 0.01})
	if err != nil {
		t.Fatalf("NewSGD(continued): %v", err)
	}
	defer continued.Close()

	gradients := [][]float64{{0.5, -1, 0.25}, {-0.25, 0.75, 1}, {1, 0.5, -0.5}}
	for _, gradient := range gradients {
		for _, item := range []struct {
			optimizer *SGD
			parameter Param
		}{{original, left}, {continued, right}} {
			ctx, release := newGradientContext(t)
			applyGradientUsingMul(t, ctx, &item.parameter, gradient)
			if err := item.optimizer.Step(ctx, []Param{item.parameter}); err != nil {
				t.Fatalf("Step: %v", err)
			}
			release()
		}
	}
	state, err := original.MarshalState([]Param{left})
	if err != nil {
		t.Fatalf("MarshalState: %v", err)
	}
	if err := continued.UnmarshalState([]Param{right}, state); err != nil {
		t.Fatalf("UnmarshalState: %v", err)
	}
	assertClose(t, "nilai setelah restore", valuesOf(t, &right), valuesOf(t, &left), 0)

	for _, gradient := range [][]float64{{0.3, -0.2, 0.4}, {-0.1, 0.6, 0.2}} {
		ctxLeft, releaseLeft := newGradientContext(t)
		applyGradientUsingMul(t, ctxLeft, &left, gradient)
		if err := original.Step(ctxLeft, []Param{left}); err != nil {
			t.Fatalf("Step(original): %v", err)
		}
		releaseLeft()

		ctxRight, releaseRight := newGradientContext(t)
		applyGradientUsingMul(t, ctxRight, &right, gradient)
		if err := continued.Step(ctxRight, []Param{right}); err != nil {
			t.Fatalf("Step(restored): %v", err)
		}
		releaseRight()
	}
	assertClose(t, "lintasan setelah restore", valuesOf(t, &right), valuesOf(t, &left), 1e-15)
	stateLeft, err := original.MarshalState([]Param{left})
	if err != nil {
		t.Fatalf("MarshalState(original): %v", err)
	}
	stateRight, err := continued.MarshalState([]Param{right})
	if err != nil {
		t.Fatalf("MarshalState(restored): %v", err)
	}
	decodedLeft, err := decodeState(stateLeft)
	if err != nil {
		t.Fatalf("decodeState(left): %v", err)
	}
	decodedRight, err := decodeState(stateRight)
	if err != nil {
		t.Fatalf("decodeState(right): %v", err)
	}
	assertClose(t, "velocity setelah restore", decodedRight["weight"].vectors[0], decodedLeft["weight"].vectors[0], 1e-15)
}

func TestSGDStateRejectsMismatch(t *testing.T) {
	weight := testParam(t, "weight", []float64{1, 2}, 2)
	defer weight.Parameter.Close()
	other := testParam(t, "other", []float64{1, 2}, 2)
	defer other.Parameter.Close()
	optimizer, err := NewSGD(SGDConfig{LearningRate: 0.1})
	if err != nil {
		t.Fatalf("NewSGD: %v", err)
	}
	defer optimizer.Close()

	state, err := optimizer.MarshalState([]Param{weight})
	if err != nil {
		t.Fatalf("MarshalState: %v", err)
	}
	if err := optimizer.UnmarshalState([]Param{other}, state); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("nama parameter berbeda: got %v", err)
	}
	if err := optimizer.UnmarshalState([]Param{weight, other}, state); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("state kehilangan parameter: got %v", err)
	}
	if err := optimizer.UnmarshalState([]Param{weight}, state[:len(state)-1]); !errors.Is(err, errs.ErrInvalidSerialization) {
		t.Fatalf("state terpotong: got %v", err)
	}
	corrupted := append([]byte(nil), state...)
	corrupted[0] ^= 0xff
	if err := optimizer.UnmarshalState([]Param{weight}, corrupted); !errors.Is(err, errs.ErrInvalidSerialization) {
		t.Fatalf("magic rusak: got %v", err)
	}
	wrongVector, err := encodeState(optimizerState{"weight": {vectors: [][]float64{{1}}}})
	if err != nil {
		t.Fatalf("encodeState: %v", err)
	}
	if err := optimizer.UnmarshalState([]Param{weight}, wrongVector); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("ukuran velocity berbeda: got %v", err)
	}
}

func TestSGDSupportsFloat32(t *testing.T) {
	value, err := tensor.NewTensor(dtype.Float32, 2)
	if err != nil {
		t.Fatalf("NewTensor: %v", err)
	}
	defer value.Close()
	if err := tensor.Write(value, []float32{1, -2}); err != nil {
		t.Fatalf("tensor.Write: %v", err)
	}
	parameter, err := nn.NewParameter("float32.weight", value, true)
	if err != nil {
		t.Fatalf("NewParameter: %v", err)
	}
	defer parameter.Close()
	optimizer, err := NewSGD(SGDConfig{LearningRate: 0.25})
	if err != nil {
		t.Fatalf("NewSGD: %v", err)
	}
	defer optimizer.Close()
	ctx, release := newGradientContext(t)
	defer release()
	applyGradientUsingMul(t, ctx, &Param{Name: "float32.weight", Parameter: parameter}, []float64{1, -0.5})
	named := Param{Name: "float32.weight", Parameter: parameter}
	if err := optimizer.Step(ctx, []Param{named}); err != nil {
		t.Fatalf("Step: %v", err)
	}
	got, err := tensor.Read[float32](named.Parameter.Value())
	if err != nil {
		t.Fatalf("tensor.Read(float32): %v", err)
	}
	assertClose(t, "update float32", []float64{float64(got[0]), float64(got[1])}, []float64{0.75, -1.875}, 1e-6)
}

func TestSGDCloseIsTerminal(t *testing.T) {
	parameter := testParam(t, "weight", []float64{1}, 1)
	defer parameter.Parameter.Close()
	optimizer, err := NewSGD(SGDConfig{LearningRate: 0.1})
	if err != nil {
		t.Fatalf("NewSGD: %v", err)
	}
	ctx, release := newGradientContext(t)
	defer release()
	applyGradientUsingMul(t, ctx, &parameter, []float64{1})
	state, err := optimizer.MarshalState([]Param{parameter})
	if err != nil {
		t.Fatalf("MarshalState: %v", err)
	}
	if err := optimizer.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := optimizer.Close(); err != nil {
		t.Fatalf("Close kedua: %v", err)
	}
	if err := optimizer.Step(ctx, []Param{parameter}); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("Step setelah Close: got %v", err)
	}
	if err := optimizer.SetLearningRate(0.2); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("SetLearningRate setelah Close: got %v", err)
	}
	if err := optimizer.UnmarshalState([]Param{parameter}, state); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("UnmarshalState setelah Close: got %v", err)
	}
}

func TestSGDHandlesEmptyParameter(t *testing.T) {
	value, err := tensor.NewTensor(dtype.Float64, 0)
	if err != nil {
		t.Fatalf("NewTensor: %v", err)
	}
	defer value.Close()
	parameter, err := nn.NewParameter("empty", value, true)
	if err != nil {
		t.Fatalf("NewParameter: %v", err)
	}
	defer parameter.Close()
	optimizer, err := NewSGD(SGDConfig{LearningRate: 0.1, Momentum: 0.9})
	if err != nil {
		t.Fatalf("NewSGD: %v", err)
	}
	defer optimizer.Close()
	ctx, release := newGradientContext(t)
	defer release()
	if err := optimizer.Step(ctx, []Param{{Name: "empty", Parameter: parameter}}); err != nil {
		t.Fatalf("Step pada parameter kosong: %v", err)
	}
}

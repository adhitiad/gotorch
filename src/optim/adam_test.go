package optim

import (
	"errors"
	"math"
	"testing"

	"github.com/adhitiad/gotorch/src/errs"
)

func TestAdamFirstStepWithBiasCorrection(t *testing.T) {
	parameter := testParam(t, "weight", []float64{1, -2}, 2)
	defer parameter.Parameter.Close()
	optimizer, err := NewAdam(AdamConfig{LearningRate: 0.1, Beta1: 0.9, Beta2: 0.999, Epsilon: 1e-8})
	if err != nil {
		t.Fatalf("NewAdam: %v", err)
	}
	defer optimizer.Close()
	ctx, release := newGradientContext(t)
	defer release()
	applyGradientUsingMul(t, ctx, &parameter, []float64{0.2, -0.1})

	if err := optimizer.Step(ctx, []Param{parameter}); err != nil {
		t.Fatalf("Step: %v", err)
	}
	// m₁ = 0,1g; v₁ = 0,001g²; m̂ ≈ g; v̂ ≈ g², sehingga Δ ≈ lr·sign(g).
	assertClose(t, "Adam langkah pertama", valuesOf(t, &parameter), []float64{0.9, -1.9}, 1e-6)

	state, err := optimizer.MarshalState([]Param{parameter})
	if err != nil {
		t.Fatalf("MarshalState: %v", err)
	}
	decoded, err := decodeState(state)
	if err != nil {
		t.Fatalf("decodeState: %v", err)
	}
	entry := decoded["weight"]
	if len(entry.scalars) != 1 || entry.scalars[0] != 1 {
		t.Fatalf("step tersimpan: got %v", entry.scalars)
	}
	assertClose(t, "momen pertama", entry.vectors[0], []float64{0.02, -0.01}, 1e-15)
	assertClose(t, "momen kedua", entry.vectors[1], []float64{0.00004, 0.00001}, 1e-18)
}

func TestAdamSecondStepUsesBiasCorrectedMoments(t *testing.T) {
	parameter := testParam(t, "weight", []float64{1, -2}, 2)
	defer parameter.Parameter.Close()
	optimizer, err := NewAdam(AdamConfig{LearningRate: 0.1, Beta1: 0.9, Beta2: 0.999, Epsilon: 1e-8})
	if err != nil {
		t.Fatalf("NewAdam: %v", err)
	}
	defer optimizer.Close()

	ctx, release := newGradientContext(t)
	applyGradientUsingMul(t, ctx, &parameter, []float64{0.2, -0.1})
	if err := optimizer.Step(ctx, []Param{parameter}); err != nil {
		t.Fatalf("Step pertama: %v", err)
	}
	release()

	ctx, release = newGradientContext(t)
	applyGradientUsingMul(t, ctx, &parameter, []float64{-0.3, 0.4})
	if err := optimizer.Step(ctx, []Param{parameter}); err != nil {
		t.Fatalf("Step kedua: %v", err)
	}
	release()

	// m₂ = [0,9·0,02 + 0,1·(-0,3), 0,9·(-0,01) + 0,1·0,4] = [-0,012, 0,031].
	// v₂ = [0,999·0,00004 + 0,001·0,09, 0,999·0,00001 + 0,001·0,16].
	beta1, beta2 := 0.9, 0.999
	step := float64(2)
	m := []float64{-0.012, 0.031}
	v := []float64{0.999*0.00004 + 0.001*0.09, 0.999*0.00001 + 0.001*0.16}
	want := make([]float64, 2)
	for index := range want {
		corrected := m[index] / (1 - math.Pow(beta1, step))
		second := v[index] / (1 - math.Pow(beta2, step))
		want[index] = []float64{0.9, -1.9}[index] - 0.1*(corrected/(math.Sqrt(second)+1e-8))
	}
	assertClose(t, "Adam langkah kedua", valuesOf(t, &parameter), want, 1e-7)

	state, err := optimizer.MarshalState([]Param{parameter})
	if err != nil {
		t.Fatalf("MarshalState: %v", err)
	}
	decoded, err := decodeState(state)
	if err != nil {
		t.Fatalf("decodeState: %v", err)
	}
	if decoded["weight"].scalars[0] != 2 {
		t.Fatalf("step kedua tidak tercatat: %v", decoded["weight"].scalars)
	}
}

func TestAdamWUsesDecoupledWeightDecay(t *testing.T) {
	makeParameter := func() Param { return testParam(t, "weight", []float64{2, -3}, 2) }
	stepOnce := func(t *testing.T, optimizer Optimizer, parameter Param) {
		t.Helper()
		ctx, release := newGradientContext(t)
		defer release()
		applyGradientUsingMul(t, ctx, &parameter, []float64{0.1, 0.2})
		if err := optimizer.Step(ctx, []Param{parameter}); err != nil {
			t.Fatalf("Step: %v", err)
		}
	}

	adamW, err := NewAdam(AdamConfig{LearningRate: 0.1, WeightDecay: 0.1, AdamW: true})
	if err != nil {
		t.Fatalf("NewAdam(AdamW): %v", err)
	}
	defer adamW.Close()
	adamWParameter := makeParameter()
	defer adamWParameter.Parameter.Close()
	stepOnce(t, adamW, adamWParameter)
	// AdamW: Δ = lr·(m̂/(√v̂+ε) + wd·θ) ≈ 0,1·(1 + 0,1·θ).
	assertClose(t, "AdamW decoupled", valuesOf(t, &adamWParameter), []float64{1.88, -3.07}, 1e-6)

	adam, err := NewAdam(AdamConfig{LearningRate: 0.1, WeightDecay: 0.1})
	if err != nil {
		t.Fatalf("NewAdam: %v", err)
	}
	defer adam.Close()
	adamParameter := makeParameter()
	defer adamParameter.Parameter.Close()
	stepOnce(t, adam, adamParameter)
	// Adam terkopel memasukkan wd·θ ke dalam gradien sebelum momen.
	assertClose(t, "Adam coupled", valuesOf(t, &adamParameter), []float64{1.9, -2.9}, 1e-6)
	if math.Abs(valuesOf(t, &adamParameter)[0]-valuesOf(t, &adamWParameter)[0]) < 1e-3 {
		t.Fatal("Adam dan AdamW tidak boleh menghasilkan update yang sama")
	}
}

func TestAdamStateRoundTripPreservesBiasCorrection(t *testing.T) {
	left := testParam(t, "weight", []float64{1, -1}, 2)
	right := testParam(t, "weight", []float64{1, -1}, 2)
	defer left.Parameter.Close()
	defer right.Parameter.Close()
	config := AdamConfig{LearningRate: 0.02, Beta1: 0.8, Beta2: 0.95, WeightDecay: 0.01, AdamW: true}
	original, err := NewAdam(config)
	if err != nil {
		t.Fatalf("NewAdam: %v", err)
	}
	defer original.Close()
	restored, err := NewAdam(config)
	if err != nil {
		t.Fatalf("NewAdam(restored): %v", err)
	}
	defer restored.Close()

	gradients := [][]float64{{0.5, -0.2}, {-0.1, 0.7}}
	for _, gradient := range gradients {
		for _, item := range []struct {
			optimizer *Adam
			parameter Param
		}{{original, left}, {restored, right}} {
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
	if err := restored.UnmarshalState([]Param{right}, state); err != nil {
		t.Fatalf("UnmarshalState: %v", err)
	}

	// Gradien nol tetap menggerakkan parameter karena momen lama dan koreksi bias.
	for index := 0; index < 3; index++ {
		ctxLeft, releaseLeft := newGradientContext(t)
		applyGradientUsingMul(t, ctxLeft, &left, []float64{0, 0})
		if err := original.Step(ctxLeft, []Param{left}); err != nil {
			t.Fatalf("Step(original): %v", err)
		}
		releaseLeft()

		ctxRight, releaseRight := newGradientContext(t)
		applyGradientUsingMul(t, ctxRight, &right, []float64{0, 0})
		if err := restored.Step(ctxRight, []Param{right}); err != nil {
			t.Fatalf("Step(restored): %v", err)
		}
		releaseRight()
	}
	assertClose(t, "lintasan Adam setelah restore", valuesOf(t, &right), valuesOf(t, &left), 1e-15)
	stateOriginal, err := original.MarshalState([]Param{left})
	if err != nil {
		t.Fatalf("MarshalState(original): %v", err)
	}
	stateRestored, err := restored.MarshalState([]Param{right})
	if err != nil {
		t.Fatalf("MarshalState(restored): %v", err)
	}
	decodedOriginal, err := decodeState(stateOriginal)
	if err != nil {
		t.Fatalf("decodeState(original): %v", err)
	}
	decodedRestored, err := decodeState(stateRestored)
	if err != nil {
		t.Fatalf("decodeState(restored): %v", err)
	}
	if decodedRestored["weight"].scalars[0] != decodedOriginal["weight"].scalars[0] {
		t.Fatalf("step berbeda setelah restore: %v vs %v", decodedRestored, decodedOriginal)
	}
	assertClose(t, "momen setelah restore", decodedRestored["weight"].vectors[0], decodedOriginal["weight"].vectors[0], 1e-15)
}

func TestAdamStateRejectsMismatch(t *testing.T) {
	weight := testParam(t, "weight", []float64{1, 2}, 2)
	defer weight.Parameter.Close()
	other := testParam(t, "other", []float64{1, 2}, 2)
	defer other.Parameter.Close()
	optimizer, err := NewAdam(AdamConfig{LearningRate: 0.1})
	if err != nil {
		t.Fatalf("NewAdam: %v", err)
	}
	defer optimizer.Close()

	state, err := optimizer.MarshalState([]Param{weight})
	if err != nil {
		t.Fatalf("MarshalState: %v", err)
	}
	if err := optimizer.UnmarshalState([]Param{other}, state); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("nama berbeda: got %v", err)
	}
	if err := optimizer.UnmarshalState([]Param{weight, other}, state); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("state kehilangan parameter: got %v", err)
	}
	if err := optimizer.UnmarshalState([]Param{weight}, state[:len(state)-1]); !errors.Is(err, errs.ErrInvalidSerialization) {
		t.Fatalf("state terpotong: got %v", err)
	}
	wrongVersion := append([]byte(nil), state...)
	wrongVersion[len(optimizerStateMagic)] = 2
	if err := optimizer.UnmarshalState([]Param{weight}, wrongVersion); !errors.Is(err, errs.ErrUnsupportedSerializationVersion) {
		t.Fatalf("versi berbeda: got %v", err)
	}

	invalidStates := []struct {
		name  string
		entry stateEntry
	}{
		{"scalar kurang", stateEntry{vectors: [][]float64{{0, 0}, {0, 0}}}},
		{"vektor kurang", stateEntry{scalars: []float64{1}, vectors: [][]float64{{0, 0}}}},
		{"ukuran vektor salah", stateEntry{scalars: []float64{1}, vectors: [][]float64{{0}, {0, 0}}}},
		{"step negatif", stateEntry{scalars: []float64{-1}, vectors: [][]float64{{0, 0}, {0, 0}}}},
		{"step pecahan", stateEntry{scalars: []float64{1.5}, vectors: [][]float64{{0, 0}, {0, 0}}}},
		{"step terlalu besar", stateEntry{scalars: []float64{math.Ldexp(1, 63)}, vectors: [][]float64{{0, 0}, {0, 0}}}},
	}
	for _, item := range invalidStates {
		t.Run(item.name, func(t *testing.T) {
			data, err := encodeState(optimizerState{"weight": item.entry})
			if err != nil {
				t.Fatalf("encodeState: %v", err)
			}
			if err := optimizer.UnmarshalState([]Param{weight}, data); !errors.Is(err, errs.ErrInvalidState) {
				t.Fatalf("got %v, want %v", err, errs.ErrInvalidState)
			}
		})
	}
}

func TestAdamWithoutGradientDoesNotAdvanceStep(t *testing.T) {
	withGradient := testParam(t, "a.weight", []float64{1}, 1)
	withoutGradient := testParam(t, "b.weight", []float64{5}, 1)
	defer withGradient.Parameter.Close()
	defer withoutGradient.Parameter.Close()
	optimizer, err := NewAdam(AdamConfig{LearningRate: 0.1})
	if err != nil {
		t.Fatalf("NewAdam: %v", err)
	}
	defer optimizer.Close()
	ctx, release := newGradientContext(t)
	defer release()
	applyGradientUsingMul(t, ctx, &withGradient, []float64{1})

	if err := optimizer.Step(ctx, []Param{withoutGradient, withGradient}); err != nil {
		t.Fatalf("Step: %v", err)
	}
	assertClose(t, "tanpa gradien", valuesOf(t, &withoutGradient), []float64{5}, 0)
	state, err := optimizer.MarshalState([]Param{withoutGradient, withGradient})
	if err != nil {
		t.Fatalf("MarshalState: %v", err)
	}
	decoded, err := decodeState(state)
	if err != nil {
		t.Fatalf("decodeState: %v", err)
	}
	if decoded["b.weight"].scalars[0] != 0 || decoded["a.weight"].scalars[0] != 1 {
		t.Fatalf("step per parameter: %v", decoded)
	}
}

func TestAdamCloseIsTerminal(t *testing.T) {
	parameter := testParam(t, "weight", []float64{1}, 1)
	defer parameter.Parameter.Close()
	optimizer, err := NewAdam(AdamConfig{LearningRate: 0.1})
	if err != nil {
		t.Fatalf("NewAdam: %v", err)
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

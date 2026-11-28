package peft

import (
	"context"
	"errors"
	"math"
	"math/rand"
	"testing"

	"github.com/adhitiad/gotorch/src/autograd"
	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/nn"
	"github.com/adhitiad/gotorch/src/tensor"
)

func newFloatTensor(t *testing.T, dims []int64, values []float64) *tensor.Tensor {
	t.Helper()
	value, err := tensor.NewTensor(dtype.Float64, dims...)
	if err != nil {
		t.Fatalf("NewTensor(%v): %v", dims, err)
	}
	if err := tensor.Write(value, values); err != nil {
		_ = value.Close()
		t.Fatalf("tensor.Write: %v", err)
	}
	return value
}

func floatValues(t *testing.T, value *tensor.Tensor) []float64 {
	t.Helper()
	values, err := tensor.Read[float64](value)
	if err != nil {
		t.Fatalf("tensor.Read: %v", err)
	}
	return values
}

func assertClose(t *testing.T, label string, got, want []float64, tolerance float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: diperoleh %d nilai, diharapkan %d: %v vs %v", label, len(got), len(want), got, want)
	}
	for index := range want {
		difference := math.Abs(got[index] - want[index])
		limit := tolerance + tolerance*math.Abs(want[index])
		if math.IsNaN(difference) || difference > limit {
			t.Fatalf("%s[%d]: diperoleh %g, diharapkan %g (selisih %g)", label, index, got[index], want[index], difference)
		}
	}
}

func assertErrorIs(t *testing.T, err, want error) {
	t.Helper()
	if !errors.Is(err, want) {
		t.Fatalf("error = %v, diharapkan tertanam %v", err, want)
	}
}

func newTestTape(t *testing.T) (*autograd.Tape, *nn.Context) {
	t.Helper()
	tape, err := nn.NewTape()
	if err != nil {
		t.Fatalf("nn.NewTape: %v", err)
	}
	ctx, err := nn.NewContext(tape)
	if err != nil {
		t.Fatalf("nn.NewContext: %v", err)
	}
	t.Cleanup(func() {
		if err := tape.Close(); err != nil {
			t.Errorf("tape.Close: %v", err)
		}
	})
	return tape, ctx
}

func newVariable(t *testing.T, tape *autograd.Tape, value *tensor.Tensor, requiresGrad bool) *autograd.Variable {
	t.Helper()
	t.Cleanup(func() { _ = value.Close() })
	variable, err := tape.NewVariable(value, requiresGrad)
	if err != nil {
		t.Fatalf("NewVariable: %v", err)
	}
	return variable
}

func newLinearWithWeight(t *testing.T, name string, in, out int64, weight []float64) *nn.Linear {
	t.Helper()
	layer, err := nn.NewLinear(name, nn.LinearConfig{
		In: in, Out: out, DType: dtype.Float64, Bias: true, RNG: rand.New(rand.NewSource(7)),
		WeightInit: func(_ *rand.Rand, value *tensor.Tensor) error {
			return tensor.Write(value, weight)
		},
	})
	if err != nil {
		t.Fatalf("nn.NewLinear(%q): %v", name, err)
	}
	t.Cleanup(func() {
		if err := layer.Close(); err != nil {
			t.Errorf("layer.Close(%q): %v", name, err)
		}
	})
	return layer
}

func parameterValues(t *testing.T, parameter *nn.Parameter) []float64 {
	t.Helper()
	return floatValues(t, parameter.Value())
}

func setParameterValues(t *testing.T, parameter *nn.Parameter, values []float64) {
	t.Helper()
	if err := tensor.Write(parameter.Value(), values); err != nil {
		t.Fatalf("tensor.Write(parameter): %v", err)
	}
}

func backwardSum(t *testing.T, tape *autograd.Tape, output *autograd.Variable) {
	t.Helper()
	scalar, err := tape.Sum(output)
	if err != nil {
		t.Fatalf("tape.Sum: %v", err)
	}
	if err := tape.Backward(context.Background(), scalar); err != nil {
		t.Fatalf("tape.Backward: %v", err)
	}
}

func gradientValues(t *testing.T, ctx *nn.Context, parameter *nn.Parameter) ([]float64, error) {
	t.Helper()
	gradient, err := ctx.Gradient(parameter)
	if err != nil {
		return nil, err
	}
	values := floatValues(t, gradient)
	if err := gradient.Close(); err != nil {
		t.Fatalf("gradient.Close: %v", err)
	}
	return values, nil
}

func assertNoGradient(t *testing.T, ctx *nn.Context, parameter *nn.Parameter) {
	t.Helper()
	_, err := gradientValues(t, ctx, parameter)
	assertErrorIs(t, err, errs.ErrNoGradient)
}

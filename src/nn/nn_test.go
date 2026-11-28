package nn

import (
	"context"
	"math"
	"testing"

	"github.com/adhitiad/gotorch/src/autograd"
	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/tensor"
)

func newTestTensor(t *testing.T, elementType dtype.DType, dims []int64, values []float64) *tensor.Tensor {
	t.Helper()
	value, err := tensor.NewTensor(elementType, dims...)
	if err != nil {
		t.Fatalf("NewTensor(%v): %v", dims, err)
	}
	if err := writeValues(value, values); err != nil {
		t.Fatalf("writeValues: %v", err)
	}
	return value
}

func newIntTensor(t *testing.T, dims []int64, values []int64) *tensor.Tensor {
	t.Helper()
	value, err := tensor.NewTensor(dtype.Int64, dims...)
	if err != nil {
		t.Fatalf("NewTensor(%v): %v", dims, err)
	}
	if err := tensor.Write(value, values); err != nil {
		t.Fatalf("tensor.Write: %v", err)
	}
	return value
}

func tensorValues(t *testing.T, value *tensor.Tensor) []float64 {
	t.Helper()
	values, err := readValues(value)
	if err != nil {
		t.Fatalf("readValues: %v", err)
	}
	return values
}

func assertClose(t *testing.T, label string, got, want []float64, tolerance float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: got %d values, want %d: %v vs %v", label, len(got), len(want), got, want)
	}
	for index := range want {
		difference := math.Abs(got[index] - want[index])
		limit := tolerance + tolerance*math.Abs(want[index])
		if math.IsNaN(difference) || difference > limit {
			t.Fatalf("%s[%d]: got %g, want %g (diff %g)", label, index, got[index], want[index], difference)
		}
	}
}

func newTestContext(t *testing.T) (*autograd.Tape, *Context) {
	t.Helper()
	tape, err := NewTape()
	if err != nil {
		t.Fatalf("NewTape: %v", err)
	}
	ctx, err := NewContext(tape)
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	return tape, ctx
}

func variableOf(t *testing.T, tape *autograd.Tape, value *tensor.Tensor, requiresGrad bool) *autograd.Variable {
	t.Helper()
	variable, err := tape.NewVariable(value, requiresGrad)
	if err != nil {
		t.Fatalf("NewVariable: %v", err)
	}
	return variable
}

func mustSum(t *testing.T, tape *autograd.Tape, value *autograd.Variable) *autograd.Variable {
	t.Helper()
	scalar, err := tape.Sum(value)
	if err != nil {
		t.Fatalf("Sum: %v", err)
	}
	return scalar
}

func scalarize(tape *autograd.Tape, value *autograd.Variable) (*autograd.Variable, error) {
	snapshot, err := value.Value()
	if err != nil {
		return nil, err
	}
	defer snapshot.Close()
	metadata, err := snapshot.Metadata()
	if err != nil {
		return nil, err
	}
	if len(metadata.Shape().Dims()) == 0 {
		return value, nil
	}
	return tape.Sum(value)
}

func checkLayerGradient(
	t *testing.T,
	input *tensor.Tensor,
	fn func(context.Context, *autograd.Tape, *autograd.Variable) (*autograd.Variable, error),
) {
	t.Helper()
	result, err := autograd.CheckGradient(context.Background(), input,
		func(ctx context.Context, tape *autograd.Tape, leaf *autograd.Variable) (*autograd.Variable, error) {
			if err := RegisterOperations(tape); err != nil {
				return nil, err
			}
			return fn(ctx, tape, leaf)
		},
		autograd.GradientCheckOptions{AbsoluteTolerance: 1e-5, RelativeTolerance: 2e-4, MaxElements: 16},
	)
	if err != nil {
		t.Fatalf("CheckGradient: %v", err)
	}
	if !result.Passed {
		for _, element := range result.Elements {
			t.Logf("index %d: analytical %g numerical %g error %g", element.Index, element.Analytical, element.Numerical, element.AbsoluteError)
		}
		t.Fatalf("gradient check failed: max abs %g max rel %g", result.MaxAbsoluteError, result.MaxRelativeError)
	}
}

func gradientsOf(t *testing.T, tape *autograd.Tape, target *autograd.Variable, seed *tensor.Tensor, variables ...*autograd.Variable) [][]float64 {
	t.Helper()
	if err := tape.BackwardWithGradient(context.Background(), target, seed); err != nil {
		t.Fatalf("BackwardWithGradient: %v", err)
	}
	result := make([][]float64, len(variables))
	for index, variable := range variables {
		gradient, err := variable.Grad()
		if err != nil {
			t.Fatalf("Grad(%d): %v", index, err)
		}
		result[index] = tensorValues(t, gradient)
		_ = gradient.Close()
	}
	return result
}

func mustValue(t *testing.T, variable *autograd.Variable) *tensor.Tensor {
	t.Helper()
	value, err := variable.Value()
	if err != nil {
		t.Fatalf("Value: %v", err)
	}
	return value
}

func TestLinearForwardAndGradients(t *testing.T) {
	inputValue := newTestTensor(t, dtype.Float64, []int64{2, 2}, []float64{1, 2, 3, 4})
	weightValue := newTestTensor(t, dtype.Float64, []int64{2, 2}, []float64{0.5, -0.25, 1.5, 0.75})
	biasValue := newTestTensor(t, dtype.Float64, []int64{2}, []float64{0.1, -0.2})

	tape, ctx := newTestContext(t)
	defer tape.Close()
	input := variableOf(t, tape, inputValue, true)
	weight := variableOf(t, tape, weightValue, true)
	bias := variableOf(t, tape, biasValue, true)
	output, err := applyOperation(ctx, OpLinear, []*autograd.Variable{input, weight, bias}, func() (tensorData, error) {
		inputData, err := variableData(input)
		if err != nil {
			return tensorData{}, err
		}
		weightData, err := variableData(weight)
		if err != nil {
			return tensorData{}, err
		}
		biasData, err := variableData(bias)
		if err != nil {
			return tensorData{}, err
		}
		return linearData(inputData, weightData, biasData, true)
	})
	if err != nil {
		t.Fatalf("linear: %v", err)
	}
	assertClose(t, "linear forward", tensorValues(t, mustValue(t, output)), []float64{0.1, 2.8, 0.6, 7.3}, 1e-12)

	seed := newTestTensor(t, dtype.Float64, []int64{2, 2}, []float64{1, 2, 3, 4})
	gradients := gradientsOf(t, tape, output, seed, input, weight, bias)
	assertClose(t, "linear grad input", gradients[0], []float64{3.5, 1.25, 7.5, 2.25}, 1e-12)
	assertClose(t, "linear grad weight", gradients[1], []float64{10, 14, 14, 20}, 1e-12)
	assertClose(t, "linear grad bias", gradients[2], []float64{4, 6}, 1e-12)
}

type activationConstructor func(string) (*Activation, error)

func activationCase(name string, constructor activationConstructor) func(context.Context, *autograd.Tape, *autograd.Variable) (*autograd.Variable, error) {
	return func(_ context.Context, tape *autograd.Tape, leaf *autograd.Variable) (*autograd.Variable, error) {
		activation, err := constructor(name)
		if err != nil {
			return nil, err
		}
		defer activation.Close()
		ctx, err := NewContext(tape)
		if err != nil {
			return nil, err
		}
		output, err := activation.Forward(ctx, leaf)
		if err != nil {
			return nil, err
		}
		return scalarize(tape, output)
	}
}

func TestActivationGradients(t *testing.T) {
	input := newTestTensor(t, dtype.Float64, []int64{2, 3}, []float64{-0.8, 0.2, 1.4, 0.5, -1.1, 0.7})
	cases := []struct {
		name string
		fn   func(context.Context, *autograd.Tape, *autograd.Variable) (*autograd.Variable, error)
	}{
		{"relu", activationCase("relu", NewReLU)},
		{"sigmoid", activationCase("sigmoid", NewSigmoid)},
		{"tanh", activationCase("tanh", NewTanh)},
		{"gelu", activationCase("gelu", NewGELU)},
		{"silu", activationCase("silu", NewSiLU)},
		{"softmax", functionalScalarCase(func(ctx *Context, leaf *autograd.Variable) (*autograd.Variable, error) {
			return Softmax(ctx, leaf)
		})},
		{"log_softmax", functionalScalarCase(func(ctx *Context, leaf *autograd.Variable) (*autograd.Variable, error) {
			return LogSoftmax(ctx, leaf)
		})},
	}
	for _, item := range cases {
		t.Run(item.name, func(t *testing.T) {
			checkLayerGradient(t, input, func(ctx context.Context, tape *autograd.Tape, leaf *autograd.Variable) (*autograd.Variable, error) {
				return item.fn(ctx, tape, leaf)
			})
		})
	}
}

type functionalFn func(*Context, *autograd.Variable) (*autograd.Variable, error)

func functionalScalarCase(fn functionalFn) func(context.Context, *autograd.Tape, *autograd.Variable) (*autograd.Variable, error) {
	return func(_ context.Context, tape *autograd.Tape, leaf *autograd.Variable) (*autograd.Variable, error) {
		ctx, err := NewContext(tape)
		if err != nil {
			return nil, err
		}
		output, err := fn(ctx, leaf)
		if err != nil {
			return nil, err
		}
		return scalarize(tape, output)
	}
}

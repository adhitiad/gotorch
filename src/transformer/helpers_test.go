package transformer

import (
	"context"
	"math"
	"testing"

	"github.com/adhitiad/gotorch/src/autograd"
	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/nn"
	"github.com/adhitiad/gotorch/src/tensor"
)

// newFloatTensor membuat tensor uji dengan dtype float yang diminta.
func newFloatTensor(t *testing.T, elementType dtype.DType, dims []int64, values []float64) *tensor.Tensor {
	t.Helper()
	value, err := tensor.NewTensor(elementType, dims...)
	if err != nil {
		t.Fatalf("NewTensor(%v): %v", dims, err)
	}
	if err := writeFloatValues(value, values); err != nil {
		_ = value.Close()
		t.Fatalf("writeFloatValues: %v", err)
	}
	return value
}

func writeFloatValues(value *tensor.Tensor, values []float64) error {
	metadata, err := value.Metadata()
	if err != nil {
		return err
	}
	if metadata.DType() == dtype.Float32 {
		converted := make([]float32, len(values))
		for index, item := range values {
			converted[index] = float32(item)
		}
		return tensor.Write(value, converted)
	}
	return tensor.Write(value, values)
}

func readFloatValues(t *testing.T, value *tensor.Tensor) []float64 {
	t.Helper()
	metadata, err := value.Metadata()
	if err != nil {
		t.Fatalf("Metadata: %v", err)
	}
	if metadata.DType() == dtype.Float32 {
		items, err := tensor.Read[float32](value)
		if err != nil {
			t.Fatalf("Read[float32]: %v", err)
		}
		result := make([]float64, len(items))
		for index, item := range items {
			result[index] = float64(item)
		}
		return result
	}
	items, err := tensor.Read[float64](value)
	if err != nil {
		t.Fatalf("Read[float64]: %v", err)
	}
	return items
}

func newIntTensor(t *testing.T, dims []int64, values []int64) *tensor.Tensor {
	t.Helper()
	value, err := tensor.NewTensor(dtype.Int64, dims...)
	if err != nil {
		t.Fatalf("NewTensor(%v): %v", dims, err)
	}
	if err := tensor.Write(value, values); err != nil {
		_ = value.Close()
		t.Fatalf("tensor.Write: %v", err)
	}
	return value
}

func variableValues(t *testing.T, variable *autograd.Variable) []float64 {
	t.Helper()
	value, err := variable.Value()
	if err != nil {
		t.Fatalf("Value: %v", err)
	}
	defer value.Close()
	return readFloatValues(t, value)
}

func assertVariableDims(t *testing.T, label string, variable *autograd.Variable, want ...int64) {
	t.Helper()
	value, err := variable.Value()
	if err != nil {
		t.Fatalf("%s: Value: %v", label, err)
	}
	defer value.Close()
	metadata, err := value.Metadata()
	if err != nil {
		t.Fatalf("%s: Metadata: %v", label, err)
	}
	if err := requireShape(label, metadata.Shape().Dims(), want...); err != nil {
		t.Fatalf("%s: %v", label, err)
	}
}

func assertVariableDType(t *testing.T, label string, variable *autograd.Variable, want dtype.DType) {
	t.Helper()
	value, err := variable.Value()
	if err != nil {
		t.Fatalf("%s: Value: %v", label, err)
	}
	defer value.Close()
	metadata, err := value.Metadata()
	if err != nil {
		t.Fatalf("%s: Metadata: %v", label, err)
	}
	if got := metadata.DType(); got != want {
		t.Fatalf("%s: dtype %s, want %s", label, got, want)
	}
}

func assertFloatClose(t *testing.T, label string, got, want []float64, tolerance float64) {
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

func newTestTape(t *testing.T) (*autograd.Tape, *nn.Context) {
	t.Helper()
	tape, err := nn.NewTape()
	if err != nil {
		t.Fatalf("nn.NewTape: %v", err)
	}
	ctx, err := nn.NewContext(tape)
	if err != nil {
		_ = tape.Close()
		t.Fatalf("nn.NewContext: %v", err)
	}
	return tape, ctx
}

func newVariable(t *testing.T, tape *autograd.Tape, value *tensor.Tensor, requiresGrad bool) *autograd.Variable {
	t.Helper()
	variable, err := tape.NewVariable(value, requiresGrad)
	if err != nil {
		t.Fatalf("NewVariable: %v", err)
	}
	return variable
}

func setParameterValues(t *testing.T, parameter *nn.Parameter, values []float64) {
	t.Helper()
	if err := parameter.Update(func(current []float64) ([]float64, error) {
		if len(current) != len(values) {
			return nil, errs.ErrGradientShapeMismatch
		}
		return append([]float64(nil), values...), nil
	}); err != nil {
		t.Fatalf("parameter.Update: %v", err)
	}
}

func setProjectorWeights(t *testing.T, projector Projector, values []float64) {
	t.Helper()
	linear, ok := projector.(*nn.Linear)
	if !ok {
		t.Fatalf("projektor %T bukan *nn.Linear", projector)
	}
	setParameterValues(t, linear.Weight, values)
}

// checkTransformerGradient menyisipkan register operasi ke tape milik CheckGradient.
func checkTransformerGradient(
	t *testing.T,
	input *tensor.Tensor,
	fn func(*nn.Context, *autograd.Variable) (*autograd.Variable, error),
) autograd.GradientCheckResult {
	t.Helper()
	result, err := autograd.CheckGradient(context.Background(), input,
		func(_ context.Context, tape *autograd.Tape, leaf *autograd.Variable) (*autograd.Variable, error) {
			if err := nn.RegisterOperations(tape); err != nil {
				return nil, err
			}
			ctx, err := nn.NewContext(tape)
			if err != nil {
				return nil, err
			}
			output, err := fn(ctx, leaf)
			if err != nil {
				return nil, err
			}
			return tape.Sum(output)
		},
		autograd.GradientCheckOptions{AbsoluteTolerance: 1e-5, RelativeTolerance: 2e-4, MaxElements: 32},
	)
	if err != nil {
		t.Fatalf("CheckGradient: %v", err)
	}
	if !result.Passed {
		for _, element := range result.Elements {
			t.Logf("indeks %d: analitik %g numerik %g galat %g", element.Index, element.Analytical, element.Numerical, element.AbsoluteError)
		}
		t.Fatalf("gradient check gagal: galat absolut maks %g, galat relatif maks %g", result.MaxAbsoluteError, result.MaxRelativeError)
	}
	return result
}

// checkParameterGradient membandingkan gradien parameter dari autograd dengan
// selisih terpusat. Parameter tidak dapat disuntikkan sebagai leaf CheckGradient
// karena nn.Context membuat variabelnya sendiri.
func checkParameterGradient(
	t *testing.T,
	parameter *nn.Parameter,
	input *tensor.Tensor,
	forward func(*nn.Context, *autograd.Variable) (*autograd.Variable, error),
) {
	t.Helper()
	original := readFloatValues(t, parameter.Value())
	defer setParameterValues(t, parameter, original)

	evaluate := func(values []float64) float64 {
		setParameterValues(t, parameter, values)
		tape, ctx := newTestTape(t)
		defer tape.Close()
		leaf := newVariable(t, tape, input, false)
		output, err := forward(ctx, leaf)
		if err != nil {
			t.Fatalf("forward: %v", err)
		}
		scalar, err := tape.Sum(output)
		if err != nil {
			t.Fatalf("Sum: %v", err)
		}
		return variableValues(t, scalar)[0]
	}

	tape, ctx := newTestTape(t)
	defer tape.Close()
	leaf := newVariable(t, tape, input, false)
	output, err := forward(ctx, leaf)
	if err != nil {
		t.Fatalf("forward analitik: %v", err)
	}
	scalar, err := tape.Sum(output)
	if err != nil {
		t.Fatalf("Sum analitik: %v", err)
	}
	if err := tape.Backward(context.Background(), scalar); err != nil {
		t.Fatalf("Backward: %v", err)
	}
	gradient, err := ctx.Gradient(parameter)
	if err != nil {
		t.Fatalf("Gradient: %v", err)
	}
	analytical := readFloatValues(t, gradient)
	_ = gradient.Close()

	step := 1e-6
	for index, value := range original {
		plus := append([]float64(nil), original...)
		minus := append([]float64(nil), original...)
		plus[index] = value + step
		minus[index] = value - step
		numerical := (evaluate(plus) - evaluate(minus)) / (2 * step)
		limit := 1e-5 + 1e-4*math.Abs(numerical)
		if math.Abs(analytical[index]-numerical) > limit {
			t.Fatalf("parameter[%d]: analitik %g, numerik %g", index, analytical[index], numerical)
		}
	}
}

func referenceSoftmax(values []float64) []float64 {
	maximum := math.Inf(-1)
	for _, value := range values {
		if value > maximum {
			maximum = value
		}
	}
	result := make([]float64, len(values))
	if math.IsInf(maximum, -1) {
		return result
	}
	var total float64
	for index, value := range values {
		result[index] = math.Exp(value - maximum)
		total += result[index]
	}
	for index := range result {
		result[index] /= total
	}
	return result
}

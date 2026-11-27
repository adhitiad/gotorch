package autograd

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/tensor"
)

func TestBackwardAddMulSumAndSnapshotOwnership(t *testing.T) {
	input, err := tensor.From([]float64{1, 2, 3}, 3)
	if err != nil {
		t.Fatal(err)
	}
	defer input.Close()

	tape := NewTape()
	defer tape.Close()
	x, err := tape.NewVariable(input, true)
	if err != nil {
		t.Fatal(err)
	}

	// Mutating the caller's tensor must not alter the value captured by the tape.
	if err := tensor.Write(input, []float64{10, 20, 30}); err != nil {
		t.Fatal(err)
	}
	squared, err := tape.Mul(x, x)
	if err != nil {
		t.Fatal(err)
	}
	shifted, err := tape.Add(squared, x)
	if err != nil {
		t.Fatal(err)
	}
	loss, err := tape.Sum(shifted)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(context.Background(), loss); err != nil {
		t.Fatal(err)
	}

	gradient, err := x.Grad()
	if err != nil {
		t.Fatal(err)
	}
	defer gradient.Close()
	got, err := tensor.Read[float64](gradient)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, []float64{3, 5, 7}) {
		t.Fatalf("gradient = %v, want [3 5 7]", got)
	}
}

func TestBackwardAccumulatesAndZeroGradClears(t *testing.T) {
	input, err := tensor.From([]float32{1, 2}, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer input.Close()
	tape := NewTape()
	defer tape.Close()
	x, err := tape.NewVariable(input, true)
	if err != nil {
		t.Fatal(err)
	}
	squared, err := tape.Mul(x, x)
	if err != nil {
		t.Fatal(err)
	}
	loss, err := tape.Sum(squared)
	if err != nil {
		t.Fatal(err)
	}

	for i := 0; i < 2; i++ {
		if err := tape.Backward(context.Background(), loss); err != nil {
			t.Fatal(err)
		}
	}
	gradient, err := x.Grad()
	if err != nil {
		t.Fatal(err)
	}
	got, err := tensor.Read[float32](gradient)
	_ = gradient.Close()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, []float32{4, 8}) {
		t.Fatalf("accumulated gradient = %v, want [4 8]", got)
	}

	if err := tape.ZeroGrad(x); err != nil {
		t.Fatal(err)
	}
	if _, err := x.Grad(); !errors.Is(err, errs.ErrNoGradient) {
		t.Fatalf("Grad after ZeroGrad = %v, want ErrNoGradient", err)
	}
	if err := tape.Backward(context.Background(), loss); err != nil {
		t.Fatal(err)
	}
	gradient, err = x.Grad()
	if err != nil {
		t.Fatal(err)
	}
	got, err = tensor.Read[float32](gradient)
	_ = gradient.Close()
	if err != nil || !reflect.DeepEqual(got, []float32{2, 4}) {
		t.Fatalf("gradient after zero and backward = %v, %v", got, err)
	}
}

func TestSharedSubgraphAccumulatesBothPaths(t *testing.T) {
	input, err := tensor.From([]float64{2, 3}, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer input.Close()
	tape := NewTape()
	defer tape.Close()
	x, err := tape.NewVariable(input, true)
	if err != nil {
		t.Fatal(err)
	}
	squared, err := tape.Mul(x, x)
	if err != nil {
		t.Fatal(err)
	}
	doubled, err := tape.Add(squared, squared)
	if err != nil {
		t.Fatal(err)
	}
	loss, err := tape.Sum(doubled)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(context.Background(), loss); err != nil {
		t.Fatal(err)
	}
	gradient, err := x.Grad()
	if err != nil {
		t.Fatal(err)
	}
	defer gradient.Close()
	got, err := tensor.Read[float64](gradient)
	if err != nil || !reflect.DeepEqual(got, []float64{8, 12}) {
		t.Fatalf("shared-subgraph gradient = %v, %v", got, err)
	}
}

func TestForwardJVP(t *testing.T) {
	input, _ := tensor.From([]float64{1, 2}, 2)
	defer input.Close()
	seed, _ := tensor.From([]float64{3, 4}, 2)
	defer seed.Close()

	tape := NewTape()
	defer tape.Close()
	x, err := tape.NewVariable(input, true)
	if err != nil {
		t.Fatal(err)
	}
	squared, err := tape.Mul(x, x)
	if err != nil {
		t.Fatal(err)
	}
	output, err := tape.Add(squared, x)
	if err != nil {
		t.Fatal(err)
	}
	tangent, err := tape.JVP(context.Background(), output, map[*Variable]*tensor.Tensor{x: seed})
	if err != nil {
		t.Fatal(err)
	}
	defer tangent.Close()
	got, err := tensor.Read[float64](tangent)
	if err != nil || !reflect.DeepEqual(got, []float64{9, 20}) {
		t.Fatalf("JVP = %v, %v; want [9 20]", got, err)
	}
}

func TestCustomOperationRegistrationAndApply(t *testing.T) {
	tape := NewTape()
	defer tape.Close()
	operation := Operation{
		Name: "scale_two",
		Backward: func(_ context.Context, upstream *tensor.Tensor, _ []*tensor.Tensor, _ *tensor.Tensor) ([]*tensor.Tensor, error) {
			values, err := tensor.Read[float64](upstream)
			if err != nil {
				return nil, err
			}
			for i := range values {
				values[i] *= 2
			}
			gradient, err := tensor.From(values, int64(len(values)))
			return []*tensor.Tensor{gradient}, err
		},
		Forward: func(_ context.Context, _ []*tensor.Tensor, tangents []*tensor.Tensor, _ *tensor.Tensor) (*tensor.Tensor, error) {
			values, err := tensor.Read[float64](tangents[0])
			if err != nil {
				return nil, err
			}
			for i := range values {
				values[i] *= 2
			}
			return tensor.From(values, int64(len(values)))
		},
	}
	if err := tape.RegisterOperation(operation); err != nil {
		t.Fatal(err)
	}
	if err := tape.RegisterOperation(operation); !errors.Is(err, errs.ErrOperationAlreadyRegistered) {
		t.Fatalf("duplicate registration = %v", err)
	}

	input, _ := tensor.From([]float64{2, 5}, 2)
	defer input.Close()
	x, err := tape.NewVariable(input, true)
	if err != nil {
		t.Fatal(err)
	}
	output, _ := tensor.From([]float64{4, 10}, 2)
	defer output.Close()
	y, err := tape.Apply("scale_two", []*Variable{x}, output)
	if err != nil {
		t.Fatal(err)
	}
	loss, err := tape.Sum(y)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(context.Background(), loss); err != nil {
		t.Fatal(err)
	}
	gradient, err := x.Grad()
	if err != nil {
		t.Fatal(err)
	}
	defer gradient.Close()
	got, err := tensor.Read[float64](gradient)
	if err != nil || !reflect.DeepEqual(got, []float64{2, 2}) {
		t.Fatalf("custom VJP = %v, %v", got, err)
	}
}

func TestFiniteDifferenceGradientCheck(t *testing.T) {
	input, err := tensor.From([]float64{1.5, -2}, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer input.Close()

	result, err := CheckGradient(context.Background(), input, func(_ context.Context, tape *Tape, x *Variable) (*Variable, error) {
		squared, err := tape.Mul(x, x)
		if err != nil {
			return nil, err
		}
		shifted, err := tape.Add(squared, x)
		if err != nil {
			return nil, err
		}
		return tape.Sum(shifted)
	}, GradientCheckOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if !result.Passed || result.CheckedElements != 2 {
		t.Fatalf("gradient check result = %+v", result)
	}
}

func TestBackwardCancellationAndGradientSeedValidation(t *testing.T) {
	input, _ := tensor.From([]float32{1, 2}, 2)
	defer input.Close()
	tape := NewTape()
	defer tape.Close()
	x, err := tape.NewVariable(input, true)
	if err != nil {
		t.Fatal(err)
	}
	loss, err := tape.Sum(x)
	if err != nil {
		t.Fatal(err)
	}
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if err := tape.Backward(cancelled, loss); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled Backward = %v", err)
	}
	if _, err := x.Grad(); !errors.Is(err, errs.ErrNoGradient) {
		t.Fatalf("gradient after cancelled Backward = %v", err)
	}
	wrongSeed, _ := tensor.From([]float32{1, 1}, 2)
	defer wrongSeed.Close()
	if err := tape.BackwardWithGradient(context.Background(), loss, wrongSeed); !errors.Is(err, errs.ErrGradientShapeMismatch) {
		t.Fatalf("wrong seed = %v", err)
	}
	if err := tape.Backward(context.Background(), loss); err != nil {
		t.Fatal(err)
	}
}

func TestTapeReleaseInvalidatesVariables(t *testing.T) {
	input, _ := tensor.From([]float32{1}, 1)
	defer input.Close()
	tape := NewTape()
	variable, err := tape.NewVariable(input, true)
	if err != nil {
		t.Fatal(err)
	}
	if tape.NodeCount() != 1 {
		t.Fatalf("NodeCount = %d, want 1", tape.NodeCount())
	}
	if err := tape.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := variable.Value(); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("Value after Tape.Close = %v", err)
	}
	if err := tape.Close(); err != nil {
		t.Fatalf("repeated Tape.Close = %v", err)
	}
}

func TestRejectsNonFloatingRequiresGrad(t *testing.T) {
	integer, _ := tensor.From([]int32{1}, 1)
	defer integer.Close()
	tape := NewTape()
	defer tape.Close()
	if _, err := tape.NewVariable(integer, true); !errors.Is(err, errs.ErrUnsupportedDType) {
		t.Fatalf("integer requires-grad = %v", err)
	}
	floating, _ := tensor.From([]float32{1}, 1)
	defer floating.Close()
	if _, err := tape.NewVariable(floating, false); err != nil {
		t.Fatalf("non-grad float variable: %v", err)
	}
	if !dtype.Float32.IsFloat() {
		t.Fatal("test dtype invariant failed")
	}
}

package optim

import (
	"context"
	"math"
	"testing"

	"github.com/adhitiad/gotorch/src/autograd"
	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/nn"
	"github.com/adhitiad/gotorch/src/tensor"
)

// lossValue membaca nilai skalar dari simpul loss autograd.
func lossValue(t *testing.T, loss *autograd.Variable) float64 {
	t.Helper()
	value, err := loss.Value()
	if err != nil {
		t.Fatalf("loss.Value: %v", err)
	}
	defer value.Close()
	values, err := tensor.Read[float64](value)
	if err != nil {
		t.Fatalf("tensor.Read(loss): %v", err)
	}
	if len(values) != 1 {
		t.Fatalf("loss bukan skalar: %v", values)
	}
	return values[0]
}

// regressionLoss menghitung MSE secara langsung sebagai pembanding numerik.
func regressionLoss(weights, inputs, targets []float64) float64 {
	const rows, columns = 4, 2
	total := 0.0
	for row := 0; row < rows; row++ {
		prediction := 0.0
		for column := 0; column < columns; column++ {
			prediction += inputs[row*columns+column] * weights[column]
		}
		difference := prediction - targets[row]
		total += difference * difference
	}
	return total / rows
}

func TestFiniteDifferenceMatchesOptimizerGradient(t *testing.T) {
	inputs := []float64{1, 2, 0.5, -1, -0.5, 2, 3, -2}
	targets := []float64{4, -2, 1, 5}

	linear, err := nn.NewLinear("regressor", nn.LinearConfig{
		In: 2, Out: 1, DType: dtype.Float64, Bias: false, RNG: nn.NewRNG(17),
		WeightInit: nn.XavierUniform,
	})
	if err != nil {
		t.Fatalf("NewLinear: %v", err)
	}
	defer linear.Close()
	if err := linear.Weight.Update(func([]float64) ([]float64, error) {
		return []float64{1.25, -0.75}, nil
	}); err != nil {
		t.Fatalf("Weight.Update: %v", err)
	}

	inputTensor := testTensor(t, inputs, 4, 2)
	defer inputTensor.Close()
	targetTensor := testTensor(t, targets, 4, 1)
	defer targetTensor.Close()
	mse, err := nn.NewMSELoss("mse", nn.ReductionMean)
	if err != nil {
		t.Fatalf("NewMSELoss: %v", err)
	}
	defer mse.Close()

	tape, err := nn.NewTape()
	if err != nil {
		t.Fatalf("nn.NewTape: %v", err)
	}
	ctx, err := nn.NewContext(tape)
	if err != nil {
		t.Fatalf("nn.NewContext: %v", err)
	}
	inputVariable, err := tape.NewVariable(inputTensor, false)
	if err != nil {
		t.Fatalf("NewVariable(input): %v", err)
	}
	targetVariable, err := tape.NewVariable(targetTensor, false)
	if err != nil {
		t.Fatalf("NewVariable(target): %v", err)
	}
	prediction, err := linear.Forward(ctx, inputVariable)
	if err != nil {
		t.Fatalf("Linear.Forward: %v", err)
	}
	loss, err := mse.Forward(ctx, prediction, targetVariable)
	if err != nil {
		t.Fatalf("MSELoss.Forward: %v", err)
	}
	if err := tape.Backward(context.Background(), loss); err != nil {
		t.Fatalf("Backward: %v", err)
	}
	gradient, err := ctx.Gradient(linear.Weight)
	if err != nil {
		t.Fatalf("Gradient: %v", err)
	}
	analytical, err := tensor.Read[float64](gradient)
	if err != nil {
		t.Fatalf("tensor.Read(gradient): %v", err)
	}
	if err := gradient.Close(); err != nil {
		t.Fatalf("gradient.Close: %v", err)
	}
	if err := tape.Close(); err != nil {
		t.Fatalf("tape.Close: %v", err)
	}

	// Turunan analitik untuk L = mean((Xw - y)²) adalah 2/N · Xᵀ(Xw - y).
	const step = 1e-5
	for index := range analytical {
		current := valuesOf(t, &Param{Name: "regressor.weight", Parameter: linear.Weight})
		plus := append([]float64(nil), current...)
		minus := append([]float64(nil), current...)
		plus[index] += step
		minus[index] -= step
		if err := linear.Weight.Update(func([]float64) ([]float64, error) { return plus, nil }); err != nil {
			t.Fatalf("Update(+h): %v", err)
		}
		upper := regressionLoss(plus, inputs, targets)
		if err := linear.Weight.Update(func([]float64) ([]float64, error) { return minus, nil }); err != nil {
			t.Fatalf("Update(-h): %v", err)
		}
		lower := regressionLoss(minus, inputs, targets)
		if err := linear.Weight.Update(func([]float64) ([]float64, error) { return current, nil }); err != nil {
			t.Fatalf("Update(restore): %v", err)
		}
		numerical := (upper - lower) / (2 * step)
		if math.Abs(analytical[index]-numerical) > 1e-6 {
			t.Fatalf("gradien[%d]: analitik %g, finite difference %g", index, analytical[index], numerical)
		}
	}
}

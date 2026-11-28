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

func TestAdamConvergesOnLinearRegression(t *testing.T) {
	inputs := []float64{-1, -0.5, 0, 0.5, 1}
	targets := []float64{-1, 0, 1, 2, 3}
	inputTensor := testTensor(t, inputs, 5, 1)
	defer inputTensor.Close()
	targetTensor := testTensor(t, targets, 5, 1)
	defer targetTensor.Close()

	linear, err := nn.NewLinear("regressor", nn.LinearConfig{
		In: 1, Out: 1, DType: dtype.Float64, Bias: true, RNG: nn.NewRNG(5),
		WeightInit: nn.XavierUniform,
	})
	if err != nil {
		t.Fatalf("NewLinear: %v", err)
	}
	defer linear.Close()
	if err := linear.Weight.Update(func([]float64) ([]float64, error) { return []float64{0}, nil }); err != nil {
		t.Fatalf("Weight.Update: %v", err)
	}
	if err := linear.Bias.Update(func([]float64) ([]float64, error) { return []float64{0}, nil }); err != nil {
		t.Fatalf("Bias.Update: %v", err)
	}
	mse, err := nn.NewMSELoss("mse", nn.ReductionMean)
	if err != nil {
		t.Fatalf("NewMSELoss: %v", err)
	}
	defer mse.Close()
	optimizer, err := NewAdam(AdamConfig{LearningRate: 0.1})
	if err != nil {
		t.Fatalf("NewAdam: %v", err)
	}
	defer optimizer.Close()
	parameters := Parameters(linear)

	evaluate := func() float64 {
		t.Helper()
		tape, err := nn.NewTape()
		if err != nil {
			t.Fatalf("nn.NewTape: %v", err)
		}
		defer tape.Close()
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
			t.Fatalf("Forward: %v", err)
		}
		loss, err := mse.Forward(ctx, prediction, targetVariable)
		if err != nil {
			t.Fatalf("MSELoss: %v", err)
		}
		return lossValue(t, loss)
	}

	initialLoss := evaluate()
	for iteration := 0; iteration < 300; iteration++ {
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
			t.Fatalf("Forward: %v", err)
		}
		loss, err := mse.Forward(ctx, prediction, targetVariable)
		if err != nil {
			t.Fatalf("MSELoss: %v", err)
		}
		if err := tape.Backward(context.Background(), loss); err != nil {
			t.Fatalf("Backward: %v", err)
		}
		if err := optimizer.Step(ctx, parameters); err != nil {
			t.Fatalf("Step: %v", err)
		}
		if err := tape.Close(); err != nil {
			t.Fatalf("tape.Close: %v", err)
		}
	}
	finalLoss := evaluate()
	if initialLoss <= finalLoss {
		t.Fatalf("loss tidak turun: awal %g, akhir %g", initialLoss, finalLoss)
	}
	assertLess(t, "loss regresi akhir", finalLoss, 1e-3)
	weight := valuesOf(t, &Param{Name: "regressor.weight", Parameter: linear.Weight})[0]
	bias := valuesOf(t, &Param{Name: "regressor.bias", Parameter: linear.Bias})[0]
	assertClose(t, "weight regresi", []float64{weight}, []float64{2}, 1e-2)
	assertClose(t, "bias regresi", []float64{bias}, []float64{1}, 1e-2)
}

func TestSGDWithMomentumConvergesOnXOR(t *testing.T) {
	inputs := []float64{0, 0, 0, 1, 1, 0, 1, 1}
	targets := []float64{0, 1, 1, 0}
	inputTensor := testTensor(t, inputs, 4, 2)
	defer inputTensor.Close()
	targetTensor := testTensor(t, targets, 4, 1)
	defer targetTensor.Close()

	module, err := nn.NewModule("xor")
	if err != nil {
		t.Fatalf("NewModule: %v", err)
	}
	defer module.Close()
	hidden, err := nn.NewLinear("hidden", nn.LinearConfig{
		In: 2, Out: 4, DType: dtype.Float64, Bias: true, RNG: nn.NewRNG(11),
		WeightInit: nn.XavierUniform,
	})
	if err != nil {
		t.Fatalf("NewLinear(hidden): %v", err)
	}
	if err := module.RegisterModule(hidden); err != nil {
		t.Fatalf("RegisterModule(hidden): %v", err)
	}
	activation, err := nn.NewTanh("tanh")
	if err != nil {
		t.Fatalf("NewTanh: %v", err)
	}
	if err := module.RegisterModule(activation); err != nil {
		t.Fatalf("RegisterModule(tanh): %v", err)
	}
	output, err := nn.NewLinear("output", nn.LinearConfig{
		In: 4, Out: 1, DType: dtype.Float64, Bias: true, RNG: nn.NewRNG(23),
		WeightInit: nn.XavierUniform,
	})
	if err != nil {
		t.Fatalf("NewLinear(output): %v", err)
	}
	if err := module.RegisterModule(output); err != nil {
		t.Fatalf("RegisterModule(output): %v", err)
	}
	sigmoid, err := nn.NewSigmoid("sigmoid")
	if err != nil {
		t.Fatalf("NewSigmoid: %v", err)
	}
	if err := module.RegisterModule(sigmoid); err != nil {
		t.Fatalf("RegisterModule(sigmoid): %v", err)
	}
	mse, err := nn.NewMSELoss("mse", nn.ReductionMean)
	if err != nil {
		t.Fatalf("NewMSELoss: %v", err)
	}
	defer mse.Close()
	optimizer, err := NewSGD(SGDConfig{LearningRate: 0.5, Momentum: 0.9})
	if err != nil {
		t.Fatalf("NewSGD: %v", err)
	}
	defer optimizer.Close()
	parameters := Parameters(module)
	if len(parameters) != 4 {
		t.Fatalf("jumlah parameter XOR: got %d, want 4", len(parameters))
	}

	forward := func(ctx *nn.Context, inputVariable *autograd.Variable) (*autograd.Variable, error) {
		hiddenValue, err := hidden.Forward(ctx, inputVariable)
		if err != nil {
			return nil, err
		}
		activated, err := activation.Forward(ctx, hiddenValue)
		if err != nil {
			return nil, err
		}
		outputValue, err := output.Forward(ctx, activated)
		if err != nil {
			return nil, err
		}
		return sigmoid.Forward(ctx, outputValue)
	}

	run := func(train bool) float64 {
		t.Helper()
		tape, err := nn.NewTape()
		if err != nil {
			t.Fatalf("nn.NewTape: %v", err)
		}
		defer tape.Close()
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
		prediction, err := forward(ctx, inputVariable)
		if err != nil {
			t.Fatalf("forward: %v", err)
		}
		loss, err := mse.Forward(ctx, prediction, targetVariable)
		if err != nil {
			t.Fatalf("MSELoss: %v", err)
		}
		if train {
			if err := tape.Backward(context.Background(), loss); err != nil {
				t.Fatalf("Backward: %v", err)
			}
			if err := optimizer.Step(ctx, parameters); err != nil {
				t.Fatalf("Step: %v", err)
			}
		}
		return lossValue(t, loss)
	}

	initialLoss := run(false)
	for iteration := 0; iteration < 500; iteration++ {
		_ = run(true)
	}
	finalLoss := run(false)
	if initialLoss <= finalLoss {
		t.Fatalf("loss XOR tidak turun: awal %g, akhir %g", initialLoss, finalLoss)
	}

	tape, err := nn.NewTape()
	if err != nil {
		t.Fatalf("nn.NewTape: %v", err)
	}
	defer tape.Close()
	ctx, err := nn.NewContext(tape)
	if err != nil {
		t.Fatalf("nn.NewContext: %v", err)
	}
	inputVariable, err := tape.NewVariable(inputTensor, false)
	if err != nil {
		t.Fatalf("NewVariable(input): %v", err)
	}
	prediction, err := forward(ctx, inputVariable)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	predictionTensor, err := prediction.Value()
	if err != nil {
		t.Fatalf("prediction.Value: %v", err)
	}
	defer predictionTensor.Close()
	predictions, err := tensor.Read[float64](predictionTensor)
	if err != nil {
		t.Fatalf("tensor.Read(prediction): %v", err)
	}
	for index, value := range predictions {
		want := targets[index]
		if math.Abs(value-want) > 0.2 {
			t.Fatalf("XOR[%d]: got %g, want %g (loss %g)", index, value, want, finalLoss)
		}
	}
	assertLess(t, "loss XOR akhir", finalLoss, 0.02)
}

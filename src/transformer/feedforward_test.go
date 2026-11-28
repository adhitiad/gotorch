package transformer

import (
	"errors"
	"math"
	"testing"

	"github.com/adhitiad/gotorch/src/autograd"
	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/nn"
)

func referenceLinear(input, weight []float64, rows, columns int) []float64 {
	result := make([]float64, len(input)/columns*rows)
	for row := 0; row < len(input)/columns; row++ {
		for output := 0; output < rows; output++ {
			var total float64
			for column := 0; column < columns; column++ {
				total += input[row*columns+column] * weight[output*columns+column]
			}
			result[row*rows+output] = total
		}
	}
	return result
}

func referenceGELU(value float64) float64 {
	cubic := value*value*value*0.044715 + value
	return 0.5 * value * (1 + math.Tanh(0.7978845608028654*cubic))
}

func referenceSiLU(value float64) float64 {
	return value / (1 + math.Exp(-value))
}

func TestFeedForwardContractAndNumerics(t *testing.T) {
	settings := FeedForwardConfig{ModelDim: 2, HiddenDim: 3, DType: dtype.Float64, RNG: nn.NewRNG(19)}
	network, err := NewFeedForward("ffn", settings)
	if err != nil {
		t.Fatalf("NewFeedForward: %v", err)
	}
	defer network.Close()
	firstWeight := []float64{0.5, -0.2, 0.1, 0.8, -0.3, 0.6}
	secondWeight := []float64{0.3, 0.2, -0.4, -0.5, 0.6, 0.1}
	setProjectorWeights(t, network.First, firstWeight)
	setProjectorWeights(t, network.Second, secondWeight)

	inputs := []float64{1, -0.5, 0.2, 0.7, -1, 0.3, 0.4, 1.2}
	tape, ctx := newTestTape(t)
	defer tape.Close()
	input := newVariable(t, tape, newFloatTensor(t, dtype.Float64, []int64{2, 2, 2}, inputs), false)
	output, err := network.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	assertVariableDims(t, "ffn", output, 2, 2, 2)
	assertVariableDType(t, "ffn", output, dtype.Float64)

	hidden := referenceLinear(inputs, firstWeight, 3, 2)
	for index := range hidden {
		hidden[index] = referenceGELU(hidden[index])
	}
	want := referenceLinear(hidden, secondWeight, 2, 3)
	assertFloatClose(t, "ffn values", variableValues(t, output), want, 1e-12)

	result := checkTransformerGradient(t, newFloatTensor(t, dtype.Float64, []int64{1, 2, 2}, []float64{0.4, -0.6, 0.9, 0.1}),
		func(ctx *nn.Context, leaf *autograd.Variable) (*autograd.Variable, error) {
			return network.Forward(ctx, leaf)
		})
	if result.CheckedElements != 4 {
		t.Fatalf("elemen gradient: %d, want 4", result.CheckedElements)
	}

	checkParameterGradient(t, network.Second.(*nn.Linear).Weight,
		newFloatTensor(t, dtype.Float64, []int64{1, 2, 2}, []float64{0.7, -0.2, 0.3, 0.5}),
		func(ctx *nn.Context, leaf *autograd.Variable) (*autograd.Variable, error) {
			return network.Forward(ctx, leaf)
		})

	invalid := settings
	invalid.HiddenDim = 0
	if _, err := NewFeedForward("invalid", invalid); !errors.Is(err, errs.ErrInvalidShape) {
		t.Fatalf("dimensi invalid: error %v", err)
	}
}

func TestSwiGLUContractAndNumerics(t *testing.T) {
	for _, elementType := range []dtype.DType{dtype.Float64, dtype.Float32} {
		t.Run(elementType.String(), func(t *testing.T) {
			settings := FeedForwardConfig{ModelDim: 2, HiddenDim: 3, DType: elementType, RNG: nn.NewRNG(23)}
			network, err := NewSwiGLU("swiglu", settings)
			if err != nil {
				t.Fatalf("NewSwiGLU: %v", err)
			}
			defer network.Close()
			gateWeight := []float64{0.4, -0.1, 0.2, 0.7, -0.6, 0.3}
			upWeight := []float64{-0.2, 0.5, 0.8, -0.3, 0.1, 0.9}
			downWeight := []float64{0.6, 0.2, -0.4, -0.5, 0.3, 0.7}
			setProjectorWeights(t, network.Gate, gateWeight)
			setProjectorWeights(t, network.Up, upWeight)
			setProjectorWeights(t, network.Down, downWeight)

			inputs := []float64{0.8, -0.4, 0.1, 1.1, -0.7, 0.2, 0.5, -0.9}
			tape, ctx := newTestTape(t)
			defer tape.Close()
			input := newVariable(t, tape, newFloatTensor(t, elementType, []int64{2, 2, 2}, inputs), false)
			output, err := network.Forward(ctx, input)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}
			assertVariableDims(t, "swiglu", output, 2, 2, 2)
			assertVariableDType(t, "swiglu", output, elementType)

			gate := referenceLinear(inputs, gateWeight, 3, 2)
			up := referenceLinear(inputs, upWeight, 3, 2)
			gated := make([]float64, len(gate))
			for index := range gate {
				gated[index] = referenceSiLU(gate[index]) * up[index]
			}
			want := referenceLinear(gated, downWeight, 2, 3)
			assertFloatClose(t, "swiglu values", variableValues(t, output), want, 1e-6)

			if elementType == dtype.Float64 {
				result := checkTransformerGradient(t, newFloatTensor(t, dtype.Float64, []int64{1, 2, 2}, []float64{0.3, -0.8, 0.6, 0.2}),
					func(ctx *nn.Context, leaf *autograd.Variable) (*autograd.Variable, error) {
						return network.Forward(ctx, leaf)
					})
				if result.CheckedElements != 4 {
					t.Fatalf("elemen gradient: %d, want 4", result.CheckedElements)
				}
				checkParameterGradient(t, network.Gate.(*nn.Linear).Weight,
					newFloatTensor(t, dtype.Float64, []int64{1, 2, 2}, []float64{0.2, 0.7, -0.5, 0.4}),
					func(ctx *nn.Context, leaf *autograd.Variable) (*autograd.Variable, error) {
						return network.Forward(ctx, leaf)
					})
			}
		})
	}

	invalid := FeedForwardConfig{ModelDim: 0, HiddenDim: 3, DType: dtype.Float64, RNG: nn.NewRNG(1)}
	if _, err := NewSwiGLU("invalid", invalid); !errors.Is(err, errs.ErrInvalidShape) {
		t.Fatalf("dimensi invalid: error %v", err)
	}
}

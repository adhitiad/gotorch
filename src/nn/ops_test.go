package nn

import (
	"context"
	"testing"

	"github.com/adhitiad/gotorch/src/autograd"
	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/tensor"
)

func checkArgumentGradient(
	t *testing.T,
	index int,
	inputs []*tensor.Tensor,
	operation func(*Context, []*autograd.Variable) (*autograd.Variable, error),
) {
	t.Helper()
	checkLayerGradient(t, inputs[index], func(_ context.Context, tape *autograd.Tape, leaf *autograd.Variable) (*autograd.Variable, error) {
		ctx, err := NewContext(tape)
		if err != nil {
			return nil, err
		}
		variables := make([]*autograd.Variable, len(inputs))
		for position, input := range inputs {
			if position == index {
				variables[position] = leaf
				continue
			}
			variables[position] = variableOf(t, tape, input, false)
		}
		output, err := operation(ctx, variables)
		if err != nil {
			return nil, err
		}
		return scalarize(tape, output)
	})
}

func mustVariableData(t *testing.T, variable *autograd.Variable) tensorData {
	t.Helper()
	data, err := variableData(variable)
	if err != nil {
		t.Fatalf("variableData: %v", err)
	}
	return data
}

func TestMathOperationGradients(t *testing.T) {
	left := newTestTensor(t, dtype.Float64, []int64{2, 2}, []float64{1, 2, 3, 4})
	right := newTestTensor(t, dtype.Float64, []int64{2, 2}, []float64{0.5, -1, 1.5, 0.25})
	t.Run("matmul", func(t *testing.T) {
		for index := range []*tensor.Tensor{left, right} {
			checkArgumentGradient(t, index, []*tensor.Tensor{left, right}, func(ctx *Context, variables []*autograd.Variable) (*autograd.Variable, error) {
				return MatMul(ctx, variables[0], variables[1])
			})
		}
	})

	matrix := newTestTensor(t, dtype.Float64, []int64{2, 3}, []float64{1, -2, 0.5, 3, 0.25, -1})
	bias := newTestTensor(t, dtype.Float64, []int64{3}, []float64{0.1, -0.2, 0.3})
	t.Run("bias_add", func(t *testing.T) {
		for index := range []*tensor.Tensor{matrix, bias} {
			checkArgumentGradient(t, index, []*tensor.Tensor{matrix, bias}, func(ctx *Context, variables []*autograd.Variable) (*autograd.Variable, error) {
				return BiasAdd(ctx, variables[0], variables[1])
			})
		}
	})

	scale := newTestTensor(t, dtype.Float64, nil, []float64{2.5})
	t.Run("scale", func(t *testing.T) {
		checkArgumentGradient(t, 0, []*tensor.Tensor{matrix, scale}, func(ctx *Context, variables []*autograd.Variable) (*autograd.Variable, error) {
			return Scale(ctx, variables[0], mustVariableData(t, variables[1]).values[0])
		})
	})

	columns := newTestTensor(t, dtype.Float64, []int64{3}, []float64{0.5, 1.5, -0.75})
	t.Run("scale_last", func(t *testing.T) {
		for index := range []*tensor.Tensor{matrix, columns} {
			checkArgumentGradient(t, index, []*tensor.Tensor{matrix, columns}, func(ctx *Context, variables []*autograd.Variable) (*autograd.Variable, error) {
				return ScaleLast(ctx, variables[0], variables[1])
			})
		}
	})

	prefix := newTestTensor(t, dtype.Float64, []int64{2, 2, 2}, []float64{1, 2, 3, 4, 5, 6, 7, 8})
	prompt := newTestTensor(t, dtype.Float64, []int64{2, 2}, []float64{0.25, -0.5, 1, 1.5})
	t.Run("concat_time", func(t *testing.T) {
		for index := range []*tensor.Tensor{prefix, prompt} {
			checkArgumentGradient(t, index, []*tensor.Tensor{prefix, prompt}, func(ctx *Context, variables []*autograd.Variable) (*autograd.Variable, error) {
				return ConcatTime(ctx, variables[0], variables[1])
			})
		}
	})
	t.Run("concat_time_prefix", func(t *testing.T) {
		for index := range []*tensor.Tensor{prompt, prefix} {
			checkArgumentGradient(t, index, []*tensor.Tensor{prompt, prefix}, func(ctx *Context, variables []*autograd.Variable) (*autograd.Variable, error) {
				return ConcatTimePrefix(ctx, variables[0], variables[1])
			})
		}
	})

	t.Run("reshape", func(t *testing.T) {
		checkArgumentGradient(t, 0, []*tensor.Tensor{prefix}, func(ctx *Context, variables []*autograd.Variable) (*autograd.Variable, error) {
			return Reshape(ctx, variables[0], 4, 2)
		})
	})
	t.Run("transpose", func(t *testing.T) {
		checkArgumentGradient(t, 0, []*tensor.Tensor{matrix}, func(ctx *Context, variables []*autograd.Variable) (*autograd.Variable, error) {
			return Transpose(ctx, variables[0], 1, 0)
		})
	})
}

func layerNormOperation(t *testing.T, ctx *Context, variables []*autograd.Variable) (*autograd.Variable, error) {
	epsilon, err := floatControl(ctx, dtype.Float64, nil, 1e-5)
	if err != nil {
		return nil, err
	}
	inputs := []*autograd.Variable{variables[0], variables[1], variables[2], epsilon}
	return applyOperation(ctx, OpLayerNorm, inputs, func() (tensorData, error) {
		input, gamma, beta := mustVariableData(t, variables[0]), mustVariableData(t, variables[1]), mustVariableData(t, variables[2])
		output, _, err := layerNormData(input, gamma, beta, 1e-5)
		return output, err
	})
}

func rmsNormOperation(t *testing.T, ctx *Context, variables []*autograd.Variable) (*autograd.Variable, error) {
	epsilon, err := floatControl(ctx, dtype.Float64, nil, 1e-5)
	if err != nil {
		return nil, err
	}
	inputs := []*autograd.Variable{variables[0], variables[1], epsilon}
	return applyOperation(ctx, OpRMSNorm, inputs, func() (tensorData, error) {
		input, gamma := mustVariableData(t, variables[0]), mustVariableData(t, variables[1])
		output, _, err := rmsNormData(input, gamma, 1e-5)
		return output, err
	})
}

func TestNormalizationGradients(t *testing.T) {
	input := newTestTensor(t, dtype.Float64, []int64{2, 3}, []float64{-1.2, 0.4, 2.1, 0.7, -0.3, 1.6})
	gamma := newTestTensor(t, dtype.Float64, []int64{3}, []float64{1.1, 0.8, -0.5})
	beta := newTestTensor(t, dtype.Float64, []int64{3}, []float64{0.2, -0.1, 0.3})
	for index := range []*tensor.Tensor{input, gamma, beta} {
		checkArgumentGradient(t, index, []*tensor.Tensor{input, gamma, beta}, func(ctx *Context, variables []*autograd.Variable) (*autograd.Variable, error) {
			return layerNormOperation(t, ctx, variables)
		})
	}
	for index := range []*tensor.Tensor{input, gamma} {
		checkArgumentGradient(t, index, []*tensor.Tensor{input, gamma}, func(ctx *Context, variables []*autograd.Variable) (*autograd.Variable, error) {
			return rmsNormOperation(t, ctx, variables)
		})
	}
}

func TestLossGradients(t *testing.T) {
	prediction := newTestTensor(t, dtype.Float64, []int64{2, 2}, []float64{1.2, -0.4, 0.3, 2.1})
	target := newTestTensor(t, dtype.Float64, []int64{2, 2}, []float64{0.5, -1, 1, 0})
	for _, reduction := range []Reduction{ReductionMean, ReductionSum, ReductionNone} {
		t.Run(string(reduction), func(t *testing.T) {
			checkArgumentGradient(t, 0, []*tensor.Tensor{prediction, target}, func(ctx *Context, variables []*autograd.Variable) (*autograd.Variable, error) {
				loss, err := NewMSELoss("mse", reduction)
				if err != nil {
					return nil, err
				}
				defer loss.Close()
				return loss.Forward(ctx, variables[0], variables[1])
			})
		})
	}

	logits := newTestTensor(t, dtype.Float64, []int64{2, 3}, []float64{2, 0.5, -1, 0.2, 1.5, -0.3})
	classes := newIntTensor(t, []int64{2}, []int64{2, 1})
	t.Run("cross_entropy", func(t *testing.T) {
		checkArgumentGradient(t, 0, []*tensor.Tensor{logits, classes}, func(ctx *Context, variables []*autograd.Variable) (*autograd.Variable, error) {
			loss, err := NewCrossEntropyLoss("cross_entropy")
			if err != nil {
				return nil, err
			}
			defer loss.Close()
			return loss.Forward(ctx, variables[0], variables[1])
		})
	})
}

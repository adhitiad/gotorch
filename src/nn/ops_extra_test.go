package nn

import (
	"testing"

	"github.com/adhitiad/gotorch/src/autograd"
	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/tensor"
)

func convOperation(t *testing.T, ctx *Context, variables []*autograd.Variable) (*autograd.Variable, error) {
	t.Helper()
	config, err := variableIntData(variables[3])
	if err != nil {
		return nil, err
	}
	return applyOperation(ctx, OpConv2D, variables, func() (tensorData, error) {
		input, err := variableData(variables[0])
		if err != nil {
			return tensorData{}, err
		}
		weight, err := variableData(variables[1])
		if err != nil {
			return tensorData{}, err
		}
		bias, err := variableData(variables[2])
		if err != nil {
			return tensorData{}, err
		}
		return conv2DData(input, weight, bias, config)
	})
}

func poolOperation(t *testing.T, op string, ctx *Context, variables []*autograd.Variable) (*autograd.Variable, error) {
	t.Helper()
	config, err := variableIntData(variables[1])
	if err != nil {
		return nil, err
	}
	return applyOperation(ctx, op, variables, func() (tensorData, error) {
		input, err := variableData(variables[0])
		if err != nil {
			return tensorData{}, err
		}
		if op == OpMaxPool2D {
			return maxPoolData(input, config)
		}
		return avgPoolData(input, config)
	})
}

func TestConv2DForwardAndGradients(t *testing.T) {
	input := newTestTensor(t, dtype.Float64, []int64{1, 1, 2, 2}, []float64{1, 2, 3, 4})
	weight := newTestTensor(t, dtype.Float64, []int64{1, 1, 2, 2}, []float64{0.5, -1, 1.5, 0.25})
	bias := newTestTensor(t, dtype.Float64, []int64{1}, []float64{0.1})
	config := newIntTensor(t, []int64{2}, []int64{1, 0})
	inputs := []*tensor.Tensor{input, weight, bias, config}

	tape, ctx := newTestContext(t)
	defer tape.Close()
	variables := make([]*autograd.Variable, len(inputs))
	for index, value := range inputs {
		variables[index] = variableOf(t, tape, value, index < 3)
	}
	output, err := convOperation(t, ctx, variables)
	if err != nil {
		t.Fatalf("conv2d: %v", err)
	}
	assertClose(t, "conv2d forward", tensorValues(t, mustValue(t, output)), []float64{4.1}, 1e-12)

	for index := 0; index < 3; index++ {
		checkArgumentGradient(t, index, inputs, func(ctx *Context, variables []*autograd.Variable) (*autograd.Variable, error) {
			return convOperation(t, ctx, variables)
		})
	}
}

func TestPoolForwardAndGradients(t *testing.T) {
	input := newTestTensor(t, dtype.Float64, []int64{1, 1, 2, 2}, []float64{1, 4, 3, 2})
	config := newIntTensor(t, []int64{4}, []int64{2, 2, 2, 0})
	inputs := []*tensor.Tensor{input, config}

	t.Run("max", func(t *testing.T) {
		tape, ctx := newTestContext(t)
		defer tape.Close()
		variables := []*autograd.Variable{
			variableOf(t, tape, input, true),
			variableOf(t, tape, config, false),
		}
		output, err := poolOperation(t, OpMaxPool2D, ctx, variables)
		if err != nil {
			t.Fatalf("max pool: %v", err)
		}
		assertClose(t, "max pool forward", tensorValues(t, mustValue(t, output)), []float64{4}, 1e-12)
		checkArgumentGradient(t, 0, inputs, func(ctx *Context, variables []*autograd.Variable) (*autograd.Variable, error) {
			return poolOperation(t, OpMaxPool2D, ctx, variables)
		})
	})

	t.Run("average", func(t *testing.T) {
		tape, ctx := newTestContext(t)
		defer tape.Close()
		variables := []*autograd.Variable{
			variableOf(t, tape, input, true),
			variableOf(t, tape, config, false),
		}
		output, err := poolOperation(t, OpAvgPool2D, ctx, variables)
		if err != nil {
			t.Fatalf("average pool: %v", err)
		}
		assertClose(t, "average pool forward", tensorValues(t, mustValue(t, output)), []float64{2.5}, 1e-12)
		checkArgumentGradient(t, 0, inputs, func(ctx *Context, variables []*autograd.Variable) (*autograd.Variable, error) {
			return poolOperation(t, OpAvgPool2D, ctx, variables)
		})
	})
}

func TestEmbeddingForwardAndGradients(t *testing.T) {
	weight := newTestTensor(t, dtype.Float64, []int64{3, 2}, []float64{1, 2, 3, 4, 5, 6})
	indices := newIntTensor(t, []int64{2}, []int64{0, 2})
	inputs := []*tensor.Tensor{weight, indices}

	tape, ctx := newTestContext(t)
	defer tape.Close()
	variables := []*autograd.Variable{
		variableOf(t, tape, weight, true),
		variableOf(t, tape, indices, false),
	}
	output, err := Embedding(ctx, variables[0], variables[1])
	if err != nil {
		t.Fatalf("embedding: %v", err)
	}
	assertClose(t, "embedding forward", tensorValues(t, mustValue(t, output)), []float64{1, 2, 5, 6}, 1e-12)

	checkArgumentGradient(t, 0, inputs, func(ctx *Context, variables []*autograd.Variable) (*autograd.Variable, error) {
		return Embedding(ctx, variables[0], variables[1])
	})
}

func TestRoPEForwardAndGradients(t *testing.T) {
	input := newTestTensor(t, dtype.Float64, []int64{1, 1, 2, 2}, []float64{1, 2, 3, 4})
	cosine := newTestTensor(t, dtype.Float64, []int64{2, 1}, []float64{0.8, 0.5})
	sine := newTestTensor(t, dtype.Float64, []int64{2, 1}, []float64{0.6, 0.8660254037844386})
	inputs := []*tensor.Tensor{input, cosine, sine}

	tape, ctx := newTestContext(t)
	defer tape.Close()
	variables := []*autograd.Variable{
		variableOf(t, tape, input, true),
		variableOf(t, tape, cosine, false),
		variableOf(t, tape, sine, false),
	}
	output, err := RoPE(ctx, variables[0], variables[1], variables[2])
	if err != nil {
		t.Fatalf("rope: %v", err)
	}
	if got := tensorValues(t, mustValue(t, output)); len(got) != 4 {
		t.Fatalf("rope forward: got %d values, want 4", len(got))
	}

	checkArgumentGradient(t, 0, inputs, func(ctx *Context, variables []*autograd.Variable) (*autograd.Variable, error) {
		return RoPE(ctx, variables[0], variables[1], variables[2])
	})
}

func TestAttentionGradients(t *testing.T) {
	query := newTestTensor(t, dtype.Float64, []int64{2, 2, 2, 2}, []float64{
		0.2, -0.4, 0.5, 0.1, 0.7, 0.3, -0.2, 0.6,
		-0.1, 0.4, -0.6, 0.2, 0.3, -0.7, 0.5, 0.9,
	})
	key := newTestTensor(t, dtype.Float64, []int64{2, 1, 2, 2}, []float64{0.9, 0.1, -0.3, 0.8, -0.2, 0.6, 0.4, -0.5})
	value := newTestTensor(t, dtype.Float64, []int64{2, 1, 2, 2}, []float64{1.2, -0.5, 0.4, 0.7, 0.8, 0.2, -0.1, 0.6})
	inputs := []*tensor.Tensor{query, key, value}
	for index := range inputs {
		checkArgumentGradient(t, index, inputs, func(ctx *Context, variables []*autograd.Variable) (*autograd.Variable, error) {
			return Attention(ctx, variables[0], variables[1], variables[2], nil)
		})
	}
}

func TestAttentionMaskForward(t *testing.T) {
	query := newTestTensor(t, dtype.Float64, []int64{1, 1, 2, 2}, []float64{1, 0, 0, 1})
	key := newTestTensor(t, dtype.Float64, []int64{1, 1, 2, 2}, []float64{1, 0, 0, 1})
	value := newTestTensor(t, dtype.Float64, []int64{1, 1, 2, 2}, []float64{5, 6, 7, 8})
	mask := newTestTensor(t, dtype.Float64, []int64{1, 2, 2}, []float64{0, float64(-1e9), float64(-1e9), 0})

	tape, ctx := newTestContext(t)
	defer tape.Close()
	output, err := Attention(ctx,
		variableOf(t, tape, query, false),
		variableOf(t, tape, key, false),
		variableOf(t, tape, value, false),
		variableOf(t, tape, mask, false),
	)
	if err != nil {
		t.Fatalf("masked attention: %v", err)
	}
	assertClose(t, "masked attention forward", tensorValues(t, mustValue(t, output)), []float64{5, 6, 7, 8}, 1e-9)
}

package nn

import (
	"github.com/adhitiad/gotorch/src/autograd"
	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/tensor"
)

func variableData(variable *autograd.Variable) (tensorData, error) {
	if variable == nil {
		return tensorData{}, errs.ErrInvalidArgument
	}
	value, err := variable.Value()
	if err != nil {
		return tensorData{}, err
	}
	defer value.Close()
	return readTensorData(value)
}

func applyOperation(ctx *Context, name string, inputs []*autograd.Variable, compute func() (tensorData, error)) (*autograd.Variable, error) {
	if ctx == nil || ctx.Tape == nil || compute == nil {
		return nil, errs.ErrInvalidArgument
	}
	output, err := compute()
	if err != nil {
		return nil, err
	}
	value, err := output.tensor()
	if err != nil {
		return nil, err
	}
	variable, err := ctx.Tape.Apply(name, inputs, value)
	_ = value.Close()
	if err != nil {
		return nil, err
	}
	return variable, nil
}

func unaryOperation(ctx *Context, name string, input *autograd.Variable, compute func(tensorData) tensorData) (*autograd.Variable, error) {
	data, err := variableData(input)
	if err != nil {
		return nil, err
	}
	return applyOperation(ctx, name, []*autograd.Variable{input}, func() (tensorData, error) { return compute(data), nil })
}

func newParameterValue(elementType dtype.DType, dims ...int64) (*tensor.Tensor, error) {
	value, err := tensor.NewTensor(elementType, dims...)
	if err != nil {
		return nil, err
	}
	return value, nil
}

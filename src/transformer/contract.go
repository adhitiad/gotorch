// Package transformer menyediakan primitive transformer berbasis nn dan autograd.
package transformer

import (
	"fmt"

	"github.com/adhitiad/gotorch/src/autograd"
	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/nn"
	"github.com/adhitiad/gotorch/src/tensor"
)

// Projector adalah layer linear yang bisa diganti adapter PEFT.
type Projector interface {
	nn.Module
	Forward(*nn.Context, *autograd.Variable) (*autograd.Variable, error)
	InputDim() int64
	OutputDim() int64
}

func variableFromTensor(ctx *nn.Context, value *tensor.Tensor, requiresGrad bool) (*autograd.Variable, error) {
	if ctx == nil || ctx.Tape == nil || value == nil {
		return nil, errs.ErrInvalidArgument
	}
	variable, err := ctx.Tape.NewVariable(value, requiresGrad)
	_ = value.Close()
	return variable, err
}

func floatTensor(elementType dtype.DType, dims []int64, values []float64) (*tensor.Tensor, error) {
	if !isFloat(elementType) {
		return nil, errs.ErrUnsupportedDType
	}
	value, err := tensor.NewTensor(elementType, dims...)
	if err != nil {
		return nil, err
	}
	switch elementType {
	case dtype.Float64:
		err = tensor.Write(value, values)
	case dtype.Float32:
		converted := make([]float32, len(values))
		for index, item := range values {
			converted[index] = float32(item)
		}
		err = tensor.Write(value, converted)
	default:
		err = errs.ErrUnsupportedDType
	}
	if err != nil {
		_ = value.Close()
		return nil, err
	}
	return value, nil
}

func isFloat(elementType dtype.DType) bool {
	return elementType == dtype.Float32 || elementType == dtype.Float64
}

func resolveDType(elementType dtype.DType) dtype.DType {
	if isFloat(elementType) {
		return elementType
	}
	return dtype.Float64
}

func tensorDims(value *tensor.Tensor) ([]int64, dtype.DType, error) {
	metadata, err := value.Metadata()
	if err != nil {
		return nil, dtype.Invalid, err
	}
	return metadata.Shape().Dims(), metadata.DType(), nil
}

func requireShape(label string, dims []int64, want ...int64) error {
	if len(dims) != len(want) {
		return fmt.Errorf("%w: %s has rank %d, want %d", errs.ErrInvalidShape, label, len(dims), len(want))
	}
	for index := range want {
		if want[index] >= 0 && dims[index] != want[index] {
			return fmt.Errorf("%w: %s dims %v, want %v", errs.ErrInvalidShape, label, dims, want)
		}
	}
	return nil
}

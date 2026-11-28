package nn

import (
	"context"

	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/tensor"
)

func reshapeData(input tensorData, dims []int64) (tensorData, error) {
	if len(dims) == 0 {
		return tensorData{}, errs.ErrInvalidShape
	}
	source, err := elementCount(input.dims)
	if err != nil {
		return tensorData{}, err
	}
	target, err := elementCount(dims)
	if err != nil {
		return tensorData{}, err
	}
	if source != target {
		return tensorData{}, errs.ErrGradientShapeMismatch
	}
	return input.withValues(floatSlice(input.values)).withDims(dims), nil
}

func transposeData(input tensorData, axes []int64) (tensorData, error) {
	rank := len(input.dims)
	if rank == 0 || len(axes) != rank {
		return tensorData{}, errs.ErrInvalidShape
	}
	used := make([]bool, rank)
	sourceStrides := make([]int, rank)
	for axis := 0; axis < rank; axis++ {
		sourceStrides[axis] = 1
		for dimension := axis + 1; dimension < rank; dimension++ {
			sourceStrides[axis] *= int(input.dims[dimension])
		}
	}
	outputDims := make([]int64, rank)
	for index, axis := range axes {
		if axis < 0 || axis >= int64(rank) || used[axis] {
			return tensorData{}, errs.ErrInvalidArgument
		}
		used[axis] = true
		outputDims[index] = input.dims[axis]
	}
	outputStrides := make([]int, rank)
	for axis := 0; axis < rank; axis++ {
		outputStrides[axis] = 1
		for dimension := axis + 1; dimension < rank; dimension++ {
			outputStrides[axis] *= int(outputDims[dimension])
		}
	}
	count, err := elementCount(input.dims)
	if err != nil {
		return tensorData{}, err
	}
	values := make([]float64, count)
	for index := range values {
		remainder := index
		source := 0
		for outputAxis := 0; outputAxis < rank; outputAxis++ {
			coordinate := remainder / outputStrides[outputAxis]
			remainder %= outputStrides[outputAxis]
			source += coordinate * sourceStrides[axes[outputAxis]]
		}
		values[index] = input.values[source]
	}
	return input.withValues(values).withDims(outputDims), nil
}

func inverseAxes(axes []int64) ([]int64, error) {
	result := make([]int64, len(axes))
	for index, axis := range axes {
		if axis < 0 || axis >= int64(len(axes)) {
			return nil, errs.ErrInvalidArgument
		}
		result[axis] = int64(index)
	}
	return result, nil
}

func reshapeBackward(_ context.Context, gradOutput *tensor.Tensor, inputs []*tensor.Tensor, _ *tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(inputs) != 2 {
		return nil, errs.ErrInvalidArgument
	}
	grad, err := dataOf(gradOutput)
	if err != nil {
		return nil, err
	}
	metadata, err := inputs[0].Metadata()
	if err != nil {
		return nil, err
	}
	restored, err := reshapeData(grad, metadata.Shape().Dims())
	if err != nil {
		return nil, err
	}
	return tensorResults(restored, tensorData{})
}

func transposeBackward(_ context.Context, gradOutput *tensor.Tensor, inputs []*tensor.Tensor, _ *tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(inputs) != 2 {
		return nil, errs.ErrInvalidArgument
	}
	grad, err := dataOf(gradOutput)
	if err != nil {
		return nil, err
	}
	axes, err := readIntTensor(inputs[1])
	if err != nil {
		return nil, err
	}
	inverse, err := inverseAxes(axes)
	if err != nil {
		return nil, err
	}
	restored, err := transposeData(grad, inverse)
	if err != nil {
		return nil, err
	}
	return tensorResults(restored, tensorData{})
}

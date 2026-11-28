package nn

import (
	"context"
	"math"

	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/tensor"
)

type poolShape struct {
	batch, channels, height, width int
	kernelH, kernelW, stride, pad  int
	outHeight, outWidth            int
}

func poolLayout(input tensorData, config []int64) (poolShape, error) {
	if err := requireRank(input, 4); err != nil {
		return poolShape{}, err
	}
	if len(config) != 4 || config[0] <= 0 || config[1] <= 0 || config[2] <= 0 || config[3] < 0 {
		return poolShape{}, errs.ErrInvalidArgument
	}
	shape := poolShape{
		batch:    int(input.dims[0]),
		channels: int(input.dims[1]),
		height:   int(input.dims[2]),
		width:    int(input.dims[3]),
		kernelH:  int(config[0]),
		kernelW:  int(config[1]),
		stride:   int(config[2]),
		pad:      int(config[3]),
	}
	if shape.batch <= 0 || shape.channels <= 0 || shape.height <= 0 || shape.width <= 0 {
		return poolShape{}, errs.ErrInvalidShape
	}
	shape.outHeight = (shape.height+2*shape.pad-shape.kernelH)/shape.stride + 1
	shape.outWidth = (shape.width+2*shape.pad-shape.kernelW)/shape.stride + 1
	if shape.outHeight <= 0 || shape.outWidth <= 0 ||
		shape.outHeight*shape.stride > shape.height+2*shape.pad ||
		shape.outWidth*shape.stride > shape.width+2*shape.pad {
		return poolShape{}, errs.ErrInvalidShape
	}
	return shape, nil
}

func maxPoolData(input tensorData, config []int64) (tensorData, error) {
	shape, err := poolLayout(input, config)
	if err != nil {
		return tensorData{}, err
	}
	output := make([]float64, shape.batch*shape.channels*shape.outHeight*shape.outWidth)
	plane := shape.height * shape.width
	outPlane := shape.outHeight * shape.outWidth
	for batch := 0; batch < shape.batch; batch++ {
		for channel := 0; channel < shape.channels; channel++ {
			base := (batch*shape.channels + channel) * plane
			for outRow := 0; outRow < shape.outHeight; outRow++ {
				for outColumn := 0; outColumn < shape.outWidth; outColumn++ {
					maximum := math.Inf(-1)
					index := -1
					for kernelRow := 0; kernelRow < shape.kernelH; kernelRow++ {
						for kernelColumn := 0; kernelColumn < shape.kernelW; kernelColumn++ {
							row := outRow*shape.stride + kernelRow - shape.pad
							column := outColumn*shape.stride + kernelColumn - shape.pad
							if row < 0 || row >= shape.height || column < 0 || column >= shape.width {
								continue
							}
							candidate := base + row*shape.width + column
							if index < 0 || input.values[candidate] > maximum {
								maximum = input.values[candidate]
								index = candidate
							}
						}
					}
					outIndex := (batch*shape.channels+channel)*outPlane + outRow*shape.outWidth + outColumn
					if index < 0 {
						return tensorData{}, errs.ErrInvalidShape
					}
					output[outIndex] = maximum
				}
			}
		}
	}
	return input.withValues(output).withDims([]int64{int64(shape.batch), int64(shape.channels), int64(shape.outHeight), int64(shape.outWidth)}), nil
}

func avgPoolData(input tensorData, config []int64) (tensorData, error) {
	shape, err := poolLayout(input, config)
	if err != nil {
		return tensorData{}, err
	}
	output := make([]float64, shape.batch*shape.channels*shape.outHeight*shape.outWidth)
	plane := shape.height * shape.width
	outPlane := shape.outHeight * shape.outWidth
	for batch := 0; batch < shape.batch; batch++ {
		for channel := 0; channel < shape.channels; channel++ {
			base := (batch*shape.channels + channel) * plane
			for outRow := 0; outRow < shape.outHeight; outRow++ {
				for outColumn := 0; outColumn < shape.outWidth; outColumn++ {
					var total float64
					var count int
					for kernelRow := 0; kernelRow < shape.kernelH; kernelRow++ {
						for kernelColumn := 0; kernelColumn < shape.kernelW; kernelColumn++ {
							row := outRow*shape.stride + kernelRow - shape.pad
							column := outColumn*shape.stride + kernelColumn - shape.pad
							if row < 0 || row >= shape.height || column < 0 || column >= shape.width {
								continue
							}
							total += input.values[base+row*shape.width+column]
							count++
						}
					}
					outIndex := (batch*shape.channels+channel)*outPlane + outRow*shape.outWidth + outColumn
					output[outIndex] = total / float64(count)
				}
			}
		}
	}
	return input.withValues(output).withDims([]int64{int64(shape.batch), int64(shape.channels), int64(shape.outHeight), int64(shape.outWidth)}), nil
}

func maxPool2DBackward(_ context.Context, gradOutput *tensor.Tensor, inputs []*tensor.Tensor, _ *tensor.Tensor) ([]*tensor.Tensor, error) {
	grad, input, _, shape, err := readPoolBackward(gradOutput, inputs)
	if err != nil {
		return nil, err
	}
	config, err := readIntTensor(inputs[1])
	if err != nil {
		return nil, err
	}
	indices, err := maxPoolIndices(input, config)
	if err != nil {
		return nil, err
	}
	if len(indices) != len(grad.values) {
		return nil, errs.ErrGradientShapeMismatch
	}
	gradInput := make([]float64, len(input.values))
	for index, position := range indices {
		if position < 0 || position >= len(gradInput) {
			return nil, errs.ErrInvalidState
		}
		gradInput[position] += grad.values[index]
	}
	_ = shape
	return tensorResults(input.withValues(gradInput), tensorData{})
}

func avgPool2DBackward(_ context.Context, gradOutput *tensor.Tensor, inputs []*tensor.Tensor, _ *tensor.Tensor) ([]*tensor.Tensor, error) {
	grad, input, _, shape, err := readPoolBackward(gradOutput, inputs)
	if err != nil {
		return nil, err
	}
	gradInput := make([]float64, len(input.values))
	plane := shape.height * shape.width
	outPlane := shape.outHeight * shape.outWidth
	for batch := 0; batch < shape.batch; batch++ {
		for channel := 0; channel < shape.channels; channel++ {
			base := (batch*shape.channels + channel) * plane
			for outRow := 0; outRow < shape.outHeight; outRow++ {
				for outColumn := 0; outColumn < shape.outWidth; outColumn++ {
					var positions []int
					for kernelRow := 0; kernelRow < shape.kernelH; kernelRow++ {
						for kernelColumn := 0; kernelColumn < shape.kernelW; kernelColumn++ {
							row := outRow*shape.stride + kernelRow - shape.pad
							column := outColumn*shape.stride + kernelColumn - shape.pad
							if row < 0 || row >= shape.height || column < 0 || column >= shape.width {
								continue
							}
							positions = append(positions, base+row*shape.width+column)
						}
					}
					if len(positions) == 0 {
						return nil, errs.ErrInvalidShape
					}
					upstream := grad.values[(batch*shape.channels+channel)*outPlane+outRow*shape.outWidth+outColumn] / float64(len(positions))
					for _, position := range positions {
						gradInput[position] += upstream
					}
				}
			}
		}
	}
	return tensorResults(input.withValues(gradInput), tensorData{})
}

func maxPoolIndices(input tensorData, config []int64) ([]int, error) {
	shape, err := poolLayout(input, config)
	if err != nil {
		return nil, err
	}
	indices := make([]int, shape.batch*shape.channels*shape.outHeight*shape.outWidth)
	plane := shape.height * shape.width
	outPlane := shape.outHeight * shape.outWidth
	for batch := 0; batch < shape.batch; batch++ {
		for channel := 0; channel < shape.channels; channel++ {
			base := (batch*shape.channels + channel) * plane
			for outRow := 0; outRow < shape.outHeight; outRow++ {
				for outColumn := 0; outColumn < shape.outWidth; outColumn++ {
					maximum := math.Inf(-1)
					index := -1
					for kernelRow := 0; kernelRow < shape.kernelH; kernelRow++ {
						for kernelColumn := 0; kernelColumn < shape.kernelW; kernelColumn++ {
							row := outRow*shape.stride + kernelRow - shape.pad
							column := outColumn*shape.stride + kernelColumn - shape.pad
							if row < 0 || row >= shape.height || column < 0 || column >= shape.width {
								continue
							}
							candidate := base + row*shape.width + column
							if index < 0 || input.values[candidate] > maximum {
								maximum = input.values[candidate]
								index = candidate
							}
						}
					}
					if index < 0 {
						return nil, errs.ErrInvalidShape
					}
					indices[(batch*shape.channels+channel)*outPlane+outRow*shape.outWidth+outColumn] = index
				}
			}
		}
	}
	return indices, nil
}

func readPoolBackward(gradOutput *tensor.Tensor, inputs []*tensor.Tensor) (tensorData, tensorData, tensorData, poolShape, error) {
	if len(inputs) != 2 {
		return tensorData{}, tensorData{}, tensorData{}, poolShape{}, errs.ErrInvalidArgument
	}
	grad, err := dataOf(gradOutput)
	if err != nil {
		return tensorData{}, tensorData{}, tensorData{}, poolShape{}, err
	}
	input, err := dataOf(inputs[0])
	if err != nil {
		return tensorData{}, tensorData{}, tensorData{}, poolShape{}, err
	}
	config, err := readIntTensor(inputs[1])
	if err != nil {
		return tensorData{}, tensorData{}, tensorData{}, poolShape{}, err
	}
	shape, err := poolLayout(input, config)
	if err != nil {
		return tensorData{}, tensorData{}, tensorData{}, poolShape{}, err
	}
	expected := shape.batch * shape.channels * shape.outHeight * shape.outWidth
	if len(grad.values) != expected {
		return tensorData{}, tensorData{}, tensorData{}, poolShape{}, errs.ErrGradientShapeMismatch
	}
	return grad, input, tensorData{}, shape, nil
}

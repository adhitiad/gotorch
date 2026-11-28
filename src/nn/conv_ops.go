package nn

import (
	"context"

	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/tensor"
)

type convShape struct {
	batch, inChannels, height, width int
	outChannels, kernelH, kernelW    int
	stride, padding                  int
	outHeight, outWidth              int
}

func convLayout(input, weight tensorData, config []int64) (convShape, error) {
	if err := requireRank(input, 4); err != nil {
		return convShape{}, err
	}
	if err := requireRank(weight, 4); err != nil {
		return convShape{}, err
	}
	if len(config) != 2 || config[0] <= 0 || config[1] < 0 {
		return convShape{}, errs.ErrInvalidArgument
	}
	shape := convShape{
		batch:       int(input.dims[0]),
		inChannels:  int(input.dims[1]),
		height:      int(input.dims[2]),
		width:       int(input.dims[3]),
		outChannels: int(weight.dims[0]),
		kernelH:     int(weight.dims[2]),
		kernelW:     int(weight.dims[3]),
		stride:      int(config[0]),
		padding:     int(config[1]),
	}
	if shape.batch <= 0 || shape.inChannels <= 0 || shape.height <= 0 || shape.width <= 0 ||
		shape.outChannels <= 0 || shape.kernelH <= 0 || shape.kernelW <= 0 {
		return convShape{}, errs.ErrInvalidShape
	}
	if weight.dims[1] != int64(shape.inChannels) {
		return convShape{}, errs.ErrGradientShapeMismatch
	}
	shape.outHeight = (shape.height+2*shape.padding-shape.kernelH)/shape.stride + 1
	shape.outWidth = (shape.width+2*shape.padding-shape.kernelW)/shape.stride + 1
	if shape.outHeight <= 0 || shape.outWidth <= 0 ||
		shape.outHeight*shape.stride > shape.height+2*shape.padding ||
		shape.outWidth*shape.stride > shape.width+2*shape.padding {
		return convShape{}, errs.ErrInvalidShape
	}
	return shape, nil
}

func conv2DData(input, weight, bias tensorData, config []int64) (tensorData, error) {
	shape, err := convLayout(input, weight, config)
	if err != nil {
		return tensorData{}, err
	}
	if len(bias.values) != shape.outChannels {
		return tensorData{}, errs.ErrGradientShapeMismatch
	}
	output := make([]float64, shape.batch*shape.outChannels*shape.outHeight*shape.outWidth)
	inputStride := shape.inChannels * shape.height * shape.width
	weightStride := shape.inChannels * shape.kernelH * shape.kernelW
	outputStride := shape.outChannels * shape.outHeight * shape.outWidth
	for batch := 0; batch < shape.batch; batch++ {
		for channel := 0; channel < shape.outChannels; channel++ {
			for outRow := 0; outRow < shape.outHeight; outRow++ {
				for outColumn := 0; outColumn < shape.outWidth; outColumn++ {
					var total float64
					for inChannel := 0; inChannel < shape.inChannels; inChannel++ {
						for kernelRow := 0; kernelRow < shape.kernelH; kernelRow++ {
							for kernelColumn := 0; kernelColumn < shape.kernelW; kernelColumn++ {
								inputRow := outRow*shape.stride + kernelRow - shape.padding
								inputColumn := outColumn*shape.stride + kernelColumn - shape.padding
								if inputRow < 0 || inputRow >= shape.height || inputColumn < 0 || inputColumn >= shape.width {
									continue
								}
								inputIndex := batch*inputStride + inChannel*shape.height*shape.width + inputRow*shape.width + inputColumn
								weightIndex := channel*weightStride + inChannel*shape.kernelH*shape.kernelW + kernelRow*shape.kernelW + kernelColumn
								total += input.values[inputIndex] * weight.values[weightIndex]
							}
						}
					}
					outputIndex := batch*outputStride + channel*shape.outHeight*shape.outWidth + outRow*shape.outWidth + outColumn
					output[outputIndex] = total + bias.values[channel]
				}
			}
		}
	}
	return input.withValues(output).withDims([]int64{int64(shape.batch), int64(shape.outChannels), int64(shape.outHeight), int64(shape.outWidth)}), nil
}

func conv2DBackward(_ context.Context, gradOutput *tensor.Tensor, inputs []*tensor.Tensor, _ *tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(inputs) != 4 {
		return nil, errs.ErrInvalidArgument
	}
	grad, err := dataOf(gradOutput)
	if err != nil {
		return nil, err
	}
	input, err := dataOf(inputs[0])
	if err != nil {
		return nil, err
	}
	weight, err := dataOf(inputs[1])
	if err != nil {
		return nil, err
	}
	bias, err := dataOf(inputs[2])
	if err != nil {
		return nil, err
	}
	config, err := readIntTensor(inputs[3])
	if err != nil {
		return nil, err
	}
	shape, err := convLayout(input, weight, config)
	if err != nil {
		return nil, err
	}
	gradInput := make([]float64, len(input.values))
	gradWeight := make([]float64, len(weight.values))
	gradBias := make([]float64, shape.outChannels)
	inputStride := shape.inChannels * shape.height * shape.width
	weightStride := shape.inChannels * shape.kernelH * shape.kernelW
	outputStride := shape.outChannels * shape.outHeight * shape.outWidth
	for batch := 0; batch < shape.batch; batch++ {
		for channel := 0; channel < shape.outChannels; channel++ {
			for outRow := 0; outRow < shape.outHeight; outRow++ {
				for outColumn := 0; outColumn < shape.outWidth; outColumn++ {
					gradIndex := batch*outputStride + channel*shape.outHeight*shape.outWidth + outRow*shape.outWidth + outColumn
					upstream := grad.values[gradIndex]
					gradBias[channel] += upstream
					for inChannel := 0; inChannel < shape.inChannels; inChannel++ {
						for kernelRow := 0; kernelRow < shape.kernelH; kernelRow++ {
							for kernelColumn := 0; kernelColumn < shape.kernelW; kernelColumn++ {
								inputRow := outRow*shape.stride + kernelRow - shape.padding
								inputColumn := outColumn*shape.stride + kernelColumn - shape.padding
								if inputRow < 0 || inputRow >= shape.height || inputColumn < 0 || inputColumn >= shape.width {
									continue
								}
								inputIndex := batch*inputStride + inChannel*shape.height*shape.width + inputRow*shape.width + inputColumn
								weightIndex := channel*weightStride + inChannel*shape.kernelH*shape.kernelW + kernelRow*shape.kernelW + kernelColumn
								gradInput[inputIndex] += upstream * weight.values[weightIndex]
								gradWeight[weightIndex] += upstream * input.values[inputIndex]
							}
						}
					}
				}
			}
		}
	}
	return tensorResults(input.withValues(gradInput), weight.withValues(gradWeight), bias.withValues(gradBias), tensorData{})
}

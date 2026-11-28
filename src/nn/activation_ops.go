package nn

import (
	"context"
	"math"

	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/tensor"
)

func sigmoid(value float64) float64 {
	if value >= 0 {
		return 1 / (1 + math.Exp(-value))
	}
	exponent := math.Exp(value)
	return exponent / (1 + exponent)
}

func applyUnary(input tensorData, transform func(float64) float64) tensorData {
	values := make([]float64, len(input.values))
	for index, value := range input.values {
		values[index] = transform(value)
	}
	return input.withValues(values)
}

func reluForward(input tensorData) tensorData {
	return applyUnary(input, func(value float64) float64 {
		if value > 0 {
			return value
		}
		return 0
	})
}

func sigmoidForward(input tensorData) tensorData {
	return applyUnary(input, sigmoid)
}

func tanhForward(input tensorData) tensorData {
	return applyUnary(input, math.Tanh)
}

const geluScale = 0.7978845608028654
const geluCoefficient = 0.044715

func geluForward(input tensorData) tensorData {
	return applyUnary(input, func(value float64) float64 {
		cubic := value*value*value*geluCoefficient + value
		return 0.5 * value * (1 + math.Tanh(geluScale*cubic))
	})
}

func geluPrime(value float64) float64 {
	cubic := value*value*value*geluCoefficient + value
	tangent := math.Tanh(geluScale * cubic)
	derivative := 1 - tangent*tangent
	return 0.5*(1+tangent) + 0.5*value*derivative*geluScale*(1+3*geluCoefficient*value*value)
}

func siluForward(input tensorData) tensorData {
	return applyUnary(input, func(value float64) float64 { return value * sigmoid(value) })
}

func reluBackward(_ context.Context, gradOutput *tensor.Tensor, inputs []*tensor.Tensor, _ *tensor.Tensor) ([]*tensor.Tensor, error) {
	grad, input, err := readBackwardPair(gradOutput, inputs)
	if err != nil {
		return nil, err
	}
	values := make([]float64, len(grad.values))
	for index, value := range input.values {
		if value > 0 {
			values[index] = grad.values[index]
		}
	}
	return tensorResults(grad.withValues(values))
}

func sigmoidBackward(_ context.Context, gradOutput *tensor.Tensor, inputs []*tensor.Tensor, output *tensor.Tensor) ([]*tensor.Tensor, error) {
	grad, _, err := readBackwardPair(gradOutput, inputs)
	if err != nil {
		return nil, err
	}
	values, err := dataOf(output)
	if err != nil {
		return nil, err
	}
	result := make([]float64, len(grad.values))
	for index, value := range values.values {
		result[index] = grad.values[index] * value * (1 - value)
	}
	return tensorResults(grad.withValues(result))
}

func tanhBackward(_ context.Context, gradOutput *tensor.Tensor, inputs []*tensor.Tensor, output *tensor.Tensor) ([]*tensor.Tensor, error) {
	grad, _, err := readBackwardPair(gradOutput, inputs)
	if err != nil {
		return nil, err
	}
	values, err := dataOf(output)
	if err != nil {
		return nil, err
	}
	result := make([]float64, len(grad.values))
	for index, value := range values.values {
		result[index] = grad.values[index] * (1 - value*value)
	}
	return tensorResults(grad.withValues(result))
}

func geluBackward(_ context.Context, gradOutput *tensor.Tensor, inputs []*tensor.Tensor, _ *tensor.Tensor) ([]*tensor.Tensor, error) {
	grad, input, err := readBackwardPair(gradOutput, inputs)
	if err != nil {
		return nil, err
	}
	result := make([]float64, len(grad.values))
	for index, value := range input.values {
		result[index] = grad.values[index] * geluPrime(value)
	}
	return tensorResults(grad.withValues(result))
}

func siluBackward(_ context.Context, gradOutput *tensor.Tensor, inputs []*tensor.Tensor, _ *tensor.Tensor) ([]*tensor.Tensor, error) {
	grad, input, err := readBackwardPair(gradOutput, inputs)
	if err != nil {
		return nil, err
	}
	result := make([]float64, len(grad.values))
	for index, value := range input.values {
		activation := sigmoid(value)
		result[index] = grad.values[index] * activation * (1 + value*(1-activation))
	}
	return tensorResults(grad.withValues(result))
}

func softmaxProbs(input tensorData) (tensorData, tensorData, error) {
	rows, columns, err := rowColumns(input.dims)
	if err != nil {
		return tensorData{}, tensorData{}, err
	}
	if columns == 0 {
		return tensorData{}, tensorData{}, errs.ErrInvalidShape
	}
	probabilities := make([]float64, len(input.values))
	output := make([]float64, len(input.values))
	for row := 0; row < rows; row++ {
		start := row * columns
		maximum := math.Inf(-1)
		for column := 0; column < columns; column++ {
			if input.values[start+column] > maximum {
				maximum = input.values[start+column]
			}
		}
		if math.IsInf(maximum, -1) {
			for column := 0; column < columns; column++ {
				probabilities[start+column] = 1 / float64(columns)
				output[start+column] = -math.Log(float64(columns))
			}
			continue
		}
		var total float64
		for column := 0; column < columns; column++ {
			exponent := math.Exp(input.values[start+column] - maximum)
			probabilities[start+column] = exponent
			total += exponent
		}
		for column := 0; column < columns; column++ {
			probabilities[start+column] /= total
			output[start+column] = math.Log(probabilities[start+column])
		}
	}
	return input.withValues(output), input.withValues(probabilities), nil
}

func softmaxForward(input tensorData) (tensorData, error) {
	_, probabilities, err := softmaxProbs(input)
	if err != nil {
		return tensorData{}, err
	}
	return probabilities, nil
}

func logSoftmaxForward(input tensorData) (tensorData, error) {
	output, _, err := softmaxProbs(input)
	if err != nil {
		return tensorData{}, err
	}
	return output, nil
}

func softmaxBackward(_ context.Context, gradOutput *tensor.Tensor, inputs []*tensor.Tensor, _ *tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(inputs) != 1 {
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
	if err := requireSameDims(grad, input); err != nil {
		return nil, err
	}
	_, probabilities, err := softmaxProbs(input)
	if err != nil {
		return nil, err
	}
	rows, columns, err := rowColumns(input.dims)
	if err != nil {
		return nil, err
	}
	result := make([]float64, len(grad.values))
	for row := 0; row < rows; row++ {
		start := row * columns
		var dot float64
		for column := 0; column < columns; column++ {
			dot += grad.values[start+column] * probabilities.values[start+column]
		}
		for column := 0; column < columns; column++ {
			result[start+column] = probabilities.values[start+column] * (grad.values[start+column] - dot)
		}
	}
	return tensorResults(grad.withValues(result))
}

func logSoftmaxBackward(_ context.Context, gradOutput *tensor.Tensor, inputs []*tensor.Tensor, _ *tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(inputs) != 1 {
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
	if err := requireSameDims(grad, input); err != nil {
		return nil, err
	}
	_, probabilities, err := softmaxProbs(input)
	if err != nil {
		return nil, err
	}
	rows, columns, err := rowColumns(input.dims)
	if err != nil {
		return nil, err
	}
	result := make([]float64, len(grad.values))
	for row := 0; row < rows; row++ {
		start := row * columns
		var upstream float64
		for column := 0; column < columns; column++ {
			upstream += grad.values[start+column]
		}
		for column := 0; column < columns; column++ {
			index := start + column
			result[index] = grad.values[index] - probabilities.values[index]*upstream
		}
	}
	return tensorResults(grad.withValues(result))
}

func scaleData(input tensorData, scale float64) tensorData {
	values := make([]float64, len(input.values))
	for index, value := range input.values {
		values[index] = value * scale
	}
	return input.withValues(values)
}

func scaleLastData(input, scale tensorData) (tensorData, error) {
	rows, columns, err := rowColumns(input.dims)
	if err != nil || len(scale.values) != columns {
		return tensorData{}, errs.ErrGradientShapeMismatch
	}
	values := make([]float64, len(input.values))
	for row := 0; row < rows; row++ {
		for column := 0; column < columns; column++ {
			values[row*columns+column] = input.values[row*columns+column] * scale.values[column]
		}
	}
	return input.withValues(values), nil
}

func concatTimeData(prefix, prompt tensorData) (tensorData, error) {
	batch, length, columns, err := concatLayout(prefix.dims, prompt.dims)
	if err != nil {
		return tensorData{}, err
	}
	promptLength := int(prompt.dims[0])
	values := make([]float64, batch*(length+promptLength)*columns)
	for row := 0; row < batch; row++ {
		copy(values[row*(length+promptLength)*columns:], prefix.values[row*length*columns:(row+1)*length*columns])
		for index := 0; index < promptLength; index++ {
			target := (row*(length+promptLength) + length + index) * columns
			copy(values[target:target+columns], prompt.values[index*columns:(index+1)*columns])
		}
	}
	return prefix.withValues(values).withDims([]int64{int64(batch), int64(length + promptLength), int64(columns)}), nil
}

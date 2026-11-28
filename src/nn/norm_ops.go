package nn

import (
	"context"
	"math"

	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/tensor"
)

func readEpsilon(value *tensor.Tensor) (float64, error) {
	data, err := dataOf(value)
	if err != nil {
		return 0, err
	}
	if len(data.values) != 1 {
		return 0, errs.ErrInvalidArgument
	}
	return data.values[0], nil
}

func layerNormData(input, gamma, beta tensorData, epsilon float64) (tensorData, tensorData, error) {
	rows, columns, err := rowColumns(input.dims)
	if err != nil {
		return tensorData{}, tensorData{}, err
	}
	if len(gamma.values) != columns || len(beta.values) != columns || epsilon <= 0 {
		return tensorData{}, tensorData{}, errs.ErrInvalidArgument
	}
	output := make([]float64, len(input.values))
	normalized := make([]float64, len(input.values))
	for row := 0; row < rows; row++ {
		start := row * columns
		var mean float64
		for column := 0; column < columns; column++ {
			mean += input.values[start+column]
		}
		mean /= float64(columns)
		var variance float64
		for column := 0; column < columns; column++ {
			difference := input.values[start+column] - mean
			variance += difference * difference
		}
		variance /= float64(columns)
		inverse := 1 / math.Sqrt(variance+epsilon)
		for column := 0; column < columns; column++ {
			index := start + column
			normalized[index] = (input.values[index] - mean) * inverse
			output[index] = gamma.values[column]*normalized[index] + beta.values[column]
		}
	}
	return input.withValues(output), input.withValues(normalized), nil
}

func layerNormBackward(_ context.Context, gradOutput *tensor.Tensor, inputs []*tensor.Tensor, _ *tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(inputs) != 4 {
		return nil, errs.ErrInvalidArgument
	}
	grad, input, err := readDataPair(gradOutput, inputs)
	if err != nil {
		return nil, err
	}
	gamma, err := dataOf(inputs[1])
	if err != nil {
		return nil, err
	}
	beta, err := dataOf(inputs[2])
	if err != nil {
		return nil, err
	}
	epsilon, err := readEpsilon(inputs[3])
	if err != nil {
		return nil, err
	}
	rows, columns, err := rowColumns(input.dims)
	if err != nil || len(gamma.values) != columns || len(beta.values) != columns {
		return nil, errs.ErrGradientShapeMismatch
	}
	if err := requireSameDims(grad, input); err != nil {
		return nil, err
	}
	_, normalized, err := layerNormData(input, gamma, beta, epsilon)
	if err != nil {
		return nil, err
	}
	gradGamma := make([]float64, columns)
	gradBeta := make([]float64, columns)
	gradInput := make([]float64, len(input.values))
	for row := 0; row < rows; row++ {
		start := row * columns
		var mean float64
		for column := 0; column < columns; column++ {
			mean += input.values[start+column]
		}
		mean /= float64(columns)
		var variance float64
		for column := 0; column < columns; column++ {
			difference := input.values[start+column] - mean
			variance += difference * difference
		}
		variance /= float64(columns)
		inverse := 1 / math.Sqrt(variance+epsilon)
		var sumWeighted float64
		var sumWeightedNormalized float64
		for column := 0; column < columns; column++ {
			index := start + column
			weighted := grad.values[index] * gamma.values[column]
			gradGamma[column] += grad.values[index] * normalized.values[index]
			gradBeta[column] += grad.values[index]
			sumWeighted += weighted
			sumWeightedNormalized += weighted * normalized.values[index]
		}
		for column := 0; column < columns; column++ {
			index := start + column
			gradInput[index] = inverse *
				(grad.values[index]*gamma.values[column] - sumWeighted/float64(columns) -
					normalized.values[index]*sumWeightedNormalized/float64(columns))
		}
	}
	return tensorResults(grad.withValues(gradInput), gamma.withValues(gradGamma), beta.withValues(gradBeta), tensorData{})
}

func rmsNormData(input, gamma tensorData, epsilon float64) (tensorData, tensorData, error) {
	rows, columns, err := rowColumns(input.dims)
	if err != nil {
		return tensorData{}, tensorData{}, err
	}
	if len(gamma.values) != columns || epsilon <= 0 {
		return tensorData{}, tensorData{}, errs.ErrInvalidArgument
	}
	output := make([]float64, len(input.values))
	inverse := make([]float64, rows)
	for row := 0; row < rows; row++ {
		start := row * columns
		var sum float64
		for column := 0; column < columns; column++ {
			value := input.values[start+column]
			sum += value * value
		}
		inverse[row] = 1 / math.Sqrt(sum/float64(columns)+epsilon)
		for column := 0; column < columns; column++ {
			output[start+column] = gamma.values[column] * input.values[start+column] * inverse[row]
		}
	}
	return input.withValues(output), input.withValues(inverse), nil
}

func rmsNormBackward(_ context.Context, gradOutput *tensor.Tensor, inputs []*tensor.Tensor, _ *tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(inputs) != 3 {
		return nil, errs.ErrInvalidArgument
	}
	grad, input, err := readDataPair(gradOutput, inputs)
	if err != nil {
		return nil, err
	}
	gamma, err := dataOf(inputs[1])
	if err != nil {
		return nil, err
	}
	epsilon, err := readEpsilon(inputs[2])
	if err != nil {
		return nil, err
	}
	rows, columns, err := rowColumns(input.dims)
	if err != nil || len(gamma.values) != columns {
		return nil, errs.ErrGradientShapeMismatch
	}
	if err := requireSameDims(grad, input); err != nil {
		return nil, err
	}
	_, inverse, err := rmsNormData(input, gamma, epsilon)
	if err != nil {
		return nil, err
	}
	gradGamma := make([]float64, columns)
	gradInput := make([]float64, len(input.values))
	for row := 0; row < rows; row++ {
		start := row * columns
		var weightedSum float64
		for column := 0; column < columns; column++ {
			index := start + column
			gradGamma[column] += grad.values[index] * input.values[index] * inverse.values[row]
			weightedSum += grad.values[index] * gamma.values[column] * input.values[index]
		}
		scale := weightedSum * inverse.values[row] * inverse.values[row] * inverse.values[row] / float64(columns)
		for column := 0; column < columns; column++ {
			index := start + column
			gradInput[index] = grad.values[index]*gamma.values[column]*inverse.values[row] - input.values[index]*scale
		}
	}
	return tensorResults(grad.withValues(gradInput), gamma.withValues(gradGamma), tensorData{})
}

package nn

import (
	"context"

	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/tensor"
)

func matMulData(left, right tensorData) (tensorData, error) {
	if err := requireRank(left, 2); err != nil {
		return tensorData{}, err
	}
	if err := requireRank(right, 2); err != nil {
		return tensorData{}, err
	}
	if left.dtype != right.dtype || left.dims[1] != right.dims[0] {
		return tensorData{}, errs.ErrGradientShapeMismatch
	}
	rows, inner := int(left.dims[0]), int(left.dims[1])
	columns := int(right.dims[1])
	if rows < 0 || inner < 0 || columns < 0 {
		return tensorData{}, errs.ErrInvalidShape
	}
	result := make([]float64, rows*columns)
	for row := 0; row < rows; row++ {
		for column := 0; column < columns; column++ {
			var total float64
			for index := 0; index < inner; index++ {
				total += left.values[row*inner+index] * right.values[index*columns+column]
			}
			result[row*columns+column] = total
		}
	}
	return left.withValues(result).withDims([]int64{int64(rows), int64(columns)}), nil
}

func (d tensorData) withDims(dims []int64) tensorData {
	return tensorData{values: d.values, dtype: d.dtype, dims: cloneDims(dims)}
}

func transpose2D(value tensorData) (tensorData, error) {
	if err := requireRank(value, 2); err != nil {
		return tensorData{}, err
	}
	rows, columns := int(value.dims[0]), int(value.dims[1])
	result := make([]float64, rows*columns)
	for row := 0; row < rows; row++ {
		for column := 0; column < columns; column++ {
			result[column*rows+row] = value.values[row*columns+column]
		}
	}
	return value.withValues(result).withDims([]int64{int64(columns), int64(rows)}), nil
}

// weight disimpan sebagai [out, in], sehingga forward menghitung input @ weightᵀ.
func linearData(input, weight, bias tensorData, withBias bool) (tensorData, error) {
	weightTransposed, err := transpose2D(weight)
	if err != nil {
		return tensorData{}, err
	}
	output, err := matMulData(input, weightTransposed)
	if err != nil {
		return tensorData{}, err
	}
	if !withBias {
		return output, nil
	}
	if err := requireRank(bias, 1); err != nil {
		return tensorData{}, err
	}
	columns := int(output.dims[1])
	if bias.dims[0] != int64(columns) {
		return tensorData{}, errs.ErrGradientShapeMismatch
	}
	values := floatSlice(output.values)
	for row := 0; row < int(output.dims[0]); row++ {
		for column := 0; column < columns; column++ {
			values[row*columns+column] += bias.values[column]
		}
	}
	return output.withValues(values), nil
}

func linearGradients(gradOutput, input, weight tensorData) (tensorData, tensorData, tensorData, error) {
	if err := requireRank(gradOutput, 2); err != nil {
		return tensorData{}, tensorData{}, tensorData{}, err
	}
	gradOutputT, err := transpose2D(gradOutput)
	if err != nil {
		return tensorData{}, tensorData{}, tensorData{}, err
	}
	gradInput, err := matMulData(gradOutput, weight)
	if err != nil {
		return tensorData{}, tensorData{}, tensorData{}, err
	}
	gradWeight, err := matMulData(gradOutputT, input)
	if err != nil {
		return tensorData{}, tensorData{}, tensorData{}, err
	}
	gradBias, err := sumRows(gradOutput)
	if err != nil {
		return tensorData{}, tensorData{}, tensorData{}, err
	}
	return gradInput, gradWeight, gradBias, nil
}

func sumRows(value tensorData) (tensorData, error) {
	if err := requireRank(value, 2); err != nil {
		return tensorData{}, err
	}
	columns := int(value.dims[1])
	result := make([]float64, columns)
	for row := 0; row < int(value.dims[0]); row++ {
		for column := 0; column < columns; column++ {
			result[column] += value.values[row*columns+column]
		}
	}
	return value.withValues(result).withDims([]int64{int64(columns)}), nil
}

func linearBackward(_ context.Context, gradOutput *tensor.Tensor, inputs []*tensor.Tensor, _ *tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(inputs) != 3 {
		return nil, errs.ErrInvalidArgument
	}
	grad, input, weight, err := readDataTriple(gradOutput, inputs)
	if err != nil {
		return nil, err
	}
	gradInput, gradWeight, gradBias, err := linearGradients(grad, input, weight)
	if err != nil {
		return nil, err
	}
	return tensorResults(gradInput, gradWeight, gradBias)
}

func linearNoBiasBackward(_ context.Context, gradOutput *tensor.Tensor, inputs []*tensor.Tensor, _ *tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(inputs) != 2 {
		return nil, errs.ErrInvalidArgument
	}
	grad, input, err := readDataPair(gradOutput, inputs)
	if err != nil {
		return nil, err
	}
	weight, err := dataOf(inputs[1])
	if err != nil {
		return nil, err
	}
	gradInput, gradWeight, _, err := linearGradients(grad, input, weight)
	if err != nil {
		return nil, err
	}
	return tensorResults(gradInput, gradWeight)
}

func matMulBackward(_ context.Context, gradOutput *tensor.Tensor, inputs []*tensor.Tensor, _ *tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(inputs) != 2 {
		return nil, errs.ErrInvalidArgument
	}
	grad, left, err := readDataPair(gradOutput, inputs)
	if err != nil {
		return nil, err
	}
	right, err := dataOf(inputs[1])
	if err != nil {
		return nil, err
	}
	leftT, err := transpose2D(left)
	if err != nil {
		return nil, err
	}
	rightT, err := transpose2D(right)
	if err != nil {
		return nil, err
	}
	gradLeft, err := matMulData(grad, rightT)
	if err != nil {
		return nil, err
	}
	gradRight, err := matMulData(leftT, grad)
	if err != nil {
		return nil, err
	}
	return tensorResults(gradLeft, gradRight)
}

func biasAddBackward(_ context.Context, gradOutput *tensor.Tensor, inputs []*tensor.Tensor, _ *tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(inputs) != 2 {
		return nil, errs.ErrInvalidArgument
	}
	grad, _, err := readBackwardPair(gradOutput, inputs)
	if err != nil {
		return nil, err
	}
	gradBias, err := sumRows(grad)
	if err != nil {
		return nil, err
	}
	return tensorResults(grad, gradBias)
}

func scaleBackward(_ context.Context, gradOutput *tensor.Tensor, inputs []*tensor.Tensor, _ *tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(inputs) != 2 {
		return nil, errs.ErrInvalidArgument
	}
	grad, err := dataOf(gradOutput)
	if err != nil {
		return nil, err
	}
	scale, err := dataOf(inputs[1])
	if err != nil {
		return nil, err
	}
	if len(scale.values) != 1 {
		return nil, errs.ErrInvalidArgument
	}
	values := make([]float64, len(grad.values))
	for index, value := range grad.values {
		values[index] = value * scale.values[0]
	}
	return tensorResults(grad.withValues(values), tensorData{})
}

func scaleLastBackward(_ context.Context, gradOutput *tensor.Tensor, inputs []*tensor.Tensor, _ *tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(inputs) != 2 {
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
	scale, err := dataOf(inputs[1])
	if err != nil {
		return nil, err
	}
	rows, columns, err := rowColumns(grad.dims)
	if err != nil || len(scale.values) != columns || len(grad.values) != len(input.values) {
		return nil, errs.ErrGradientShapeMismatch
	}
	values := make([]float64, len(grad.values))
	gradScale := make([]float64, columns)
	for row := 0; row < rows; row++ {
		for column := 0; column < columns; column++ {
			index := row*columns + column
			values[index] = grad.values[index] * scale.values[column]
			gradScale[column] += grad.values[index] * input.values[index]
		}
	}
	return tensorResults(grad.withValues(values), scale.withValues(gradScale))
}

func concatTimeBackward(_ context.Context, gradOutput *tensor.Tensor, inputs []*tensor.Tensor, _ *tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(inputs) != 2 {
		return nil, errs.ErrInvalidArgument
	}
	grad, err := dataOf(gradOutput)
	if err != nil {
		return nil, err
	}
	prefix, err := dataOf(inputs[0])
	if err != nil {
		return nil, err
	}
	prompt, err := dataOf(inputs[1])
	if err != nil {
		return nil, err
	}
	batch, length, columns, err := concatLayout(prefix.dims, prompt.dims)
	if err != nil {
		return nil, err
	}
	promptLength := int(prompt.dims[0])
	if len(grad.values) != batch*(length+promptLength)*columns {
		return nil, errs.ErrGradientShapeMismatch
	}
	prefixValues := make([]float64, batch*length*columns)
	promptValues := make([]float64, promptLength*columns)
	copy(prefixValues, grad.values[:len(prefixValues)])
	for row := 0; row < batch; row++ {
		start := (row*(length+promptLength) + length) * columns
		for index := 0; index < len(promptValues); index++ {
			promptValues[index] += grad.values[start+index]
		}
	}
	promptData := prompt.withValues(promptValues)
	return tensorResults(grad.withValues(prefixValues).withDims(prefix.dims), promptData)
}

func concatTimePrefixData(sequence, prompt tensorData) (tensorData, error) {
	batch, length, columns, err := concatLayout(sequence.dims, prompt.dims)
	if err != nil {
		return tensorData{}, err
	}
	promptLength := int(prompt.dims[0])
	total := promptLength + length
	values := make([]float64, batch*total*columns)
	for row := 0; row < batch; row++ {
		base := row * total * columns
		copy(values[base:base+promptLength*columns], prompt.values)
		copy(values[base+promptLength*columns:base+total*columns], sequence.values[row*length*columns:(row+1)*length*columns])
	}
	return sequence.withValues(values).withDims([]int64{int64(batch), int64(total), int64(columns)}), nil
}

func concatTimePrefixBackward(_ context.Context, gradOutput *tensor.Tensor, inputs []*tensor.Tensor, _ *tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(inputs) != 2 {
		return nil, errs.ErrInvalidArgument
	}
	grad, err := dataOf(gradOutput)
	if err != nil {
		return nil, err
	}
	sequence, err := dataOf(inputs[0])
	if err != nil {
		return nil, err
	}
	prompt, err := dataOf(inputs[1])
	if err != nil {
		return nil, err
	}
	batch, length, columns, err := concatLayout(sequence.dims, prompt.dims)
	if err != nil {
		return nil, err
	}
	promptLength := int(prompt.dims[0])
	total := promptLength + length
	if len(grad.values) != batch*total*columns {
		return nil, errs.ErrGradientShapeMismatch
	}
	sequenceValues := make([]float64, batch*length*columns)
	promptValues := make([]float64, promptLength*columns)
	for row := 0; row < batch; row++ {
		base := row * total * columns
		copy(sequenceValues[row*length*columns:(row+1)*length*columns], grad.values[base+promptLength*columns:base+total*columns])
		for index := range promptValues {
			promptValues[index] += grad.values[base+index]
		}
	}
	return tensorResults(
		sequence.withValues(sequenceValues),
		prompt.withValues(promptValues),
	)
}

func concatLayout(prefixDims, promptDims []int64) (int, int, int, error) {
	if len(prefixDims) != 3 || len(promptDims) != 2 || prefixDims[2] != promptDims[1] {
		return 0, 0, 0, errs.ErrInvalidShape
	}
	count, err := elementCount(prefixDims)
	if err != nil {
		return 0, 0, 0, err
	}
	if prefixDims[2] <= 0 || int64(count)%prefixDims[2] != 0 {
		return 0, 0, 0, errs.ErrInvalidShape
	}
	batchLength := int64(count) / prefixDims[2]
	if prefixDims[1] <= 0 || batchLength%prefixDims[1] != 0 {
		return 0, 0, 0, errs.ErrInvalidShape
	}
	return int(batchLength / prefixDims[1]), int(prefixDims[1]), int(prefixDims[2]), nil
}

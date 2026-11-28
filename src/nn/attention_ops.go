package nn

import (
	"context"
	"math"

	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/tensor"
)

func embeddingData(weight tensorData, indices []int64) (tensorData, error) {
	if err := requireRank(weight, 2); err != nil {
		return tensorData{}, err
	}
	if len(indices) == 0 || weight.dims[0] <= 0 || weight.dims[1] <= 0 {
		return tensorData{}, errs.ErrInvalidShape
	}
	columns := int(weight.dims[1])
	output := make([]float64, len(indices)*columns)
	for row, index := range indices {
		if index < 0 || index >= weight.dims[0] {
			return tensorData{}, errs.ErrOutOfBounds
		}
		copy(output[row*columns:(row+1)*columns], weight.values[int(index)*columns:(int(index)+1)*columns])
	}
	return weight.withValues(output).withDims([]int64{int64(len(indices)), int64(columns)}), nil
}

func embeddingBackward(_ context.Context, gradOutput *tensor.Tensor, inputs []*tensor.Tensor, _ *tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(inputs) != 2 {
		return nil, errs.ErrInvalidArgument
	}
	grad, err := dataOf(gradOutput)
	if err != nil {
		return nil, err
	}
	weight, err := dataOf(inputs[0])
	if err != nil {
		return nil, err
	}
	indices, err := readIntTensor(inputs[1])
	if err != nil {
		return nil, err
	}
	embedded, err := embeddingData(weight, indices)
	if err != nil {
		return nil, err
	}
	if err := requireSameDims(grad, embedded); err != nil {
		return nil, err
	}
	gradWeight := make([]float64, len(weight.values))
	columns := int(weight.dims[1])
	for row, index := range indices {
		base := int(index) * columns
		for column := 0; column < columns; column++ {
			gradWeight[base+column] += grad.values[row*columns+column]
		}
	}
	return tensorResults(weight.withValues(gradWeight), tensorData{})
}

func ropeData(input, cosine, sine tensorData) (tensorData, error) {
	if err := requireRank(input, 4); err != nil {
		return tensorData{}, err
	}
	if err := requireRank(cosine, 2); err != nil {
		return tensorData{}, err
	}
	batch, heads, length, columns := int(input.dims[0]), int(input.dims[1]), int(input.dims[2]), int(input.dims[3])
	if columns%2 != 0 || columns == 0 || length <= 0 {
		return tensorData{}, errs.ErrInvalidShape
	}
	if cosine.dims[0] != int64(length) || cosine.dims[1] != int64(columns/2) ||
		!dimsEqual(cosine.dims, sine.dims) || cosine.dtype != input.dtype || sine.dtype != input.dtype {
		return tensorData{}, errs.ErrGradientShapeMismatch
	}
	output := make([]float64, len(input.values))
	rows := batch * heads * length
	for row := 0; row < rows; row++ {
		position := row % length
		base := row * columns
		for pair := 0; pair < columns/2; pair++ {
			first := base + pair*2
			second := first + 1
			angleIndex := position*(columns/2) + pair
			cosineValue := cosine.values[angleIndex]
			sineValue := sine.values[angleIndex]
			left := input.values[first]
			right := input.values[second]
			output[first] = left*cosineValue - right*sineValue
			output[second] = right*cosineValue + left*sineValue
		}
	}
	return input.withValues(output), nil
}

func ropeBackward(_ context.Context, gradOutput *tensor.Tensor, inputs []*tensor.Tensor, _ *tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(inputs) != 3 {
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
	cosine, err := dataOf(inputs[1])
	if err != nil {
		return nil, err
	}
	sine, err := dataOf(inputs[2])
	if err != nil {
		return nil, err
	}
	if _, err := ropeData(input, cosine, sine); err != nil {
		return nil, err
	}
	if err := requireSameDims(grad, input); err != nil {
		return nil, err
	}
	columns := int(input.dims[3])
	length := int(input.dims[2])
	result := make([]float64, len(grad.values))
	for row := 0; row < len(result)/columns; row++ {
		position := row % length
		base := row * columns
		for pair := 0; pair < columns/2; pair++ {
			first := base + pair*2
			second := first + 1
			angleIndex := position*(columns/2) + pair
			cosineValue := cosine.values[angleIndex]
			sineValue := sine.values[angleIndex]
			firstGradient := grad.values[first]
			secondGradient := grad.values[second]
			result[first] = firstGradient*cosineValue + secondGradient*sineValue
			result[second] = secondGradient*cosineValue - firstGradient*sineValue
		}
	}
	return tensorResults(input.withValues(result), tensorData{}, tensorData{})
}

type attentionShape struct {
	batch, queryHeads, keyHeads, queryLength, keyLength, columns, group int
}

func attentionLayout(query, key, value tensorData) (attentionShape, error) {
	if err := requireRank(query, 4); err != nil {
		return attentionShape{}, err
	}
	if err := requireRank(key, 4); err != nil {
		return attentionShape{}, err
	}
	if err := requireRank(value, 4); err != nil {
		return attentionShape{}, err
	}
	if query.dtype != key.dtype || key.dtype != value.dtype ||
		query.dims[2] <= 0 || query.dims[3] <= 0 || key.dims[2] <= 0 {
		return attentionShape{}, errs.ErrInvalidShape
	}
	if !dimsEqual(key.dims, value.dims) || query.dims[3] != key.dims[3] {
		return attentionShape{}, errs.ErrGradientShapeMismatch
	}
	if query.dims[0] != key.dims[0] || key.dims[1] <= 0 || query.dims[1]%key.dims[1] != 0 {
		return attentionShape{}, errs.ErrInvalidShape
	}
	shape := attentionShape{
		batch:       int(query.dims[0]),
		queryHeads:  int(query.dims[1]),
		keyHeads:    int(key.dims[1]),
		queryLength: int(query.dims[2]),
		keyLength:   int(key.dims[2]),
		columns:     int(query.dims[3]),
	}
	shape.group = shape.queryHeads / shape.keyHeads
	return shape, nil
}

func maskValue(mask tensorData, batch, queryIndex, keyIndex int) float64 {
	if len(mask.values) == 0 {
		return 0
	}
	if mask.dims[0] == 1 {
		batch = 0
	}
	index := (batch*int(mask.dims[1])+queryIndex)*int(mask.dims[2]) + keyIndex
	if index < 0 || index >= len(mask.values) {
		return math.Inf(-1)
	}
	return mask.values[index]
}

func attentionProbabilities(query, key tensorData, mask tensorData, shape attentionShape) (tensorData, error) {
	scoreCount := shape.batch * shape.queryHeads * shape.queryLength * shape.keyLength
	scores := make([]float64, scoreCount)
	probabilities := make([]float64, scoreCount)
	scale := 1 / math.Sqrt(float64(shape.columns))
	keyPlane := shape.keyHeads * shape.keyLength * shape.columns
	queryPlane := shape.queryHeads * shape.queryLength * shape.columns
	scorePlane := shape.queryHeads * shape.queryLength * shape.keyLength
	if len(mask.values) > 0 && (len(mask.dims) != 3 || mask.dims[1] != int64(shape.queryLength) || mask.dims[2] != int64(shape.keyLength) ||
		(mask.dims[0] != 1 && mask.dims[0] != int64(shape.batch))) {
		return tensorData{}, errs.ErrInvalidShape
	}
	for batch := 0; batch < shape.batch; batch++ {
		for head := 0; head < shape.queryHeads; head++ {
			keyHead := head / shape.group
			for queryIndex := 0; queryIndex < shape.queryLength; queryIndex++ {
				base := batch*scorePlane + (head*shape.queryLength+queryIndex)*shape.keyLength
				maximum := math.Inf(-1)
				for keyIndex := 0; keyIndex < shape.keyLength; keyIndex++ {
					var dot float64
					for column := 0; column < shape.columns; column++ {
						queryValue := query.values[batch*queryPlane+head*shape.queryLength*shape.columns+queryIndex*shape.columns+column]
						keyValue := key.values[batch*keyPlane+keyHead*shape.keyLength*shape.columns+keyIndex*shape.columns+column]
						dot += queryValue * keyValue
					}
					score := dot*scale + maskValue(mask, batch, queryIndex, keyIndex)
					scores[base+keyIndex] = score
					if score > maximum {
						maximum = score
					}
				}
				if math.IsInf(maximum, -1) {
					continue
				}
				var total float64
				for keyIndex := 0; keyIndex < shape.keyLength; keyIndex++ {
					exponent := math.Exp(scores[base+keyIndex] - maximum)
					probabilities[base+keyIndex] = exponent
					total += exponent
				}
				for keyIndex := 0; keyIndex < shape.keyLength; keyIndex++ {
					probabilities[base+keyIndex] /= total
				}
			}
		}
	}
	return query.withValues(probabilities).withDims([]int64{int64(shape.batch), int64(shape.queryHeads), int64(shape.queryLength), int64(shape.keyLength)}), nil
}

func attentionData(query, key, value, mask tensorData) (tensorData, error) {
	shape, err := attentionLayout(query, key, value)
	if err != nil {
		return tensorData{}, err
	}
	probabilities, err := attentionProbabilities(query, key, mask, shape)
	if err != nil {
		return tensorData{}, err
	}
	output := make([]float64, shape.batch*shape.queryHeads*shape.queryLength*shape.columns)
	keyPlane := shape.keyHeads * shape.keyLength * shape.columns
	queryPlane := shape.queryHeads * shape.queryLength * shape.columns
	scorePlane := shape.queryHeads * shape.queryLength * shape.keyLength
	for batch := 0; batch < shape.batch; batch++ {
		for head := 0; head < shape.queryHeads; head++ {
			keyHead := head / shape.group
			for queryIndex := 0; queryIndex < shape.queryLength; queryIndex++ {
				scoreBase := batch*scorePlane + (head*shape.queryLength+queryIndex)*shape.keyLength
				outputBase := batch*queryPlane + head*shape.queryLength*shape.columns + queryIndex*shape.columns
				for keyIndex := 0; keyIndex < shape.keyLength; keyIndex++ {
					probability := probabilities.values[scoreBase+keyIndex]
					if probability == 0 {
						continue
					}
					valueBase := batch*keyPlane + keyHead*shape.keyLength*shape.columns + keyIndex*shape.columns
					for column := 0; column < shape.columns; column++ {
						output[outputBase+column] += probability * value.values[valueBase+column]
					}
				}
			}
		}
	}
	return query.withValues(output), nil
}

func attentionBackward(_ context.Context, gradOutput *tensor.Tensor, inputs []*tensor.Tensor, _ *tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(inputs) != 3 && len(inputs) != 4 {
		return nil, errs.ErrInvalidArgument
	}
	grad, err := dataOf(gradOutput)
	if err != nil {
		return nil, err
	}
	query, err := dataOf(inputs[0])
	if err != nil {
		return nil, err
	}
	key, err := dataOf(inputs[1])
	if err != nil {
		return nil, err
	}
	value, err := dataOf(inputs[2])
	if err != nil {
		return nil, err
	}
	mask := tensorData{}
	if len(inputs) == 4 {
		mask, err = dataOf(inputs[3])
		if err != nil {
			return nil, err
		}
	}
	shape, err := attentionLayout(query, key, value)
	if err != nil {
		return nil, err
	}
	output, err := attentionData(query, key, value, mask)
	if err != nil {
		return nil, err
	}
	if err := requireSameDims(grad, output); err != nil {
		return nil, err
	}
	probabilities, err := attentionProbabilities(query, key, mask, shape)
	if err != nil {
		return nil, err
	}
	gradQuery := make([]float64, len(query.values))
	gradKey := make([]float64, len(key.values))
	gradValue := make([]float64, len(value.values))
	keyPlane := shape.keyHeads * shape.keyLength * shape.columns
	queryPlane := shape.queryHeads * shape.queryLength * shape.columns
	scorePlane := shape.queryHeads * shape.queryLength * shape.keyLength
	for batch := 0; batch < shape.batch; batch++ {
		for head := 0; head < shape.queryHeads; head++ {
			keyHead := head / shape.group
			for queryIndex := 0; queryIndex < shape.queryLength; queryIndex++ {
				scoreBase := batch*scorePlane + (head*shape.queryLength+queryIndex)*shape.keyLength
				var dotTotal float64
				for keyIndex := 0; keyIndex < shape.keyLength; keyIndex++ {
					var dot float64
					outputBase := batch*queryPlane + head*shape.queryLength*shape.columns + queryIndex*shape.columns
					valueBase := batch*keyPlane + keyHead*shape.keyLength*shape.columns + keyIndex*shape.columns
					for column := 0; column < shape.columns; column++ {
						dot += grad.values[outputBase+column] * value.values[valueBase+column]
					}
					dotTotal += dot * probabilities.values[scoreBase+keyIndex]
				}
				scale := 1 / math.Sqrt(float64(shape.columns))
				for keyIndex := 0; keyIndex < shape.keyLength; keyIndex++ {
					scoreGradient := scale * probabilities.values[scoreBase+keyIndex] *
						(dotOf(grad, value, shape, batch, head, queryIndex, keyHead, keyIndex) - dotTotal)
					queryBase := batch*queryPlane + head*shape.queryLength*shape.columns + queryIndex*shape.columns
					valueBase := batch*keyPlane + keyHead*shape.keyLength*shape.columns + keyIndex*shape.columns
					for column := 0; column < shape.columns; column++ {
						queryValue := query.values[queryBase+column]
						keyValue := key.values[valueBase+column]
						gradQuery[queryBase+column] += scoreGradient * keyValue
						gradKey[valueBase+column] += scoreGradient * queryValue
						gradValue[valueBase+column] += probabilities.values[scoreBase+keyIndex] * grad.values[queryBase+column]
					}
				}
			}
		}
	}
	gradients := []tensorData{query.withValues(gradQuery), key.withValues(gradKey), value.withValues(gradValue)}
	if len(inputs) == 4 {
		gradients = append(gradients, tensorData{})
	}
	return tensorResults(gradients...)
}

func dotOf(grad, value tensorData, shape attentionShape, batch, head, queryIndex, keyHead, keyIndex int) float64 {
	outputBase := batch*shape.queryHeads*shape.queryLength*shape.columns + head*shape.queryLength*shape.columns + queryIndex*shape.columns
	valueBase := batch*shape.keyHeads*shape.keyLength*shape.columns + keyHead*shape.keyLength*shape.columns + keyIndex*shape.columns
	var total float64
	for column := 0; column < shape.columns; column++ {
		total += grad.values[outputBase+column] * value.values[valueBase+column]
	}
	return total
}

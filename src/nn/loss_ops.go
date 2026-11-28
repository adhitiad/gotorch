package nn

import (
	"context"
	"fmt"
	"math"

	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/tensor"
)

// Reduction menentukan cara menggabungkan loss per elemen.
type Reduction string

// Nilai reduction yang didukung layer loss P5.
const (
	ReductionNone Reduction = "none"
	ReductionMean Reduction = "mean"
	ReductionSum  Reduction = "sum"
)

func (r Reduction) valid() bool {
	return r == ReductionNone || r == ReductionMean || r == ReductionSum
}

func mseData(prediction, target tensorData, reduction Reduction) (tensorData, error) {
	if err := requireSameDims(prediction, target); err != nil {
		return tensorData{}, err
	}
	count, err := elementCount(prediction.dims)
	if err != nil {
		return tensorData{}, err
	}
	if count == 0 {
		return tensorData{}, errs.ErrInvalidShape
	}
	values := make([]float64, len(prediction.values))
	var total float64
	for index, value := range prediction.values {
		difference := value - target.values[index]
		values[index] = difference * difference
		total += values[index]
	}
	switch reduction {
	case ReductionNone:
		return prediction.withValues(values), nil
	case ReductionMean:
		return prediction.withValues([]float64{total / float64(count)}).withDims(nil), nil
	case ReductionSum:
		return prediction.withValues([]float64{total}).withDims(nil), nil
	default:
		return tensorData{}, errs.ErrInvalidArgument
	}
}

func mseGradient(prediction, target tensorData, scale float64) (tensorData, error) {
	if err := requireSameDims(prediction, target); err != nil {
		return tensorData{}, err
	}
	values := make([]float64, len(prediction.values))
	for index, value := range prediction.values {
		values[index] = 2 * scale * (value - target.values[index])
	}
	return prediction.withValues(values), nil
}

func mseBackwardWith(_ context.Context, gradOutput *tensor.Tensor, inputs []*tensor.Tensor, reduction Reduction) ([]*tensor.Tensor, error) {
	if len(inputs) != 2 {
		return nil, errs.ErrInvalidArgument
	}
	grad, err := dataOf(gradOutput)
	if err != nil {
		return nil, err
	}
	prediction, err := dataOf(inputs[0])
	if err != nil {
		return nil, err
	}
	target, err := dataOf(inputs[1])
	if err != nil {
		return nil, err
	}
	count, err := elementCount(prediction.dims)
	if err != nil || count == 0 {
		return nil, errs.ErrInvalidShape
	}
	if reduction == ReductionNone {
		if err := requireSameDims(grad, prediction); err != nil {
			return nil, err
		}
		values := make([]float64, len(grad.values))
		for index := range values {
			values[index] = 2 * grad.values[index] * (prediction.values[index] - target.values[index])
		}
		return tensorResults(prediction.withValues(values), prediction.withValues(negateValues(values)))
	}
	if len(grad.values) != 1 {
		return nil, errs.ErrGradientShapeMismatch
	}
	scale := grad.values[0]
	if reduction == ReductionMean {
		scale /= float64(count)
	}
	gradPrediction, err := mseGradient(prediction, target, scale)
	if err != nil {
		return nil, err
	}
	gradTarget, err := mseGradient(prediction, target, -scale)
	if err != nil {
		return nil, err
	}
	return tensorResults(gradPrediction, gradTarget)
}

func negateValues(values []float64) []float64 {
	result := make([]float64, len(values))
	for index, value := range values {
		result[index] = -value
	}
	return result
}

func mseMeanBackward(ctx context.Context, gradOutput *tensor.Tensor, inputs []*tensor.Tensor, _ *tensor.Tensor) ([]*tensor.Tensor, error) {
	return mseBackwardWith(ctx, gradOutput, inputs, ReductionMean)
}

func mseSumBackward(ctx context.Context, gradOutput *tensor.Tensor, inputs []*tensor.Tensor, _ *tensor.Tensor) ([]*tensor.Tensor, error) {
	return mseBackwardWith(ctx, gradOutput, inputs, ReductionSum)
}

func mseNoneBackward(ctx context.Context, gradOutput *tensor.Tensor, inputs []*tensor.Tensor, _ *tensor.Tensor) ([]*tensor.Tensor, error) {
	return mseBackwardWith(ctx, gradOutput, inputs, ReductionNone)
}

func crossEntropyData(logits tensorData, targets []int64) (tensorData, tensorData, error) {
	if err := requireRank(logits, 2); err != nil {
		return tensorData{}, tensorData{}, err
	}
	batch := int(logits.dims[0])
	classes := int(logits.dims[1])
	if batch <= 0 || classes <= 0 || len(targets) != batch {
		return tensorData{}, tensorData{}, errs.ErrInvalidShape
	}
	_, probabilities, err := softmaxProbs(logits)
	if err != nil {
		return tensorData{}, tensorData{}, err
	}
	values := make([]float64, batch)
	for row := 0; row < batch; row++ {
		target := targets[row]
		if target < 0 || target >= int64(classes) {
			return tensorData{}, tensorData{}, fmtErrorInvalidTarget(target)
		}
		probability := probabilities.values[row*classes+int(target)]
		if probability <= 0 {
			values[row] = float64(math.Inf(1))
		} else {
			values[row] = -math.Log(probability)
		}
	}
	loss := tensorData{values: []float64{average(values)}, dtype: logits.dtype, dims: nil}
	return loss, probabilities, nil
}

func fmtErrorInvalidTarget(target int64) error {
	return fmt.Errorf("%w: class target %d", errs.ErrOutOfBounds, target)
}

func average(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	var total float64
	for _, value := range values {
		total += value
	}
	return total / float64(len(values))
}

func crossEntropyBackward(_ context.Context, gradOutput *tensor.Tensor, inputs []*tensor.Tensor, _ *tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(inputs) != 2 {
		return nil, errs.ErrInvalidArgument
	}
	grad, err := dataOf(gradOutput)
	if err != nil {
		return nil, err
	}
	logits, err := dataOf(inputs[0])
	if err != nil {
		return nil, err
	}
	targets, err := readIntTensor(inputs[1])
	if err != nil {
		return nil, err
	}
	if err := requireRank(logits, 2); err != nil {
		return nil, err
	}
	batch, classes := int(logits.dims[0]), int(logits.dims[1])
	if batch <= 0 || classes <= 0 || len(targets) != batch || len(grad.values) != 1 {
		return nil, errs.ErrInvalidShape
	}
	_, probabilities, err := softmaxProbs(logits)
	if err != nil {
		return nil, err
	}
	values := floatSlice(probabilities.values)
	for row := 0; row < batch; row++ {
		target := targets[row]
		if target < 0 || target >= int64(classes) {
			return nil, fmtErrorInvalidTarget(target)
		}
		index := row*classes + int(target)
		values[index] -= 1
		for column := 0; column < classes; column++ {
			values[row*classes+column] *= grad.values[0] / float64(batch)
		}
	}
	return tensorResults(logits.withValues(values), tensorData{})
}

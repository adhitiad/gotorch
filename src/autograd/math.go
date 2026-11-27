package autograd

import (
	"fmt"

	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/tensor"
)

func readNumbers(value *tensor.Tensor) ([]float64, dtype.DType, []int64, error) {
	if value == nil {
		return nil, dtype.Invalid, nil, errs.ErrInvalidArgument
	}
	metadata, err := value.Metadata()
	if err != nil {
		return nil, dtype.Invalid, nil, err
	}
	switch metadata.DType() {
	case dtype.Float32:
		values, readErr := tensor.Read[float32](value)
		if readErr != nil {
			return nil, dtype.Invalid, nil, readErr
		}
		result := make([]float64, len(values))
		for i, item := range values {
			result[i] = float64(item)
		}
		return result, dtype.Float32, metadata.Shape().Dims(), nil
	case dtype.Float64:
		values, readErr := tensor.Read[float64](value)
		if readErr != nil {
			return nil, dtype.Invalid, nil, readErr
		}
		result := make([]float64, len(values))
		for i, item := range values {
			result[i] = item
		}
		return result, dtype.Float64, metadata.Shape().Dims(), nil
	default:
		return nil, dtype.Invalid, nil, fmt.Errorf("%w: autograd supports float32 and float64, got %s", errs.ErrUnsupportedDType, metadata.DType())
	}
}

func numberTensor(elementType dtype.DType, values []float64, dimensions ...int64) (*tensor.Tensor, error) {
	switch elementType {
	case dtype.Float32:
		converted := make([]float32, len(values))
		for i, value := range values {
			converted[i] = float32(value)
		}
		return tensor.From(converted, dimensions...)
	case dtype.Float64:
		return tensor.From(values, dimensions...)
	default:
		return nil, errs.ErrUnsupportedDType
	}
}

func addNumberTensors(left, right *tensor.Tensor) (*tensor.Tensor, error) {
	leftValues, leftType, leftShape, err := readNumbers(left)
	if err != nil {
		return nil, err
	}
	rightValues, rightType, rightShape, err := readNumbers(right)
	if err != nil {
		return nil, err
	}
	if leftType != rightType || !equalDimensions(leftShape, rightShape) || len(leftValues) != len(rightValues) {
		return nil, errs.ErrGradientShapeMismatch
	}
	result := make([]float64, len(leftValues))
	for i := range result {
		result[i] = leftValues[i] + rightValues[i]
	}
	return numberTensor(leftType, result, leftShape...)
}

func multiplyNumberTensors(left, right *tensor.Tensor) (*tensor.Tensor, error) {
	leftValues, leftType, leftShape, err := readNumbers(left)
	if err != nil {
		return nil, err
	}
	rightValues, rightType, rightShape, err := readNumbers(right)
	if err != nil {
		return nil, err
	}
	if leftType != rightType || !equalDimensions(leftShape, rightShape) || len(leftValues) != len(rightValues) {
		return nil, errs.ErrGradientShapeMismatch
	}
	result := make([]float64, len(leftValues))
	for i := range result {
		result[i] = leftValues[i] * rightValues[i]
	}
	return numberTensor(leftType, result, leftShape...)
}

func sumNumberTensor(input *tensor.Tensor) (*tensor.Tensor, error) {
	values, elementType, _, err := readNumbers(input)
	if err != nil {
		return nil, err
	}
	var total float64
	for _, value := range values {
		total += value
	}
	return numberTensor(elementType, []float64{total})
}

func fillLike(reference *tensor.Tensor, scalar float64) (*tensor.Tensor, error) {
	metadata, err := reference.Metadata()
	if err != nil {
		return nil, err
	}
	numel, err := metadata.Numel()
	if err != nil {
		return nil, err
	}
	if uint64(numel) > uint64(^uint(0)>>1) {
		return nil, errs.ErrShapeOverflow
	}
	values := make([]float64, int(numel))
	for i := range values {
		values[i] = scalar
	}
	return numberTensor(metadata.DType(), values, metadata.Shape().Dims()...)
}

func onesLike(reference *tensor.Tensor) (*tensor.Tensor, error) {
	return fillLike(reference, 1)
}

func zerosLike(reference *tensor.Tensor) (*tensor.Tensor, error) {
	return fillLike(reference, 0)
}

func sameSignature(left, right *tensor.Tensor) bool {
	if left == nil || right == nil {
		return false
	}
	leftMetadata, leftErr := left.Metadata()
	rightMetadata, rightErr := right.Metadata()
	return leftErr == nil && rightErr == nil &&
		leftMetadata.DType() == rightMetadata.DType() &&
		leftMetadata.Shape().Equal(rightMetadata.Shape())
}

func equalDimensions(left, right []int64) bool {
	if len(left) != len(right) {
		return false
	}
	for i := range left {
		if left[i] != right[i] {
			return false
		}
	}
	return true
}

func addGradientTensors(left, right *tensor.Tensor) (*tensor.Tensor, error) {
	if !sameSignature(left, right) {
		return nil, errs.ErrGradientShapeMismatch
	}
	return addNumberTensors(left, right)
}

func accumulateGradient(gradients map[int]*tensor.Tensor, id int, contribution *tensor.Tensor) error {
	if contribution == nil {
		return errs.ErrGradientShapeMismatch
	}
	existing := gradients[id]
	if existing == nil {
		gradients[id] = contribution
		return nil
	}
	combined, err := addGradientTensors(existing, contribution)
	if err != nil {
		return err
	}
	_ = existing.Close()
	_ = contribution.Close()
	gradients[id] = combined
	return nil
}

func closeTensorMap(values map[int]*tensor.Tensor) {
	for id, value := range values {
		if value != nil {
			_ = value.Close()
		}
		delete(values, id)
	}
}

func closeTensors(values []*tensor.Tensor) error {
	var result error
	for _, value := range values {
		if value != nil {
			result = errorsJoin(result, value.Close())
		}
	}
	return result
}

func errorsJoin(left, right error) error {
	if left == nil {
		return right
	}
	if right == nil {
		return left
	}
	return fmt.Errorf("%v; %w", left, right)
}


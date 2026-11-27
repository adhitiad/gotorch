package autograd

import (
	"context"
	"fmt"

	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/tensor"
)

func builtinOperations() []Operation {
	return []Operation{
		{Name: "add", Backward: addBackward, Forward: addForward},
		{Name: "mul", Backward: mulBackward, Forward: mulForward},
		{Name: "sum", Backward: sumBackward, Forward: sumForward},
	}
}

// Add records element-wise addition. The initial P2 primitive requires equal
// shapes and dtypes; broadcasted arithmetic is intentionally left to a later
// compute/API task so its gradient reduction semantics can be designed once.
func (t *Tape) Add(left, right *Variable) (*Variable, error) {
	leftValue, err := left.Value()
	if err != nil {
		return nil, err
	}
	defer leftValue.Close()
	rightValue, err := right.Value()
	if err != nil {
		return nil, err
	}
	defer rightValue.Close()
	output, err := addNumberTensors(leftValue, rightValue)
	if err != nil {
		return nil, err
	}
	defer output.Close()
	return t.Apply("add", []*Variable{left, right}, output)
}

// Mul records element-wise multiplication for equal-shaped float32/float64 tensors.
func (t *Tape) Mul(left, right *Variable) (*Variable, error) {
	leftValue, err := left.Value()
	if err != nil {
		return nil, err
	}
	defer leftValue.Close()
	rightValue, err := right.Value()
	if err != nil {
		return nil, err
	}
	defer rightValue.Close()
	output, err := multiplyNumberTensors(leftValue, rightValue)
	if err != nil {
		return nil, err
	}
	defer output.Close()
	return t.Apply("mul", []*Variable{left, right}, output)
}

// Sum records a reduction over every input element and returns a rank-zero scalar.
func (t *Tape) Sum(input *Variable) (*Variable, error) {
	value, err := input.Value()
	if err != nil {
		return nil, err
	}
	defer value.Close()
	output, err := sumNumberTensor(value)
	if err != nil {
		return nil, err
	}
	defer output.Close()
	return t.Apply("sum", []*Variable{input}, output)
}

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

func addBackward(_ context.Context, gradOutput *tensor.Tensor, inputs []*tensor.Tensor, _ *tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(inputs) != 2 {
		return nil, errs.ErrInvalidArgument
	}
	left, err := gradOutput.Copy()
	if err != nil {
		return nil, err
	}
	right, err := gradOutput.Copy()
	if err != nil {
		_ = left.Close()
		return nil, err
	}
	return []*tensor.Tensor{left, right}, nil
}

func addForward(_ context.Context, _ []*tensor.Tensor, tangents []*tensor.Tensor, _ *tensor.Tensor) (*tensor.Tensor, error) {
	if len(tangents) != 2 {
		return nil, errs.ErrInvalidArgument
	}
	switch {
	case tangents[0] == nil && tangents[1] == nil:
		return nil, errs.ErrNoGradient
	case tangents[0] == nil:
		return tangents[1].Copy()
	case tangents[1] == nil:
		return tangents[0].Copy()
	default:
		return addNumberTensors(tangents[0], tangents[1])
	}
}

func mulBackward(_ context.Context, gradOutput *tensor.Tensor, inputs []*tensor.Tensor, _ *tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(inputs) != 2 {
		return nil, errs.ErrInvalidArgument
	}
	left, err := multiplyNumberTensors(gradOutput, inputs[1])
	if err != nil {
		return nil, err
	}
	right, err := multiplyNumberTensors(gradOutput, inputs[0])
	if err != nil {
		_ = left.Close()
		return nil, err
	}
	return []*tensor.Tensor{left, right}, nil
}

func mulForward(_ context.Context, inputs []*tensor.Tensor, tangents []*tensor.Tensor, _ *tensor.Tensor) (*tensor.Tensor, error) {
	if len(inputs) != 2 || len(tangents) != 2 {
		return nil, errs.ErrInvalidArgument
	}
	var result *tensor.Tensor
	if tangents[0] != nil {
		var err error
		result, err = multiplyNumberTensors(tangents[0], inputs[1])
		if err != nil {
			return nil, err
		}
	}
	if tangents[1] != nil {
		term, err := multiplyNumberTensors(inputs[0], tangents[1])
		if err != nil {
			if result != nil {
				_ = result.Close()
			}
			return nil, err
		}
		if result == nil {
			return term, nil
		}
		combined, addErr := addNumberTensors(result, term)
		_ = result.Close()
		_ = term.Close()
		if addErr != nil {
			return nil, addErr
		}
		result = combined
	}
	if result == nil {
		return nil, errs.ErrNoGradient
	}
	return result, nil
}

func sumBackward(_ context.Context, gradOutput *tensor.Tensor, inputs []*tensor.Tensor, _ *tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(inputs) != 1 {
		return nil, errs.ErrInvalidArgument
	}
	seed, _, _, err := readNumbers(gradOutput)
	if err != nil {
		return nil, err
	}
	if len(seed) != 1 {
		return nil, errs.ErrGradientShapeMismatch
	}
	gradient, err := fillLike(inputs[0], seed[0])
	if err != nil {
		return nil, err
	}
	return []*tensor.Tensor{gradient}, nil
}

func sumForward(_ context.Context, _ []*tensor.Tensor, tangents []*tensor.Tensor, _ *tensor.Tensor) (*tensor.Tensor, error) {
	if len(tangents) != 1 || tangents[0] == nil {
		return nil, errs.ErrNoGradient
	}
	return sumNumberTensor(tangents[0])
}

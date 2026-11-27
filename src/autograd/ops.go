package autograd

import (
	"github.com/adhitiad/gotorch/src/errs"
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
	if t == nil {
		return nil, errs.ErrInvalidArgument
	}
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
	if t == nil {
		return nil, errs.ErrInvalidArgument
	}
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
	if t == nil {
		return nil, errs.ErrInvalidArgument
	}
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

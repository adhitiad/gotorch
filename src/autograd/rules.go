package autograd

import (
	"context"

	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/tensor"
)

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

package nn

import (
	"fmt"

	"github.com/adhitiad/gotorch/src/autograd"
	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/tensor"
)

// MSELoss menghitung mean squared error dengan reduction yang eksplisit.
type MSELoss struct {
	*ModuleBase
	Reduction Reduction
}

// NewMSELoss membuat loss MSE.
func NewMSELoss(name string, reduction Reduction) (*MSELoss, error) {
	if !reduction.valid() {
		return nil, fmt.Errorf("%w: unsupported reduction %q", errs.ErrInvalidArgument, reduction)
	}
	base, err := NewModule(name)
	if err != nil {
		return nil, err
	}
	return &MSELoss{ModuleBase: base, Reduction: reduction}, nil
}

// Forward menghitung loss dan mencatat gradient pada autograd tape.
func (l *MSELoss) Forward(ctx *Context, prediction, target *autograd.Variable) (*autograd.Variable, error) {
	if l == nil || l.ModuleBase == nil {
		return nil, errs.ErrInvalidArgument
	}
	predictionData, err := variableData(prediction)
	if err != nil {
		return nil, err
	}
	targetData, err := variableData(target)
	if err != nil {
		return nil, err
	}
	operation := OpMSELossMean
	switch l.Reduction {
	case ReductionSum:
		operation = OpMSELossSum
	case ReductionNone:
		operation = OpMSELossNone
	}
	return applyOperation(ctx, operation, []*autograd.Variable{prediction, target}, func() (tensorData, error) {
		return mseData(predictionData, targetData, l.Reduction)
	})
}

// CrossEntropyLoss menghitung softmax cross entropy dengan reduction mean.
type CrossEntropyLoss struct {
	*ModuleBase
}

// NewCrossEntropyLoss membuat loss cross entropy.
func NewCrossEntropyLoss(name string) (*CrossEntropyLoss, error) {
	base, err := NewModule(name)
	if err != nil {
		return nil, err
	}
	return &CrossEntropyLoss{ModuleBase: base}, nil
}

// Forward menerima logits [batch, classes] dan target int64 [batch].
func (l *CrossEntropyLoss) Forward(ctx *Context, logits, target *autograd.Variable) (*autograd.Variable, error) {
	if l == nil || l.ModuleBase == nil || ctx == nil || ctx.Tape == nil {
		return nil, errs.ErrInvalidArgument
	}
	logitsData, err := variableData(logits)
	if err != nil {
		return nil, err
	}
	targetTensor, err := target.Value()
	if err != nil {
		return nil, err
	}
	defer targetTensor.Close()
	metadata, err := targetTensor.Metadata()
	if err != nil {
		return nil, err
	}
	if metadata.DType() != dtype.Int64 {
		return nil, errs.ErrUnsupportedDType
	}
	targets, err := tensor.Read[int64](targetTensor)
	if err != nil {
		return nil, err
	}
	return applyOperation(ctx, OpCrossEntropy, []*autograd.Variable{logits, target}, func() (tensorData, error) {
		loss, _, err := crossEntropyData(logitsData, targets)
		return loss, err
	})
}

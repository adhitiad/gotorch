// Package autograd provides tape-based automatic differentiation for GoTorch tensors.
package autograd

import (
	"context"

	"github.com/adhitiad/gotorch/src/tensor"
)

// BackwardFunc computes vector-Jacobian products for one operation.
//
// gradOutput, inputs, and output are independent snapshots owned by the caller
// for the duration of the callback. The callback must not close them. It must
// return one newly owned gradient Tensor per input; nil is allowed only for
// inputs that are not differentiable. The engine validates and releases every
// returned Tensor after accumulation.
type BackwardFunc func(
	ctx context.Context,
	gradOutput *tensor.Tensor,
	inputs []*tensor.Tensor,
	output *tensor.Tensor,
) ([]*tensor.Tensor, error)

// ForwardFunc computes a Jacobian-vector product for one operation.
//
// tangents has the same length/order as inputs. A nil entry means that the
// corresponding input has no tangent. The returned Tensor is newly owned by
// the engine. Callback inputs, tangents, and output are read-only snapshots.
type ForwardFunc func(
	ctx context.Context,
	inputs []*tensor.Tensor,
	tangents []*tensor.Tensor,
	output *tensor.Tensor,
) (*tensor.Tensor, error)

// Operation defines the differentiation rules registered under a stable name.
// Backward is required. Forward is optional until forward-mode AD is requested.
type Operation struct {
	Name     string
	Backward BackwardFunc
	Forward  ForwardFunc
}

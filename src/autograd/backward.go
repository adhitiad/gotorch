package autograd

import (
	"context"
	"fmt"

	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/tensor"
)

// Backward seeds target with ones and accumulates gradients into every
// reachable requires-grad Variable. Repeated calls accumulate until ZeroGrad.
func (t *Tape) Backward(ctx context.Context, target *Variable) error {
	if t == nil {
		return errs.ErrInvalidArgument
	}
	return t.BackwardWithGradient(ctx, target, nil)
}

// BackwardWithGradient is Backward with an explicit upstream seed. seed must
// have the same shape and dtype as target. A nil seed is equivalent to ones.
func (t *Tape) BackwardWithGradient(ctx context.Context, target *Variable, seed *tensor.Tensor) error {
	if t == nil || ctx == nil || target == nil {
		return errs.ErrInvalidArgument
	}
	t.mu.Lock()
	if err := t.mutableLocked(); err != nil {
		t.mu.Unlock()
		return err
	}
	if err := t.validateVariableLocked(target); err != nil {
		t.mu.Unlock()
		return err
	}
	targetNode := t.nodes[target.id]
	if !targetNode.requiresGrad {
		t.mu.Unlock()
		return errs.ErrNoGradient
	}
	t.busy = true
	nodes := append([]*graphNode(nil), t.nodes[:target.id+1]...)
	t.mu.Unlock()
	defer t.finishBackward()

	if err := ctx.Err(); err != nil {
		return err
	}
	var initial *tensor.Tensor
	var err error
	if seed == nil {
		initial, err = onesLike(targetNode.value)
	} else {
		if !sameSignature(seed, targetNode.value) {
			return errs.ErrGradientShapeMismatch
		}
		initial, err = seed.Copy()
	}
	if err != nil {
		return err
	}
	gradients := map[int]*tensor.Tensor{target.id: initial}
	defer closeTensorMap(gradients)

	for id := target.id; id >= 0; id-- {
		if err := ctx.Err(); err != nil {
			return err
		}
		upstream := gradients[id]
		if upstream == nil {
			continue
		}
		node := nodes[id]
		if node.backward == nil || len(node.inputs) == 0 {
			continue
		}

		inputValues := make([]*tensor.Tensor, len(node.inputs))
		for i, parentID := range node.inputs {
			inputValues[i], err = nodes[parentID].value.Copy()
			if err != nil {
				_ = closeTensors(inputValues)
				return err
			}
		}
		outputValue, copyErr := node.value.Copy()
		if copyErr != nil {
			_ = closeTensors(inputValues)
			return copyErr
		}
		upstreamValue, copyErr := upstream.Copy()
		if copyErr != nil {
			_ = closeTensors(inputValues)
			_ = outputValue.Close()
			return copyErr
		}
		contributions, backwardErr := node.backward(ctx, upstreamValue, inputValues, outputValue)
		_ = upstreamValue.Close()
		_ = outputValue.Close()
		_ = closeTensors(inputValues)
		if backwardErr != nil {
			_ = closeTensors(contributions)
			return fmt.Errorf("autograd: backward %s: %w", node.opName, backwardErr)
		}
		if len(contributions) != len(node.inputs) {
			_ = closeTensors(contributions)
			return fmt.Errorf("%w: operation %q returned %d gradients for %d inputs", errs.ErrInvalidArgument, node.opName, len(contributions), len(node.inputs))
		}
		for i, contribution := range contributions {
			parentID := node.inputs[i]
			if !nodes[parentID].requiresGrad {
				if contribution != nil {
					_ = contribution.Close()
					contributions[i] = nil
				}
				continue
			}
			if contribution == nil || !sameSignature(contribution, nodes[parentID].value) {
				_ = closeTensors(contributions)
				return fmt.Errorf("%w: operation %q returned an invalid gradient for input %d", errs.ErrGradientShapeMismatch, node.opName, i)
			}
			if err := accumulateGradient(gradients, parentID, contribution); err != nil {
				_ = closeTensors(contributions)
				return err
			}
			contributions[i] = nil // ownership moved into the gradient map
		}
	}

	if err := ctx.Err(); err != nil {
		return err
	}
	staged := make(map[int]*tensor.Tensor, len(gradients))
	defer closeTensorMap(staged)
	for id, contribution := range gradients {
		if nodes[id].grad == nil {
			staged[id], err = contribution.Copy()
		} else {
			staged[id], err = addGradientTensors(nodes[id].grad, contribution)
		}
		if err != nil {
			return err
		}
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	t.mu.Lock()
	defer t.mu.Unlock()
	if t.closed {
		return errs.ErrInvalidState
	}
	for id, stagedGradient := range staged {
		if old := t.nodes[id].grad; old != nil {
			_ = old.Close()
		}
		t.nodes[id].grad = stagedGradient
		staged[id] = nil
	}
	return nil
}

func (t *Tape) finishBackward() {
	t.mu.Lock()
	t.busy = false
	t.mu.Unlock()
}


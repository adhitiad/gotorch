package autograd

import (
	"context"
	"fmt"

	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/tensor"
)

// JVP evaluates the forward-mode directional derivative of target.
// seeds maps leaf/intermediate Variables to tangent tensors; unseeded values
// contribute zero tangents. The returned Tensor is independently owned.
func (t *Tape) JVP(ctx context.Context, target *Variable, seeds map[*Variable]*tensor.Tensor) (*tensor.Tensor, error) {
	if ctx == nil || target == nil {
		return nil, errs.ErrInvalidArgument
	}
	t.mu.Lock()
	if err := t.mutableLocked(); err != nil {
		t.mu.Unlock()
		return nil, err
	}
	if err := t.validateVariableLocked(target); err != nil {
		t.mu.Unlock()
		return nil, err
	}
	t.busy = true
	nodes := append([]*graphNode(nil), t.nodes[:target.id+1]...)
	t.mu.Unlock()
	defer t.finishBackward()

	tangents := make(map[int]*tensor.Tensor, len(seeds)+1)
	defer closeTensorMap(tangents)
	for variable, tangent := range seeds {
		if variable == nil || tangent == nil {
			return nil, errs.ErrInvalidArgument
		}
		t.mu.Lock()
		validErr := t.validateVariableLocked(variable)
		t.mu.Unlock()
		if validErr != nil {
			return nil, validErr
		}
		if variable.id > target.id || !sameSignature(tangent, nodes[variable.id].value) {
			return nil, errs.ErrGradientShapeMismatch
		}
		copyValue, err := tangent.Copy()
		if err != nil {
			return nil, err
		}
		if old := tangents[variable.id]; old != nil {
			combined, addErr := addGradientTensors(old, copyValue)
			_ = old.Close()
			_ = copyValue.Close()
			if addErr != nil {
				return nil, addErr
			}
			tangents[variable.id] = combined
		} else {
			tangents[variable.id] = copyValue
		}
	}

	for id := 0; id <= target.id; id++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		node := nodes[id]
		if node.forward == nil || len(node.inputs) == 0 {
			continue
		}
		hasTangent := false
		for _, parentID := range node.inputs {
			if tangents[parentID] != nil {
				hasTangent = true
				break
			}
		}
		if !hasTangent {
			continue
		}
		inputValues := make([]*tensor.Tensor, len(node.inputs))
		inputTangents := make([]*tensor.Tensor, len(node.inputs))
		for i, parentID := range node.inputs {
			var err error
			inputValues[i], err = nodes[parentID].value.Copy()
			if err != nil {
				_ = closeTensors(inputValues)
				_ = closeTensors(inputTangents)
				return nil, err
			}
			if tangents[parentID] != nil {
				inputTangents[i], err = tangents[parentID].Copy()
				if err != nil {
					_ = closeTensors(inputValues)
					_ = closeTensors(inputTangents)
					return nil, err
				}
			}
		}
		output, err := node.value.Copy()
		if err != nil {
			_ = closeTensors(inputValues)
			_ = closeTensors(inputTangents)
			return nil, err
		}
		result, forwardErr := node.forward(ctx, inputValues, inputTangents, output)
		_ = output.Close()
		_ = closeTensors(inputValues)
		_ = closeTensors(inputTangents)
		if forwardErr != nil {
			_ = result.Close()
			return nil, fmt.Errorf("autograd: JVP %s: %w", node.opName, forwardErr)
		}
		if result == nil || !sameSignature(result, node.value) {
			_ = result.Close()
			return nil, fmt.Errorf("%w: operation %q returned an invalid tangent", errs.ErrGradientShapeMismatch, node.opName)
		}
		if previous := tangents[id]; previous != nil {
			combined, addErr := addGradientTensors(previous, result)
			_ = previous.Close()
			_ = result.Close()
			if addErr != nil {
				return nil, addErr
			}
			tangents[id] = combined
		} else {
			tangents[id] = result
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if result := tangents[target.id]; result != nil {
		return result.Copy()
	}
	return zerosLike(nodes[target.id].value)
}

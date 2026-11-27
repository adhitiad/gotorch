package autograd

import (
	"errors"

	"github.com/adhitiad/gotorch/src/errs"
)

// ZeroGrad clears all gradients when called without arguments, or only the
// specified Variables. It cannot run concurrently with graph execution.
func (t *Tape) ZeroGrad(variables ...*Variable) error {
	if t == nil {
		return errs.ErrInvalidArgument
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if err := t.mutableLocked(); err != nil {
		return err
	}
	ids := make(map[int]struct{}, len(variables))
	if len(variables) == 0 {
		for id := range t.nodes {
			ids[id] = struct{}{}
		}
	} else {
		for _, variable := range variables {
			if err := t.validateVariableLocked(variable); err != nil {
				return err
			}
			ids[variable.id] = struct{}{}
		}
	}
	var closeErr error
	for id := range ids {
		if gradient := t.nodes[id].grad; gradient != nil {
			closeErr = errors.Join(closeErr, gradient.Close())
			t.nodes[id].grad = nil
		}
	}
	return closeErr
}

// NodeCount reports the number of recorded graph nodes.
func (t *Tape) NodeCount() int {
	if t == nil {
		return 0
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.closed {
		return 0
	}
	return len(t.nodes)
}

// Close releases every graph value and gradient. It is idempotent. A Tape
// cannot be closed while Backward or JVP is executing.
func (t *Tape) Close() error {
	if t == nil {
		return errs.ErrInvalidArgument
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.closed {
		return nil
	}
	if t.busy {
		return errs.ErrInvalidState
	}
	var closeErr error
	for _, node := range t.nodes {
		closeErr = errors.Join(closeErr, node.value.Close())
		if node.grad != nil {
			closeErr = errors.Join(closeErr, node.grad.Close())
		}
	}
	t.nodes = nil
	t.operations = nil
	t.closed = true
	return closeErr
}

func (t *Tape) mutableLocked() error {
	if t == nil || t.closed || t.busy {
		return errs.ErrInvalidState
	}
	return nil
}

func (t *Tape) validateVariableLocked(variable *Variable) error {
	if t == nil || t.closed || variable == nil || variable.tape != t {
		return errs.ErrInvalidArgument
	}
	if variable.id < 0 || variable.id >= len(t.nodes) {
		return errs.ErrInvalidState
	}
	return nil
}

func isDifferentiable(elementType dtype.DType) bool {
	return elementType == dtype.Float32 || elementType == dtype.Float64
}

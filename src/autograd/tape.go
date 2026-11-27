package autograd

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"

	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/tensor"
)

type graphNode struct {
	value        *tensor.Tensor
	grad         *tensor.Tensor
	requiresGrad bool
	inputs       []int
	opName       string
	backward     BackwardFunc
	forward      ForwardFunc
}

// Tape owns an append-only computation graph and its gradient buffers.
// Graph nodes keep private tensor snapshots, so mutating the caller's original
// tensors cannot silently change values saved for backward.
type Tape struct {
	mu         sync.Mutex
	nodes      []*graphNode
	operations map[string]Operation
	busy       bool
	closed     bool
}

// Variable is an immutable handle to one value in a Tape. It must not be copied
// by value; use the pointer returned by Tape.NewVariable or Tape.Apply.
type Variable struct {
	tape *Tape
	id   int
}

// NewTape creates an independent graph and registers the built-in add, mul and
// sum operations. No global registry or mutable process-wide state is used.
func NewTape() *Tape {
	t := &Tape{operations: make(map[string]Operation)}
	for _, operation := range builtinOperations() {
		t.operations[operation.Name] = operation
	}
	return t
}

// NewVariable records an input/leaf value. The Tape stores a deep snapshot.
// Only float32 and float64 leaves may require gradients.
func (t *Tape) NewVariable(value *tensor.Tensor, requiresGrad bool) (*Variable, error) {
	if value == nil {
		return nil, errs.ErrInvalidArgument
	}
	metadata, err := value.Metadata()
	if err != nil {
		return nil, err
	}
	if requiresGrad && !isDifferentiable(metadata.DType()) {
		return nil, fmt.Errorf("%w: gradients require float32 or float64, got %s", errs.ErrUnsupportedDType, metadata.DType())
	}
	snapshot, err := value.Copy()
	if err != nil {
		return nil, err
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if err := t.mutableLocked(); err != nil {
		_ = snapshot.Close()
		return nil, err
	}
	id := len(t.nodes)
	t.nodes = append(t.nodes, &graphNode{value: snapshot, requiresGrad: requiresGrad})
	return &Variable{tape: t, id: id}, nil
}

// RegisterOperation registers a custom differentiable operation on this Tape.
// Names are local to the Tape; duplicate and empty names are rejected.
func (t *Tape) RegisterOperation(operation Operation) error {
	name := strings.TrimSpace(operation.Name)
	if name == "" || name != operation.Name || operation.Backward == nil {
		return errs.ErrInvalidArgument
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if err := t.mutableLocked(); err != nil {
		return err
	}
	if _, exists := t.operations[name]; exists {
		return errs.ErrOperationAlreadyRegistered
	}
	t.operations[name] = operation
	return nil
}

// Apply records the output of a registered operation and its parent Variables.
// output is snapshotted before the method returns. All parents must already
// exist on this Tape, which makes cycles impossible and preserves topological
// append order.
func (t *Tape) Apply(name string, inputs []*Variable, output *tensor.Tensor) (*Variable, error) {
	if output == nil || len(inputs) == 0 {
		return nil, errs.ErrInvalidArgument
	}
	outputMetadata, err := output.Metadata()
	if err != nil {
		return nil, err
	}

	t.mu.Lock()
	defer t.mu.Unlock()
	if err := t.mutableLocked(); err != nil {
		return nil, err
	}
	operation, ok := t.operations[name]
	if !ok {
		return nil, fmt.Errorf("%w: operation %q is not registered", errs.ErrUnsupportedOperation, name)
	}

	parentIDs := make([]int, len(inputs))
	requiresGrad := false
	for i, input := range inputs {
		if err := t.validateVariableLocked(input); err != nil {
			return nil, err
		}
		if input.id >= len(t.nodes) {
			return nil, errs.ErrInvalidState
		}
		parentIDs[i] = input.id
		parent := t.nodes[input.id]
		if parent.requiresGrad {
			requiresGrad = true
			parentMetadata, metaErr := parent.value.Metadata()
			if metaErr != nil {
				return nil, metaErr
			}
			if !isDifferentiable(parentMetadata.DType()) || parentMetadata.DType() != outputMetadata.DType() {
				return nil, fmt.Errorf("%w: operation %q mixes incompatible gradient dtypes", errs.ErrUnsupportedDType, name)
			}
		}
	}
	if requiresGrad && !isDifferentiable(outputMetadata.DType()) {
		return nil, fmt.Errorf("%w: output dtype %s is not differentiable", errs.ErrUnsupportedDType, outputMetadata.DType())
	}

	snapshot, err := output.Copy()
	if err != nil {
		return nil, err
	}
	id := len(t.nodes)
	t.nodes = append(t.nodes, &graphNode{
		value:        snapshot,
		requiresGrad: requiresGrad,
		inputs:       parentIDs,
		opName:       name,
		backward:     operation.Backward,
		forward:      operation.Forward,
	})
	return &Variable{tape: t, id: id}, nil
}

// Value returns an independent copy of the Variable's forward value.
func (v *Variable) Value() (*tensor.Tensor, error) {
	if v == nil || v.tape == nil {
		return nil, errs.ErrInvalidArgument
	}
	v.tape.mu.Lock()
	defer v.tape.mu.Unlock()
	if err := v.tape.validateVariableLocked(v); err != nil {
		return nil, err
	}
	return v.tape.nodes[v.id].value.Copy()
}

// Grad returns an independent copy of the accumulated gradient. It returns
// errs.ErrNoGradient before the first successful Backward or after ZeroGrad.
func (v *Variable) Grad() (*tensor.Tensor, error) {
	if v == nil || v.tape == nil {
		return nil, errs.ErrInvalidArgument
	}
	v.tape.mu.Lock()
	defer v.tape.mu.Unlock()
	if err := v.tape.validateVariableLocked(v); err != nil {
		return nil, err
	}
	gradient := v.tape.nodes[v.id].grad
	if gradient == nil {
		return nil, errs.ErrNoGradient
	}
	return gradient.Copy()
}

// RequiresGrad reports whether this Variable participates in reverse-mode AD.
func (v *Variable) RequiresGrad() bool {
	if v == nil || v.tape == nil {
		return false
	}
	v.tape.mu.Lock()
	defer v.tape.mu.Unlock()
	if v.tape.validateVariableLocked(v) != nil {
		return false
	}
	return v.tape.nodes[v.id].requiresGrad
}

// Backward seeds target with ones and accumulates gradients into every
// reachable requires-grad Variable. Repeated calls accumulate until ZeroGrad.
func (t *Tape) Backward(ctx context.Context, target *Variable) error {
	return t.BackwardWithGradient(ctx, target, nil)
}

// BackwardWithGradient is Backward with an explicit upstream seed. seed must
// have the same shape and dtype as target. A nil seed is equivalent to ones.
func (t *Tape) BackwardWithGradient(ctx context.Context, target *Variable, seed *tensor.Tensor) error {
	if ctx == nil || target == nil {
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
				}
				continue
			}
			if contribution == nil || !sameSignature(contribution, nodes[parentID].value) {
				_ = closeTensors(contributions)
				return fmt.Errorf("%w: operation %q returned an invalid gradient for input %d", errs.ErrGradientShapeMismatch, node.opName, i)
			}
			contributions[i] = nil // ownership moves into the gradient map
			if err := accumulateGradient(gradients, parentID, contribution); err != nil {
				_ = closeTensors(contributions)
				return err
			}
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

// ZeroGrad clears all gradients when called without arguments, or only the
// specified Variables. It cannot run concurrently with graph execution.
func (t *Tape) ZeroGrad(variables ...*Variable) error {
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

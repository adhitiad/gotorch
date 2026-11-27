package autograd

import (
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
	if t == nil || value == nil {
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
	if t == nil {
		return errs.ErrInvalidArgument
	}
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
	if t == nil || output == nil || len(inputs) == 0 {
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


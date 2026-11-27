package compute

import (
	"context"
	"fmt"
	"reflect"
	"strings"
	"sync"

	"github.com/adhitiad/gotorch/src/errs"
)

// Registry selects the first compatible backend in deterministic registration order.
type Registry struct {
	mu sync.RWMutex
	backends []Backend
	names map[string]struct{}
}

func NewRegistry() *Registry { return &Registry{names: make(map[string]struct{})} }

func (r *Registry) Register(backend Backend) error {
	if r == nil || nilInterface(backend) { return errs.ErrInvalidArgument }
	name := strings.TrimSpace(backend.Name())
	if name == "" || !backend.Kind().Valid() { return errs.ErrInvalidArgument }
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.names == nil { r.names = make(map[string]struct{}) }
	if _, ok := r.names[name]; ok { return fmt.Errorf("%w: duplicate backend %q", errs.ErrInvalidState, name) }
	r.names[name] = struct{}{}
	r.backends = append(r.backends, backend)
	return nil
}

func (r *Registry) Select(spec KernelSpec, call Invocation) (Backend, error) {
	if r == nil { return nil, errs.ErrInvalidArgument }
	if err := spec.Validate(); err != nil { return nil, err }
	if err := call.Validate(); err != nil { return nil, err }
	r.mu.RLock()
	defer r.mu.RUnlock()
	for _, backend := range r.backends {
		if backend.Supports(spec, call) { return backend, nil }
	}
	return nil, fmt.Errorf("%w: no backend supports %q", errs.ErrUnsupportedBackend, spec.Operation)
}

func (r *Registry) Execute(ctx context.Context, spec KernelSpec, call Invocation) error {
	if nilInterface(ctx) { return errs.ErrInvalidArgument }
	if err := ctx.Err(); err != nil { return err }
	backend, err := r.Select(spec, call)
	if err != nil { return err }
	return Run(ctx, backend, spec, call)
}

func (r *Registry) Backends() []string {
	if r == nil { return nil }
	r.mu.RLock()
	defer r.mu.RUnlock()
	names := make([]string, len(r.backends))
	for i, backend := range r.backends { names[i] = backend.Name() }
	return names
}

func nilInterface(value any) bool {
	if value == nil { return true }
	v := reflect.ValueOf(value)
	switch v.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return v.IsNil()
	default:
		return false
	}
}

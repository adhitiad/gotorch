package compute

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"sync"

	"github.com/adhitiad/gotorch/src/errs"
)

// Registry selects the first compatible backend in deterministic registration order.
type Registry struct {
	mu       sync.RWMutex
	backends []Backend
	names    map[string]struct{}
	closed   bool
	closeErr error
}

func NewRegistry() *Registry { return &Registry{names: make(map[string]struct{})} }

func (r *Registry) Register(backend Backend) error {
	if r == nil || nilInterface(backend) {
		return errs.ErrInvalidArgument
	}
	name := strings.TrimSpace(backend.Name())
	if name == "" || !backend.Kind().Valid() {
		return errs.ErrInvalidArgument
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return fmt.Errorf("%w: compute registry is closed", errs.ErrInvalidState)
	}
	if r.names == nil {
		r.names = make(map[string]struct{})
	}
	if _, ok := r.names[name]; ok {
		return fmt.Errorf("%w: duplicate backend %q", errs.ErrInvalidState, name)
	}
	r.names[name] = struct{}{}
	r.backends = append(r.backends, backend)
	return nil
}

func (r *Registry) Select(spec KernelSpec, call Invocation) (Backend, error) {
	if r == nil {
		return nil, errs.ErrInvalidArgument
	}
	if err := spec.Validate(); err != nil {
		return nil, err
	}
	if err := call.Validate(); err != nil {
		return nil, err
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	if r.closed {
		return nil, fmt.Errorf("%w: compute registry is closed", errs.ErrInvalidState)
	}
	for _, backend := range r.backends {
		if backend.Supports(spec, call) {
			return backend, nil
		}
	}
	return nil, fmt.Errorf("%w: no backend supports %q", errs.ErrUnsupportedBackend, spec.Operation)
}

func (r *Registry) Execute(ctx context.Context, spec KernelSpec, call Invocation) error {
	if nilInterface(ctx) {
		return errs.ErrInvalidArgument
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	backend, err := r.Select(spec, call)
	if err != nil {
		return err
	}
	return Run(ctx, backend, spec, call)
}

// Close prevents new dispatch, closes registered backends in reverse order, and
// releases the registry's backend references. Backends without a Close() error
// method require no additional lifecycle action. It waits for each backend's own
// Close implementation to drain in-flight work. Repeated calls return the first
// close result without closing backends again.
func (r *Registry) Close() error {
	if r == nil {
		return errs.ErrInvalidArgument
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return r.closeErr
	}
	var closeErrors []error
	for i := len(r.backends) - 1; i >= 0; i-- {
		backend := r.backends[i]
		r.backends[i] = nil
		closer, ok := backend.(interface{ Close() error })
		if !ok {
			continue
		}
		name := backend.Name()
		if err := closer.Close(); err != nil {
			closeErrors = append(closeErrors, fmt.Errorf("close backend %q: %w", name, err))
		}
	}
	r.backends = nil
	r.names = nil
	r.closeErr = errors.Join(closeErrors...)
	r.closed = true
	return r.closeErr
}

// Backends returns registered backend names in registration order. It returns
// nil once the registry is closed because Close releases the backend references.
func (r *Registry) Backends() []string {
	if r == nil {
		return nil
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	if r.closed {
		return nil
	}
	names := make([]string, len(r.backends))
	for i, backend := range r.backends {
		names[i] = backend.Name()
	}
	return names
}

func nilInterface(value any) bool {
	if value == nil {
		return true
	}
	v := reflect.ValueOf(value)
	switch v.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return v.IsNil()
	default:
		return false
	}
}

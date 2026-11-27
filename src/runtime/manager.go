// Package runtime defines managed component startup and shutdown contracts.
package runtime

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"

	"github.com/adhitiad/gotorch/src/errs"
)

// Component is a runtime subsystem with an explicit start/stop lifecycle.
// Start must clean up its own partial work before returning an error.
type Component interface {
	Name() string
	Start(context.Context) error
	Stop(context.Context) error
}

// State describes a Manager lifecycle state.
type State uint8

const (
	StateNew State = iota
	StateStarting
	StateRunning
	StateClosing
	StateClosed
	StateFailed
)

func (s State) String() string {
	names := [...]string{"new", "starting", "running", "closing", "closed", "failed"}
	if int(s) >= len(names) {
		return "unknown"
	}
	return names[s]
}

// Manager starts components in declaration order and stops them in reverse.
// It serializes lifecycle transitions and attempts every stop during cleanup.
type Manager struct {
	operation  sync.Mutex
	mu         sync.RWMutex
	components []Component
	started    int
	state      State
}

// NewManager validates and copies the ordered component list.
func NewManager(components ...Component) (*Manager, error) {
	seen := make(map[string]struct{}, len(components))
	copied := append([]Component(nil), components...)
	for i, component := range copied {
		if component == nil || strings.TrimSpace(component.Name()) == "" {
			return nil, fmt.Errorf("%w: component %d is nil or unnamed", errs.ErrInvalidArgument, i)
		}
		name := component.Name()
		if _, exists := seen[name]; exists {
			return nil, fmt.Errorf("%w: duplicate component %q", errs.ErrInvalidArgument, name)
		}
		seen[name] = struct{}{}
	}
	return &Manager{components: copied, state: StateNew}, nil
}

// State returns the current lifecycle state.
func (m *Manager) State() State {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.state
}

// Start initializes components in order. A failed start triggers reverse cleanup
// of components that already started; a failed manager cannot be restarted.
func (m *Manager) Start(ctx context.Context) error {
	if ctx == nil {
		return fmt.Errorf("%w: nil context", errs.ErrInvalidArgument)
	}
	m.operation.Lock()
	defer m.operation.Unlock()

	m.mu.Lock()
	switch m.state {
	case StateRunning:
		m.mu.Unlock()
		return nil
	case StateNew:
		m.state = StateStarting
	default:
		state := m.state
		m.mu.Unlock()
		return fmt.Errorf("%w: cannot start from %s", errs.ErrInvalidState, state)
	}
	m.mu.Unlock()

	for index, component := range m.components {
		if err := ctx.Err(); err != nil {
			return m.rollback(ctx, fmt.Errorf("start cancelled before %q: %w", component.Name(), err))
		}
		if err := component.Start(ctx); err != nil {
			return m.rollback(ctx, fmt.Errorf("start component %q: %w", component.Name(), err))
		}
		m.started = index + 1
	}
	m.setState(StateRunning)
	return nil
}

// Close stops every started component in reverse order. It is idempotent.
func (m *Manager) Close(ctx context.Context) error {
	if ctx == nil {
		return fmt.Errorf("%w: nil context", errs.ErrInvalidArgument)
	}
	m.operation.Lock()
	defer m.operation.Unlock()

	m.mu.Lock()
	switch m.state {
	case StateClosed:
		m.mu.Unlock()
		return nil
	case StateNew, StateFailed:
		m.state = StateClosed
		m.mu.Unlock()
		return nil
	case StateRunning:
		m.state = StateClosing
	default:
		state := m.state
		m.mu.Unlock()
		return fmt.Errorf("%w: cannot close from %s", errs.ErrInvalidState, state)
	}
	m.mu.Unlock()

	stopErr := m.stopStarted(ctx)
	m.setState(StateClosed)
	return stopErr
}

func (m *Manager) rollback(ctx context.Context, cause error) error {
	stopErr := m.stopStarted(ctx)
	m.setState(StateFailed)
	return errors.Join(cause, stopErr)
}

func (m *Manager) stopStarted(ctx context.Context) error {
	var stopErrors []error
	for index := m.started - 1; index >= 0; index-- {
		component := m.components[index]
		if err := component.Stop(ctx); err != nil {
			stopErrors = append(stopErrors, fmt.Errorf("stop component %q: %w", component.Name(), err))
		}
	}
	m.started = 0
	return errors.Join(stopErrors...)
}

func (m *Manager) setState(state State) {
	m.mu.Lock()
	m.state = state
	m.mu.Unlock()
}

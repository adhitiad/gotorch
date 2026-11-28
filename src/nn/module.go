package nn

import (
	"errors"
	"fmt"
	"path"
	"sort"
	"sync"

	"github.com/adhitiad/gotorch/src/errs"
)

// Module adalah simpul pohon model yang memiliki parameter, buffer, dan modul anak.
type Module interface {
	Name() string
	Parameters() []NamedParameter
	Buffers() []NamedBuffer
	Train(enabled bool)
	IsTraining() bool
	Close() error
}

// ModuleBase adalah implementasi Module yang bisa disisipkan ke layer buatan pengguna.
type ModuleBase struct {
	mu       sync.RWMutex
	name     string
	children []Module
	params   map[string]*Parameter
	buffers  map[string]*Buffer
	training bool
	closed   bool
}

// NewModule membuat ModuleBase dengan nama yang divalidasi.
func NewModule(name string) (*ModuleBase, error) {
	if err := validateName(name); err != nil {
		return nil, err
	}
	return &ModuleBase{
		name:     name,
		params:   make(map[string]*Parameter),
		buffers:  make(map[string]*Buffer),
		training: true,
	}, nil
}

// Name mengembalikan nama modul.
func (m *ModuleBase) Name() string {
	if m == nil {
		return ""
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.name
}

// RegisterModule menambahkan modul anak dengan nama unik.
func (m *ModuleBase) RegisterModule(child Module) error {
	if m == nil || child == nil || child.Name() == "" {
		return errs.ErrInvalidArgument
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return errs.ErrInvalidState
	}
	name := child.Name()
	if err := validateName(name); err != nil {
		return err
	}
	if _, exists := m.params[name]; exists {
		return fmt.Errorf("%w: duplicate module name %q", errs.ErrInvalidArgument, name)
	}
	if _, exists := m.buffers[name]; exists {
		return fmt.Errorf("%w: duplicate module name %q", errs.ErrInvalidArgument, name)
	}
	for _, existing := range m.children {
		if existing.Name() == name {
			return fmt.Errorf("%w: duplicate module name %q", errs.ErrInvalidArgument, name)
		}
	}
	m.children = append(m.children, child)
	child.Train(m.training)
	return nil
}

// RegisterParameter menambahkan parameter dengan nama lokal yang unik.
func (m *ModuleBase) RegisterParameter(name string, parameter *Parameter) error {
	if m == nil || parameter == nil {
		return errs.ErrInvalidArgument
	}
	if err := validateName(name); err != nil {
		return err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return errs.ErrInvalidState
	}
	if _, exists := m.params[name]; exists {
		return fmt.Errorf("%w: duplicate parameter %q", errs.ErrInvalidArgument, name)
	}
	if _, exists := m.buffers[name]; exists {
		return fmt.Errorf("%w: duplicate parameter %q", errs.ErrInvalidArgument, name)
	}
	m.params[name] = parameter
	return nil
}

// RegisterBuffer menambahkan buffer dengan nama lokal yang unik.
func (m *ModuleBase) RegisterBuffer(name string, buffer *Buffer) error {
	if m == nil || buffer == nil {
		return errs.ErrInvalidArgument
	}
	if err := validateName(name); err != nil {
		return err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return errs.ErrInvalidState
	}
	if _, exists := m.buffers[name]; exists {
		return fmt.Errorf("%w: duplicate buffer %q", errs.ErrInvalidArgument, name)
	}
	if _, exists := m.params[name]; exists {
		return fmt.Errorf("%w: duplicate buffer %q", errs.ErrInvalidArgument, name)
	}
	m.buffers[name] = buffer
	return nil
}

// Parameters mengembalikan seluruh parameter pohon modul, diurutkan berdasarkan nama.
func (m *ModuleBase) Parameters() []NamedParameter {
	if m == nil {
		return nil
	}
	m.mu.RLock()
	local := make([]NamedParameter, 0, len(m.params))
	for name, parameter := range m.params {
		local = append(local, NamedParameter{Name: joinName(m.name, name), Parameter: parameter})
	}
	children := append([]Module(nil), m.children...)
	m.mu.RUnlock()
	result := local
	for _, child := range children {
		for _, named := range child.Parameters() {
			result = append(result, NamedParameter{Name: joinName(m.name, named.Name), Parameter: named.Parameter})
		}
	}
	sort.Slice(result, func(i, j int) bool { return result[i].Name < result[j].Name })
	return result
}

// Buffers mengembalikan seluruh buffer persisten pohon modul, diurutkan berdasarkan nama.
func (m *ModuleBase) Buffers() []NamedBuffer {
	if m == nil {
		return nil
	}
	m.mu.RLock()
	local := make([]NamedBuffer, 0, len(m.buffers))
	for name, buffer := range m.buffers {
		if buffer.Persistent() {
			local = append(local, NamedBuffer{Name: joinName(m.name, name), Buffer: buffer})
		}
	}
	children := append([]Module(nil), m.children...)
	m.mu.RUnlock()
	result := local
	for _, child := range children {
		for _, named := range child.Buffers() {
			result = append(result, NamedBuffer{Name: joinName(m.name, named.Name), Buffer: named.Buffer})
		}
	}
	sort.Slice(result, func(i, j int) bool { return result[i].Name < result[j].Name })
	return result
}

// Train mengatur mode latih untuk modul dan seluruh anaknya.
func (m *ModuleBase) Train(enabled bool) {
	if m == nil {
		return
	}
	m.mu.Lock()
	m.training = enabled
	children := append([]Module(nil), m.children...)
	m.mu.Unlock()
	for _, child := range children {
		child.Train(enabled)
	}
}

// IsTraining melaporkan mode modul.
func (m *ModuleBase) IsTraining() bool {
	if m == nil {
		return false
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.training
}

// Close menutup anak, parameter, dan buffer dalam urutan terbalik.
func (m *ModuleBase) Close() error {
	if m == nil {
		return errs.ErrInvalidArgument
	}
	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		return nil
	}
	children := append([]Module(nil), m.children...)
	m.children = nil
	parameters := make([]*Parameter, 0, len(m.params))
	for _, parameter := range m.params {
		parameters = append(parameters, parameter)
	}
	m.params = make(map[string]*Parameter)
	buffers := make([]*Buffer, 0, len(m.buffers))
	for _, buffer := range m.buffers {
		buffers = append(buffers, buffer)
	}
	m.buffers = make(map[string]*Buffer)
	m.mu.Unlock()

	var closeErr error
	for index := len(children) - 1; index >= 0; index-- {
		closeErr = errors.Join(closeErr, children[index].Close())
	}
	for index := len(parameters) - 1; index >= 0; index-- {
		closeErr = errors.Join(closeErr, parameters[index].Close())
	}
	for index := len(buffers) - 1; index >= 0; index-- {
		closeErr = errors.Join(closeErr, buffers[index].Close())
	}
	if closeErr != nil {
		return closeErr
	}
	m.mu.Lock()
	m.closed = true
	m.mu.Unlock()
	return nil
}

func joinName(prefix, name string) string {
	if prefix == "" {
		return name
	}
	return prefix + "." + name
}

func matchName(name, pattern string) bool {
	matched, err := path.Match(pattern, name)
	return err == nil && matched
}

// Package nn menyediakan modul, parameter, buffer, layer, loss, dan state
// dictionary dengan kepemilikan eksplisit di atas Tensor dan autograd Tape.
package nn

import (
	"fmt"
	"strings"
	"sync"

	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/tensor"
)

// NamedParameter mengikat parameter dengan nama lengkapnya di dalam pohon modul.
type NamedParameter struct {
	Name      string
	Parameter *Parameter
}

// NamedBuffer mengikat buffer dengan nama lengkapnya di dalam pohon modul.
type NamedBuffer struct {
	Name   string
	Buffer *Buffer
}

// Parameter adalah tensor trainable yang dimiliki sebuah modul.
type Parameter struct {
	mu           sync.RWMutex
	name         string
	value        *tensor.Tensor
	requiresGrad bool
	closed       bool
}

// Buffer adalah tensor non-trainable yang ikut state dictionary, misalnya statistik running.
type Buffer struct {
	mu         sync.RWMutex
	name       string
	value      *tensor.Tensor
	persistent bool
	closed     bool
}

// NewParameter membuat parameter dari salinan value.
func NewParameter(name string, value *tensor.Tensor, requiresGrad bool) (*Parameter, error) {
	if err := validateName(name); err != nil {
		return nil, err
	}
	if value == nil {
		return nil, errs.ErrInvalidArgument
	}
	metadata, err := value.Metadata()
	if err != nil {
		return nil, err
	}
	if requiresGrad && !isDifferentiable(metadata.DType()) {
		return nil, fmt.Errorf("%w: parameter %q dtype %s", errs.ErrUnsupportedDType, name, metadata.DType())
	}
	snapshot, err := value.Copy()
	if err != nil {
		return nil, err
	}
	return &Parameter{name: name, value: snapshot, requiresGrad: requiresGrad}, nil
}

// Name mengembalikan nama parameter.
func (p *Parameter) Name() string {
	if p == nil {
		return ""
	}
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.name
}

// Value mengembalikan tensor milik parameter. Pemanggil tidak boleh menutupnya.
func (p *Parameter) Value() *tensor.Tensor {
	if p == nil {
		return nil
	}
	p.mu.RLock()
	defer p.mu.RUnlock()
	if p.closed {
		return nil
	}
	return p.value
}

// Snapshot mengembalikan salinan tensor milik parameter.
func (p *Parameter) Snapshot() (*tensor.Tensor, error) {
	value := p.Value()
	if value == nil {
		return nil, errs.ErrInvalidState
	}
	return value.Copy()
}

// RequiresGrad melaporkan apakah parameter ikut pelatihan.
func (p *Parameter) RequiresGrad() bool {
	if p == nil {
		return false
	}
	p.mu.RLock()
	defer p.mu.RUnlock()
	return !p.closed && p.requiresGrad
}

// SetRequiresGrad mengubah status trainable parameter.
func (p *Parameter) SetRequiresGrad(enabled bool) error {
	if p == nil {
		return errs.ErrInvalidArgument
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closed {
		return errs.ErrInvalidState
	}
	if enabled {
		metadata, err := p.value.Metadata()
		if err != nil {
			return err
		}
		if !isDifferentiable(metadata.DType()) {
			return fmt.Errorf("%w: parameter %q dtype %s", errs.ErrUnsupportedDType, p.name, metadata.DType())
		}
	}
	p.requiresGrad = enabled
	return nil
}

// Update menerapkan fungsi pembaruan secara atomik terhadap nilai parameter.
// Fungsi menerima salinan float64 dan mengembalikan jumlah elemen yang sama.
func (p *Parameter) Update(update func([]float64) ([]float64, error)) error {
	if p == nil || update == nil {
		return errs.ErrInvalidArgument
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closed {
		return errs.ErrInvalidState
	}
	values, err := readValues(p.value)
	if err != nil {
		return err
	}
	next, err := update(values)
	if err != nil {
		return err
	}
	if len(next) != len(values) {
		return fmt.Errorf("%w: update returned %d values for %d elements", errs.ErrGradientShapeMismatch, len(next), len(values))
	}
	return writeValues(p.value, next)
}

// Close melepaskan tensor milik parameter dan idempoten.
func (p *Parameter) Close() error {
	if p == nil {
		return errs.ErrInvalidArgument
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closed {
		return nil
	}
	if err := p.value.Close(); err != nil {
		return err
	}
	p.closed = true
	return nil
}

// NewBuffer membuat buffer dari salinan value.
func NewBuffer(name string, value *tensor.Tensor, persistent bool) (*Buffer, error) {
	if err := validateName(name); err != nil {
		return nil, err
	}
	if value == nil {
		return nil, errs.ErrInvalidArgument
	}
	snapshot, err := value.Copy()
	if err != nil {
		return nil, err
	}
	return &Buffer{name: name, value: snapshot, persistent: persistent}, nil
}

// Name mengembalikan nama buffer.
func (b *Buffer) Name() string {
	if b == nil {
		return ""
	}
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.name
}

// Value mengembalikan tensor milik buffer. Pemanggil tidak boleh menutupnya.
func (b *Buffer) Value() *tensor.Tensor {
	if b == nil {
		return nil
	}
	b.mu.RLock()
	defer b.mu.RUnlock()
	if b.closed {
		return nil
	}
	return b.value
}

// Persistent melaporkan apakah buffer disimpan dalam state dictionary.
func (b *Buffer) Persistent() bool {
	if b == nil {
		return false
	}
	b.mu.RLock()
	defer b.mu.RUnlock()
	return !b.closed && b.persistent
}

// Snapshot mengembalikan salinan tensor milik buffer.
func (b *Buffer) Snapshot() (*tensor.Tensor, error) {
	value := b.Value()
	if value == nil {
		return nil, errs.ErrInvalidState
	}
	return value.Copy()
}

// Update menerapkan fungsi pembaruan secara atomik terhadap nilai buffer.
func (b *Buffer) Update(update func([]float64) ([]float64, error)) error {
	if b == nil || update == nil {
		return errs.ErrInvalidArgument
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed {
		return errs.ErrInvalidState
	}
	values, err := readValues(b.value)
	if err != nil {
		return err
	}
	next, err := update(values)
	if err != nil {
		return err
	}
	if len(next) != len(values) {
		return fmt.Errorf("%w: update returned %d values for %d elements", errs.ErrGradientShapeMismatch, len(next), len(values))
	}
	return writeValues(b.value, next)
}

// Close melepaskan tensor milik buffer dan idempoten.
func (b *Buffer) Close() error {
	if b == nil {
		return errs.ErrInvalidArgument
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed {
		return nil
	}
	if err := b.value.Close(); err != nil {
		return err
	}
	b.closed = true
	return nil
}

func readValues(value *tensor.Tensor) ([]float64, error) {
	if value == nil {
		return nil, errs.ErrInvalidArgument
	}
	metadata, err := value.Metadata()
	if err != nil {
		return nil, err
	}
	switch metadata.DType() {
	case dtype.Float32:
		items, err := tensor.Read[float32](value)
		if err != nil {
			return nil, err
		}
		result := make([]float64, len(items))
		for i, item := range items {
			result[i] = float64(item)
		}
		return result, nil
	case dtype.Float64:
		return tensor.Read[float64](value)
	default:
		return nil, fmt.Errorf("%w: expected float32 or float64, got %s", errs.ErrUnsupportedDType, metadata.DType())
	}
}

func writeValues(value *tensor.Tensor, values []float64) error {
	metadata, err := value.Metadata()
	if err != nil {
		return err
	}
	switch metadata.DType() {
	case dtype.Float32:
		converted := make([]float32, len(values))
		for i, item := range values {
			converted[i] = float32(item)
		}
		return tensor.Write(value, converted)
	case dtype.Float64:
		return tensor.Write(value, values)
	default:
		return fmt.Errorf("%w: expected float32 or float64, got %s", errs.ErrUnsupportedDType, metadata.DType())
	}
}

func isDifferentiable(elementType dtype.DType) bool {
	return elementType == dtype.Float32 || elementType == dtype.Float64
}

func validateName(name string) error {
	trimmed := strings.TrimSpace(name)
	if trimmed == "" || trimmed != name || strings.HasPrefix(name, ".") || strings.HasSuffix(name, ".") || strings.Contains(name, "..") {
		return fmt.Errorf("%w: invalid module name %q", errs.ErrInvalidArgument, name)
	}
	return nil
}

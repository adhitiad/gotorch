package nn

import (
	"encoding/binary"
	"errors"
	"fmt"
	"sort"

	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/tensor"
)

var stateMagic = [8]byte{'G', 'O', 'T', 'S', 'T', 'A', 'T', 'E'}

const maxStateEntryBytes = 1 << 30

// StateDict menyimpan salinan tensor parameter dan buffer persisten.
type StateDict struct {
	values map[string]*tensor.Tensor
}

// NewStateDict membuat state dictionary kosong.
func NewStateDict() *StateDict {
	return &StateDict{values: make(map[string]*tensor.Tensor)}
}

// Put menyalin tensor ke dalam state dictionary.
func (d *StateDict) Put(name string, value *tensor.Tensor) error {
	if d == nil {
		return errs.ErrInvalidArgument
	}
	if d.values == nil {
		d.values = make(map[string]*tensor.Tensor)
	}
	if err := validateName(name); err != nil {
		return err
	}
	if value == nil {
		return errs.ErrInvalidArgument
	}
	snapshot, err := value.Copy()
	if err != nil {
		return err
	}
	if old := d.values[name]; old != nil {
		_ = old.Close()
	}
	d.values[name] = snapshot
	return nil
}

// Get mengembalikan salinan tensor dari state dictionary.
func (d *StateDict) Get(name string) (*tensor.Tensor, error) {
	if d == nil || d.values == nil {
		return nil, errs.ErrInvalidState
	}
	value := d.values[name]
	if value == nil {
		return nil, fmt.Errorf("%w: state entry %q", errs.ErrInvalidArgument, name)
	}
	return value.Copy()
}

// Names mengembalikan nama state yang diurutkan.
func (d *StateDict) Names() []string {
	if d == nil {
		return nil
	}
	result := make([]string, 0, len(d.values))
	for name := range d.values {
		result = append(result, name)
	}
	sort.Strings(result)
	return result
}

// Close melepaskan seluruh tensor di dalam state dictionary.
func (d *StateDict) Close() error {
	if d == nil {
		return errs.ErrInvalidArgument
	}
	var closeErr error
	for name, value := range d.values {
		closeErr = errors.Join(closeErr, value.Close())
		delete(d.values, name)
	}
	return closeErr
}

// MarshalBinary menyandikan state dictionary menjadi format biner yang stabil.
func (d *StateDict) MarshalBinary() ([]byte, error) {
	if d == nil {
		return nil, errs.ErrInvalidArgument
	}
	names := d.Names()
	if uint64(len(names)) > uint64(^uint32(0)) {
		return nil, errs.ErrShapeOverflow
	}
	buffer := make([]byte, 0, 64)
	buffer = append(buffer, stateMagic[:]...)
	buffer = appendUint32(buffer, uint32(len(names)))
	for _, name := range names {
		value := d.values[name]
		encoded, err := value.MarshalBinary()
		if err != nil {
			return nil, err
		}
		if uint64(len(name)) > uint64(^uint32(0)) || len(encoded) > maxStateEntryBytes {
			return nil, errs.ErrShapeOverflow
		}
		buffer = appendUint32(buffer, uint32(len(name)))
		buffer = append(buffer, name...)
		buffer = appendUint64(buffer, uint64(len(encoded)))
		buffer = append(buffer, encoded...)
	}
	return buffer, nil
}

// UnmarshalBinary mendekode state dictionary dari format biner yang divalidasi.
func (d *StateDict) UnmarshalBinary(data []byte) error {
	if d == nil || len(data) < len(stateMagic)+4 {
		return errs.ErrInvalidArgument
	}
	for index, item := range stateMagic {
		if data[index] != item {
			return fmt.Errorf("%w: invalid state magic", errs.ErrInvalidArgument)
		}
	}
	count := binary.LittleEndian.Uint32(data[len(stateMagic):])
	offset := len(stateMagic) + 4
	decoded := make(map[string]*tensor.Tensor, int(count))
	for entry := uint32(0); entry < count; entry++ {
		if offset+4 > len(data) {
			return errs.ErrInvalidArgument
		}
		nameLength := int(binary.LittleEndian.Uint32(data[offset:]))
		offset += 4
		if nameLength < 1 || nameLength > len(data)-offset {
			return errs.ErrInvalidArgument
		}
		name := string(data[offset : offset+nameLength])
		offset += nameLength
		if err := validateName(name); err != nil {
			return err
		}
		if offset+8 > len(data) {
			return errs.ErrInvalidArgument
		}
		valueLength := binary.LittleEndian.Uint64(data[offset:])
		offset += 8
		if valueLength == 0 || valueLength > uint64(maxStateEntryBytes) || valueLength > uint64(len(data)-offset) {
			return errs.ErrInvalidArgument
		}
		value := new(tensor.Tensor)
		if err := value.UnmarshalBinary(data[offset : offset+int(valueLength)]); err != nil {
			_ = value.Close()
			return err
		}
		offset += int(valueLength)
		if _, exists := decoded[name]; exists {
			_ = value.Close()
			return fmt.Errorf("%w: duplicate state entry %q", errs.ErrInvalidArgument, name)
		}
		decoded[name] = value
	}
	if offset != len(data) {
		for _, value := range decoded {
			_ = value.Close()
		}
		return errs.ErrInvalidArgument
	}
	_ = d.Close()
	d.values = decoded
	return nil
}

// StateDictOf mengambil salinan state dari parameter dan buffer persisten modul.
func StateDictOf(module Module) (*StateDict, error) {
	if module == nil {
		return nil, errs.ErrInvalidArgument
	}
	state := NewStateDict()
	for _, named := range module.Parameters() {
		value, err := named.Parameter.Snapshot()
		if err != nil {
			_ = state.Close()
			return nil, err
		}
		err = state.Put(named.Name, value)
		_ = value.Close()
		if err != nil {
			_ = state.Close()
			return nil, err
		}
	}
	for _, named := range module.Buffers() {
		value, err := named.Buffer.Snapshot()
		if err != nil {
			_ = state.Close()
			return nil, err
		}
		err = state.Put(named.Name, value)
		_ = value.Close()
		if err != nil {
			_ = state.Close()
			return nil, err
		}
	}
	return state, nil
}

// LoadStateDict menyalin state ke parameter dan buffer yang bernama sama.
func LoadStateDict(module Module, state *StateDict) error {
	if module == nil || state == nil || state.values == nil {
		return errs.ErrInvalidArgument
	}
	parameters := make(map[string]*Parameter)
	for _, named := range module.Parameters() {
		parameters[named.Name] = named.Parameter
	}
	buffers := make(map[string]*Buffer)
	for _, named := range module.Buffers() {
		buffers[named.Name] = named.Buffer
	}
	for _, name := range state.Names() {
		value := state.values[name]
		if parameter := parameters[name]; parameter != nil {
			if err := assignTensor(parameter.Value(), value); err != nil {
				return fmt.Errorf("%w: parameter %q: %w", errs.ErrInvalidState, name, err)
			}
			continue
		}
		if buffer := buffers[name]; buffer != nil {
			if err := assignTensor(buffer.Value(), value); err != nil {
				return fmt.Errorf("%w: buffer %q: %w", errs.ErrInvalidState, name, err)
			}
			continue
		}
		return fmt.Errorf("%w: unknown state entry %q", errs.ErrInvalidArgument, name)
	}
	return nil
}

func assignTensor(target, source *tensor.Tensor) error {
	if target == nil || source == nil {
		return errs.ErrInvalidArgument
	}
	targetMeta, err := target.Metadata()
	if err != nil {
		return err
	}
	sourceMeta, err := source.Metadata()
	if err != nil {
		return err
	}
	if targetMeta.DType() != sourceMeta.DType() || !targetMeta.Shape().Equal(sourceMeta.Shape()) {
		return errs.ErrGradientShapeMismatch
	}
	switch targetMeta.DType() {
	case dtype.Bool:
		return copyTyped[bool](target, source)
	case dtype.Int8:
		return copyTyped[int8](target, source)
	case dtype.Int16:
		return copyTyped[int16](target, source)
	case dtype.Int32:
		return copyTyped[int32](target, source)
	case dtype.Int64:
		return copyTyped[int64](target, source)
	case dtype.Uint8:
		return copyTyped[uint8](target, source)
	case dtype.Uint16:
		return copyTyped[uint16](target, source)
	case dtype.Uint32:
		return copyTyped[uint32](target, source)
	case dtype.Uint64:
		return copyTyped[uint64](target, source)
	case dtype.Float16:
		return copyTyped[tensor.Float16Bits](target, source)
	case dtype.BFloat16:
		return copyTyped[tensor.BFloat16Bits](target, source)
	case dtype.Float32:
		return copyTyped[float32](target, source)
	case dtype.Float64:
		return copyTyped[float64](target, source)
	case dtype.Complex64:
		return copyTyped[complex64](target, source)
	case dtype.Complex128:
		return copyTyped[complex128](target, source)
	default:
		return errs.ErrUnsupportedDType
	}
}

func copyTyped[T tensor.Element](target, source *tensor.Tensor) error {
	values, err := tensor.Read[T](source)
	if err != nil {
		return err
	}
	return tensor.Write(target, values)
}

func appendUint32(buffer []byte, value uint32) []byte {
	var encoded [4]byte
	binary.LittleEndian.PutUint32(encoded[:], value)
	return append(buffer, encoded[:]...)
}

func appendUint64(buffer []byte, value uint64) []byte {
	var encoded [8]byte
	binary.LittleEndian.PutUint64(encoded[:], value)
	return append(buffer, encoded[:]...)
}

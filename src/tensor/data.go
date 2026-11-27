package tensor

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"

	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/storage"
)

// Float16Bits stores the raw IEEE 754 binary16 bit pattern.
// Conversion to/from float32 is intentionally left to a later numerical API.
type Float16Bits uint16

// BFloat16Bits stores the raw bfloat16 bit pattern.
type BFloat16Bits uint16

// Element is the set of fixed-width Go element types supported by typed Tensor
// constructors and accessors. Float16Bits/BFloat16Bits carry raw 16-bit payloads.
type Element interface {
	bool | int8 | int16 | int32 | int64 |
		uint8 | uint16 | uint32 | uint64 |
	float32 | float64 | complex64 | complex128 |
	Float16Bits | BFloat16Bits
}

// From constructs a CPU tensor from a copy of values. If dimensions are omitted,
// values form a scalar and must contain exactly one element.
func From[T Element](values []T, dimensions ...int64) (*Tensor, error) {
	return FromWithAllocator(storage.NewHeapAllocator(), values, dimensions...)
}

// FromWithAllocator constructs a tensor using allocator and copies values into it.
// The caller may freely mutate values after this function returns.
func FromWithAllocator[T Element](allocator storage.Allocator, values []T, dimensions ...int64) (*Tensor, error) {
	elementType, err := elementDType[T]()
	if err != nil {
		return nil, err
	}
	t, err := NewTensorWithAllocator(allocator, elementType, dimensions...)
	if err != nil {
		return nil, err
	}
	if err := Write(t, values); err != nil {
		_ = t.Close()
		return nil, err
	}
	return t, nil
}

// Read returns a new slice containing a copy of the tensor's values.
// The returned slice never aliases tensor storage.
func Read[T Element](t *Tensor) ([]T, error) {
	if t == nil {
		return nil, errs.ErrInvalidArgument
	}
	metadata, err := t.Metadata()
	if err != nil {
		return nil, err
	}
	requested, err := elementDType[T]()
	if err != nil {
		return nil, err
	}
	if metadata.DType() != requested {
		return nil, fmt.Errorf("%w: tensor dtype is %s, requested %s", errs.ErrUnsupportedDType, metadata.DType(), requested)
	}
	raw, err := t.readAll()
	if err != nil {
		return nil, err
	}
	values, err := decodeElements[T](raw)
	if err != nil {
		return nil, err
	}
	numel, err := metadata.Numel()
	if err != nil {
		return nil, err
	}
	if int64(len(values)) != numel {
		return nil, fmt.Errorf("%w: decoded %d values for %d elements", errs.ErrInvalidState, len(values), numel)
	}
	return values, nil
}

// Write copies values into a tensor. The tensor dtype and element count must
// match exactly; no implicit numeric conversion is performed.
func Write[T Element](t *Tensor, values []T) error {
	if t == nil {
		return errs.ErrInvalidArgument
	}
	metadata, err := t.Metadata()
	if err != nil {
		return err
	}
	requested, err := elementDType[T]()
	if err != nil {
		return err
	}
	if metadata.DType() != requested {
		return fmt.Errorf("%w: tensor dtype is %s, data dtype is %s", errs.ErrUnsupportedDType, metadata.DType(), requested)
	}
	numel, err := metadata.Numel()
	if err != nil {
		return err
	}
	if int64(len(values)) != numel {
		return fmt.Errorf("%w: got %d elements, want %d", errs.ErrInvalidArgument, len(values), numel)
	}
	raw, err := encodeElements(values)
	if err != nil {
		return err
	}
	return t.writeAll(raw)
}

func elementDType[T Element]() (dtype.DType, error) {
	var zero T
	switch any(zero).(type) {
	case bool:
		return dtype.Bool, nil
	case int8:
		return dtype.Int8, nil
	case int16:
		return dtype.Int16, nil
	case int32:
		return dtype.Int32, nil
	case int64:
		return dtype.Int64, nil
	case uint8:
		return dtype.Uint8, nil
	case uint16:
		return dtype.Uint16, nil
	case uint32:
		return dtype.Uint32, nil
	case uint64:
		return dtype.Uint64, nil
	case Float16Bits:
		return dtype.Float16, nil
	case BFloat16Bits:
		return dtype.BFloat16, nil
	case float32:
		return dtype.Float32, nil
	case float64:
		return dtype.Float64, nil
	case complex64:
		return dtype.Complex64, nil
	case complex128:
		return dtype.Complex128, nil
	default:
		return dtype.Invalid, errs.ErrUnsupportedDType
	}
}

func encodeElements[T Element](values []T) ([]byte, error) {
	switch typed := any(values).(type) {
	case []bool:
		raw := make([]byte, len(typed))
		for i, value := range typed {
			if value {
				raw[i] = 1
			}
		}
		return raw, nil
	case []complex64:
		length, err := checkedByteLength(len(typed), 8)
		if err != nil {
			return nil, err
		}
		raw := make([]byte, length)
		for i, value := range typed {
			offset := i * 8
			binary.LittleEndian.PutUint32(raw[offset:], math.Float32bits(real(value)))
			binary.LittleEndian.PutUint32(raw[offset+4:], math.Float32bits(imag(value)))
		}
		return raw, nil
	case []complex128:
		length, err := checkedByteLength(len(typed), 16)
		if err != nil {
			return nil, err
		}
		raw := make([]byte, length)
		for i, value := range typed {
			offset := i * 16
			binary.LittleEndian.PutUint64(raw[offset:], math.Float64bits(real(value)))
			binary.LittleEndian.PutUint64(raw[offset+8:], math.Float64bits(imag(value)))
		}
		return raw, nil
	default:
		var buffer bytes.Buffer
		if err := binary.Write(&buffer, binary.LittleEndian, values); err != nil {
			return nil, fmt.Errorf("%w: encode tensor elements: %v", errs.ErrInvalidArgument, err)
		}
		return buffer.Bytes(), nil
	}
}

func checkedByteLength(elements, width int) (int, error) {
	maxInt := int(^uint(0) >> 1)
	if elements < 0 || width <= 0 || elements > maxInt/width {
		return 0, errs.ErrShapeOverflow
	}
	return elements * width, nil
}

func decodeElements[T Element](raw []byte) ([]T, error) {
	elementType, err := elementDType[T]()
	if err != nil {
		return nil, err
	}
	width, err := elementType.Size()
	if err != nil {
		return nil, err
	}
	if int64(len(raw))%width != 0 {
		return nil, errs.ErrInvalidState
	}
	count := len(raw) / int(width)
	values := make([]T, count)
	switch typed := any(values).(type) {
	case []bool:
		for i, value := range raw {
			if value > 1 {
				return nil, fmt.Errorf("%w: invalid bool payload at element %d", errs.ErrInvalidState, i)
			}
			typed[i] = value == 1
		}
	case []complex64:
		for i := range typed {
			offset := i * 8
			typed[i] = complex(
				math.Float32frombits(binary.LittleEndian.Uint32(raw[offset:])),
				math.Float32frombits(binary.LittleEndian.Uint32(raw[offset+4:])),
			)
		}
	case []complex128:
		for i := range typed {
			offset := i * 16
			typed[i] = complex(
				math.Float64frombits(binary.LittleEndian.Uint64(raw[offset:])),
				math.Float64frombits(binary.LittleEndian.Uint64(raw[offset+8:])),
			)
		}
	default:
		if err := binary.Read(bytes.NewReader(raw), binary.LittleEndian, &values); err != nil {
			return nil, fmt.Errorf("%w: decode tensor elements: %v", errs.ErrInvalidState, err)
		}
	}
	return values, nil
}

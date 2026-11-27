// Package dtype defines the numeric element types supported by GoTorch metadata.
package dtype

import (
	"fmt"

	"github.com/adhitiad/gotorch/errs"
)

// DType identifies the in-memory representation of one tensor element.
type DType uint8

const (
	Invalid DType = iota
	Bool
	Int8
	Int16
	Int32
	Int64
	Uint8
	Uint16
	Uint32
	Uint64
	Float16
	BFloat16
	Float32
	Float64
	Complex64
	Complex128
)

// Valid reports whether d is a defined GoTorch dtype.
func (d DType) Valid() bool { return d >= Bool && d <= Complex128 }

// Size returns the number of bytes in one element.
func (d DType) Size() (int64, error) {
	switch d {
	case Bool, Int8, Uint8:
		return 1, nil
	case Int16, Uint16, Float16, BFloat16:
		return 2, nil
	case Int32, Uint32, Float32:
		return 4, nil
	case Int64, Uint64, Float64, Complex64:
		return 8, nil
	case Complex128:
		return 16, nil
	default:
		return 0, fmt.Errorf("%w: %d", errs.ErrUnsupportedDType, d)
	}
}

// IsInteger reports whether d is a signed or unsigned integer dtype.
func (d DType) IsInteger() bool {
	return d >= Int8 && d <= Int64 || d >= Uint8 && d <= Uint64
}

// IsFloat reports whether d is a real floating-point dtype.
func (d DType) IsFloat() bool { return d >= Float16 && d <= Float64 }

// IsComplex reports whether d is a complex floating-point dtype.
func (d DType) IsComplex() bool { return d == Complex64 || d == Complex128 }

// IsNumeric reports whether d supports numeric arithmetic (excluding Bool).
func (d DType) IsNumeric() bool { return d.IsInteger() || d.IsFloat() || d.IsComplex() }

// String returns the canonical dtype name.
func (d DType) String() string {
	names := [...]string{"invalid", "bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float16", "bfloat16", "float32", "float64", "complex64", "complex128"}
	if int(d) >= len(names) {
		return fmt.Sprintf("dtype(%d)", d)
	}
	return names[d]
}

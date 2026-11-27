// Package shape provides immutable tensor-dimension metadata and checked size math.
package shape

import (
	"fmt"

	"github.com/adhitiad/gotorch/src/errs"
)

const maxInt64 = int64(^uint64(0) >> 1)

// Shape stores non-negative dimensions. Its zero value represents a scalar.
type Shape struct{ dims []int64 }

// New constructs a shape and rejects negative dimensions or overflowing element counts.
func New(dims ...int64) (Shape, error) {
	copyDims := append([]int64(nil), dims...)
	for axis, dim := range copyDims {
		if dim < 0 {
			return Shape{}, fmt.Errorf("%w: dimension %d at axis %d is negative", errs.ErrInvalidShape, dim, axis)
		}
	}
	s := Shape{dims: copyDims}
	if _, err := s.Numel(); err != nil {
		return Shape{}, err
	}
	return s, nil
}

// Scalar returns the rank-zero shape, which contains exactly one element.
func Scalar() Shape { return Shape{} }

// Rank returns the number of dimensions.
func (s Shape) Rank() int { return len(s.dims) }

// Dims returns a copy of the dimensions.
func (s Shape) Dims() []int64 { return append([]int64(nil), s.dims...) }

// Dimension returns the dimension at axis.
func (s Shape) Dimension(axis int) (int64, error) {
	if axis < 0 || axis >= len(s.dims) {
		return 0, fmt.Errorf("%w: axis %d outside rank %d", errs.ErrInvalidArgument, axis, len(s.dims))
	}
	return s.dims[axis], nil
}

// Numel returns the checked product of dimensions. A zero-sized dimension yields zero.
func (s Shape) Numel() (int64, error) {
	product := int64(1)
	zero := false
	for axis, dim := range s.dims {
		if dim < 0 {
			return 0, fmt.Errorf("%w: negative dimension at axis %d", errs.ErrInvalidShape, axis)
		}
		if dim == 0 {
			zero = true
		}
	}
	if zero {
		return 0, nil
	}
	for _, dim := range s.dims {
		if dim != 0 && product > maxInt64/dim {
			return 0, errs.ErrShapeOverflow
		}
		product *= dim
	}
	return product, nil
}

// Equal reports whether two shapes have identical dimensions.
func (s Shape) Equal(other Shape) bool {
	if len(s.dims) != len(other.dims) {
		return false
	}
	for i := range s.dims {
		if s.dims[i] != other.dims[i] {
			return false
		}
	}
	return true
}

// String returns dimensions in bracket notation; [] denotes a scalar.
func (s Shape) String() string { return fmt.Sprint(s.dims) }

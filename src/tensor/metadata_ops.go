package tensor

import (
	"fmt"
	"sort"

	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/shape"
)

// IsContiguous reports whether elements occupy canonical row-major order.
func (m Metadata) IsContiguous() bool { return m.isContiguous() }

// ElementOffset returns the element offset from the tensor's first logical
// element for a multidimensional index. Strides are measured in elements.
func (m Metadata) ElementOffset(indices ...int64) (int64, error) {
	if err := m.Validate(); err != nil {
		return 0, err
	}
	dims, strides := m.shape.Dims(), m.strides
	if len(indices) != len(dims) {
		return 0, fmt.Errorf("%w: got %d indices for rank %d", errs.ErrInvalidArgument, len(indices), len(dims))
	}
	var offset int64
	for axis, index := range indices {
		if index < 0 || index >= dims[axis] {
			return 0, fmt.Errorf("%w: index %d outside axis %d with size %d", errs.ErrOutOfBounds, index, axis, dims[axis])
		}
		stride := strides[axis]
		if stride != 0 && index > maxInt64/stride {
			return 0, errs.ErrShapeOverflow
		}
		term := index * stride
		if offset > maxInt64-term {
			return 0, errs.ErrShapeOverflow
		}
		offset += term
	}
	return offset, nil
}

// LinearIndex returns the row-major logical index, independent of physical
// strides. The result is useful for indexing a detached Read result.
func (m Metadata) LinearIndex(indices ...int64) (int64, error) {
	if err := m.Validate(); err != nil {
		return 0, err
	}
	dims := m.shape.Dims()
	if len(indices) != len(dims) {
		return 0, fmt.Errorf("%w: got %d indices for rank %d", errs.ErrInvalidArgument, len(indices), len(dims))
	}
	var linear int64
	for axis, index := range indices {
		if index < 0 || index >= dims[axis] {
			return 0, fmt.Errorf("%w: index %d outside axis %d with size %d", errs.ErrOutOfBounds, index, axis, dims[axis])
		}
		dim := dims[axis]
		if linear > (maxInt64-index)/dim {
			return 0, errs.ErrShapeOverflow
		}
		linear = linear*dim + index
	}
	return linear, nil
}

// Slice returns metadata for a positive-step, end-exclusive slice on one axis.
// The returned metadata retains the source element strides; the owning Tensor
// operation separately advances the storage base offset.
func (m Metadata) Slice(axis int, start, end, step int64) (Metadata, error) {
	if err := m.Validate(); err != nil {
		return Metadata{}, err
	}
	dims := m.shape.Dims()
	if axis < 0 || axis >= len(dims) || step <= 0 {
		return Metadata{}, errs.ErrInvalidArgument
	}
	if start < 0 || end < start || end > dims[axis] {
		return Metadata{}, errs.ErrOutOfBounds
	}
	distance := end - start
	length := distance / step
	if distance%step != 0 {
		length++
	}
	dims[axis] = length
	strides := append([]int64(nil), m.strides...)
	if length > 1 {
		if strides[axis] != 0 && step > maxInt64/strides[axis] {
			return Metadata{}, errs.ErrShapeOverflow
		}
		strides[axis] *= step
	}
	newShape, err := shape.New(dims...)
	if err != nil {
		return Metadata{}, err
	}
	return NewStridedMetadata(m.dtype, newShape, strides)
}

// Narrow returns metadata for a length-sized unit-step range on one axis.
func (m Metadata) Narrow(axis int, start, length int64) (Metadata, error) {
	if err := m.Validate(); err != nil {
		return Metadata{}, err
	}
	if axis < 0 || axis >= m.shape.Rank() || start < 0 || length < 0 {
		return Metadata{}, errs.ErrInvalidArgument
	}
	dim, _ := m.shape.Dimension(axis)
	if start > dim || length > dim-start {
		return Metadata{}, errs.ErrOutOfBounds
	}
	return m.Slice(axis, start, start+length, 1)
}

// Select removes one axis after selecting a single index along it.
func (m Metadata) Select(axis int, index int64) (Metadata, error) {
	if err := m.Validate(); err != nil {
		return Metadata{}, err
	}
	if axis < 0 || axis >= m.shape.Rank() {
		return Metadata{}, errs.ErrInvalidArgument
	}
	dims := m.shape.Dims()
	if index < 0 || index >= dims[axis] {
		return Metadata{}, errs.ErrOutOfBounds
	}
	dims = append(dims[:axis], dims[axis+1:]...)
	strides := append([]int64(nil), m.strides...)
	strides = append(strides[:axis], strides[axis+1:]...)
	newShape, err := shape.New(dims...)
	if err != nil {
		return Metadata{}, err
	}
	return NewStridedMetadata(m.dtype, newShape, strides)
}

// Reshape returns contiguous metadata with the same logical element count.
// A non-contiguous source cannot be reshaped as a view without copying.
func (m Metadata) Reshape(dimensions ...int64) (Metadata, error) {
	if err := m.Validate(); err != nil {
		return Metadata{}, err
	}
	if !m.isContiguous() {
		return Metadata{}, errs.ErrUnsupportedOperation
	}
	target, err := shape.New(dimensions...)
	if err != nil {
		return Metadata{}, err
	}
	sourceCount, _ := m.shape.Numel()
	targetCount, _ := target.Numel()
	if sourceCount != targetCount {
		return Metadata{}, fmt.Errorf("%w: reshape changes element count from %d to %d", errs.ErrInvalidShape, sourceCount, targetCount)
	}
	return NewMetadata(m.dtype, target)
}

// Transpose permutes dimensions and strides. With no axes, it reverses all axes.
// Explicit axes map each output axis to its input axis.
func (m Metadata) Transpose(axes ...int) (Metadata, error) {
	if err := m.Validate(); err != nil {
		return Metadata{}, err
	}
	rank := m.shape.Rank()
	permutation := make([]int, rank)
	if len(axes) == 0 {
		for axis := range permutation {
			permutation[axis] = rank - 1 - axis
		}
	} else {
		if len(axes) != rank {
			return Metadata{}, fmt.Errorf("%w: got %d axes for rank %d", errs.ErrInvalidArgument, len(axes), rank)
		}
		copy(permutation, axes)
	}
	seen := make([]bool, rank)
	dims, sourceDims := make([]int64, rank), m.shape.Dims()
	strides := make([]int64, rank)
	for outputAxis, inputAxis := range permutation {
		if inputAxis < 0 || inputAxis >= rank || seen[inputAxis] {
			return Metadata{}, fmt.Errorf("%w: axes must be a permutation of [0,%d)", errs.ErrInvalidArgument, rank)
		}
		seen[inputAxis] = true
		dims[outputAxis] = sourceDims[inputAxis]
		strides[outputAxis] = m.strides[inputAxis]
	}
	newShape, err := shape.New(dims...)
	if err != nil {
		return Metadata{}, err
	}
	return NewStridedMetadata(m.dtype, newShape, strides)
}

// BroadcastTo returns zero-stride metadata that broadcasts this tensor to target.
// Expanded axes alias the same source element; writes through a view created by
// Tensor.BroadcastTo are therefore guarded against ambiguous bulk scatter.
func (m Metadata) BroadcastTo(target shape.Shape) (Metadata, error) {
	if err := m.Validate(); err != nil {
		return Metadata{}, err
	}
	sourceDims, targetDims := m.shape.Dims(), target.Dims()
	if len(targetDims) < len(sourceDims) {
		return Metadata{}, errs.ErrInvalidShape
	}
	strides := make([]int64, len(targetDims))
	lead := len(targetDims) - len(sourceDims)
	for targetAxis, targetDim := range targetDims {
		sourceAxis := targetAxis - lead
		if sourceAxis < 0 {
			strides[targetAxis] = 0
			continue
		}
		sourceDim := sourceDims[sourceAxis]
		switch {
		case sourceDim == targetDim:
			strides[targetAxis] = m.strides[sourceAxis]
		case sourceDim == 1:
			strides[targetAxis] = 0
		default:
			return Metadata{}, fmt.Errorf("%w: cannot broadcast %v to %v", errs.ErrInvalidShape, sourceDims, targetDims)
		}
	}
	return NewStridedMetadata(m.dtype, target, strides)
}

// BroadcastShapes computes the common broadcast shape using right-aligned
// dimensions. A dimension may match or expand from one; zero only matches zero
// or expands from one.
func BroadcastShapes(shapes ...shape.Shape) (shape.Shape, error) {
	maxRank := 0
	for _, candidate := range shapes {
		if candidate.Rank() > maxRank {
			maxRank = candidate.Rank()
		}
	}
	dims := make([]int64, maxRank)
	for i := range dims {
		dims[i] = 1
	}
	for _, candidate := range shapes {
		source := candidate.Dims()
		lead := maxRank - len(source)
		for axis, dim := range source {
			targetAxis := lead + axis
			current := dims[targetAxis]
			if current == dim || dim == 1 {
				continue
			}
			if current == 1 {
				dims[targetAxis] = dim
				continue
			}
			return shape.Shape{}, fmt.Errorf("%w: incompatible broadcast dimensions %d and %d", errs.ErrInvalidShape, current, dim)
		}
	}
	return shape.New(dims...)
}

type strideAxis struct {
	stride int64
	dim    int64
}

// hasOverlappingElements conservatively identifies layouts whose logical
// elements may share storage. It proves common dense/gapped permutations safe;
// uncertain layouts are treated as overlapping to keep bulk writes deterministic.
func (m Metadata) hasOverlappingElements() bool {
	dims := m.shape.Dims()
	axes := make([]strideAxis, 0, len(dims))
	for axis, dim := range dims {
		if dim > 1 {
			axes = append(axes, strideAxis{stride: m.strides[axis], dim: dim})
		}
	}
	sort.Slice(axes, func(i, j int) bool { return axes[i].stride < axes[j].stride })
	span := int64(1)
	for _, axis := range axes {
		if axis.stride < span {
			return true
		}
		steps := axis.dim - 1
		if axis.stride != 0 && steps > (maxInt64-span)/axis.stride {
			return true
		}
		span += steps * axis.stride
	}
	return false
}

package tensor

import "github.com/adhitiad/gotorch/src/errs"

// Copy returns an independent, row-major tensor containing the same logical
// values. It never aliases the source, including when the source is contiguous.
func (t *Tensor) Copy() (*Tensor, error) {
	if t == nil {
		return nil, errs.ErrInvalidArgument
	}
	metadata, err := t.Metadata()
	if err != nil {
		return nil, err
	}
	values, err := t.readAll()
	if err != nil {
		return nil, err
	}
	result, err := NewTensor(metadata.DType(), metadata.Shape().Dims()...)
	if err != nil {
		return nil, err
	}
	if err := result.writeAll(values); err != nil {
		_ = result.Close()
		return nil, err
	}
	return result, nil
}

// Contiguous returns an independent row-major copy. It intentionally copies
// even if the source is already contiguous so ownership and aliasing are explicit.
func (t *Tensor) Contiguous() (*Tensor, error) { return t.Copy() }

package compute

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"

	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/tensor"
)

const (
	OpAdd    = "add"
	OpSub    = "sub"
	OpMul    = "mul"
	OpDiv    = "div"
	OpNeg    = "neg"
	OpExp    = "exp"
	OpLog    = "log"
	OpReLU   = "relu"
	OpSum    = "sum"
	OpMean   = "mean"
	OpMatMul = "matmul"
)

// ReferenceCPUBackend is the mandatory Pure Go scalar reference implementation.
// It supports float32/float64, strided inputs and contiguous outputs.
type ReferenceCPUBackend struct{}

func NewReferenceCPUBackend() *ReferenceCPUBackend { return &ReferenceCPUBackend{} }
func (*ReferenceCPUBackend) Name() string           { return "purego-reference" }
func (*ReferenceCPUBackend) Kind() BackendKind      { return BackendReference }

func (*ReferenceCPUBackend) Supports(spec KernelSpec, call Invocation) bool {
	if (spec.DType != dtype.Float32 && spec.DType != dtype.Float64) ||
		spec.Layout != tensor.LayoutContiguous || len(call.Outputs) != 1 {
		return false
	}
	out := call.Outputs[0].Metadata
	if out.DType() != spec.DType || out.Layout() != tensor.LayoutContiguous {
		return false
	}
	for _, in := range call.Inputs {
		if in.Metadata.DType() != spec.DType {
			return false
		}
	}
	switch spec.Operation {
	case OpAdd, OpSub, OpMul, OpDiv:
		return len(call.Inputs) == 2 && sameShape(call.Inputs[0].Metadata, call.Inputs[1].Metadata) && sameShape(call.Inputs[0].Metadata, out)
	case OpNeg, OpExp, OpLog, OpReLU:
		return len(call.Inputs) == 1 && sameShape(call.Inputs[0].Metadata, out)
	case OpSum, OpMean:
		return len(call.Inputs) == 1 && out.Shape().Rank() == 0
	case OpMatMul:
		if len(call.Inputs) != 2 || call.Inputs[0].Metadata.Shape().Rank() != 2 || call.Inputs[1].Metadata.Shape().Rank() != 2 || out.Shape().Rank() != 2 {
			return false
		}
		a, b, c := call.Inputs[0].Metadata.Shape().Dims(), call.Inputs[1].Metadata.Shape().Dims(), out.Shape().Dims()
		return a[1] == b[0] && c[0] == a[0] && c[1] == b[1]
	default:
		return false
	}
}

func (b *ReferenceCPUBackend) Execute(ctx context.Context, spec KernelSpec, call Invocation) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if !b.Supports(spec, call) {
		return errs.ErrUnsupportedBackend
	}
	out := call.Outputs[0]
	count, err := out.Metadata.Numel()
	if err != nil {
		return err
	}
	switch spec.Operation {
	case OpAdd, OpSub, OpMul, OpDiv, OpNeg, OpExp, OpLog, OpReLU:
		for i := int64(0); i < count; i++ {
			if i&1023 == 0 {
				if err := ctx.Err(); err != nil {
					return err
				}
			}
			x, err := readFloat(call.Inputs[0], i, spec.DType)
			if err != nil {
				return err
			}
			var y float64
			if len(call.Inputs) == 2 {
				z, err := readFloat(call.Inputs[1], i, spec.DType)
				if err != nil {
					return err
				}
				switch spec.Operation {
				case OpAdd:
					y = x + z
				case OpSub:
					y = x - z
				case OpMul:
					y = x * z
				case OpDiv:
					y = x / z
				}
			} else {
				switch spec.Operation {
				case OpNeg:
					y = -x
				case OpExp:
					y = math.Exp(x)
				case OpLog:
					y = math.Log(x)
				case OpReLU:
					if math.IsNaN(x) || x > 0 {
						y = x
					}
				}
			}
			if err := writeFloat(out, i, spec.DType, y); err != nil {
				return err
			}
		}
	case OpSum, OpMean:
		n, _ := call.Inputs[0].Metadata.Numel()
		var total float64
		for i := int64(0); i < n; i++ {
			if i&1023 == 0 {
				if err := ctx.Err(); err != nil {
					return err
				}
			}
			x, err := readFloat(call.Inputs[0], i, spec.DType)
			if err != nil {
				return err
			}
			total += x
		}
		if spec.Operation == OpMean {
			if n == 0 {
				return fmt.Errorf("%w: mean of empty tensor", errs.ErrInvalidArgument)
			}
			total /= float64(n)
		}
		return writeFloat(out, 0, spec.DType, total)
	case OpMatMul:
		a, b := call.Inputs[0].Metadata.Shape().Dims(), call.Inputs[1].Metadata.Shape().Dims()
		for row := int64(0); row < a[0]; row++ {
			for col := int64(0); col < b[1]; col++ {
				var total float64
				for k := int64(0); k < a[1]; k++ {
					if k&1023 == 0 {
						if err := ctx.Err(); err != nil {
							return err
						}
					}
					x, err := readFloatAt(call.Inputs[0], []int64{row, k}, spec.DType)
					if err != nil {
						return err
					}
					y, err := readFloatAt(call.Inputs[1], []int64{k, col}, spec.DType)
					if err != nil {
						return err
					}
					total += x * y
				}
				if err := writeFloat(out, row*b[1]+col, spec.DType, total); err != nil {
					return err
				}
			}
			if err := ctx.Err(); err != nil {
				return err
			}
		}
	default:
		return errs.ErrUnsupportedOperation
	}
	return nil
}

func sameShape(a, b tensor.Metadata) bool { return a.Shape().Equal(b.Shape()) }

func readFloat(arg Argument, linear int64, dt dtype.DType) (float64, error) {
	dims := arg.Metadata.Shape().Dims()
	coords := make([]int64, len(dims))
	remaining := linear
	for axis := len(dims) - 1; axis >= 0; axis-- {
		if dims[axis] == 0 {
			return 0, errs.ErrOutOfBounds
		}
		coords[axis] = remaining % dims[axis]
		remaining /= dims[axis]
	}
	return readFloatAt(arg, coords, dt)
}

func readFloatAt(arg Argument, coords []int64, dt dtype.DType) (float64, error) {
	elem, err := arg.Metadata.ElementOffset(coords...)
	if err != nil {
		return 0, err
	}
	width, err := dt.Size()
	if err != nil {
		return 0, err
	}
	offset, err := byteOffset(arg.Offset, elem, width)
	if err != nil {
		return 0, err
	}
	raw := make([]byte, int(width))
	n, err := arg.Storage.ReadAt(raw, offset)
	if err != nil {
		return 0, fmt.Errorf("compute read: %w", err)
	}
	if n != len(raw) {
		return 0, errs.ErrInvalidState
	}
	if dt == dtype.Float32 {
		return float64(math.Float32frombits(binary.LittleEndian.Uint32(raw))), nil
	}
	return math.Float64frombits(binary.LittleEndian.Uint64(raw)), nil
}

func writeFloat(arg Argument, linear int64, dt dtype.DType, value float64) error {
	dims := arg.Metadata.Shape().Dims()
	coords := make([]int64, len(dims))
	remaining := linear
	for axis := len(dims) - 1; axis >= 0; axis-- {
		if dims[axis] == 0 {
			return errs.ErrOutOfBounds
		}
		coords[axis] = remaining % dims[axis]
		remaining /= dims[axis]
	}
	elem, err := arg.Metadata.ElementOffset(coords...)
	if err != nil {
		return err
	}
	width, err := dt.Size()
	if err != nil {
		return err
	}
	offset, err := byteOffset(arg.Offset, elem, width)
	if err != nil {
		return err
	}
	raw := make([]byte, int(width))
	if dt == dtype.Float32 {
		binary.LittleEndian.PutUint32(raw, math.Float32bits(float32(value)))
	} else {
		binary.LittleEndian.PutUint64(raw, math.Float64bits(value))
	}
	n, err := arg.Storage.WriteAt(raw, offset)
	if err != nil {
		return fmt.Errorf("compute write: %w", err)
	}
	if n != len(raw) {
		return errs.ErrInvalidState
	}
	return nil
}

func byteOffset(base, element, width int64) (int64, error) {
	if base < 0 || element < 0 || width <= 0 || element > (math.MaxInt64-base)/width {
		return 0, errs.ErrShapeOverflow
	}
	delta := element * width
	if base > math.MaxInt64-delta {
		return 0, errs.ErrShapeOverflow
	}
	return base + delta, nil
}

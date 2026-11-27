//go:build goexperiment.simd && (amd64 || arm64)

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

const simdMinElements = 32

type simdCPUBackend struct{}

func (*simdCPUBackend) Name() string      { return "archsimd-cpu" }
func (*simdCPUBackend) Kind() BackendKind { return BackendSIMD }

func NewSIMDCPUBackend() Backend {
	features := DetectCPUFeatures()
	switch features.Architecture {
	case "amd64":
		if features.Initialized && features.AVX2 {
			return &simdCPUBackend{}
		}
	case "arm64":
		if features.Initialized && features.ASIMD {
			return &simdCPUBackend{}
		}
	}
	return nil
}

func (*simdCPUBackend) Supports(spec KernelSpec, call Invocation) bool {
	if spec.Layout != tensor.LayoutContiguous ||
		(spec.DType != dtype.Float32 && spec.DType != dtype.Float64) ||
		len(call.Outputs) != 1 || len(call.Inputs) != 2 {
		return false
	}
	if spec.Operation != OpAdd && spec.Operation != OpSub && spec.Operation != OpMul {
		return false
	}
	out := call.Outputs[0].Metadata
	if out.DType() != spec.DType || out.Layout() != tensor.LayoutContiguous {
		return false
	}
	for _, in := range call.Inputs {
		if in.Metadata.DType() != spec.DType ||
			in.Metadata.Layout() != tensor.LayoutContiguous ||
			!sameShape(in.Metadata, out) {
			return false
		}
	}
	return true
}

func (b *simdCPUBackend) Execute(ctx context.Context, spec KernelSpec, call Invocation) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if !b.Supports(spec, call) {
		return errs.ErrUnsupportedBackend
	}
	count, err := call.Outputs[0].Metadata.Numel()
	if err != nil {
		return err
	}
	if count < simdMinElements {
		return (&ReferenceCPUBackend{}).Execute(ctx, spec, call)
	}
	if spec.DType == dtype.Float32 {
		a, err := readSIMDFloat32(call.Inputs[0], count)
		if err != nil {
			return err
		}
		b, err := readSIMDFloat32(call.Inputs[1], count)
		if err != nil {
			return err
		}
		dst := make([]float32, int(count))
		if err := simdBinaryFloat32(ctx, dst, a, b, spec.Operation); err != nil {
			return err
		}
		return writeSIMDFloat32(call.Outputs[0], dst)
	}
	a, err := readSIMDFloat64(call.Inputs[0], count)
	if err != nil {
		return err
	}
	b, err := readSIMDFloat64(call.Inputs[1], count)
	if err != nil {
		return err
	}
	dst := make([]float64, int(count))
	if err := simdBinaryFloat64(ctx, dst, a, b, spec.Operation); err != nil {
		return err
	}
	return writeSIMDFloat64(call.Outputs[0], dst)
}

func simdByteSize(count, width int64) (int, error) {
	if count < 0 || width <= 0 || count > int64(int(^uint(0)>>1))/width {
		return 0, errs.ErrShapeOverflow
	}
	return int(count * width), nil
}

func readSIMDBytes(arg Argument, count, width int64) ([]byte, error) {
	size, err := simdByteSize(count, width)
	if err != nil {
		return nil, err
	}
	if err := storageRange(arg, int64(size)); err != nil {
		return nil, err
	}
	buf := make([]byte, size)
	if size == 0 {
		return buf, nil
	}
	n, err := arg.Storage.ReadAt(buf, arg.Offset)
	if err != nil {
		return nil, fmt.Errorf("SIMD input read: %w", err)
	}
	if n != size {
		return nil, errs.ErrInvalidState
	}
	return buf, nil
}

func storageRange(arg Argument, length int64) error {
	if arg.Offset < 0 || length < 0 || arg.Offset > arg.Storage.Size() || length > arg.Storage.Size()-arg.Offset {
		return errs.ErrOutOfBounds
	}
	return nil
}

func readSIMDFloat32(arg Argument, count int64) ([]float32, error) {
	raw, err := readSIMDBytes(arg, count, 4)
	if err != nil {
		return nil, err
	}
	values := make([]float32, int(count))
	for i := range values {
		values[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return values, nil
}

func readSIMDFloat64(arg Argument, count int64) ([]float64, error) {
	raw, err := readSIMDBytes(arg, count, 8)
	if err != nil {
		return nil, err
	}
	values := make([]float64, int(count))
	for i := range values {
		values[i] = math.Float64frombits(binary.LittleEndian.Uint64(raw[i*8:]))
	}
	return values, nil
}

func writeSIMDBytes(arg Argument, raw []byte) error {
	if err := storageRange(arg, int64(len(raw))); err != nil {
		return err
	}
	n, err := arg.Storage.WriteAt(raw, arg.Offset)
	if err != nil {
		return fmt.Errorf("SIMD output write: %w", err)
	}
	if n != len(raw) {
		return errs.ErrInvalidState
	}
	return nil
}

func writeSIMDFloat32(arg Argument, values []float32) error {
	raw := make([]byte, len(values)*4)
	for i, value := range values {
		binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(value))
	}
	return writeSIMDBytes(arg, raw)
}

func writeSIMDFloat64(arg Argument, values []float64) error {
	raw := make([]byte, len(values)*8)
	for i, value := range values {
		binary.LittleEndian.PutUint64(raw[i*8:], math.Float64bits(value))
	}
	return writeSIMDBytes(arg, raw)
}

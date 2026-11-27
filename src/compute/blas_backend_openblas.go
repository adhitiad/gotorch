//go:build gotorch_blas && (linux || windows)

package compute

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"runtime"
	"sync"
	"unsafe"

	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/tensor"
)

const (
	cblasRowMajor = int32(101)
	cblasNoTrans  = int32(111)
)

var openBLASProcessMu sync.Mutex

type openBLASAPI struct {
	sgemm      func(order, transA, transB, m, n, k int32, alpha float32, a unsafe.Pointer, lda int32, b unsafe.Pointer, ldb int32, beta float32, c unsafe.Pointer, ldc int32)
	dgemm      func(order, transA, transB, m, n, k int32, alpha float64, a unsafe.Pointer, lda int32, b unsafe.Pointer, ldb int32, beta float64, c unsafe.Pointer, ldc int32)
	getThreads func() int32
	setThreads func(int32)
	close      func() error
}

type openBLASCPUBackend struct {
	mu     sync.RWMutex
	api    *openBLASAPI
	policy ExecutionPolicy
	closed bool
}

func (*openBLASCPUBackend) Name() string      { return "openblas-cpu" }
func (*openBLASCPUBackend) Kind() BackendKind { return BackendBLAS }

// NewOpenBLASCPUBackend loads the exact library named in config. It never searches
// implicitly and never makes OpenBLAS a requirement for ordinary GoTorch builds.
func NewOpenBLASCPUBackend(config OpenBLASConfig) (Backend, error) {
	if config.LibraryPath == "" {
		return nil, nil
	}
	if err := config.Policy.Validate(); err != nil {
		return nil, err
	}
	api, err := loadOpenBLAS(config.LibraryPath)
	if err != nil {
		return nil, fmt.Errorf("load OpenBLAS %q: %w", config.LibraryPath, err)
	}
	return &openBLASCPUBackend{api: api, policy: config.Policy}, nil
}

func newOpenBLASCPUBackend(api *openBLASAPI, policy ExecutionPolicy) (*openBLASCPUBackend, error) {
	if api == nil || api.sgemm == nil || api.dgemm == nil || api.getThreads == nil || api.setThreads == nil {
		return nil, fmt.Errorf("%w: incomplete OpenBLAS function table", errs.ErrInvalidArgument)
	}
	if err := policy.Validate(); err != nil {
		return nil, err
	}
	return &openBLASCPUBackend{api: api, policy: policy}, nil
}

func (*openBLASCPUBackend) Supports(spec KernelSpec, call Invocation) bool {
	if spec.Operation != OpMatMul || spec.Layout != tensor.LayoutContiguous ||
		(spec.DType != dtype.Float32 && spec.DType != dtype.Float64) ||
		len(call.Inputs) != 2 || len(call.Outputs) != 1 {
		return false
	}
	a, b, out := call.Inputs[0].Metadata, call.Inputs[1].Metadata, call.Outputs[0].Metadata
	if a.DType() != spec.DType || b.DType() != spec.DType || out.DType() != spec.DType ||
		a.Layout() != tensor.LayoutContiguous || b.Layout() != tensor.LayoutContiguous ||
		out.Layout() != tensor.LayoutContiguous || a.Shape().Rank() != 2 ||
		b.Shape().Rank() != 2 || out.Shape().Rank() != 2 {
		return false
	}
	ad, bd, cd := a.Shape().Dims(), b.Shape().Dims(), out.Shape().Dims()
	if ad[1] != bd[0] || cd[0] != ad[0] || cd[1] != bd[1] {
		return false
	}
	for _, dim := range []int64{ad[0], ad[1], bd[1]} {
		if dim > math.MaxInt32 {
			return false
		}
	}
	return true
}

func (b *openBLASCPUBackend) Execute(ctx context.Context, spec KernelSpec, call Invocation) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if !b.Supports(spec, call) {
		return errs.ErrUnsupportedBackend
	}
	b.mu.RLock()
	defer b.mu.RUnlock()
	if b.closed {
		return errs.ErrInvalidState
	}

	ad, bd := call.Inputs[0].Metadata.Shape().Dims(), call.Inputs[1].Metadata.Shape().Dims()
	m, k, n := int32(ad[0]), int32(ad[1]), int32(bd[1])
	if m == 0 || n == 0 {
		return nil
	}
	if k == 0 {
		if spec.DType == dtype.Float32 {
			return writeBLASFloat32(call.Outputs[0], make([]float32, int(m)*int(n)))
		}
		return writeBLASFloat64(call.Outputs[0], make([]float64, int(m)*int(n)))
	}

	if spec.DType == dtype.Float32 {
		a, err := readBLASFloat32(call.Inputs[0], int64(m)*int64(k))
		if err != nil {
			return err
		}
		bb, err := readBLASFloat32(call.Inputs[1], int64(k)*int64(n))
		if err != nil {
			return err
		}
		out := make([]float32, int(m)*int(n))
		if err := b.runBLAS(ctx, func() {
			b.api.sgemm(cblasRowMajor, cblasNoTrans, cblasNoTrans, m, n, k, 1, unsafe.Pointer(unsafe.SliceData(a)), k, unsafe.Pointer(unsafe.SliceData(bb)), n, 0, unsafe.Pointer(unsafe.SliceData(out)), n)
		}); err != nil {
			return err
		}
		runtime.KeepAlive(a)
		runtime.KeepAlive(bb)
		runtime.KeepAlive(out)
		return writeBLASFloat32(call.Outputs[0], out)
	}

	a, err := readBLASFloat64(call.Inputs[0], int64(m)*int64(k))
	if err != nil {
		return err
	}
	bb, err := readBLASFloat64(call.Inputs[1], int64(k)*int64(n))
	if err != nil {
		return err
	}
	out := make([]float64, int(m)*int(n))
	if err := b.runBLAS(ctx, func() {
		b.api.dgemm(cblasRowMajor, cblasNoTrans, cblasNoTrans, m, n, k, 1, unsafe.Pointer(unsafe.SliceData(a)), k, unsafe.Pointer(unsafe.SliceData(bb)), n, 0, unsafe.Pointer(unsafe.SliceData(out)), n)
	}); err != nil {
		return err
	}
	runtime.KeepAlive(a)
	runtime.KeepAlive(bb)
	runtime.KeepAlive(out)
	return writeBLASFloat64(call.Outputs[0], out)
}

// OpenBLAS exposes its thread setter as process-global state. Serialize each
// set/call/restore transaction so concurrent GoTorch callers cannot race settings.
func (b *openBLASCPUBackend) runBLAS(ctx context.Context, call func()) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	openBLASProcessMu.Lock()
	defer openBLASProcessMu.Unlock()
	previous := b.api.getThreads()
	if previous < 1 {
		return fmt.Errorf("%w: OpenBLAS reported invalid thread count %d", errs.ErrInvalidState, previous)
	}
	b.api.setThreads(int32(b.policy.EffectiveBLASThreads()))
	defer b.api.setThreads(previous)
	call()
	return ctx.Err()
}

// Close releases the dynamically loaded library after all in-flight calls finish.
// The caller must stop dispatching to this backend before closing it.
func (b *openBLASCPUBackend) Close() error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed {
		return nil
	}
	openBLASProcessMu.Lock()
	defer openBLASProcessMu.Unlock()
	if b.api.close != nil {
		if err := b.api.close(); err != nil {
			return err
		}
	}
	b.closed = true
	return nil
}

func readBLASBytes(arg Argument, count, width int64) ([]byte, error) {
	if count < 0 || width <= 0 || count > int64(int(^uint(0)>>1))/width {
		return nil, errs.ErrShapeOverflow
	}
	size := int(count * width)
	if arg.Offset < 0 || arg.Offset > arg.Storage.Size() || int64(size) > arg.Storage.Size()-arg.Offset {
		return nil, errs.ErrOutOfBounds
	}
	raw := make([]byte, size)
	if size == 0 {
		return raw, nil
	}
	n, err := arg.Storage.ReadAt(raw, arg.Offset)
	if err != nil {
		return nil, fmt.Errorf("OpenBLAS input read: %w", err)
	}
	if n != size {
		return nil, errs.ErrInvalidState
	}
	return raw, nil
}

func readBLASFloat32(arg Argument, count int64) ([]float32, error) {
	raw, err := readBLASBytes(arg, count, 4)
	if err != nil {
		return nil, err
	}
	values := make([]float32, int(count))
	for i := range values {
		values[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return values, nil
}

func readBLASFloat64(arg Argument, count int64) ([]float64, error) {
	raw, err := readBLASBytes(arg, count, 8)
	if err != nil {
		return nil, err
	}
	values := make([]float64, int(count))
	for i := range values {
		values[i] = math.Float64frombits(binary.LittleEndian.Uint64(raw[i*8:]))
	}
	return values, nil
}

func writeBLASBytes(arg Argument, raw []byte) error {
	if arg.Offset < 0 || arg.Offset > arg.Storage.Size() || int64(len(raw)) > arg.Storage.Size()-arg.Offset {
		return errs.ErrOutOfBounds
	}
	if len(raw) == 0 {
		return nil
	}
	n, err := arg.Storage.WriteAt(raw, arg.Offset)
	if err != nil {
		return fmt.Errorf("OpenBLAS output write: %w", err)
	}
	if n != len(raw) {
		return errs.ErrInvalidState
	}
	return nil
}

func writeBLASFloat32(arg Argument, values []float32) error {
	raw := make([]byte, len(values)*4)
	for i, value := range values {
		binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(value))
	}
	return writeBLASBytes(arg, raw)
}

func writeBLASFloat64(arg Argument, values []float64) error {
	raw := make([]byte, len(values)*8)
	for i, value := range values {
		binary.LittleEndian.PutUint64(raw[i*8:], math.Float64bits(value))
	}
	return writeBLASBytes(arg, raw)
}

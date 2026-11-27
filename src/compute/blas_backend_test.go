//go:build gotorch_blas && (linux || windows)

package compute

import (
	"context"
	"math"
	"runtime"
	"testing"
	"unsafe"

	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/shape"
	"github.com/adhitiad/gotorch/src/tensor"
)

func mockOpenBLAS() *openBLASAPI {
	threads := int32(8)
	api := &openBLASAPI{
		getThreads: func() int32 { return threads },
		setThreads: func(n int32) { threads = n },
	}
	api.sgemm = func(_, _, _ int32, m, n, k int32, alpha float32, ap unsafe.Pointer, lda int32, bp unsafe.Pointer, ldb int32, beta float32, cp unsafe.Pointer, ldc int32) {
		a := unsafe.Slice((*float32)(ap), int(m*k))
		b := unsafe.Slice((*float32)(bp), int(k*n))
		c := unsafe.Slice((*float32)(cp), int(m*n))
		for i := int32(0); i < m; i++ {
			for j := int32(0); j < n; j++ {
				var sum float32
				for q := int32(0); q < k; q++ {
					sum += a[i*lda+q] * b[q*ldb+j]
				}
				c[i*ldc+j] = alpha*sum + beta*c[i*ldc+j]
			}
		}
	}
	api.dgemm = func(_, _, _ int32, m, n, k int32, alpha float64, ap unsafe.Pointer, lda int32, bp unsafe.Pointer, ldb int32, beta float64, cp unsafe.Pointer, ldc int32) {
		a := unsafe.Slice((*float64)(ap), int(m*k))
		b := unsafe.Slice((*float64)(bp), int(k*n))
		c := unsafe.Slice((*float64)(cp), int(m*n))
		for i := int32(0); i < m; i++ {
			for j := int32(0); j < n; j++ {
				var sum float64
				for q := int32(0); q < k; q++ {
					sum += a[i*lda+q] * b[q*ldb+j]
				}
				c[i*ldc+j] = alpha*sum + beta*c[i*ldc+j]
			}
		}
	}
	return api
}

func matrixArg(t *testing.T, dt dtype.DType, values []float64, rows, cols int64, output bool) Argument {
	t.Helper()
	var arg Argument
	if output {
		arg = typedFloatOutput(t, dt, len(values))
	} else {
		arg = typedFloatArg(t, dt, values)
	}
	sh, err := shape.New(rows, cols)
	if err != nil {
		t.Fatal(err)
	}
	arg.Metadata, err = tensor.NewMetadata(dt, sh)
	if err != nil {
		t.Fatal(err)
	}
	return arg
}

func TestOpenBLASMatMulDifferentialAgainstReference(t *testing.T) {
	api := mockOpenBLAS()
	backend, err := newOpenBLASCPUBackend(api, ExecutionPolicy{GoWorkers: 4, BLASThreads: 8})
	if err != nil {
		t.Fatal(err)
	}
	for _, dt := range []dtype.DType{dtype.Float32, dtype.Float64} {
		t.Run(dt.String(), func(t *testing.T) {
			a := matrixArg(t, dt, []float64{1, 2, 3, 4, 5, 6}, 2, 3, false)
			b := matrixArg(t, dt, []float64{7, 8, 9, 10, 11, 12}, 3, 2, false)
			got := matrixArg(t, dt, make([]float64, 4), 2, 2, true)
			want := matrixArg(t, dt, make([]float64, 4), 2, 2, true)
			spec := KernelSpec{Operation: OpMatMul, DType: dt, Layout: tensor.LayoutContiguous}
			if err := Run(context.Background(), backend, spec, Invocation{Inputs: []Argument{a, b}, Outputs: []Argument{got}}); err != nil {
				t.Fatal(err)
			}
			if err := Run(context.Background(), NewReferenceCPUBackend(), spec, Invocation{Inputs: []Argument{a, b}, Outputs: []Argument{want}}); err != nil {
				t.Fatal(err)
			}
			tol, ok := DefaultTolerance(OpMatMul, dt)
			if !ok {
				t.Fatal("missing matmul tolerance")
			}
			for i := int64(0); i < 4; i++ {
				x, err := readFloat(got, i, dt)
				if err != nil {
					t.Fatal(err)
				}
				y, err := readFloat(want, i, dt)
				if err != nil {
					t.Fatal(err)
				}
				if !tol.Close(x, y) {
					t.Fatalf("index %d: OpenBLAS=%g reference=%g tolerance=%+v", i, x, y, tol)
				}
			}
		})
	}
	if api.getThreads() != 8 {
		t.Fatalf("thread count=%d, want restored value 8", api.getThreads())
	}
}

func TestExecutionPolicyPreventsNestedOversubscription(t *testing.T) {
	policy := ExecutionPolicy{GoWorkers: 4, BLASThreads: 16}
	want := runtime.GOMAXPROCS(0) / policy.GoWorkers
	if want < 1 {
		want = 1
	}
	if want > policy.BLASThreads {
		want = policy.BLASThreads
	}
	if got := policy.EffectiveBLASThreads(); got != want {
		t.Fatalf("effective threads=%d, want %d", got, want)
	}
}

func TestOpenBLASRejectsUnsupportedKernelAndCancellation(t *testing.T) {
	backend, err := newOpenBLASCPUBackend(mockOpenBLAS(), ExecutionPolicy{GoWorkers: 1, BLASThreads: 1})
	if err != nil {
		t.Fatal(err)
	}
	a := matrixArg(t, dtype.Float64, []float64{1, 2, 3, 4}, 2, 2, false)
	out := matrixArg(t, dtype.Float64, make([]float64, 4), 2, 2, true)
	call := Invocation{Inputs: []Argument{a, a}, Outputs: []Argument{out}}
	if backend.Supports(KernelSpec{Operation: OpAdd, DType: dtype.Float64, Layout: tensor.LayoutContiguous}, call) {
		t.Fatal("BLAS unexpectedly supports elementwise add")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := Run(ctx, backend, KernelSpec{Operation: OpMatMul, DType: dtype.Float64, Layout: tensor.LayoutContiguous}, call); err == nil {
		t.Fatal("cancelled invocation unexpectedly succeeded")
	}
}

func TestOpenBLASZeroInnerDimensionWritesZeros(t *testing.T) {
	backend, err := newOpenBLASCPUBackend(mockOpenBLAS(), ExecutionPolicy{GoWorkers: 1, BLASThreads: 1})
	if err != nil {
		t.Fatal(err)
	}
	a := matrixArg(t, dtype.Float64, []float64{}, 2, 0, false)
	b := matrixArg(t, dtype.Float64, []float64{}, 0, 3, false)
	out := matrixArg(t, dtype.Float64, make([]float64, 6), 2, 3, true)
	spec := KernelSpec{Operation: OpMatMul, DType: dtype.Float64, Layout: tensor.LayoutContiguous}
	if err := Run(context.Background(), backend, spec, Invocation{Inputs: []Argument{a, b}, Outputs: []Argument{out}}); err != nil {
		t.Fatal(err)
	}
	for i := int64(0); i < 6; i++ {
		got, err := readFloat(out, i, dtype.Float64)
		if err != nil {
			t.Fatal(err)
		}
		if math.Float64bits(got) != 0 {
			t.Fatalf("output[%d]=%g, want zero", i, got)
		}
	}
}

package compute

import (
	"context"
	"fmt"
	"runtime"
	"testing"

	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/shape"
	"github.com/adhitiad/gotorch/src/storage"
	"github.com/adhitiad/gotorch/src/tensor"
)

func benchmarkMeta(b *testing.B, dt dtype.DType, dims, strides []int64) tensor.Metadata {
	b.Helper()
	sh, err := shape.New(dims...)
	if err != nil {
		b.Fatal(err)
	}
	var meta tensor.Metadata
	if strides == nil {
		meta, err = tensor.NewMetadata(dt, sh)
	} else {
		meta, err = tensor.NewStridedMetadata(dt, sh, strides)
	}
	if err != nil {
		b.Fatal(err)
	}
	return meta
}

func benchmarkArgument(b *testing.B, meta tensor.Metadata, seed float64) Argument {
	b.Helper()
	size, err := meta.RequiredBytes()
	if err != nil {
		b.Fatal(err)
	}
	backing, err := storage.NewHeapAllocator().Allocate(context.Background(), size)
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { _ = backing.Release() })
	arg := Argument{Metadata: meta, Storage: backing}
	count, err := meta.Numel()
	if err != nil {
		b.Fatal(err)
	}
	for i := int64(0); i < count; i++ {
		value := seed + float64((i%31)-15)/32
		if err := writeFloat(arg, i, meta.DType(), value); err != nil {
			b.Fatal(err)
		}
	}
	return arg
}

func benchmarkOutput(b *testing.B, dt dtype.DType, dims ...int64) Argument {
	b.Helper()
	return benchmarkArgument(b, benchmarkMeta(b, dt, dims, nil), 0)
}

func benchmarkEnvironment(b *testing.B) {
	b.Helper()
	features := DetectCPUFeatures()
	b.Logf("go=%s os=%s arch=%s gomaxprocs=%d cpu_features={initialized:%t avx2:%t asimd:%t aes:%t}",
		runtime.Version(), runtime.GOOS, runtime.GOARCH, runtime.GOMAXPROCS(0),
		features.Initialized, features.AVX2, features.ASIMD, features.AES)
}

func benchmarkCheckDifferential(b *testing.B, backend Backend, spec KernelSpec, inputs []Argument, output Argument) {
	b.Helper()
	reference := benchmarkOutput(b, spec.DType, output.Metadata.Shape().Dims()...)
	if err := Run(context.Background(), backend, spec, Invocation{Inputs: inputs, Outputs: []Argument{output}}); err != nil {
		b.Fatal(err)
	}
	if err := Run(context.Background(), NewReferenceCPUBackend(), spec, Invocation{Inputs: inputs, Outputs: []Argument{reference}}); err != nil {
		b.Fatal(err)
	}
	tolerance, ok := DefaultTolerance(spec.Operation, spec.DType)
	if !ok {
		b.Fatalf("missing tolerance for %s/%s", spec.Operation, spec.DType)
	}
	count, _ := output.Metadata.Numel()
	for i := int64(0); i < count; i++ {
		got, err := readFloat(output, i, spec.DType)
		if err != nil {
			b.Fatal(err)
		}
		want, err := readFloat(reference, i, spec.DType)
		if err != nil {
			b.Fatal(err)
		}
		if !tolerance.Close(got, want) {
			b.Fatalf("differential mismatch at %d: got=%g want=%g tolerance=%+v", i, got, want, tolerance)
		}
	}
}

func BenchmarkCPUElementwise(b *testing.B) {
	benchmarkEnvironment(b)
	backends := []Backend{NewReferenceCPUBackend()}
	if simd := NewSIMDCPUBackend(); simd != nil {
		backends = append(backends, simd)
	}
	for _, backend := range backends {
		for _, operation := range []string{OpAdd, OpMul} {
			for _, dt := range []dtype.DType{dtype.Float32, dtype.Float64} {
				for _, layout := range []string{"contiguous", "strided-input"} {
					for _, size := range []int64{256, 4096, 65536} {
						name := fmt.Sprintf("%s/%s/%s/%s/n=%d", backend.Name(), operation, dt, layout, size)
						b.Run(name, func(b *testing.B) {
							var inputMeta tensor.Metadata
							if layout == "contiguous" {
								inputMeta = benchmarkMeta(b, dt, []int64{size}, nil)
							} else {
								inputMeta = benchmarkMeta(b, dt, []int64{size}, []int64{2})
							}
							a := benchmarkArgument(b, inputMeta, 1)
							bb := benchmarkArgument(b, inputMeta, 2)
							out := benchmarkOutput(b, dt, size)
							spec := KernelSpec{Operation: operation, DType: dt, Layout: tensor.LayoutContiguous}
							call := Invocation{Inputs: []Argument{a, bb}, Outputs: []Argument{out}}
							if !backend.Supports(spec, call) {
								b.Skip("backend does not support this dtype/layout")
							}
							benchmarkCheckDifferential(b, backend, spec, call.Inputs, out)
							width, _ := dt.Size()
							b.SetBytes(size * width * 3)
							b.ReportAllocs()
							b.ResetTimer()
							for i := 0; i < b.N; i++ {
								if err := Run(context.Background(), backend, spec, call); err != nil {
									b.Fatal(err)
								}
							}
							b.ReportMetric(float64(size)*float64(b.N)/b.Elapsed().Seconds(), "elem/s")
						})
					}
				}
			}
		}
	}
}

func BenchmarkCPUMatMulReference(b *testing.B) {
	benchmarkEnvironment(b)
	backend := NewReferenceCPUBackend()
	for _, dt := range []dtype.DType{dtype.Float32, dtype.Float64} {
		for _, layout := range []string{"contiguous", "strided-input"} {
			for _, n := range []int64{8, 32, 64} {
				b.Run(fmt.Sprintf("%s/%s/n=%d", dt, layout, n), func(b *testing.B) {
					var aMeta, bMeta tensor.Metadata
					if layout == "contiguous" {
						aMeta = benchmarkMeta(b, dt, []int64{n, n}, nil)
						bMeta = benchmarkMeta(b, dt, []int64{n, n}, nil)
					} else {
						aMeta = benchmarkMeta(b, dt, []int64{n, n}, []int64{n + 1, 1})
						bMeta = benchmarkMeta(b, dt, []int64{n, n}, []int64{n + 1, 1})
					}
					a := benchmarkArgument(b, aMeta, 1)
					bb := benchmarkArgument(b, bMeta, 2)
					out := benchmarkOutput(b, dt, n, n)
					spec := KernelSpec{Operation: OpMatMul, DType: dt, Layout: tensor.LayoutContiguous}
					call := Invocation{Inputs: []Argument{a, bb}, Outputs: []Argument{out}}
					benchmarkCheckDifferential(b, backend, spec, call.Inputs, out)
					width, _ := dt.Size()
					b.SetBytes(2*n*n*width + n*n*width)
					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						if err := Run(context.Background(), backend, spec, call); err != nil {
							b.Fatal(err)
						}
					}
					b.ReportMetric(2*float64(n*n*n)*float64(b.N)/b.Elapsed().Seconds(), "FLOP/s")
				})
			}
		}
	}
}

// BenchmarkCPUOpenBLASMatMul is active only when built with gotorch_blas and
// GOTORCH_OPENBLAS_LIBRARY points to a loadable OpenBLAS shared library.
func BenchmarkCPUOpenBLASMatMul(b *testing.B) {
	benchmarkEnvironment(b)
	config := DefaultOpenBLASConfig()
	if config.LibraryPath == "" {
		b.Skip("set GOTORCH_OPENBLAS_LIBRARY to benchmark a real OpenBLAS library")
	}
	for _, dt := range []dtype.DType{dtype.Float32, dtype.Float64} {
		for _, n := range []int64{32, 128, 256} {
			for _, workers := range []int{1, 2, 4} {
				for _, threads := range []int{1, 2, 4} {
					name := fmt.Sprintf("%s/n=%d/go-workers=%d/blas-threads=%d", dt, n, workers, threads)
					b.Run(name, func(b *testing.B) {
						config.Policy = ExecutionPolicy{GoWorkers: workers, BLASThreads: threads}
						backend, err := NewOpenBLASCPUBackend(config)
						if err != nil {
							b.Fatalf("load OpenBLAS: %v", err)
						}
						if closer, ok := backend.(interface{ Close() error }); ok {
							b.Cleanup(func() { _ = closer.Close() })
						}
						spec := KernelSpec{Operation: OpMatMul, DType: dt, Layout: tensor.LayoutContiguous}
						verifyA := benchmarkArgument(b, benchmarkMeta(b, dt, []int64{8, 8}, nil), 1)
						verifyB := benchmarkArgument(b, benchmarkMeta(b, dt, []int64{8, 8}, nil), 2)
						verifyOut := benchmarkOutput(b, dt, 8, 8)
						verifyCall := Invocation{Inputs: []Argument{verifyA, verifyB}, Outputs: []Argument{verifyOut}}
						if !backend.Supports(spec, verifyCall) {
							b.Skip("OpenBLAS backend does not support this matrix")
						}
						benchmarkCheckDifferential(b, backend, spec, verifyCall.Inputs, verifyOut)
						a := benchmarkArgument(b, benchmarkMeta(b, dt, []int64{n, n}, nil), 1)
						bb := benchmarkArgument(b, benchmarkMeta(b, dt, []int64{n, n}, nil), 2)
						out := benchmarkOutput(b, dt, n, n)
						call := Invocation{Inputs: []Argument{a, bb}, Outputs: []Argument{out}}
						width, _ := dt.Size()
						b.SetBytes(2*n*n*width + n*n*width)
						effectiveThreads := (ExecutionPolicy{GoWorkers: workers, BLASThreads: threads}).EffectiveBLASThreads()
						b.ReportAllocs()
						b.ResetTimer()
						for i := 0; i < b.N; i++ {
							if err := Run(context.Background(), backend, spec, call); err != nil {
								b.Fatal(err)
							}
						}
						b.ReportMetric(float64(effectiveThreads), "effective-blas-threads")
						b.ReportMetric(2*float64(n*n*n)*float64(b.N)/b.Elapsed().Seconds(), "FLOP/s")
					})
				}
			}
		}
	}
}

package compute

import (
	"context"
	"math"
	"math/rand"
	"testing"

	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/shape"
	"github.com/adhitiad/gotorch/src/storage"
	"github.com/adhitiad/gotorch/src/tensor"
)

func typedFloatArg(t *testing.T, dt dtype.DType, values []float64) Argument {
	t.Helper()
	sh, err := shape.New(int64(len(values)))
	if err != nil {
		t.Fatal(err)
	}
	meta, err := tensor.NewMetadata(dt, sh)
	if err != nil {
		t.Fatal(err)
	}
	size, err := meta.RequiredBytes()
	if err != nil {
		t.Fatal(err)
	}
	backing, err := storage.NewHeapAllocator().Allocate(context.Background(), size)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = backing.Release() })
	arg := Argument{Metadata: meta, Storage: backing}
	for i, value := range values {
		if err := writeFloat(arg, int64(i), dt, value); err != nil {
			t.Fatal(err)
		}
	}
	return arg
}

func typedFloatOutput(t *testing.T, dt dtype.DType, count int) Argument {
	t.Helper()
	sh, err := shape.New(int64(count))
	if err != nil {
		t.Fatal(err)
	}
	meta, err := tensor.NewMetadata(dt, sh)
	if err != nil {
		t.Fatal(err)
	}
	size, err := meta.RequiredBytes()
	if err != nil {
		t.Fatal(err)
	}
	backing, err := storage.NewHeapAllocator().Allocate(context.Background(), size)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = backing.Release() })
	return Argument{Metadata: meta, Storage: backing}
}

func TestSIMDBackendDifferentialAgainstReference(t *testing.T) {
	optimized := NewSIMDCPUBackend()
	if optimized == nil {
		t.Skip("SIMD experiment disabled or CPU feature unavailable")
	}
	if optimized.Kind() != BackendSIMD {
		t.Fatalf("kind=%v, want SIMD", optimized.Kind())
	}

	rng := rand.New(rand.NewSource(0x503006))
	for _, dt := range []dtype.DType{dtype.Float32, dtype.Float64} {
		for _, operation := range []string{OpAdd, OpSub, OpMul} {
			t.Run(dt.String()+"/"+operation, func(t *testing.T) {
				const count = 1031 // exercises vector body and scalar tail
				aValues, bValues := make([]float64, count), make([]float64, count)
				for i := range aValues {
					aValues[i] = (rng.Float64()*2 - 1) * math.Pow10(rng.Intn(12)-6)
					bValues[i] = (rng.Float64()*2 - 1) * math.Pow10(rng.Intn(12)-6)
				}
				a, b := typedFloatArg(t, dt, aValues), typedFloatArg(t, dt, bValues)
				simdOut, refOut := typedFloatOutput(t, dt, count), typedFloatOutput(t, dt, count)
				spec := KernelSpec{Operation: operation, DType: dt, Layout: tensor.LayoutContiguous}
				callSIMD := Invocation{Inputs: []Argument{a, b}, Outputs: []Argument{simdOut}}
				callRef := Invocation{Inputs: []Argument{a, b}, Outputs: []Argument{refOut}}
				if err := Run(context.Background(), optimized, spec, callSIMD); err != nil {
					t.Fatal(err)
				}
				if err := Run(context.Background(), NewReferenceCPUBackend(), spec, callRef); err != nil {
					t.Fatal(err)
				}
				tolerance, ok := DefaultTolerance(operation, dt)
				if !ok {
					t.Fatal("missing differential tolerance")
				}
				for i := 0; i < count; i++ {
					got, err := readFloat(simdOut, int64(i), dt)
					if err != nil {
						t.Fatal(err)
					}
					want, err := readFloat(refOut, int64(i), dt)
					if err != nil {
						t.Fatal(err)
					}
					if !tolerance.Close(got, want) {
						t.Fatalf("index %d: SIMD=%g reference=%g tolerance=%+v", i, got, want, tolerance)
					}
				}
			})
		}
	}
}

func TestDefaultCPURegistryAlwaysHasScalarFallback(t *testing.T) {
	registry, err := NewDefaultCPURegistry()
	if err != nil {
		t.Fatal(err)
	}
	names := registry.Backends()
	if len(names) == 0 || names[len(names)-1] != "purego-reference" {
		t.Fatalf("backend order=%v, scalar fallback must be last", names)
	}
}

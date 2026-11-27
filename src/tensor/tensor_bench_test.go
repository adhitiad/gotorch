package tensor

import (
	"testing"

	"github.com/adhitiad/gotorch/src/dtype"
)

func BenchmarkTensorAllocation(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		value, err := NewTensor(dtype.Float32, 128, 128)
		if err != nil {
			b.Fatal(err)
		}
		if err := value.Close(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkTensorViewCreation(b *testing.B) {
	source, err := NewTensor(dtype.Float32, 128, 128)
	if err != nil {
		b.Fatal(err)
	}
	defer source.Close()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		view, err := source.Transpose()
		if err != nil {
			b.Fatal(err)
		}
		if err := view.Close(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkTensorIndexing(b *testing.B) {
	source, err := From(make([]float32, 128*128), 128, 128)
	if err != nil {
		b.Fatal(err)
	}
	defer source.Close()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := At[float32](source, int64(i%128), int64((i/128)%128)); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkTensorCopy(b *testing.B) {
	source, err := From(make([]float32, 128*128), 128, 128)
	if err != nil {
		b.Fatal(err)
	}
	defer source.Close()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copied, err := source.Copy()
		if err != nil {
			b.Fatal(err)
		}
		if err := copied.Close(); err != nil {
			b.Fatal(err)
		}
	}
}

package tensor

import (
	"reflect"
	"testing"
)

func TestCopyAndContiguousProduceIndependentDenseTensors(t *testing.T) {
	source, err := From([]float32{0, 1, 2, 3, 4, 5}, 2, 3)
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	transposed, err := source.Transpose()
	if err != nil {
		t.Fatal(err)
	}
	defer transposed.Close()

	copied, err := transposed.Copy()
	if err != nil {
		t.Fatal(err)
	}
	defer copied.Close()
	contiguous, err := transposed.Contiguous()
	if err != nil {
		t.Fatal(err)
	}
	defer contiguous.Close()

	for label, value := range map[string]*Tensor{"Copy": copied, "Contiguous": contiguous} {
		metadata, err := value.Metadata()
		if err != nil {
			t.Fatal(err)
		}
		if !metadata.IsContiguous() {
			t.Errorf("%s result is not contiguous", label)
		}
		got, err := Read[float32](value)
		if err != nil || !reflect.DeepEqual(got, []float32{0, 3, 1, 4, 2, 5}) {
			t.Errorf("%s values = %v, %v", label, got, err)
		}
	}
	if err := Set(copied, float32(77), 0, 0); err != nil {
		t.Fatal(err)
	}
	if got, err := At[float32](source, 0, 0); err != nil || got != 0 {
		t.Fatalf("copy mutation affected source: %v, %v", got, err)
	}
	if got, err := At[float32](contiguous, 0, 0); err != nil || got != 0 {
		t.Fatalf("copy mutation affected another copy: %v, %v", got, err)
	}
}

func TestBulkWriteScattersThroughNonOverlappingTranspose(t *testing.T) {
	base, err := From([]int32{0, 0, 0, 0, 0, 0}, 2, 3)
	if err != nil {
		t.Fatal(err)
	}
	defer base.Close()
	transposed, err := base.Transpose()
	if err != nil {
		t.Fatal(err)
	}
	defer transposed.Close()

	if err := Write(transposed, []int32{1, 2, 3, 4, 5, 6}); err != nil {
		t.Fatal(err)
	}
	got, err := Read[int32](base)
	if err != nil || !reflect.DeepEqual(got, []int32{1, 3, 5, 2, 4, 6}) {
		t.Fatalf("scattered base values = %v, %v", got, err)
	}
}

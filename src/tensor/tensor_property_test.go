package tensor

import (
	"math/rand"
	"reflect"
	"testing"
)

func TestPropertyTensorReshapeAndTransposeRoundTrips(t *testing.T) {
	random := rand.New(rand.NewSource(0x70_31))
	for iteration := 0; iteration < 75; iteration++ {
		rank := random.Intn(3) + 1
		dimensions := make([]int64, rank)
		numel := int64(1)
		for axis := range dimensions {
			dimensions[axis] = int64(random.Intn(4) + 1)
			numel *= dimensions[axis]
		}
		values := make([]int32, int(numel))
		for i := range values {
			values[i] = int32(iteration*100 + i)
		}
		source, err := From(values, dimensions...)
		if err != nil {
			t.Fatal(err)
		}

		flattened, err := source.Reshape(numel)
		if err != nil {
			_ = source.Close()
			t.Fatal(err)
		}
		restored, err := flattened.Reshape(dimensions...)
		if err != nil {
			_ = flattened.Close()
			_ = source.Close()
			t.Fatal(err)
		}
		roundTrip, err := Read[int32](restored)
		if err != nil || !reflect.DeepEqual(roundTrip, values) {
			t.Fatalf("iteration %d reshape round-trip = %v, %v; want %v", iteration, roundTrip, err, values)
		}

		transposed, err := source.Transpose()
		if err != nil {
			t.Fatal(err)
		}
		restoredTranspose, err := transposed.Transpose()
		if err != nil {
			t.Fatal(err)
		}
		transposedRoundTrip, err := Read[int32](restoredTranspose)
		if err != nil || !reflect.DeepEqual(transposedRoundTrip, values) {
			t.Fatalf("iteration %d transpose round-trip = %v, %v; want %v", iteration, transposedRoundTrip, err, values)
		}
		_ = restoredTranspose.Close()
		_ = transposed.Close()
		_ = restored.Close()
		_ = flattened.Close()
		_ = source.Close()
	}
}

package tensor

import (
	"errors"
	"reflect"
	"testing"

	"github.com/adhitiad/gotorch/src/errs"
)

func TestScalarViewsAndBroadcast(t *testing.T) {
	scalar, err := From([]int32{7})
	if err != nil {
		t.Fatal(err)
	}
	defer scalar.Close()

	transposed, err := scalar.Transpose()
	if err != nil {
		t.Fatal(err)
	}
	defer transposed.Close()
	if got := mustShape(t, transposed); len(got) != 0 {
		t.Fatalf("scalar transpose shape = %v, want rank 0", got)
	}
	if got, err := At[int32](transposed); err != nil || got != 7 {
		t.Fatalf("scalar transpose value = %d, %v; want 7", got, err)
	}

	reshaped, err := scalar.Reshape()
	if err != nil {
		t.Fatal(err)
	}
	defer reshaped.Close()
	if got := mustShape(t, reshaped); len(got) != 0 {
		t.Fatalf("scalar reshape shape = %v, want rank 0", got)
	}

	broadcast, err := scalar.BroadcastTo(2, 3)
	if err != nil {
		t.Fatal(err)
	}
	defer broadcast.Close()
	if got, err := Read[int32](broadcast); err != nil || !reflect.DeepEqual(got, []int32{7, 7, 7, 7, 7, 7}) {
		t.Fatalf("scalar broadcast = %v, %v", got, err)
	}
	if err := Write(broadcast, make([]int32, 6)); !errors.Is(err, errs.ErrOverlappingWrite) {
		t.Fatalf("scalar broadcast bulk write error = %v", err)
	}
	if err := Set(broadcast, int32(9), 1, 2); err != nil {
		t.Fatal(err)
	}
	if got, err := At[int32](scalar); err != nil || got != 9 {
		t.Fatalf("scalar after broadcast Set = %d, %v; want 9", got, err)
	}
}

func TestEmptyTensorViewsAndCopy(t *testing.T) {
	empty, err := From([]float32{}, 0, 3)
	if err != nil {
		t.Fatal(err)
	}
	defer empty.Close()

	sliced, err := empty.Slice(1, 1, 3, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer sliced.Close()
	if got, err := Read[float32](sliced); err != nil || len(got) != 0 {
		t.Fatalf("empty slice read = %v, %v", got, err)
	}

	transposed, err := empty.Transpose()
	if err != nil {
		t.Fatal(err)
	}
	defer transposed.Close()
	if got := mustShape(t, transposed); !equal(got, []int64{3, 0}) {
		t.Fatalf("empty transpose shape = %v, want [3 0]", got)
	}
	copied, err := transposed.Copy()
	if err != nil {
		t.Fatal(err)
	}
	defer copied.Close()
	if got, err := Read[float32](copied); err != nil || len(got) != 0 {
		t.Fatalf("empty copy read = %v, %v", got, err)
	}

	reshaped, err := empty.Reshape(0)
	if err != nil {
		t.Fatal(err)
	}
	defer reshaped.Close()
	if got, err := Read[float32](reshaped); err != nil || len(got) != 0 {
		t.Fatalf("empty reshape read = %v, %v", got, err)
	}

	if _, err := empty.Reshape(6); !errors.Is(err, errs.ErrInvalidShape) {
		t.Fatalf("empty reshape element-count error = %v", err)
	}
	if _, err := empty.Slice(0, 1, 1, 1); !errors.Is(err, errs.ErrOutOfBounds) {
		t.Fatalf("empty slice beyond zero axis error = %v", err)
	}
}

package tensor

import (
	"errors"
	"reflect"
	"testing"

	"github.com/adhitiad/gotorch/src/errs"
)

func TestTensorTransposeSliceNarrowAndSelectViews(t *testing.T) {
	source, err := From([]int32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, 3, 4)
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()

	transposed, err := source.Transpose()
	if err != nil {
		t.Fatal(err)
	}
	defer transposed.Close()
	if got, want := mustShape(t, transposed), []int64{4, 3}; !equal(got, want) {
		t.Fatalf("transpose shape = %v, want %v", got, want)
	}
	if got, err := Read[int32](transposed); err != nil || !reflect.DeepEqual(got, []int32{0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11}) {
		t.Fatalf("transpose values = %v, %v", got, err)
	}

	sliced, err := source.Slice(1, 1, 4, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer sliced.Close()
	if got, err := Read[int32](sliced); err != nil || !reflect.DeepEqual(got, []int32{1, 3, 5, 7, 9, 11}) {
		t.Fatalf("slice values = %v, %v", got, err)
	}

	narrowed, err := source.Narrow(0, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer narrowed.Close()
	if got, err := Read[int32](narrowed); err != nil || !reflect.DeepEqual(got, []int32{4, 5, 6, 7}) {
		t.Fatalf("narrow values = %v, %v", got, err)
	}

	selected, err := source.Select(0, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer selected.Close()
	if got, err := Read[int32](selected); err != nil || !reflect.DeepEqual(got, []int32{8, 9, 10, 11}) {
		t.Fatalf("select values = %v, %v", got, err)
	}
}

func TestTensorViewsAliasAndRetainStorageLifetime(t *testing.T) {
	source, err := From([]float32{1, 2, 3, 4}, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	transposed, err := source.Transpose()
	if err != nil {
		t.Fatal(err)
	}
	if err := Set(transposed, float32(99), 1, 0); err != nil {
		t.Fatal(err)
	}
	if got, err := At[float32](source, 0, 1); err != nil || got != 99 {
		t.Fatalf("source alias value = %v, %v; want 99", got, err)
	}
	if err := source.Close(); err != nil {
		t.Fatal(err)
	}
	if got, err := At[float32](transposed, 1, 0); err != nil || got != 99 {
		t.Fatalf("view after source Close = %v, %v; want 99", got, err)
	}
	if err := transposed.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestTensorReshapeViewAndNonContiguousRejection(t *testing.T) {
	source, err := From([]float32{0, 1, 2, 3, 4, 5}, 2, 3)
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()

	reshaped, err := source.Reshape(3, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer reshaped.Close()
	if err := Set(reshaped, float32(42), 2, 1); err != nil {
		t.Fatal(err)
	}
	if got, err := At[float32](source, 1, 2); err != nil || got != 42 {
		t.Fatalf("reshape view alias = %v, %v; want 42", got, err)
	}

	transposed, err := source.Transpose()
	if err != nil {
		t.Fatal(err)
	}
	defer transposed.Close()
	if _, err := transposed.Reshape(6); !errors.Is(err, errs.ErrUnsupportedOperation) {
		t.Fatalf("non-contiguous reshape error = %v", err)
	}
}

func TestTensorBroadcastViewReadsAndGuardsBulkWrite(t *testing.T) {
	source, err := From([]float32{7, 8}, 1, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()

	broadcast, err := source.BroadcastTo(3, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer broadcast.Close()

	want := []float32{7, 8, 7, 8, 7, 8}
	if got, err := Read[float32](broadcast); err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("broadcast Read = %v, %v; want %v", got, err, want)
	}
	if err := Write(broadcast, []float32{1, 2, 3, 4, 5, 6}); !errors.Is(err, errs.ErrOverlappingWrite) {
		t.Fatalf("broadcast bulk Write error = %v", err)
	}
	if err := Set(broadcast, float32(99), 2, 0); err != nil {
		t.Fatal(err)
	}
	if got, err := At[float32](source, 0, 0); err != nil || got != 99 {
		t.Fatalf("broadcast Set alias = %v, %v; want 99", got, err)
	}
	if got, err := At[float32](broadcast, 0, 0); err != nil || got != 99 {
		t.Fatalf("broadcast aliased read = %v, %v; want 99", got, err)
	}
}

func TestTensorByteOffsetUsesPhysicalStrides(t *testing.T) {
	value, err := From([]float32{0, 1, 2, 3, 4, 5}, 2, 3)
	if err != nil {
		t.Fatal(err)
	}
	defer value.Close()
	view, err := value.Transpose()
	if err != nil {
		t.Fatal(err)
	}
	defer view.Close()
	if got, err := view.ByteOffset(2, 1); err != nil || got != 20 {
		t.Fatalf("transposed ByteOffset = %d, %v; want 20", got, err)
	}
}

func TestTensorViewRejectsInvalidRanges(t *testing.T) {
	source, err := From([]int32{1, 2, 3, 4}, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	if _, err := source.Slice(1, 0, 3, 1); !errors.Is(err, errs.ErrOutOfBounds) {
		t.Fatalf("invalid Slice error = %v", err)
	}
	if _, err := source.Narrow(0, 2, 1); !errors.Is(err, errs.ErrOutOfBounds) {
		t.Fatalf("invalid Narrow error = %v", err)
	}
	if _, err := source.Select(0, -1); !errors.Is(err, errs.ErrOutOfBounds) {
		t.Fatalf("invalid Select error = %v", err)
	}
	if _, err := source.Transpose(0, 0); err == nil {
		t.Fatal("duplicate transpose axes accepted")
	}
}

func mustShape(t *testing.T, value *Tensor) []int64 {
	t.Helper()
	metadata, err := value.Metadata()
	if err != nil {
		t.Fatal(err)
	}
	return metadata.Shape().Dims()
}

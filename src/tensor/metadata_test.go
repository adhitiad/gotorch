package tensor

import (
	"math"
	"testing"

	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/shape"
)

func TestContiguousMetadata(t *testing.T) {
	s, _ := shape.New(2, 3, 4)
	m, err := NewMetadata(dtype.Float32, s)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := m.Strides(), []int64{12, 4, 1}; !equal(got, want) {
		t.Fatalf("strides = %v, want %v", got, want)
	}
	if m.Layout() != LayoutContiguous {
		t.Fatalf("layout = %v", m.Layout())
	}
	if got, _ := m.RequiredBytes(); got != 96 {
		t.Fatalf("required bytes = %d, want 96", got)
	}
}

func TestStridedMetadataSpan(t *testing.T) {
	s, _ := shape.New(2, 2)
	m, err := NewStridedMetadata(dtype.Float32, s, []int64{4, 1})
	if err != nil {
		t.Fatal(err)
	}
	if m.Layout() != LayoutStrided {
		t.Fatalf("layout = %v", m.Layout())
	}
	if got, _ := m.RequiredBytes(); got != 24 {
		t.Fatalf("required bytes = %d, want 24", got)
	}
}

func TestScalarAndEmptyMetadata(t *testing.T) {
	scalar, err := NewMetadata(dtype.Float64, shape.Scalar())
	if err != nil {
		t.Fatal(err)
	}
	if got, _ := scalar.RequiredBytes(); got != 8 {
		t.Fatalf("scalar span = %d", got)
	}
	emptyShape, _ := shape.New(2, 0, 3)
	empty, err := NewMetadata(dtype.Float32, emptyShape)
	if err != nil {
		t.Fatal(err)
	}
	if got, _ := empty.RequiredBytes(); got != 0 {
		t.Fatalf("empty span = %d", got)
	}
}

func TestMetadataRejectsInvalidAndOverflow(t *testing.T) {
	s, _ := shape.New(2, 2)
	if _, err := NewStridedMetadata(dtype.Float32, s, []int64{1}); err == nil {
		t.Fatal("stride rank mismatch accepted")
	}
	if _, err := NewStridedMetadata(dtype.Float32, s, []int64{-1, 1}); err == nil {
		t.Fatal("negative stride accepted")
	}
	_, err := NewStridedMetadata(dtype.Float32, s, []int64{math.MaxInt64, 1})
	if err == nil || err != errs.ErrShapeOverflow {
		t.Fatalf("got %v, want ErrShapeOverflow", err)
	}
}

func equal(a, b []int64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

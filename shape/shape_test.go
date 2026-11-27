package shape

import (
	"math"
	"testing"

	"github.com/adhitiad/gotorch/errs"
)

func TestShapeNumel(t *testing.T) {
	a, err := New(2, 3, 4)
	if err != nil {
		t.Fatal(err)
	}
	b, err := New(2, 0, 4)
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		shape Shape
		want  int64
	}{{a, 24}, {b, 0}, {Scalar(), 1}}
	for _, tc := range cases {
		got, err := tc.shape.Numel()
		if err != nil || got != tc.want {
			t.Fatalf("Numel() = %d, %v; want %d", got, err, tc.want)
		}
	}
}

func TestShapeValidationAndCopy(t *testing.T) {
	if _, err := New(2, -1); err == nil {
		t.Fatal("negative dimension accepted")
	}
	s, err := New(2, 3)
	if err != nil {
		t.Fatal(err)
	}
	dims := s.Dims()
	dims[0] = 99
	if got, _ := s.Dimension(0); got != 2 {
		t.Fatal("Dims exposed internal storage")
	}
	if _, err := s.Dimension(-1); err == nil {
		t.Fatal("negative axis accepted")
	}
}

func TestShapeOverflow(t *testing.T) {
	_, err := New(math.MaxInt64, 2)
	if err == nil || err != errs.ErrShapeOverflow {
		t.Fatalf("got %v, want ErrShapeOverflow", err)
	}
}

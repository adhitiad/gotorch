package compute

import (
	"math"
	"testing"

	"github.com/adhitiad/gotorch/src/dtype"
)

func TestDefaultToleranceByOperationAndDType(t *testing.T) {
	for _, op := range []string{OpAdd, OpSub, OpMul, OpDiv, OpNeg, OpExp, OpLog, OpReLU, OpSum, OpMean, OpMatMul} {
		for _, dt := range []dtype.DType{dtype.Float32, dtype.Float64} {
			got, ok := DefaultTolerance(op, dt)
			if !ok || !got.Valid() { t.Fatalf("missing/invalid tolerance for %s %v: %+v", op, dt, got) }
		}
	}
	if _, ok := DefaultTolerance("unknown", dtype.Float64); ok { t.Fatal("unknown operation accepted") }
	if _, ok := DefaultTolerance(OpAdd, dtype.Int32); ok { t.Fatal("integer dtype accepted") }
}

func TestToleranceCloseSpecialValues(t *testing.T) {
	tol := Tolerance{Abs: 1e-6, Rel: 1e-5}
	cases := []struct{ got, want float64; ok bool }{
		{1, 1+1e-7, true}, {1, 1.1, false},
		{math.NaN(), math.NaN(), true}, {math.NaN(), 1, false},
		{math.Inf(1), math.Inf(1), true}, {math.Inf(1), math.Inf(-1), false},
		{math.Inf(1), 1, false}, {0, 0, true},
	}
	for _, tc := range cases { if got := tol.Close(tc.got, tc.want); got != tc.ok { t.Errorf("Close(%v,%v)=%v want %v", tc.got, tc.want, got, tc.ok) } }
	if (Tolerance{Abs: math.NaN()}).Valid() || (Tolerance{Rel: math.Inf(1)}).Valid() { t.Fatal("non-finite tolerance accepted") }
}

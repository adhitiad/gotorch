package compute

import (
	"math"

	"github.com/adhitiad/gotorch/src/dtype"
)

// Tolerance defines the acceptance budget |got-want| <= Abs + Rel*max(|got|, |want|).
// Both fields must be finite and non-negative.
type Tolerance struct {
	Abs float64
	Rel float64
}

// DefaultTolerance returns the documented comparison budget for an operation and dtype.
func DefaultTolerance(operation string, dt dtype.DType) (Tolerance, bool) {
	if dt != dtype.Float32 && dt != dtype.Float64 {
		return Tolerance{}, false
	}
	switch operation {
	case OpAdd, OpSub, OpMul, OpDiv, OpNeg, OpReLU:
		if dt == dtype.Float32 {
			return Tolerance{Abs: 1e-6, Rel: 1e-5}, true
		}
		return Tolerance{Abs: 1e-12, Rel: 1e-10}, true
	case OpExp, OpLog:
		if dt == dtype.Float32 {
			return Tolerance{Abs: 2e-6, Rel: 2e-5}, true
		}
		return Tolerance{Abs: 2e-12, Rel: 2e-10}, true
	case OpSum, OpMean:
		if dt == dtype.Float32 {
			return Tolerance{Abs: 2e-5, Rel: 2e-5}, true
		}
		return Tolerance{Abs: 2e-11, Rel: 2e-10}, true
	case OpMatMul:
		if dt == dtype.Float32 {
			return Tolerance{Abs: 5e-5, Rel: 5e-5}, true
		}
		return Tolerance{Abs: 1e-10, Rel: 1e-9}, true
	default:
		return Tolerance{}, false
	}
}

// Valid reports whether both tolerance terms are finite and non-negative.
func (t Tolerance) Valid() bool {
	return !math.IsNaN(t.Abs) && !math.IsInf(t.Abs, 0) && t.Abs >= 0 &&
		!math.IsNaN(t.Rel) && !math.IsInf(t.Rel, 0) && t.Rel >= 0
}

// Close reports whether got and want satisfy this tolerance. NaNs compare equal only
// to NaNs; infinities compare equal only when their signs match.
func (t Tolerance) Close(got, want float64) bool {
	if !t.Valid() {
		return false
	}
	if math.IsNaN(got) || math.IsNaN(want) {
		return math.IsNaN(got) && math.IsNaN(want)
	}
	if math.IsInf(got, 0) || math.IsInf(want, 0) {
		return got == want
	}
	diff := math.Abs(got - want)
	scale := math.Max(math.Abs(got), math.Abs(want))
	return diff <= t.Abs+t.Rel*scale
}

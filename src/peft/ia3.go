package peft

import (
	"fmt"
	"math/rand"

	"github.com/adhitiad/gotorch/src/autograd"
	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/nn"
	"github.com/adhitiad/gotorch/src/tensor"
)

// IA3Config mengonfigurasi adapter IA3. Nilai awal skala defaultnya adalah satu
// agar forward awal identik dengan base.
type IA3Config struct {
	DType dtype.DType
	RNG   *rand.Rand
	Init  float64
}

// IA3Linear membungkus projector linear dengan skala output yang trainable.
type IA3Linear struct {
	*nn.ModuleBase
	Base  Projector
	Scale *nn.Parameter
}

// NewIA3Linear membuat adapter IA3 untuk base yang diberikan.
func NewIA3Linear(name string, base Projector, config IA3Config) (*IA3Linear, error) {
	module, err := nn.NewModule(name)
	if err != nil {
		return nil, err
	}
	if base == nil || base.InputDim() <= 0 || base.OutputDim() <= 0 {
		_ = module.Close()
		return nil, fmt.Errorf("%w: invalid IA3 base for %q", errs.ErrInvalidArgument, name)
	}
	elementType := config.DType
	if elementType == dtype.Invalid {
		elementType = dtype.Float64
	}
	if !isFloat(elementType) {
		_ = module.Close()
		return nil, errs.ErrUnsupportedDType
	}
	rng := config.RNG
	if rng == nil {
		rng = nn.NewRNG(1)
	}
	initial := config.Init
	if initial == 0 {
		initial = 1
	}
	scale, err := newVectorParameter("scale", elementType, base.OutputDim(), rng, initial)
	if err != nil {
		_ = module.Close()
		return nil, err
	}
	if err := module.RegisterParameter("scale", scale); err != nil {
		_ = scale.Close()
		_ = module.Close()
		return nil, err
	}
	return &IA3Linear{ModuleBase: module, Base: base, Scale: scale}, nil
}

func newVectorParameter(name string, elementType dtype.DType, length int64, rng *rand.Rand, initial float64) (*nn.Parameter, error) {
	if length <= 0 || rng == nil {
		return nil, errs.ErrInvalidArgument
	}
	value, err := tensor.NewTensor(elementType, length)
	if err != nil {
		return nil, err
	}
	values := make([]float64, int(length))
	for index := range values {
		values[index] = initial
	}
	if err := writeFloatValues(value, values); err != nil {
		_ = value.Close()
		return nil, err
	}
	parameter, err := nn.NewParameter(name, value, true)
	_ = value.Close()
	return parameter, err
}

// InputDim mengembalikan dimensi masukan base.
func (a *IA3Linear) InputDim() int64 {
	if a == nil {
		return 0
	}
	return a.Base.InputDim()
}

// OutputDim mengembalikan dimensi keluaran base.
func (a *IA3Linear) OutputDim() int64 {
	if a == nil {
		return 0
	}
	return a.Base.OutputDim()
}

// Forward menghitung scale * base(x) dengan skala per kolom keluaran.
func (a *IA3Linear) Forward(ctx *nn.Context, input *autograd.Variable) (*autograd.Variable, error) {
	if a == nil || a.Base == nil || ctx == nil || ctx.Tape == nil {
		return nil, errs.ErrInvalidArgument
	}
	if err := validateLinearInput(input, a.InputDim()); err != nil {
		return nil, err
	}
	baseOutput, err := a.Base.Forward(ctx, input)
	if err != nil {
		return nil, err
	}
	scale, err := ctx.Variable(a.Scale)
	if err != nil {
		return nil, err
	}
	return nn.ScaleLast(ctx, baseOutput, scale)
}

package nn

import (
	"fmt"
	"math/rand"

	"github.com/adhitiad/gotorch/src/autograd"
	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/tensor"
)

// DropoutConfig mengonfigurasi dropout.
type DropoutConfig struct {
	Probability float64
	RNG         *rand.Rand
}

// Dropout menerapkan inverted dropout hanya saat mode latih.
type Dropout struct {
	*ModuleBase
	Probability float64
	rng         *rand.Rand
}

// NewDropout membuat modul dropout dengan RNG deterministik.
func NewDropout(name string, config DropoutConfig) (*Dropout, error) {
	base, err := NewModule(name)
	if err != nil {
		return nil, err
	}
	if config.Probability < 0 || config.Probability >= 1 {
		_ = base.Close()
		return nil, fmt.Errorf("%w: dropout probability %g outside [0,1)", errs.ErrInvalidArgument, config.Probability)
	}
	rng := config.RNG
	if rng == nil {
		rng = NewRNG(1)
	}
	return &Dropout{ModuleBase: base, Probability: config.Probability, rng: rng}, nil
}

// Forward menghasilkan topeng dropout dan mengalikannya dengan input.
func (d *Dropout) Forward(ctx *Context, input *autograd.Variable) (*autograd.Variable, error) {
	if d == nil || d.ModuleBase == nil || ctx == nil || ctx.Tape == nil {
		return nil, errs.ErrInvalidArgument
	}
	if !d.IsTraining() || d.Probability == 0 {
		return input, nil
	}
	data, err := variableData(input)
	if err != nil {
		return nil, err
	}
	if data.dtype != dtype.Float32 && data.dtype != dtype.Float64 {
		return nil, errs.ErrUnsupportedDType
	}
	mask, err := tensor.NewTensor(data.dtype, data.dims...)
	if err != nil {
		return nil, err
	}
	values := make([]float64, len(data.values))
	keepProbability := 1 - d.Probability
	scale := 1 / keepProbability
	for index := range values {
		if d.rng.Float64() < keepProbability {
			values[index] = scale
		}
	}
	if err := writeValues(mask, values); err != nil {
		_ = mask.Close()
		return nil, err
	}
	maskVariable, err := ctx.Tape.NewVariable(mask, false)
	_ = mask.Close()
	if err != nil {
		return nil, err
	}
	return ctx.Tape.Mul(input, maskVariable)
}

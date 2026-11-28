package nn

import (
	"fmt"
	"math/rand"

	"github.com/adhitiad/gotorch/src/autograd"
	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/tensor"
)

func epsilonVariable(ctx *Context, epsilon float64) (*autograd.Variable, error) {
	if ctx == nil || ctx.Tape == nil || epsilon <= 0 {
		return nil, errs.ErrInvalidArgument
	}
	value, err := tensor.From([]float64{epsilon})
	if err != nil {
		return nil, err
	}
	variable, err := ctx.Tape.NewVariable(value, false)
	_ = value.Close()
	return variable, err
}

// LayerNormConfig mengonfigurasi LayerNorm pada sumbu terakhir.
type LayerNormConfig struct {
	Features int64
	Epsilon  float64
	DType    dtype.DType
	RNG      *rand.Rand
}

// LayerNorm menormalkan sumbu terakhir dengan gamma dan beta trainable.
type LayerNorm struct {
	*ModuleBase
	Gamma   *Parameter
	Beta    *Parameter
	Epsilon float64
}

// NewLayerNorm membuat modul LayerNorm.
func NewLayerNorm(name string, config LayerNormConfig) (*LayerNorm, error) {
	if err := validateName(name); err != nil {
		return nil, err
	}
	if config.Features <= 0 {
		return nil, fmt.Errorf("%w: layer norm %q needs positive features", errs.ErrInvalidShape, name)
	}
	elementType := config.DType
	if elementType == dtype.Invalid {
		elementType = dtype.Float64
	}
	if !isDifferentiable(elementType) {
		return nil, errs.ErrUnsupportedDType
	}
	epsilon := config.Epsilon
	if epsilon <= 0 {
		epsilon = 1e-5
	}
	rng := config.RNG
	if rng == nil {
		rng = NewRNG(1)
	}
	base, err := NewModule(name)
	if err != nil {
		return nil, err
	}
	ones := func(_ *rand.Rand, value *tensor.Tensor) error { return Ones(value) }
	zeros := func(_ *rand.Rand, value *tensor.Tensor) error { return Zeros(value) }
	gamma, err := createParameter("gamma", elementType, []int64{config.Features}, rng, ones)
	if err != nil {
		_ = base.Close()
		return nil, err
	}
	if err := base.RegisterParameter("gamma", gamma); err != nil {
		_ = gamma.Close()
		_ = base.Close()
		return nil, err
	}
	beta, err := createParameter("beta", elementType, []int64{config.Features}, rng, zeros)
	if err != nil {
		_ = base.Close()
		return nil, err
	}
	if err := base.RegisterParameter("beta", beta); err != nil {
		_ = beta.Close()
		_ = base.Close()
		return nil, err
	}
	return &LayerNorm{ModuleBase: base, Gamma: gamma, Beta: beta, Epsilon: epsilon}, nil
}

// Forward menormalkan input dan mencatat operasinya pada autograd tape.
func (l *LayerNorm) Forward(ctx *Context, input *autograd.Variable) (*autograd.Variable, error) {
	if l == nil || l.ModuleBase == nil {
		return nil, errs.ErrInvalidArgument
	}
	inputData, err := variableData(input)
	if err != nil {
		return nil, err
	}
	gammaVariable, err := ctx.Variable(l.Gamma)
	if err != nil {
		return nil, err
	}
	betaVariable, err := ctx.Variable(l.Beta)
	if err != nil {
		return nil, err
	}
	gammaData, err := variableData(gammaVariable)
	if err != nil {
		return nil, err
	}
	if len(inputData.dims) == 0 || inputData.dims[len(inputData.dims)-1] != int64(len(gammaData.values)) {
		return nil, errs.ErrGradientShapeMismatch
	}
	betaData, err := variableData(betaVariable)
	if err != nil {
		return nil, err
	}
	epsilon, err := epsilonVariable(ctx, l.Epsilon)
	if err != nil {
		return nil, err
	}
	inputs := []*autograd.Variable{input, gammaVariable, betaVariable, epsilon}
	return applyOperation(ctx, OpLayerNorm, inputs, func() (tensorData, error) {
		output, _, err := layerNormData(inputData, gammaData, betaData, l.Epsilon)
		return output, err
	})
}

// RMSNormConfig mengonfigurasi RMSNorm pada sumbu terakhir.
type RMSNormConfig struct {
	Features int64
	Epsilon  float64
	DType    dtype.DType
	RNG      *rand.Rand
}

// RMSNorm menormalkan input dengan akar rata-rata kuadrat dan gamma trainable.
type RMSNorm struct {
	*ModuleBase
	Gamma   *Parameter
	Epsilon float64
}

// NewRMSNorm membuat modul RMSNorm.
func NewRMSNorm(name string, config RMSNormConfig) (*RMSNorm, error) {
	if err := validateName(name); err != nil {
		return nil, err
	}
	if config.Features <= 0 {
		return nil, fmt.Errorf("%w: rms norm %q needs positive features", errs.ErrInvalidShape, name)
	}
	elementType := config.DType
	if elementType == dtype.Invalid {
		elementType = dtype.Float64
	}
	if !isDifferentiable(elementType) {
		return nil, errs.ErrUnsupportedDType
	}
	epsilon := config.Epsilon
	if epsilon <= 0 {
		epsilon = 1e-5
	}
	rng := config.RNG
	if rng == nil {
		rng = NewRNG(1)
	}
	base, err := NewModule(name)
	if err != nil {
		return nil, err
	}
	gamma, err := createParameter("gamma", elementType, []int64{config.Features}, rng,
		func(_ *rand.Rand, value *tensor.Tensor) error { return Ones(value) })
	if err != nil {
		_ = base.Close()
		return nil, err
	}
	if err := base.RegisterParameter("gamma", gamma); err != nil {
		_ = gamma.Close()
		_ = base.Close()
		return nil, err
	}
	return &RMSNorm{ModuleBase: base, Gamma: gamma, Epsilon: epsilon}, nil
}

// Forward menormalkan input dengan RMS dan mencatat operasinya.
func (r *RMSNorm) Forward(ctx *Context, input *autograd.Variable) (*autograd.Variable, error) {
	if r == nil || r.ModuleBase == nil {
		return nil, errs.ErrInvalidArgument
	}
	inputData, err := variableData(input)
	if err != nil {
		return nil, err
	}
	if len(inputData.dims) == 0 || inputData.dims[len(inputData.dims)-1] != r.Features() {
		return nil, errs.ErrGradientShapeMismatch
	}
	gammaVariable, err := ctx.Variable(r.Gamma)
	if err != nil {
		return nil, err
	}
	gammaData, err := variableData(gammaVariable)
	if err != nil {
		return nil, err
	}
	epsilon, err := epsilonVariable(ctx, r.Epsilon)
	if err != nil {
		return nil, err
	}
	inputs := []*autograd.Variable{input, gammaVariable, epsilon}
	return applyOperation(ctx, OpRMSNorm, inputs, func() (tensorData, error) {
		output, _, err := rmsNormData(inputData, gammaData, r.Epsilon)
		return output, err
	})
}

// Features mengembalikan jumlah fitur pada sumbu terakhir.
func (r *RMSNorm) Features() int64 {
	if r == nil || r.Gamma == nil {
		return 0
	}
	value := r.Gamma.Value()
	if value == nil {
		return 0
	}
	metadata, err := value.Metadata()
	if err != nil || len(metadata.Shape().Dims()) == 0 {
		return 0
	}
	return metadata.Shape().Dims()[0]
}

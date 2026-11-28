package nn

import (
	"fmt"
	"math/rand"

	"github.com/adhitiad/gotorch/src/autograd"
	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/tensor"
)

// ParameterInit mengisi tensor parameter menggunakan rng yang disuntikkan.
type ParameterInit func(*rand.Rand, *tensor.Tensor) error

func createParameter(name string, elementType dtype.DType, dims []int64, rng *rand.Rand, initialize ParameterInit) (*Parameter, error) {
	if rng == nil || initialize == nil {
		return nil, errs.ErrInvalidArgument
	}
	value, err := tensor.NewTensor(elementType, dims...)
	if err != nil {
		return nil, err
	}
	if err := initialize(rng, value); err != nil {
		_ = value.Close()
		return nil, err
	}
	return NewParameter(name, value, true)
}

// LinearConfig mengonfigurasi layer Linear.
type LinearConfig struct {
	In         int64
	Out        int64
	DType      dtype.DType
	Bias       bool
	RNG        *rand.Rand
	WeightInit ParameterInit
	BiasInit   ParameterInit
}

// Linear adalah transformasi affine y = xWᵀ + b.
type Linear struct {
	*ModuleBase
	Weight  *Parameter
	Bias    *Parameter
	In      int64
	Out     int64
	useBias bool
}

// NewLinear membuat layer Linear dengan inisialisasi deterministik.
func NewLinear(name string, config LinearConfig) (*Linear, error) {
	if err := validateName(name); err != nil {
		return nil, err
	}
	if config.In <= 0 || config.Out <= 0 {
		return nil, fmt.Errorf("%w: linear %q needs positive In and Out", errs.ErrInvalidShape, name)
	}
	elementType := config.DType
	if elementType == dtype.Invalid {
		elementType = dtype.Float64
	}
	if !isDifferentiable(elementType) {
		return nil, errs.ErrUnsupportedDType
	}
	rng := config.RNG
	if rng == nil {
		rng = NewRNG(1)
	}
	weightInit := config.WeightInit
	if weightInit == nil {
		weightInit = KaimingUniform
	}
	biasInit := config.BiasInit
	if biasInit == nil {
		biasInit = func(_ *rand.Rand, value *tensor.Tensor) error { return Zeros(value) }
	}
	base, err := NewModule(name)
	if err != nil {
		return nil, err
	}
	weight, err := createParameter("weight", elementType, []int64{config.Out, config.In}, rng, weightInit)
	if err != nil {
		_ = base.Close()
		return nil, err
	}
	if err := base.RegisterParameter("weight", weight); err != nil {
		_ = weight.Close()
		_ = base.Close()
		return nil, err
	}
	layer := &Linear{ModuleBase: base, Weight: weight, In: config.In, Out: config.Out, useBias: config.Bias}
	if config.Bias {
		bias, err := createParameter("bias", elementType, []int64{config.Out}, rng, biasInit)
		if err != nil {
			_ = base.Close()
			return nil, err
		}
		if err := base.RegisterParameter("bias", bias); err != nil {
			_ = bias.Close()
			_ = base.Close()
			return nil, err
		}
		layer.Bias = bias
	}
	return layer, nil
}

// InputDim mengembalikan dimensi masukan layer.
func (l *Linear) InputDim() int64 {
	if l == nil {
		return 0
	}
	return l.In
}

// OutputDim mengembalikan dimensi keluaran layer.
func (l *Linear) OutputDim() int64 {
	if l == nil {
		return 0
	}
	return l.Out
}

// WeightParameter mengembalikan parameter weight milik layer.
func (l *Linear) WeightParameter() *Parameter {
	if l == nil {
		return nil
	}
	return l.Weight
}

// Forward menjalankan transformasi affine dan mencatatnya pada autograd tape.
func (l *Linear) Forward(ctx *Context, input *autograd.Variable) (*autograd.Variable, error) {
	if l == nil || l.ModuleBase == nil {
		return nil, errs.ErrInvalidArgument
	}
	inputData, err := variableData(input)
	if err != nil {
		return nil, err
	}
	if len(inputData.dims) != 2 || inputData.dims[1] != l.In {
		return nil, fmt.Errorf("%w: linear input %v, want (*, %d)", errs.ErrInvalidShape, inputData.dims, l.In)
	}
	weightVariable, err := ctx.Variable(l.Weight)
	if err != nil {
		return nil, err
	}
	weightData, err := variableData(weightVariable)
	if err != nil {
		return nil, err
	}
	variables := []*autograd.Variable{input, weightVariable}
	var biasData tensorData
	operation := OpLinearNoBias
	if l.useBias {
		biasVariable, err := ctx.Variable(l.Bias)
		if err != nil {
			return nil, err
		}
		biasData, err = variableData(biasVariable)
		if err != nil {
			return nil, err
		}
		variables = append(variables, biasVariable)
		operation = OpLinear
	}
	return applyOperation(ctx, operation, variables, func() (tensorData, error) {
		return linearData(inputData, weightData, biasData, l.useBias)
	})
}

// Activation adalah layer element-wise tanpa parameter.
type Activation struct {
	*ModuleBase
	op      string
	compute func(tensorData) tensorData
}

func newActivation(name, op string, compute func(tensorData) tensorData) (*Activation, error) {
	base, err := NewModule(name)
	if err != nil {
		return nil, err
	}
	return &Activation{ModuleBase: base, op: op, compute: compute}, nil
}

// NewReLU membuat layer ReLU.
func NewReLU(name string) (*Activation, error) { return newActivation(name, OpReLU, reluForward) }

// NewSigmoid membuat layer sigmoid.
func NewSigmoid(name string) (*Activation, error) {
	return newActivation(name, OpSigmoid, sigmoidForward)
}

// NewTanh membuat layer tanh.
func NewTanh(name string) (*Activation, error) { return newActivation(name, OpTanh, tanhForward) }

// NewGELU membuat layer GELU dengan pendekatan tanh.
func NewGELU(name string) (*Activation, error) { return newActivation(name, OpGELU, geluForward) }

// NewSiLU membuat layer SiLU/Swish.
func NewSiLU(name string) (*Activation, error) { return newActivation(name, OpSiLU, siluForward) }

// Forward menjalankan aktivasi element-wise.
func (a *Activation) Forward(ctx *Context, input *autograd.Variable) (*autograd.Variable, error) {
	if a == nil || a.ModuleBase == nil {
		return nil, errs.ErrInvalidArgument
	}
	return unaryOperation(ctx, a.op, input, a.compute)
}

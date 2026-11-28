package nn

import (
	"fmt"
	"math/rand"

	"github.com/adhitiad/gotorch/src/autograd"
	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/tensor"
)

// Conv2DConfig mengonfigurasi konvolusi 2D NCHW.
type Conv2DConfig struct {
	InChannels  int64
	OutChannels int64
	KernelH     int64
	KernelW     int64
	Stride      int64
	Padding     int64
	DType       dtype.DType
	Bias        bool
	RNG         *rand.Rand
}

// Conv2D adalah konvolusi 2D cross-correlation pada layout NCHW.
type Conv2D struct {
	*ModuleBase
	Weight  *Parameter
	Bias    *Parameter
	Stride  int64
	Padding int64
	useBias bool
}

// NewConv2D membuat layer Conv2D dengan inisialisasi Kaiming.
func NewConv2D(name string, config Conv2DConfig) (*Conv2D, error) {
	if err := validateName(name); err != nil {
		return nil, err
	}
	if config.InChannels <= 0 || config.OutChannels <= 0 || config.KernelH <= 0 || config.KernelW <= 0 ||
		config.Stride <= 0 || config.Padding < 0 {
		return nil, fmt.Errorf("%w: invalid conv2d configuration for %q", errs.ErrInvalidShape, name)
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
	base, err := NewModule(name)
	if err != nil {
		return nil, err
	}
	weight, err := createParameter("weight", elementType,
		[]int64{config.OutChannels, config.InChannels, config.KernelH, config.KernelW}, rng, KaimingUniform)
	if err != nil {
		_ = base.Close()
		return nil, err
	}
	if err := base.RegisterParameter("weight", weight); err != nil {
		_ = weight.Close()
		_ = base.Close()
		return nil, err
	}
	layer := &Conv2D{ModuleBase: base, Weight: weight, Stride: config.Stride, Padding: config.Padding, useBias: config.Bias}
	if config.Bias {
		bias, err := createParameter("bias", elementType, []int64{config.OutChannels}, rng,
			func(_ *rand.Rand, value *tensor.Tensor) error { return Zeros(value) })
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

// Forward menjalankan konvolusi dan mencatatnya pada autograd tape.
func (c *Conv2D) Forward(ctx *Context, input *autograd.Variable) (*autograd.Variable, error) {
	if c == nil || c.ModuleBase == nil {
		return nil, errs.ErrInvalidArgument
	}
	inputData, err := variableData(input)
	if err != nil {
		return nil, err
	}
	weightVariable, err := ctx.Variable(c.Weight)
	if err != nil {
		return nil, err
	}
	weightData, err := variableData(weightVariable)
	if err != nil {
		return nil, err
	}
	biasData := tensorData{dtype: weightData.dtype, dims: []int64{weightData.dims[0]}}
	biasData.values = make([]float64, int(weightData.dims[0]))
	variables := []*autograd.Variable{input, weightVariable}
	if c.useBias {
		biasVariable, err := ctx.Variable(c.Bias)
		if err != nil {
			return nil, err
		}
		biasData, err = variableData(biasVariable)
		if err != nil {
			return nil, err
		}
		variables = append(variables, biasVariable)
	}
	configTensor, err := intTensor(c.Stride, c.Padding)
	if err != nil {
		return nil, err
	}
	configVariable, err := ctx.Tape.NewVariable(configTensor, false)
	_ = configTensor.Close()
	if err != nil {
		return nil, err
	}
	variables = append(variables, configVariable)
	return applyOperation(ctx, OpConv2D, variables, func() (tensorData, error) {
		return conv2DData(inputData, weightData, biasData, []int64{c.Stride, c.Padding})
	})
}

// Pool2DConfig mengonfigurasi pooling 2D NCHW.
type Pool2DConfig struct {
	KernelH int64
	KernelW int64
	Stride  int64
	Padding int64
}

// Pool2D adalah layer max atau average pooling pada layout NCHW.
type Pool2D struct {
	*ModuleBase
	KernelH int64
	KernelW int64
	Stride  int64
	Padding int64
	op      string
	compute func(tensorData, []int64) (tensorData, error)
}

// NewMaxPool2D membuat layer max pooling.
func NewMaxPool2D(name string, config Pool2DConfig) (*Pool2D, error) {
	return newPool2D(name, config, OpMaxPool2D, maxPoolData)
}

// NewAvgPool2D membuat layer average pooling.
func NewAvgPool2D(name string, config Pool2DConfig) (*Pool2D, error) {
	return newPool2D(name, config, OpAvgPool2D, func(data tensorData, config []int64) (tensorData, error) {
		return avgPoolData(data, config)
	})
}

func newPool2D(name string, config Pool2DConfig, op string, compute func(tensorData, []int64) (tensorData, error)) (*Pool2D, error) {
	base, err := NewModule(name)
	if err != nil {
		return nil, err
	}
	if config.KernelH <= 0 || config.KernelW <= 0 || config.Stride <= 0 || config.Padding < 0 {
		_ = base.Close()
		return nil, fmt.Errorf("%w: invalid pool configuration for %q", errs.ErrInvalidShape, name)
	}
	return &Pool2D{
		ModuleBase: base,
		KernelH:    config.KernelH,
		KernelW:    config.KernelW,
		Stride:     config.Stride,
		Padding:    config.Padding,
		op:         op,
		compute:    compute,
	}, nil
}

// Forward menjalankan pooling dan mencatatnya pada autograd tape.
func (p *Pool2D) Forward(ctx *Context, input *autograd.Variable) (*autograd.Variable, error) {
	if p == nil || p.ModuleBase == nil || ctx == nil || ctx.Tape == nil {
		return nil, errs.ErrInvalidArgument
	}
	inputData, err := variableData(input)
	if err != nil {
		return nil, err
	}
	if len(inputData.dims) != 4 {
		return nil, errs.ErrInvalidShape
	}
	settings := []int64{p.KernelH, p.KernelW, p.Stride, p.Padding}
	configTensor, err := intTensor(settings...)
	if err != nil {
		return nil, err
	}
	configVariable, err := ctx.Tape.NewVariable(configTensor, false)
	_ = configTensor.Close()
	if err != nil {
		return nil, err
	}
	return applyOperation(ctx, p.op, []*autograd.Variable{input, configVariable}, func() (tensorData, error) {
		return p.compute(inputData, settings)
	})
}

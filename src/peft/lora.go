package peft

import (
	"fmt"
	"math"
	"math/rand"
	"sync"

	"github.com/adhitiad/gotorch/src/autograd"
	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/nn"
	"github.com/adhitiad/gotorch/src/tensor"
)

// LoRAConfig mengonfigurasi adapter low-rank.
type LoRAConfig struct {
	Rank  int64
	Alpha float64
	DType dtype.DType
	RNG   *rand.Rand
}

// LoRALinear membungkus layer linear dengan delta B@A berskala alpha/rank.
type LoRALinear struct {
	*nn.ModuleBase
	Base   Projector
	A      *nn.Parameter
	B      *nn.Parameter
	Scale  float64
	mu     sync.RWMutex
	merged bool
}

// NewLoRALinear membuat adapter LoRA untuk base yang diberikan.
func NewLoRALinear(name string, base Projector, config LoRAConfig) (*LoRALinear, error) {
	module, err := nn.NewModule(name)
	if err != nil {
		return nil, err
	}
	if base == nil || config.Rank <= 0 || config.Rank > base.InputDim() || config.Rank > base.OutputDim() || config.RNG == nil {
		_ = module.Close()
		return nil, fmt.Errorf("%w: invalid LoRA configuration for %q", errs.ErrInvalidArgument, name)
	}
	elementType := config.DType
	if elementType == dtype.Invalid {
		elementType = dtype.Float64
	}
	if !isFloat(elementType) {
		_ = module.Close()
		return nil, errs.ErrUnsupportedDType
	}
	alpha := config.Alpha
	if alpha == 0 {
		alpha = float64(config.Rank)
	}
	a, err := newMatrixParameter("a", elementType, []int64{config.Rank, base.InputDim()}, config.RNG,
		func() float64 { return config.RNG.NormFloat64() / math.Sqrt(float64(base.InputDim())) })
	if err != nil {
		_ = module.Close()
		return nil, err
	}
	if err := module.RegisterParameter("a", a); err != nil {
		_ = a.Close()
		_ = module.Close()
		return nil, err
	}
	b, err := newMatrixParameter("b", elementType, []int64{base.OutputDim(), config.Rank}, config.RNG,
		func() float64 { return 0 })
	if err != nil {
		_ = module.Close()
		return nil, err
	}
	if err := module.RegisterParameter("b", b); err != nil {
		_ = b.Close()
		_ = module.Close()
		return nil, err
	}
	return &LoRALinear{
		ModuleBase: module,
		Base:       base,
		A:          a,
		B:          b,
		Scale:      alpha / float64(config.Rank),
	}, nil
}

func isFloat(elementType dtype.DType) bool {
	return elementType == dtype.Float32 || elementType == dtype.Float64
}

func newMatrixParameter(name string, elementType dtype.DType, dims []int64, rng *rand.Rand, generator func() float64) (*nn.Parameter, error) {
	_ = rng
	value, err := tensor.NewTensor(elementType, dims...)
	if err != nil {
		return nil, err
	}
	count, err := paramCountForDims(dims)
	if err != nil {
		_ = value.Close()
		return nil, err
	}
	values := make([]float64, count)
	for index := range values {
		values[index] = generator()
	}
	if err := writeFloatValues(value, values); err != nil {
		_ = value.Close()
		return nil, err
	}
	parameter, err := nn.NewParameter(name, value, true)
	_ = value.Close()
	return parameter, err
}

func paramCountForDims(dims []int64) (int, error) {
	count := 1
	for _, dim := range dims {
		if dim <= 0 || int64(count) > int64(^uint(0)>>1)/dim {
			return 0, errs.ErrShapeOverflow
		}
		count *= int(dim)
	}
	return count, nil
}

// InputDim mengembalikan dimensi masukan base.
func (l *LoRALinear) InputDim() int64 {
	if l == nil {
		return 0
	}
	return l.Base.InputDim()
}

// OutputDim mengembalikan dimensi keluaran base.
func (l *LoRALinear) OutputDim() int64 {
	if l == nil {
		return 0
	}
	return l.Base.OutputDim()
}

// IsMerged melaporkan apakah delta sudah disatukan ke base weight.
func (l *LoRALinear) IsMerged() bool {
	if l == nil {
		return false
	}
	l.mu.RLock()
	defer l.mu.RUnlock()
	return l.merged
}

// Forward menghitung base(x) + scale * xAB, atau base(x) saat sudah di-merge.
func (l *LoRALinear) Forward(ctx *nn.Context, input *autograd.Variable) (*autograd.Variable, error) {
	if l == nil || l.Base == nil || ctx == nil || ctx.Tape == nil {
		return nil, errs.ErrInvalidArgument
	}
	if err := validateLinearInput(input, l.InputDim()); err != nil {
		return nil, err
	}
	baseOutput, err := l.Base.Forward(ctx, input)
	if err != nil {
		return nil, err
	}
	l.mu.RLock()
	merged := l.merged
	l.mu.RUnlock()
	if merged {
		return baseOutput, nil
	}
	aVariable, err := ctx.Variable(l.A)
	if err != nil {
		return nil, err
	}
	bVariable, err := ctx.Variable(l.B)
	if err != nil {
		return nil, err
	}
	aTransposed, err := nn.Transpose(ctx, aVariable, 1, 0)
	if err != nil {
		return nil, err
	}
	lowRank, err := nn.MatMul(ctx, input, aTransposed)
	if err != nil {
		return nil, err
	}
	bTransposed, err := nn.Transpose(ctx, bVariable, 1, 0)
	if err != nil {
		return nil, err
	}
	delta, err := nn.MatMul(ctx, lowRank, bTransposed)
	if err != nil {
		return nil, err
	}
	scaled, err := nn.Scale(ctx, delta, l.Scale)
	if err != nil {
		return nil, err
	}
	return ctx.Tape.Add(baseOutput, scaled)
}

func validateLinearInput(input *autograd.Variable, inputDim int64) error {
	if input == nil {
		return errs.ErrInvalidArgument
	}
	value, err := input.Value()
	if err != nil {
		return err
	}
	defer value.Close()
	metadata, err := value.Metadata()
	if err != nil {
		return err
	}
	dims := metadata.Shape().Dims()
	if len(dims) != 2 || dims[0] <= 0 || dims[1] != inputDim {
		return fmt.Errorf("%w: LoRA input must be [batch, %d]", errs.ErrInvalidShape, inputDim)
	}
	return nil
}

// Merge menjumlahkan delta ke base weight. Pemanggilan ulang tidak melakukan apa-apa.
func (l *LoRALinear) Merge() error {
	return l.setMerged(true)
}

// Unmerge mengurangi delta dari base weight. Pemanggilan ulang tidak melakukan apa-apa.
func (l *LoRALinear) Unmerge() error {
	return l.setMerged(false)
}

func (l *LoRALinear) setMerged(target bool) error {
	if l == nil {
		return errs.ErrInvalidArgument
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.merged == target {
		return nil
	}
	linear, ok := l.Base.(LinearLike)
	if !ok || linear.WeightParameter() == nil {
		return fmt.Errorf("%w: LoRA merge requires a linear base with weight access", errs.ErrUnsupportedOperation)
	}
	weight := linear.WeightParameter()
	metadata, err := weight.Value().Metadata()
	if err != nil {
		return err
	}
	dims := metadata.Shape().Dims()
	if len(dims) != 2 || dims[0] != l.OutputDim() || dims[1] != l.InputDim() {
		return errs.ErrGradientShapeMismatch
	}
	rank := l.A.Value()
	if rank == nil || l.B.Value() == nil {
		return errs.ErrInvalidState
	}
	aValues, err := readFloatValues(rank)
	if err != nil {
		return err
	}
	bValues, err := readFloatValues(l.B.Value())
	if err != nil {
		return err
	}
	delta, err := l.deltaWeight(aValues, bValues)
	if err != nil {
		return err
	}
	err = weight.Update(func(values []float64) ([]float64, error) {
		if target {
			for index := range values {
				values[index] += delta[index]
			}
		} else {
			for index := range values {
				values[index] -= delta[index]
			}
		}
		return values, nil
	})
	if err != nil {
		return err
	}
	l.merged = target
	return nil
}

func rankDim(l *LoRALinear) int64 {
	metadata, err := l.A.Value().Metadata()
	if err != nil {
		return 0
	}
	dims := metadata.Shape().Dims()
	if len(dims) != 2 {
		return 0
	}
	return dims[0]
}

func (l *LoRALinear) deltaWeight(aValues, bValues []float64) ([]float64, error) {
	dims, err := l.A.Value().Metadata()
	if err != nil {
		return nil, err
	}
	shape := dims.Shape().Dims()
	rank, inputDim := int(shape[0]), int(shape[1])
	outputDim := int(l.OutputDim())
	if len(aValues) != rank*inputDim || len(bValues) != outputDim*rank {
		return nil, errs.ErrGradientShapeMismatch
	}
	delta := make([]float64, outputDim*inputDim)
	for row := 0; row < outputDim; row++ {
		for column := 0; column < inputDim; column++ {
			var total float64
			for index := 0; index < rank; index++ {
				total += bValues[row*rank+index] * aValues[index*inputDim+column]
			}
			delta[row*inputDim+column] = l.Scale * total
		}
	}
	return delta, nil
}

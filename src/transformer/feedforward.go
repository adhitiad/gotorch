package transformer

import (
	"fmt"
	"math/rand"

	"github.com/adhitiad/gotorch/src/autograd"
	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/nn"
)

// FeedForwardConfig mengonfigurasi jaringan posisi dua lapis.
type FeedForwardConfig struct {
	ModelDim  int64
	HiddenDim int64
	DType     dtype.DType
	RNG       *rand.Rand
}

// FeedForward adalah MLP posisi Linear -> aktivasi -> Linear.
type FeedForward struct {
	*nn.ModuleBase
	First      Projector
	Activation *nn.Activation
	Second     Projector
}

// NewFeedForward membuat MLP posisi dengan GELU.
func NewFeedForward(name string, config FeedForwardConfig) (*FeedForward, error) {
	base, err := nn.NewModule(name)
	if err != nil {
		return nil, err
	}
	if config.ModelDim <= 0 || config.HiddenDim <= 0 {
		_ = base.Close()
		return nil, fmt.Errorf("%w: invalid feed-forward dimensions for %q", errs.ErrInvalidShape, name)
	}
	elementType := resolveDType(config.DType)
	rng := config.RNG
	if rng == nil {
		rng = nn.NewRNG(1)
	}
	first, err := nn.NewLinear("first", nn.LinearConfig{In: config.ModelDim, Out: config.HiddenDim, DType: elementType, RNG: rng})
	if err != nil {
		_ = base.Close()
		return nil, err
	}
	activation, err := nn.NewGELU("activation")
	if err != nil {
		_ = base.Close()
		return nil, err
	}
	second, err := nn.NewLinear("second", nn.LinearConfig{In: config.HiddenDim, Out: config.ModelDim, DType: elementType, RNG: rng})
	if err != nil {
		_ = base.Close()
		return nil, err
	}
	for _, child := range []nn.Module{first, activation, second} {
		if err := base.RegisterModule(child); err != nil {
			_ = base.Close()
			return nil, err
		}
	}
	return &FeedForward{ModuleBase: base, First: first, Activation: activation, Second: second}, nil
}

// Forward menerima input [batch, time, modelDim] dan mengembalikan bentuk yang sama.
func (f *FeedForward) Forward(ctx *nn.Context, input *autograd.Variable) (*autograd.Variable, error) {
	if f == nil || f.ModuleBase == nil || ctx == nil {
		return nil, errs.ErrInvalidArgument
	}
	dims, _, err := sequenceShape(input, f.First.InputDim())
	if err != nil {
		return nil, err
	}
	flat, err := nn.Reshape(ctx, input, dims[0]*dims[1], dims[2])
	if err != nil {
		return nil, err
	}
	hidden, err := f.First.Forward(ctx, flat)
	if err != nil {
		return nil, err
	}
	activated, err := f.Activation.Forward(ctx, hidden)
	if err != nil {
		return nil, err
	}
	projected, err := f.Second.Forward(ctx, activated)
	if err != nil {
		return nil, err
	}
	return nn.Reshape(ctx, projected, dims[0], dims[1], dims[2])
}

// SwiGLU menggunakan dua proyeksi gerbang dan perkalian dengan SiLU.
type SwiGLU struct {
	*nn.ModuleBase
	Gate       Projector
	Up         Projector
	Down       Projector
	Activation *nn.Activation
}

// NewSwiGLU membuat modul feed-forward SwiGLU.
func NewSwiGLU(name string, config FeedForwardConfig) (*SwiGLU, error) {
	base, err := nn.NewModule(name)
	if err != nil {
		return nil, err
	}
	if config.ModelDim <= 0 || config.HiddenDim <= 0 {
		_ = base.Close()
		return nil, fmt.Errorf("%w: invalid SwiGLU dimensions for %q", errs.ErrInvalidShape, name)
	}
	elementType := resolveDType(config.DType)
	rng := config.RNG
	if rng == nil {
		rng = nn.NewRNG(1)
	}
	makeLinear := func(suffix string, in, out int64) (Projector, error) {
		return nn.NewLinear(suffix, nn.LinearConfig{In: in, Out: out, DType: elementType, RNG: rng})
	}
	gate, err := makeLinear("gate", config.ModelDim, config.HiddenDim)
	if err != nil {
		_ = base.Close()
		return nil, err
	}
	up, err := makeLinear("up", config.ModelDim, config.HiddenDim)
	if err != nil {
		_ = base.Close()
		return nil, err
	}
	down, err := makeLinear("down", config.HiddenDim, config.ModelDim)
	if err != nil {
		_ = base.Close()
		return nil, err
	}
	activation, err := nn.NewSiLU("silu")
	if err != nil {
		_ = base.Close()
		return nil, err
	}
	for _, child := range []nn.Module{gate, up, down, activation} {
		if err := base.RegisterModule(child); err != nil {
			_ = base.Close()
			return nil, err
		}
	}
	return &SwiGLU{ModuleBase: base, Gate: gate, Up: up, Down: down, Activation: activation}, nil
}

// Forward menghitung down(silu(gate(x)) * up(x)).
func (s *SwiGLU) Forward(ctx *nn.Context, input *autograd.Variable) (*autograd.Variable, error) {
	if s == nil || s.ModuleBase == nil || ctx == nil || ctx.Tape == nil {
		return nil, errs.ErrInvalidArgument
	}
	dims, _, err := sequenceShape(input, s.Gate.InputDim())
	if err != nil {
		return nil, err
	}
	flat, err := nn.Reshape(ctx, input, dims[0]*dims[1], dims[2])
	if err != nil {
		return nil, err
	}
	gate, err := s.Gate.Forward(ctx, flat)
	if err != nil {
		return nil, err
	}
	up, err := s.Up.Forward(ctx, flat)
	if err != nil {
		return nil, err
	}
	activated, err := s.Activation.Forward(ctx, gate)
	if err != nil {
		return nil, err
	}
	gated, err := ctx.Tape.Mul(activated, up)
	if err != nil {
		return nil, err
	}
	projected, err := s.Down.Forward(ctx, gated)
	if err != nil {
		return nil, err
	}
	return nn.Reshape(ctx, projected, dims[0], dims[1], dims[2])
}

package transformer

import (
	"fmt"

	"github.com/adhitiad/gotorch/src/autograd"
	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/nn"
)

// Norm adalah modul normalisasi yang bisa dipakai blok transformer.
type Norm interface {
	nn.Module
	Forward(*nn.Context, *autograd.Variable) (*autograd.Variable, error)
}

// Jenis normalisasi yang didukung blok transformer.
const (
	NormLayer = "layer"
	NormRMS   = "rms"
)

// Jenis feed-forward yang didukung blok transformer.
const (
	FeedForwardGELU   = "gelu"
	FeedForwardSwiGLU = "swiglu"
)

// BlockConfig mengonfigurasi satu blok encoder/dekoder.
type BlockConfig struct {
	Attention   AttentionConfig
	HiddenDim   int64
	Norm        string
	FeedForward string
}

func normalized(mode string) string {
	if mode == "" {
		return NormLayer
	}
	return mode
}

func newNorm(name, mode string, features int64, elementType dtype.DType) (Norm, error) {
	switch normalized(mode) {
	case NormLayer:
		return nn.NewLayerNorm(name, nn.LayerNormConfig{Features: features, DType: elementType})
	case NormRMS:
		return nn.NewRMSNorm(name, nn.RMSNormConfig{Features: features, DType: elementType})
	default:
		return nil, fmt.Errorf("%w: unsupported norm %q", errs.ErrInvalidArgument, mode)
	}
}

func newFeedForward(name string, config BlockConfig) (nn.Module, func(*nn.Context, *autograd.Variable) (*autograd.Variable, error), error) {
	settings := FeedForwardConfig{ModelDim: config.Attention.ModelDim, HiddenDim: config.HiddenDim, DType: config.Attention.DType, RNG: config.Attention.RNG}
	mode := config.FeedForward
	if mode == "" {
		mode = FeedForwardGELU
	}
	switch mode {
	case FeedForwardGELU:
		module, err := NewFeedForward(name, settings)
		if err != nil {
			return nil, nil, err
		}
		return module, module.Forward, nil
	case FeedForwardSwiGLU:
		module, err := NewSwiGLU(name, settings)
		if err != nil {
			return nil, nil, err
		}
		return module, module.Forward, nil
	default:
		return nil, nil, fmt.Errorf("%w: unsupported feed-forward %q", errs.ErrInvalidArgument, config.FeedForward)
	}
}

// EncoderBlock adalah pre-norm self-attention plus feed-forward dengan residual.
type EncoderBlock struct {
	*nn.ModuleBase
	Attention   *MultiHeadAttention
	Norm1       Norm
	Norm2       Norm
	FeedForward func(*nn.Context, *autograd.Variable) (*autograd.Variable, error)
}

// NewEncoderBlock membuat blok encoder.
func NewEncoderBlock(name string, config BlockConfig) (*EncoderBlock, error) {
	base, err := nn.NewModule(name)
	if err != nil {
		return nil, err
	}
	attention, err := NewMultiHeadAttention("attention", config.Attention)
	if err != nil {
		_ = base.Close()
		return nil, err
	}
	norm1, err := newNorm("norm1", config.Norm, config.Attention.ModelDim, config.Attention.DType)
	if err != nil {
		_ = base.Close()
		return nil, err
	}
	norm2, err := newNorm("norm2", config.Norm, config.Attention.ModelDim, config.Attention.DType)
	if err != nil {
		_ = base.Close()
		return nil, err
	}
	feedForward, forward, err := newFeedForward("feed_forward", config)
	if err != nil {
		_ = base.Close()
		return nil, err
	}
	for _, child := range []nn.Module{attention, norm1, feedForward, norm2} {
		if err := base.RegisterModule(child); err != nil {
			_ = base.Close()
			return nil, err
		}
	}
	return &EncoderBlock{
		ModuleBase:  base,
		Attention:   attention,
		Norm1:       norm1,
		Norm2:       norm2,
		FeedForward: forward,
	}, nil
}

// Forward menjalankan blok encoder pada input [batch, time, modelDim].
func (b *EncoderBlock) Forward(ctx *nn.Context, input *autograd.Variable, paddingMask []int64) (*autograd.Variable, error) {
	if b == nil || ctx == nil || ctx.Tape == nil {
		return nil, errs.ErrInvalidArgument
	}
	attention, err := b.Attention.Forward(ctx, input, input, input, paddingMask)
	if err != nil {
		return nil, err
	}
	residual, err := ctx.Tape.Add(input, attention)
	if err != nil {
		return nil, err
	}
	normalized, err := b.Norm1.Forward(ctx, residual)
	if err != nil {
		return nil, err
	}
	projected, err := b.FeedForward(ctx, normalized)
	if err != nil {
		return nil, err
	}
	residual, err = ctx.Tape.Add(residual, projected)
	if err != nil {
		return nil, err
	}
	return b.Norm2.Forward(ctx, residual)
}

// DecoderBlock adalah blok dengan self-attention kausal, cross-attention, dan feed-forward.
type DecoderBlock struct {
	*nn.ModuleBase
	SelfAttention  *MultiHeadAttention
	CrossAttention *MultiHeadAttention
	Norm1          Norm
	Norm2          Norm
	Norm3          Norm
	FeedForward    func(*nn.Context, *autograd.Variable) (*autograd.Variable, error)
}

// NewDecoderBlock membuat blok dekoder.
func NewDecoderBlock(name string, config BlockConfig) (*DecoderBlock, error) {
	base, err := nn.NewModule(name)
	if err != nil {
		return nil, err
	}
	selfConfig := config.Attention
	selfConfig.Causal = true
	crossConfig := config.Attention
	crossConfig.Causal = false
	selfAttention, err := NewMultiHeadAttention("self_attention", selfConfig)
	if err != nil {
		_ = base.Close()
		return nil, err
	}
	crossAttention, err := NewMultiHeadAttention("cross_attention", crossConfig)
	if err != nil {
		_ = base.Close()
		return nil, err
	}
	norm1, err := newNorm("norm1", config.Norm, config.Attention.ModelDim, config.Attention.DType)
	if err != nil {
		_ = base.Close()
		return nil, err
	}
	norm2, err := newNorm("norm2", config.Norm, config.Attention.ModelDim, config.Attention.DType)
	if err != nil {
		_ = base.Close()
		return nil, err
	}
	norm3, err := newNorm("norm3", config.Norm, config.Attention.ModelDim, config.Attention.DType)
	if err != nil {
		_ = base.Close()
		return nil, err
	}
	feedForward, forward, err := newFeedForward("feed_forward", config)
	if err != nil {
		_ = base.Close()
		return nil, err
	}
	for _, child := range []nn.Module{selfAttention, crossAttention, norm1, norm2, feedForward, norm3} {
		if err := base.RegisterModule(child); err != nil {
			_ = base.Close()
			return nil, err
		}
	}
	return &DecoderBlock{
		ModuleBase:     base,
		SelfAttention:  selfAttention,
		CrossAttention: crossAttention,
		Norm1:          norm1,
		Norm2:          norm2,
		Norm3:          norm3,
		FeedForward:    forward,
	}, nil
}

// Forward menjalankan blok dekoder dengan memori encoder opsional.
func (b *DecoderBlock) Forward(ctx *nn.Context, input, memory *autograd.Variable, selfPadding, crossPadding []int64) (*autograd.Variable, error) {
	if b == nil || ctx == nil || ctx.Tape == nil {
		return nil, errs.ErrInvalidArgument
	}
	selfAttended, err := b.SelfAttention.Forward(ctx, input, input, input, selfPadding)
	if err != nil {
		return nil, err
	}
	residual, err := ctx.Tape.Add(input, selfAttended)
	if err != nil {
		return nil, err
	}
	normalized, err := b.Norm1.Forward(ctx, residual)
	if err != nil {
		return nil, err
	}
	if memory == nil {
		memory = normalized
	}
	crossAttended, err := b.CrossAttention.Forward(ctx, normalized, memory, memory, crossPadding)
	if err != nil {
		return nil, err
	}
	residual, err = ctx.Tape.Add(residual, crossAttended)
	if err != nil {
		return nil, err
	}
	normalized, err = b.Norm2.Forward(ctx, residual)
	if err != nil {
		return nil, err
	}
	projected, err := b.FeedForward(ctx, normalized)
	if err != nil {
		return nil, err
	}
	residual, err = ctx.Tape.Add(residual, projected)
	if err != nil {
		return nil, err
	}
	return b.Norm3.Forward(ctx, residual)
}

// Stack adalah kumpulan blok yang dapat dipanggil berurutan.
type Stack struct {
	*nn.ModuleBase
	blocks []func(*nn.Context, *autograd.Variable, []int64) (*autograd.Variable, error)
}

// NewEncoderStack membuat tumpukan blok encoder.
func NewEncoderStack(name string, config BlockConfig, layers int) (*Stack, error) {
	if layers <= 0 {
		return nil, fmt.Errorf("%w: transformer stack needs positive layer count", errs.ErrInvalidShape)
	}
	base, err := nn.NewModule(name)
	if err != nil {
		return nil, err
	}
	stack := &Stack{ModuleBase: base}
	for index := 0; index < layers; index++ {
		block, err := NewEncoderBlock(fmt.Sprintf("layer%d", index), config)
		if err != nil {
			_ = base.Close()
			return nil, err
		}
		if err := base.RegisterModule(block); err != nil {
			_ = base.Close()
			return nil, err
		}
		current := block
		stack.blocks = append(stack.blocks, func(ctx *nn.Context, input *autograd.Variable, mask []int64) (*autograd.Variable, error) {
			return current.Forward(ctx, input, mask)
		})
	}
	return stack, nil
}

// Forward menjalankan seluruh blok encoder secara berurutan.
func (s *Stack) Forward(ctx *nn.Context, input *autograd.Variable, paddingMask []int64) (*autograd.Variable, error) {
	if s == nil || s.ModuleBase == nil {
		return nil, errs.ErrInvalidArgument
	}
	value := input
	var err error
	for _, block := range s.blocks {
		value, err = block(ctx, value, paddingMask)
		if err != nil {
			return nil, err
		}
	}
	return value, nil
}

package peft

import (
	"fmt"
	"math"

	"github.com/adhitiad/gotorch/src/autograd"
	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/nn"
	"github.com/adhitiad/gotorch/src/tensor"
	"github.com/adhitiad/gotorch/src/transformer"
)

// PromptTuningConfig mengonfigurasi prompt trainable yang disisipkan di awal urutan.
type PromptTuningConfig struct {
	PromptLength int64
	DType        dtype.DType
	RNG          interface{ NormFloat64() float64 }
}

// PromptTuning membungkus TokenEmbedding dengan prompt trainable [length, dim].
// Pemilik tetap menyimpan embedding; modul ini hanya memiliki parameter prompt.
type PromptTuning struct {
	*nn.ModuleBase
	Embedding *transformer.TokenEmbedding
	Prompt    *nn.Parameter
}

// NewPromptTuning membuat prompt dengan inisialisasi normal deterministik.
func NewPromptTuning(name string, embedding *transformer.TokenEmbedding, config PromptTuningConfig) (*PromptTuning, error) {
	module, err := nn.NewModule(name)
	if err != nil {
		return nil, err
	}
	if embedding == nil || embedding.Weight == nil || config.PromptLength <= 0 || config.RNG == nil {
		_ = module.Close()
		return nil, fmt.Errorf("%w: invalid prompt tuning configuration for %q", errs.ErrInvalidArgument, name)
	}
	elementType := config.DType
	if elementType == dtype.Invalid {
		elementType = dtype.Float64
	}
	if !isFloat(elementType) {
		_ = module.Close()
		return nil, errs.ErrUnsupportedDType
	}
	value, err := tensor.NewTensor(elementType, config.PromptLength, embedding.Dim)
	if err != nil {
		_ = module.Close()
		return nil, err
	}
	values := make([]float64, int(config.PromptLength*embedding.Dim))
	scale := 1 / math.Sqrt(float64(embedding.Dim))
	for index := range values {
		values[index] = config.RNG.NormFloat64() * scale
	}
	if err := writeFloatValues(value, values); err != nil {
		_ = value.Close()
		_ = module.Close()
		return nil, err
	}
	parameter, err := nn.NewParameter("prompt", value, true)
	_ = value.Close()
	if err != nil {
		_ = module.Close()
		return nil, err
	}
	if err := module.RegisterParameter("prompt", parameter); err != nil {
		_ = parameter.Close()
		_ = module.Close()
		return nil, err
	}
	return &PromptTuning{ModuleBase: module, Embedding: embedding, Prompt: parameter}, nil
}

// Forward menghasilkan [batch, promptLength+time, dim] dari token int64.
func (p *PromptTuning) Forward(ctx *nn.Context, tokens *autograd.Variable) (*autograd.Variable, error) {
	if p == nil || p.ModuleBase == nil || p.Embedding == nil || ctx == nil {
		return nil, errs.ErrInvalidArgument
	}
	embedded, err := p.Embedding.Forward(ctx, tokens)
	if err != nil {
		return nil, err
	}
	prompt, err := ctx.Variable(p.Prompt)
	if err != nil {
		return nil, err
	}
	return nn.ConcatTimePrefix(ctx, prompt, embedded)
}

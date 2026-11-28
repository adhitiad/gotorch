package transformer

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/adhitiad/gotorch/src/autograd"
	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/nn"
)

// AttentionConfig mengonfigurasi attention. KeyValueHeads=1 berarti MQA,
// KeyValueHeads=QueryHeads berarti MHA, dan nilai di antaranya berarti GQA.
type AttentionConfig struct {
	ModelDim      int64
	QueryHeads    int64
	KeyValueHeads int64
	HeadDim       int64
	Causal        bool
	DType         dtype.DType
	RNG           *rand.Rand
}

// MultiHeadAttention mencakup MHA, GQA, dan MQA dalam satu implementasi.
type MultiHeadAttention struct {
	*nn.ModuleBase
	Query      Projector
	Key        Projector
	Value      Projector
	Output     Projector
	QueryHeads int64
	KVHeads    int64
	HeadDim    int64
	Causal     bool
}

// NewMultiHeadAttention membuat proyeksi q/k/v/o dengan RNG deterministik.
func NewMultiHeadAttention(name string, config AttentionConfig) (*MultiHeadAttention, error) {
	base, err := nn.NewModule(name)
	if err != nil {
		return nil, err
	}
	queryHeads, kvHeads := config.QueryHeads, config.KeyValueHeads
	if kvHeads == 0 {
		kvHeads = queryHeads
	}
	elementType := resolveDType(config.DType)
	rng := config.RNG
	if rng == nil {
		rng = nn.NewRNG(1)
	}
	if config.ModelDim <= 0 || queryHeads <= 0 || kvHeads <= 0 || kvHeads > queryHeads ||
		queryHeads%kvHeads != 0 || config.HeadDim <= 0 || config.ModelDim != queryHeads*config.HeadDim {
		_ = base.Close()
		return nil, fmt.Errorf("%w: invalid attention configuration for %q", errs.ErrInvalidShape, name)
	}
	makeLinear := func(suffix string, in, out int64) (Projector, error) {
		return nn.NewLinear(suffix, nn.LinearConfig{
			In: in, Out: out, DType: elementType, Bias: false, RNG: rng,
		})
	}
	queryProjection, err := makeLinear("query", config.ModelDim, queryHeads*config.HeadDim)
	if err != nil {
		_ = base.Close()
		return nil, err
	}
	keyProjection, err := makeLinear("key", config.ModelDim, kvHeads*config.HeadDim)
	if err != nil {
		_ = base.Close()
		return nil, err
	}
	valueProjection, err := makeLinear("value", config.ModelDim, kvHeads*config.HeadDim)
	if err != nil {
		_ = base.Close()
		return nil, err
	}
	outputProjection, err := makeLinear("output", queryHeads*config.HeadDim, config.ModelDim)
	if err != nil {
		_ = base.Close()
		return nil, err
	}
	for _, child := range []nn.Module{queryProjection, keyProjection, valueProjection, outputProjection} {
		if err := base.RegisterModule(child); err != nil {
			_ = base.Close()
			return nil, err
		}
	}
	return &MultiHeadAttention{
		ModuleBase: base,
		Query:      queryProjection,
		Key:        keyProjection,
		Value:      valueProjection,
		Output:     outputProjection,
		QueryHeads: queryHeads,
		KVHeads:    kvHeads,
		HeadDim:    config.HeadDim,
		Causal:     config.Causal,
	}, nil
}

// Forward menjalankan attention pada input [batch, time, modelDim].
// paddingMask berisi 1 untuk token aktif dan 0 untuk padding; nil berarti tanpa mask.
func (a *MultiHeadAttention) Forward(ctx *nn.Context, query, key, value *autograd.Variable, paddingMask []int64) (*autograd.Variable, error) {
	if a == nil || a.ModuleBase == nil || ctx == nil {
		return nil, errs.ErrInvalidArgument
	}
	queryShape, queryType, err := sequenceShape(query, a.ModelDim())
	if err != nil {
		return nil, err
	}
	keyShape, _, err := sequenceShape(key, a.ModelDim())
	if err != nil {
		return nil, err
	}
	valueShape, _, err := sequenceShape(value, a.ModelDim())
	if err != nil {
		return nil, err
	}
	if keyShape[0] != queryShape[0] || valueShape[0] != queryShape[0] ||
		valueShape[1] != keyShape[1] {
		return nil, errs.ErrGradientShapeMismatch
	}
	if len(paddingMask) > 0 && int64(len(paddingMask)) != queryShape[0]*keyShape[1] {
		return nil, fmt.Errorf("%w: padding mask has %d values, want %d", errs.ErrInvalidShape, len(paddingMask), queryShape[0]*keyShape[1])
	}
	batch, queryLength := queryShape[0], queryShape[1]
	queryHeads, kvHeads, headDim := a.QueryHeads, a.KVHeads, a.HeadDim
	project := func(layer Projector, input *autograd.Variable, length int64, heads int64) (*autograd.Variable, error) {
		flat, err := nn.Reshape(ctx, input, batch*length, a.ModelDim())
		if err != nil {
			return nil, err
		}
		projected, err := layer.Forward(ctx, flat)
		if err != nil {
			return nil, err
		}
		shaped, err := nn.Reshape(ctx, projected, batch, length, heads, headDim)
		if err != nil {
			return nil, err
		}
		return nn.Transpose(ctx, shaped, 0, 2, 1, 3)
	}
	queryHeadsVar, err := project(a.Query, query, queryLength, queryHeads)
	if err != nil {
		return nil, err
	}
	keyHeadsVar, err := project(a.Key, key, keyShape[1], kvHeads)
	if err != nil {
		return nil, err
	}
	valueHeadsVar, err := project(a.Value, value, keyShape[1], kvHeads)
	if err != nil {
		return nil, err
	}
	mask, err := a.attentionMask(ctx, batch, queryLength, keyShape[1], paddingMask, queryType)
	if err != nil {
		return nil, err
	}
	attended, err := nn.Attention(ctx, queryHeadsVar, keyHeadsVar, valueHeadsVar, mask)
	if err != nil {
		return nil, err
	}
	restored, err := nn.Transpose(ctx, attended, 0, 2, 1, 3)
	if err != nil {
		return nil, err
	}
	flat, err := nn.Reshape(ctx, restored, batch*queryLength, queryHeads*headDim)
	if err != nil {
		return nil, err
	}
	output, err := a.Output.Forward(ctx, flat)
	if err != nil {
		return nil, err
	}
	return nn.Reshape(ctx, output, batch, queryLength, a.ModelDim())
}

func (a *MultiHeadAttention) attentionMask(ctx *nn.Context, batch, queryLength, keyLength int64, padding []int64, elementType dtype.DType) (*autograd.Variable, error) {
	if !a.Causal && len(padding) == 0 {
		return nil, nil
	}
	values := make([]float64, int(batch*queryLength*keyLength))
	for row := 0; row < int(batch); row++ {
		for queryIndex := 0; queryIndex < int(queryLength); queryIndex++ {
			for keyIndex := 0; keyIndex < int(keyLength); keyIndex++ {
				index := (row*int(queryLength)+queryIndex)*int(keyLength) + keyIndex
				allowed := true
				if len(padding) > 0 && padding[row*int(keyLength)+keyIndex] == 0 {
					allowed = false
				}
				if a.Causal && keyIndex > queryIndex {
					allowed = false
				}
				if !allowed {
					values[index] = math.Inf(-1)
				}
			}
		}
	}
	value, err := floatTensor(elementType, []int64{batch, queryLength, keyLength}, values)
	if err != nil {
		return nil, err
	}
	return variableFromTensor(ctx, value, false)
}

// ModelDim mengembalikan dimensi model dari proyeksi keluaran.
func (a *MultiHeadAttention) ModelDim() int64 {
	if a == nil || a.Output == nil {
		return 0
	}
	return a.Output.OutputDim()
}

func sequenceShape(variable *autograd.Variable, modelDim int64) ([]int64, dtype.DType, error) {
	if variable == nil {
		return nil, dtype.Invalid, errs.ErrInvalidArgument
	}
	value, err := variable.Value()
	if err != nil {
		return nil, dtype.Invalid, err
	}
	defer value.Close()
	dims, elementType, err := tensorDims(value)
	if err != nil {
		return nil, dtype.Invalid, err
	}
	if err := requireShape("attention input", dims, -1, -1, modelDim); err != nil {
		return nil, dtype.Invalid, err
	}
	if dims[0] <= 0 || dims[1] <= 0 {
		return nil, dtype.Invalid, errs.ErrInvalidShape
	}
	return dims, elementType, nil
}

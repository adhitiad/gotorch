package transformer

import (
	"errors"
	"math"
	"testing"

	"github.com/adhitiad/gotorch/src/autograd"
	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/nn"
)

func takePositions(values []float64, dim int, positions ...int) []float64 {
	result := make([]float64, 0, len(positions)*dim)
	for _, position := range positions {
		result = append(result, values[position*dim:(position+1)*dim]...)
	}
	return result
}

func assertAnyDifference(t *testing.T, label string, left, right []float64, epsilon float64) {
	t.Helper()
	for index := range left {
		if math.Abs(left[index]-right[index]) > epsilon {
			return
		}
	}
	t.Fatalf("%s: kedua keluaran identik", label)
}

func smallBlockConfig() BlockConfig {
	return BlockConfig{
		Attention: AttentionConfig{ModelDim: 2, QueryHeads: 1, HeadDim: 2, DType: dtype.Float64, RNG: nn.NewRNG(29)},
		HiddenDim: 3,
		Norm:      NormLayer,
	}
}

func TestEncoderBlockContractAndPadding(t *testing.T) {
	config := smallBlockConfig()
	config.HiddenDim = 6
	config.Attention.ModelDim = 4
	config.Attention.QueryHeads = 2
	config.Attention.HeadDim = 2
	block, err := NewEncoderBlock("encoder", config)
	if err != nil {
		t.Fatalf("NewEncoderBlock: %v", err)
	}
	defer block.Close()
	named := block.Parameters()
	if len(named) == 0 || named[0].Name == "" {
		t.Fatalf("parameter blok kosong")
	}

	inputs := []float64{1, 0.5, -0.5, 2, 0.2, 1.5, -1, 0.7, 9, 9, 9, 9}
	perturbed := append([]float64(nil), inputs...)
	perturbed[8], perturbed[9], perturbed[10], perturbed[11] = -3, 4, 1, 2

	run := func(values []float64) []float64 {
		tape, ctx := newTestTape(t)
		defer tape.Close()
		input := newVariable(t, tape, newFloatTensor(t, dtype.Float64, []int64{1, 3, 4}, values), false)
		output, err := block.Forward(ctx, input, []int64{1, 1, 0})
		if err != nil {
			t.Fatalf("Forward: %v", err)
		}
		assertVariableDims(t, "encoder", output, 1, 3, 4)
		assertVariableDType(t, "encoder", output, dtype.Float64)
		return variableValues(t, output)
	}
	base := run(inputs)
	changed := run(perturbed)
	assertFloatClose(t, "token aktif tidak boleh berubah", takePositions(base, 4, 0, 1), takePositions(changed, 4, 0, 1), 1e-12)

	tape, ctx := newTestTape(t)
	defer tape.Close()
	input := newVariable(t, tape, newFloatTensor(t, dtype.Float64, []int64{1, 3, 4}, inputs), false)
	if _, err := block.Forward(ctx, input, []int64{1, 1}); !errors.Is(err, errs.ErrInvalidShape) {
		t.Fatalf("mask pendek: error %v", err)
	}

	badNorm := config
	badNorm.Norm = "unknown"
	if _, err := NewEncoderBlock("bad_norm", badNorm); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("norm tidak dikenal: error %v", err)
	}
	badFFN := config
	badFFN.FeedForward = "unknown"
	if _, err := NewEncoderBlock("bad_ffn", badFFN); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("feed-forward tidak dikenal: error %v", err)
	}

	small, err := NewEncoderBlock("small", smallBlockConfig())
	if err != nil {
		t.Fatalf("NewEncoderBlock kecil: %v", err)
	}
	defer small.Close()
	result := checkTransformerGradient(t, newFloatTensor(t, dtype.Float64, []int64{1, 2, 2}, []float64{0.4, -0.7, 0.9, 0.2}),
		func(ctx *nn.Context, leaf *autograd.Variable) (*autograd.Variable, error) {
			return small.Forward(ctx, leaf, []int64{1, 1})
		})
	if result.CheckedElements != 4 {
		t.Fatalf("elemen gradient: %d, want 4", result.CheckedElements)
	}
}

func TestDecoderBlockContractAndCausality(t *testing.T) {
	config := smallBlockConfig()
	config.Attention.ModelDim = 4
	config.Attention.QueryHeads = 2
	config.Attention.HeadDim = 2
	config.Attention.Causal = true
	config.HiddenDim = 6
	config.Norm = NormRMS
	config.FeedForward = FeedForwardSwiGLU
	block, err := NewDecoderBlock("decoder", config)
	if err != nil {
		t.Fatalf("NewDecoderBlock: %v", err)
	}
	defer block.Close()
	if !block.SelfAttention.Causal || block.CrossAttention.Causal {
		t.Fatalf("konfigurasi kausal: self=%v cross=%v", block.SelfAttention.Causal, block.CrossAttention.Causal)
	}

	inputs := []float64{1, 0.5, -0.5, 2, 0.2, 1.5, -1, 0.7, 9, 9, 9, 9}
	perturbed := append([]float64(nil), inputs...)
	perturbed[8], perturbed[9], perturbed[10], perturbed[11] = -3, 4, 1, 2
	memory := newFloatTensor(t, dtype.Float64, []int64{1, 4, 4}, []float64{
		0.5, 1, -1, 0.2, 0.3, -0.4, 1.2, 0.8, -0.7, 0.1, 0.6, -0.2, 1.1, -0.5, 0.4, 0.9,
	})
	defer memory.Close()

	run := func(values []float64, crossPadding []int64) []float64 {
		tape, ctx := newTestTape(t)
		defer tape.Close()
		input := newVariable(t, tape, newFloatTensor(t, dtype.Float64, []int64{1, 3, 4}, values), false)
		memoryVariable := newVariable(t, tape, memory, false)
		output, err := block.Forward(ctx, input, memoryVariable, []int64{1, 1, 0}, crossPadding)
		if err != nil {
			t.Fatalf("Forward: %v", err)
		}
		assertVariableDims(t, "decoder", output, 1, 3, 4)
		assertVariableDType(t, "decoder", output, dtype.Float64)
		return variableValues(t, output)
	}
	base := run(inputs, []int64{1, 1, 1, 0})
	changed := run(perturbed, []int64{1, 1, 1, 0})
	assertFloatClose(t, "kausalitas self-attention", takePositions(base, 4, 0, 1), takePositions(changed, 4, 0, 1), 1e-12)
	maskedMemory := run(inputs, []int64{1, 1, 0, 0})
	assertAnyDifference(t, "cross-attention harus memakai memory", base, maskedMemory, 1e-9)

	tape, ctx := newTestTape(t)
	defer tape.Close()
	input := newVariable(t, tape, newFloatTensor(t, dtype.Float64, []int64{1, 3, 4}, inputs), false)
	memoryVariable := newVariable(t, tape, memory, false)
	if _, err := block.Forward(ctx, input, memoryVariable, []int64{1, 1}, nil); !errors.Is(err, errs.ErrInvalidShape) {
		t.Fatalf("self mask pendek: error %v", err)
	}
	if _, err := block.Forward(ctx, input, memoryVariable, nil, []int64{1, 1}); !errors.Is(err, errs.ErrInvalidShape) {
		t.Fatalf("cross mask pendek: error %v", err)
	}
	withoutMemory, err := block.Forward(ctx, input, nil, nil, nil)
	if err != nil {
		t.Fatalf("memory nil: %v", err)
	}
	assertVariableDims(t, "tanpa memory", withoutMemory, 1, 3, 4)

	small := smallBlockConfig()
	small.FeedForward = FeedForwardSwiGLU
	smallDecoder, err := NewDecoderBlock("small_decoder", small)
	if err != nil {
		t.Fatalf("NewDecoderBlock kecil: %v", err)
	}
	defer smallDecoder.Close()
	result := checkTransformerGradient(t, newFloatTensor(t, dtype.Float64, []int64{1, 2, 2}, []float64{0.3, -0.6, 0.8, 0.1}),
		func(ctx *nn.Context, leaf *autograd.Variable) (*autograd.Variable, error) {
			return smallDecoder.Forward(ctx, leaf, nil, nil, nil)
		})
	if result.CheckedElements != 4 {
		t.Fatalf("elemen gradient: %d, want 4", result.CheckedElements)
	}
}

func TestEncoderStackContractAndDeterminism(t *testing.T) {
	config := smallBlockConfig()
	config.Attention.QueryHeads = 1
	config.Attention.HeadDim = 2
	stack, err := NewEncoderStack("stack", config, 2)
	if err != nil {
		t.Fatalf("NewEncoderStack: %v", err)
	}
	defer stack.Close()

	run := func() []float64 {
		tape, ctx := newTestTape(t)
		defer tape.Close()
		input := newVariable(t, tape, newFloatTensor(t, dtype.Float64, []int64{1, 2, 2}, []float64{0.5, -0.2, 0.9, 0.3}), false)
		output, err := stack.Forward(ctx, input, []int64{1, 1})
		if err != nil {
			t.Fatalf("Forward: %v", err)
		}
		assertVariableDims(t, "stack", output, 1, 2, 2)
		assertVariableDType(t, "stack", output, dtype.Float64)
		return variableValues(t, output)
	}
	first := run()
	second := run()
	assertFloatClose(t, "deterministik", first, second, 0)

	if _, err := NewEncoderStack("invalid", config, 0); !errors.Is(err, errs.ErrInvalidShape) {
		t.Fatalf("jumlah lapisan nol: error %v", err)
	}
}

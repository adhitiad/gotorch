package peft

import (
	"math/rand"
	"testing"

	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/nn"
	"github.com/adhitiad/gotorch/src/tensor"
	"github.com/adhitiad/gotorch/src/transformer"
)

func newEmbeddingFixture(t *testing.T) *transformer.TokenEmbedding {
	t.Helper()
	embedding, err := transformer.NewTokenEmbedding("embedding", 4, 2, dtype.Float64, rand.New(rand.NewSource(3)))
	if err != nil {
		t.Fatalf("NewTokenEmbedding: %v", err)
	}
	t.Cleanup(func() {
		if err := embedding.Close(); err != nil {
			t.Errorf("embedding.Close: %v", err)
		}
	})
	setParameterValues(t, embedding.Weight, []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8})
	if err := Freeze(embedding); err != nil {
		t.Fatalf("Freeze(embedding): %v", err)
	}
	return embedding
}

func newPromptTuning(t *testing.T, name string, embedding *transformer.TokenEmbedding, length int64) *PromptTuning {
	t.Helper()
	module, err := NewPromptTuning(name, embedding, PromptTuningConfig{
		PromptLength: length, DType: dtype.Float64, RNG: rand.New(rand.NewSource(11)),
	})
	if err != nil {
		t.Fatalf("NewPromptTuning(%q): %v", name, err)
	}
	t.Cleanup(func() {
		if err := module.Close(); err != nil {
			t.Errorf("module.Close(%q): %v", name, err)
		}
	})
	return module
}

func intTensor(t *testing.T, dims []int64, values []int64) *tensor.Tensor {
	t.Helper()
	value, err := tensor.NewTensor(dtype.Int64, dims...)
	if err != nil {
		t.Fatalf("NewTensor(int64): %v", err)
	}
	t.Cleanup(func() { _ = value.Close() })
	if err := tensor.Write(value, values); err != nil {
		t.Fatalf("tensor.Write(int64): %v", err)
	}
	return value
}

func TestPromptTuningForwardAndGradient(t *testing.T) {
	embedding := newEmbeddingFixture(t)
	prompt := newPromptTuning(t, "prompt", embedding, 2)
	setParameterValues(t, prompt.Prompt, []float64{0.9, -0.9, 0.4, -0.4})

	tape, ctx := newTestTape(t)
	tokens := newVariable(t, tape, intTensor(t, []int64{2, 2}, []int64{0, 1, 2, 3}), false)
	output, err := prompt.Forward(ctx, tokens)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	value, err := output.Value()
	if err != nil {
		t.Fatalf("output.Value: %v", err)
	}
	assertShape(t, value, []int64{2, 4, 2})
	assertClose(t, "forward prompt tuning", floatValues(t, value), []float64{
		0.9, -0.9, 0.4, -0.4, 0.1, 0.2, 0.3, 0.4,
		0.9, -0.9, 0.4, -0.4, 0.5, 0.6, 0.7, 0.8,
	}, 1e-12)
	_ = value.Close()

	backwardSum(t, tape, output)
	gradient, err := gradientValues(t, ctx, prompt.Prompt)
	if err != nil {
		t.Fatalf("gradien prompt: %v", err)
	}
	assertClose(t, "gradien prompt", gradient, []float64{2, 2, 2, 2}, 1e-12)
	assertNoGradient(t, ctx, embedding.Weight)
}

func TestPromptTuningValidationAndDeterministicInit(t *testing.T) {
	embedding := newEmbeddingFixture(t)
	first := newPromptTuning(t, "prompt_pertama", embedding, 3)
	second := newPromptTuning(t, "prompt_kedua", embedding, 3)
	assertClose(t, "inisialisasi deterministik", parameterValues(t, first.Prompt), parameterValues(t, second.Prompt), 0)
	if !first.Prompt.RequiresGrad() {
		t.Fatal("prompt seharusnya trainable")
	}

	config := PromptTuningConfig{PromptLength: 2, DType: dtype.Float64, RNG: rand.New(rand.NewSource(11))}
	_, err := NewPromptTuning("prompt_nil", nil, config)
	assertErrorIs(t, err, errs.ErrInvalidArgument)
	_, err = NewPromptTuning("prompt_panjang", embedding, PromptTuningConfig{PromptLength: 0, DType: dtype.Float64, RNG: rand.New(rand.NewSource(11))})
	assertErrorIs(t, err, errs.ErrInvalidArgument)
	_, err = NewPromptTuning("prompt_rng", embedding, PromptTuningConfig{PromptLength: 2, DType: dtype.Float64})
	assertErrorIs(t, err, errs.ErrInvalidArgument)
	_, err = NewPromptTuning("prompt_dtype", embedding, PromptTuningConfig{PromptLength: 2, DType: dtype.Int64, RNG: rand.New(rand.NewSource(11))})
	assertErrorIs(t, err, errs.ErrUnsupportedDType)
	_, err = NewPromptTuning("", embedding, config)
	assertErrorIs(t, err, errs.ErrInvalidArgument)

	prompt := newPromptTuning(t, "prompt_salah", embedding, 2)
	tape, ctx := newTestTape(t)
	tokens := newVariable(t, tape, intTensor(t, []int64{2}, []int64{0, 1}), false)
	_, err = prompt.Forward(ctx, tokens)
	assertErrorIs(t, err, errs.ErrInvalidShape)
}

func TestPrefixTuningPrototypeUsesExistingContracts(t *testing.T) {
	attention, err := transformer.NewMultiHeadAttention("attention", transformer.AttentionConfig{
		ModelDim: 2, QueryHeads: 1, KeyValueHeads: 1, HeadDim: 2,
		Causal: false, DType: dtype.Float64, RNG: rand.New(rand.NewSource(13)),
	})
	if err != nil {
		t.Fatalf("NewMultiHeadAttention: %v", err)
	}
	t.Cleanup(func() {
		if err := attention.Close(); err != nil {
			t.Errorf("attention.Close: %v", err)
		}
	})

	prefixValue, err := tensor.NewTensor(dtype.Float64, 2, attention.Key.OutputDim())
	if err != nil {
		t.Fatalf("NewTensor(prefix): %v", err)
	}
	t.Cleanup(func() { _ = prefixValue.Close() })
	if err := tensor.Write(prefixValue, []float64{0.2, -0.1, 0.3, 0.4}); err != nil {
		t.Fatalf("tensor.Write(prefix): %v", err)
	}
	prefixParameter, err := nn.NewParameter("prefix", prefixValue, true)
	if err != nil {
		t.Fatalf("NewParameter(prefix): %v", err)
	}
	t.Cleanup(func() { _ = prefixParameter.Close() })

	tape, ctx := newTestTape(t)
	sequence := newVariable(t, tape, newFloatTensor(t, []int64{1, 2, 2}, []float64{1, 0, 0, 1}), false)
	prefix, err := ctx.Variable(prefixParameter)
	if err != nil {
		t.Fatalf("ctx.Variable(prefix): %v", err)
	}
	prefixKey, err := nn.ConcatTimePrefix(ctx, prefix, sequence)
	if err != nil {
		t.Fatalf("ConcatTimePrefix(key): %v", err)
	}
	prefixedValue, err := nn.ConcatTimePrefix(ctx, prefix, sequence)
	if err != nil {
		t.Fatalf("ConcatTimePrefix(value): %v", err)
	}
	output, err := attention.Forward(ctx, sequence, prefixKey, prefixedValue, nil)
	if err != nil {
		t.Fatalf("attention.Forward: %v", err)
	}
	result, err := output.Value()
	if err != nil {
		t.Fatalf("output.Value: %v", err)
	}
	assertShape(t, result, []int64{1, 2, 2})
	_ = result.Close()

	backwardSum(t, tape, output)
	gradient, err := ctx.Gradient(prefixParameter)
	if err != nil {
		t.Fatalf("gradien prefix: %v", err)
	}
	assertShape(t, gradient, []int64{2, 2})
	values := floatValues(t, gradient)
	_ = gradient.Close()
	nonzero := false
	for _, value := range values {
		if value != 0 {
			nonzero = true
			break
		}
	}
	if !nonzero {
		t.Fatal("gradien prefix tidak boleh seluruhnya nol")
	}
}

package transformer

import (
	"errors"
	"math"
	"testing"

	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/nn"
)

func identityValues(rows, columns int64) []float64 {
	values := make([]float64, rows*columns)
	limit := int(rows)
	if int(columns) < limit {
		limit = int(columns)
	}
	for index := 0; index < limit; index++ {
		values[index*int(columns)+index] = 1
	}
	return values
}

func useIdentityProjections(t *testing.T, attention *MultiHeadAttention, modelDim, queryHeads, kvHeads, headDim int64) {
	t.Helper()
	setProjectorWeights(t, attention.Query, identityValues(queryHeads*headDim, modelDim))
	setProjectorWeights(t, attention.Key, identityValues(kvHeads*headDim, modelDim))
	setProjectorWeights(t, attention.Value, identityValues(kvHeads*headDim, modelDim))
	setProjectorWeights(t, attention.Output, identityValues(modelDim, queryHeads*headDim))
}

// referenceAttentionIdentity menghitung attention langsung dari proyeksi identitas.
func referenceAttentionIdentity(values []float64, batch, length, heads, kvHeads, headDim int, mask []int64, causal bool) []float64 {
	result := make([]float64, len(values))
	group := heads / kvHeads
	scale := 1 / math.Sqrt(float64(headDim))
	for row := 0; row < batch; row++ {
		for head := 0; head < heads; head++ {
			keyHead := head / group
			for queryIndex := 0; queryIndex < length; queryIndex++ {
				scores := make([]float64, length)
				for keyIndex := 0; keyIndex < length; keyIndex++ {
					allowed := !(causal && keyIndex > queryIndex)
					if len(mask) > 0 && mask[row*length+keyIndex] == 0 {
						allowed = false
					}
					if !allowed {
						scores[keyIndex] = math.Inf(-1)
						continue
					}
					var dot float64
					for column := 0; column < headDim; column++ {
						queryAt := (row*length+queryIndex)*heads*headDim + head*headDim + column
						keyAt := (row*length+keyIndex)*heads*headDim + keyHead*headDim + column
						dot += values[queryAt] * values[keyAt]
					}
					scores[keyIndex] = dot * scale
				}
				probabilities := referenceSoftmax(scores)
				for keyIndex := 0; keyIndex < length; keyIndex++ {
					for column := 0; column < headDim; column++ {
						valueAt := (row*length+keyIndex)*heads*headDim + keyHead*headDim + column
						outputAt := (row*length+queryIndex)*heads*headDim + head*headDim + column
						result[outputAt] += probabilities[keyIndex] * values[valueAt]
					}
				}
			}
		}
	}
	return result
}

func TestMultiHeadAttentionConfigValidation(t *testing.T) {
	cases := []struct {
		name   string
		config AttentionConfig
	}{
		{"model dim", AttentionConfig{ModelDim: 3, QueryHeads: 2, HeadDim: 2, DType: dtype.Float64}},
		{"head dim", AttentionConfig{ModelDim: 4, QueryHeads: 2, HeadDim: 0, DType: dtype.Float64}},
		{"kv heads lebih besar", AttentionConfig{ModelDim: 4, QueryHeads: 2, KeyValueHeads: 3, HeadDim: 2, DType: dtype.Float64}},
		{"kv heads bukan pembagi", AttentionConfig{ModelDim: 6, QueryHeads: 3, KeyValueHeads: 2, HeadDim: 2, DType: dtype.Float64}},
	}
	for _, item := range cases {
		t.Run(item.name, func(t *testing.T) {
			if _, err := NewMultiHeadAttention(item.name, item.config); !errors.Is(err, errs.ErrInvalidShape) {
				t.Fatalf("konfigurasi ditolak dengan error %v", err)
			}
		})
	}
}

func TestMultiHeadAttentionShapeContract(t *testing.T) {
	for _, elementType := range []dtype.DType{dtype.Float64, dtype.Float32} {
		t.Run(elementType.String(), func(t *testing.T) {
			attention, err := NewMultiHeadAttention("attention", AttentionConfig{
				ModelDim: 4, QueryHeads: 2, HeadDim: 2, DType: elementType, RNG: nn.NewRNG(7),
			})
			if err != nil {
				t.Fatalf("NewMultiHeadAttention: %v", err)
			}
			defer attention.Close()

			tape, ctx := newTestTape(t)
			defer tape.Close()
			query := newVariable(t, tape, newFloatTensor(t, elementType, []int64{2, 3, 4}, make([]float64, 24)), false)
			key := newVariable(t, tape, newFloatTensor(t, elementType, []int64{2, 5, 4}, make([]float64, 40)), false)
			value := newVariable(t, tape, newFloatTensor(t, elementType, []int64{2, 5, 4}, make([]float64, 40)), false)
			output, err := attention.Forward(ctx, query, key, value, nil)
			if err != nil {
				t.Fatalf("cross attention: %v", err)
			}
			assertVariableDims(t, "cross attention", output, 2, 3, 4)
			assertVariableDType(t, "cross attention", output, elementType)

			wrongRank := newVariable(t, tape, newFloatTensor(t, elementType, []int64{1, 4}, make([]float64, 4)), false)
			if _, err := attention.Forward(ctx, wrongRank, wrongRank, wrongRank, nil); !errors.Is(err, errs.ErrInvalidShape) {
				t.Fatalf("rank salah: error %v", err)
			}
			wrongDim := newVariable(t, tape, newFloatTensor(t, elementType, []int64{1, 3, 5}, make([]float64, 15)), false)
			if _, err := attention.Forward(ctx, wrongDim, wrongDim, wrongDim, nil); !errors.Is(err, errs.ErrInvalidShape) {
				t.Fatalf("dimensi model salah: error %v", err)
			}
			wrongBatch := newVariable(t, tape, newFloatTensor(t, elementType, []int64{3, 5, 4}, make([]float64, 60)), false)
			if _, err := attention.Forward(ctx, query, wrongBatch, value, nil); !errors.Is(err, errs.ErrGradientShapeMismatch) {
				t.Fatalf("batch key berbeda: error %v", err)
			}
			wrongValueLength := newVariable(t, tape, newFloatTensor(t, elementType, []int64{2, 4, 4}, make([]float64, 32)), false)
			if _, err := attention.Forward(ctx, query, key, wrongValueLength, nil); !errors.Is(err, errs.ErrGradientShapeMismatch) {
				t.Fatalf("panjang value berbeda: error %v", err)
			}
			mask := []int64{1, 1, 1, 1}
			if _, err := attention.Forward(ctx, query, key, value, mask); !errors.Is(err, errs.ErrInvalidShape) {
				t.Fatalf("panjang mask salah: error %v", err)
			}
			if attention.ModelDim() != 4 {
				t.Fatalf("ModelDim: %d, want 4", attention.ModelDim())
			}
		})
	}
}

func TestMultiHeadAttentionMasksNumerically(t *testing.T) {
	inputs := []float64{1, 0, 0, 1, 1, 1}
	cases := []struct {
		name   string
		causal bool
		mask   []int64
	}{
		{name: "tanpa mask"},
		{name: "kausal", causal: true},
		{name: "padding", mask: []int64{1, 1, 0}},
		{name: "kausal dan padding", causal: true, mask: []int64{1, 1, 0}},
	}
	for _, item := range cases {
		t.Run(item.name, func(t *testing.T) {
			attention, err := NewMultiHeadAttention("attention", AttentionConfig{
				ModelDim: 2, QueryHeads: 1, HeadDim: 2, Causal: item.causal, DType: dtype.Float64, RNG: nn.NewRNG(3),
			})
			if err != nil {
				t.Fatalf("NewMultiHeadAttention: %v", err)
			}
			defer attention.Close()
			useIdentityProjections(t, attention, 2, 1, 1, 2)

			tape, ctx := newTestTape(t)
			defer tape.Close()
			input := newVariable(t, tape, newFloatTensor(t, dtype.Float64, []int64{1, 3, 2}, inputs), false)
			output, err := attention.Forward(ctx, input, input, input, item.mask)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}
			want := referenceAttentionIdentity(inputs, 1, 3, 1, 1, 2, item.mask, item.causal)
			assertFloatClose(t, "attention", variableValues(t, output), want, 1e-12)
		})
	}
}

func TestMultiHeadAttentionBatchValues(t *testing.T) {
	attention, err := NewMultiHeadAttention("attention", AttentionConfig{
		ModelDim: 4, QueryHeads: 2, KeyValueHeads: 1, HeadDim: 2, DType: dtype.Float64, RNG: nn.NewRNG(11),
	})
	if err != nil {
		t.Fatalf("NewMultiHeadAttention: %v", err)
	}
	defer attention.Close()
	useIdentityProjections(t, attention, 4, 2, 1, 2)
	inputs := []float64{
		1, 0, 0, 1, 0.5, 1, -0.5, 0.2,
		-1, 0.4, 0.7, -0.3, 0.2, -0.6, 1.1, 0.8,
	}
	mask := []int64{1, 1, 0, 1}
	tape, ctx := newTestTape(t)
	defer tape.Close()
	input := newVariable(t, tape, newFloatTensor(t, dtype.Float64, []int64{2, 2, 4}, inputs), false)
	output, err := attention.Forward(ctx, input, input, input, mask)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	assertVariableDims(t, "output", output, 2, 2, 4)
	want := referenceAttentionIdentity(inputs, 2, 2, 2, 1, 2, mask, false)
	assertFloatClose(t, "batch attention", variableValues(t, output), want, 1e-12)
}

func TestMultiHeadAttentionGQAAndMQA(t *testing.T) {
	cases := []struct {
		name          string
		queryHeads    int64
		keyValueHeads int64
	}{
		{name: "MHA", queryHeads: 4, keyValueHeads: 4},
		{name: "GQA", queryHeads: 4, keyValueHeads: 2},
		{name: "MQA", queryHeads: 4, keyValueHeads: 1},
	}
	for _, item := range cases {
		t.Run(item.name, func(t *testing.T) {
			modelDim := item.queryHeads
			attention, err := NewMultiHeadAttention("attention", AttentionConfig{
				ModelDim: modelDim, QueryHeads: item.queryHeads, KeyValueHeads: item.keyValueHeads,
				HeadDim: 1, DType: dtype.Float64, RNG: nn.NewRNG(5),
			})
			if err != nil {
				t.Fatalf("NewMultiHeadAttention: %v", err)
			}
			defer attention.Close()
			if attention.QueryHeads != item.queryHeads || attention.KVHeads != item.keyValueHeads {
				t.Fatalf("heads: %d/%d, want %d/%d", attention.QueryHeads, attention.KVHeads, item.queryHeads, item.keyValueHeads)
			}
			keyLinear := attention.Key.(*nn.Linear)
			if keyLinear.InputDim() != modelDim || keyLinear.OutputDim() != item.keyValueHeads {
				t.Fatalf("key projection: %d -> %d", keyLinear.InputDim(), keyLinear.OutputDim())
			}
			useIdentityProjections(t, attention, modelDim, item.queryHeads, item.keyValueHeads, 1)

			inputs := []float64{1, 0.5, -0.5, 2, 0.2, 1.5, -1, 0.7}
			tape, ctx := newTestTape(t)
			defer tape.Close()
			input := newVariable(t, tape, newFloatTensor(t, dtype.Float64, []int64{1, 2, modelDim}, inputs), false)
			output, err := attention.Forward(ctx, input, input, input, nil)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}
			assertVariableDims(t, "output", output, 1, 2, modelDim)
			want := referenceAttentionIdentity(inputs, 1, 2, int(item.queryHeads), int(item.keyValueHeads), 1, nil, false)
			assertFloatClose(t, "GQA/MQA values", variableValues(t, output), want, 1e-12)
		})
	}
}

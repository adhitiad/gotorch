package transformer

import (
	"testing"

	"github.com/adhitiad/gotorch/src/autograd"
	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/nn"
)

func TestMultiHeadAttentionGradientInput(t *testing.T) {
	attention, err := NewMultiHeadAttention("attention", AttentionConfig{
		ModelDim: 2, QueryHeads: 1, HeadDim: 2, Causal: true, DType: dtype.Float64, RNG: nn.NewRNG(13),
	})
	if err != nil {
		t.Fatalf("NewMultiHeadAttention: %v", err)
	}
	defer attention.Close()
	useIdentityProjections(t, attention, 2, 1, 1, 2)

	result := checkTransformerGradient(t,
		newFloatTensor(t, dtype.Float64, []int64{1, 2, 2}, []float64{0.4, -0.7, 0.9, 0.2}),
		func(ctx *nn.Context, leaf *autograd.Variable) (*autograd.Variable, error) {
			return attention.Forward(ctx, leaf, leaf, leaf, []int64{1, 1})
		})
	if result.CheckedElements != 4 {
		t.Fatalf("elemen gradient: %d, want 4", result.CheckedElements)
	}
}

func TestMultiHeadAttentionGradientWeights(t *testing.T) {
	cases := []struct {
		name      string
		projector func(*MultiHeadAttention) Projector
	}{
		{name: "query", projector: func(a *MultiHeadAttention) Projector { return a.Query }},
		{name: "output", projector: func(a *MultiHeadAttention) Projector { return a.Output }},
	}
	for _, item := range cases {
		t.Run(item.name, func(t *testing.T) {
			attention, err := NewMultiHeadAttention("attention", AttentionConfig{
				ModelDim: 2, QueryHeads: 1, HeadDim: 2, DType: dtype.Float64, RNG: nn.NewRNG(17),
			})
			if err != nil {
				t.Fatalf("NewMultiHeadAttention: %v", err)
			}
			defer attention.Close()
			useIdentityProjections(t, attention, 2, 1, 1, 2)

			linear := item.projector(attention).(*nn.Linear)
			input := newFloatTensor(t, dtype.Float64, []int64{1, 2, 2}, []float64{0.8, -0.3, 0.5, 1.1})
			defer input.Close()

			checkParameterGradient(t, linear.Weight, input,
				func(ctx *nn.Context, leaf *autograd.Variable) (*autograd.Variable, error) {
					return attention.Forward(ctx, leaf, leaf, leaf, nil)
				})
		})
	}
}

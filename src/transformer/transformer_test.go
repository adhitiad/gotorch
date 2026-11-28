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

func TestTokenEmbeddingContract(t *testing.T) {
	for _, elementType := range []dtype.DType{dtype.Float64, dtype.Float32} {
		t.Run(elementType.String(), func(t *testing.T) {
			embedding, err := NewTokenEmbedding("token", 4, 3, elementType, nn.NewRNG(11))
			if err != nil {
				t.Fatalf("NewTokenEmbedding: %v", err)
			}
			defer embedding.Close()
			setParameterValues(t, embedding.Weight, []float64{
				1, 2, 3,
				4, 5, 6,
				7, 8, 9,
				10, 11, 12,
			})

			tape, ctx := newTestTape(t)
			defer tape.Close()
			tokens := newVariable(t, tape, newIntTensor(t, []int64{2, 3}, []int64{0, 1, 2, 3, 0, 1}), false)
			output, err := embedding.Forward(ctx, tokens)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}
			assertVariableDims(t, "embedding", output, 2, 3, 3)
			assertVariableDType(t, "embedding", output, elementType)
			assertFloatClose(t, "embedding values", variableValues(t, output), []float64{
				1, 2, 3, 4, 5, 6, 7, 8, 9,
				10, 11, 12, 1, 2, 3, 4, 5, 6,
			}, 1e-6)

			badTokens := newVariable(t, tape, newFloatTensor(t, dtype.Float64, []int64{2, 3}, []float64{0, 1, 2, 3, 0, 1}), false)
			if _, err := embedding.Forward(ctx, badTokens); !errors.Is(err, errs.ErrInvalidShape) {
				t.Fatalf("token dtype bukan int64: error %v", err)
			}
			flatTokens := newVariable(t, tape, newIntTensor(t, []int64{6}, []int64{0, 1, 2, 3, 0, 1}), false)
			if _, err := embedding.Forward(ctx, flatTokens); !errors.Is(err, errs.ErrInvalidShape) {
				t.Fatalf("token rank bukan dua: error %v", err)
			}
			outside := newVariable(t, tape, newIntTensor(t, []int64{1, 1}, []int64{9}), false)
			if _, err := embedding.Forward(ctx, outside); !errors.Is(err, errs.ErrOutOfBounds) {
				t.Fatalf("indeks di luar kosakata: error %v", err)
			}
			named := embedding.Parameters()
			if len(named) != 1 || named[0].Name != "token.weight" {
				t.Fatalf("parameter embedding: %v", named)
			}
		})
	}
}

func TestRotaryEmbeddingContract(t *testing.T) {
	rope, err := NewRotaryEmbedding("rope", 4, 4, dtype.Float64, 10000)
	if err != nil {
		t.Fatalf("NewRotaryEmbedding: %v", err)
	}
	defer rope.Close()
	assertFloatClose(t, "cosine", readFloatValues(t, rope.Cosine.Value()), []float64{
		1, 1,
		math.Cos(1), math.Cos(0.01),
		math.Cos(2), math.Cos(0.02),
		math.Cos(3), math.Cos(0.03),
	}, 1e-12)
	assertFloatClose(t, "sine", readFloatValues(t, rope.Sine.Value()), []float64{
		0, 0,
		math.Sin(1), math.Sin(0.01),
		math.Sin(2), math.Sin(0.02),
		math.Sin(3), math.Sin(0.03),
	}, 1e-12)

	values := []float64{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	}
	tape, ctx := newTestTape(t)
	defer tape.Close()
	input := newVariable(t, tape, newFloatTensor(t, dtype.Float64, []int64{1, 2, 2, 4}, values), true)
	output, err := rope.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	assertVariableDims(t, "rope", output, 1, 2, 2, 4)
	assertVariableDType(t, "rope", output, dtype.Float64)

	want := make([]float64, len(values))
	cosines := readFloatValues(t, rope.Cosine.Value())
	sines := readFloatValues(t, rope.Sine.Value())
	for index := range values {
		position := (index % (2 * 4)) / 4
		pair := (index % 4) / 2
		angleIndex := position*2 + pair
		left := values[index-index%2]
		right := values[index-index%2+1]
		if index%2 == 0 {
			want[index] = left*cosines[angleIndex] - right*sines[angleIndex]
		} else {
			want[index] = right*cosines[angleIndex] + left*sines[angleIndex]
		}
	}
	assertFloatClose(t, "rope values", variableValues(t, output), want, 1e-12)

	checkTransformerGradient(t, newFloatTensor(t, dtype.Float64, []int64{1, 1, 2, 4}, []float64{1, 2, 3, 4, 5, 6, 7, 8}),
		func(ctx *nn.Context, leaf *autograd.Variable) (*autograd.Variable, error) {
			return rope.Forward(ctx, leaf)
		})

	if _, err := NewRotaryEmbedding("ganjil", 4, 3, dtype.Float64, 10000); !errors.Is(err, errs.ErrInvalidShape) {
		t.Fatalf("headDim ganjil: error %v", err)
	}
	long := newVariable(t, tape, newFloatTensor(t, dtype.Float64, []int64{1, 1, 5, 4}, make([]float64, 20)), false)
	if _, err := rope.Forward(ctx, long); !errors.Is(err, errs.ErrInvalidShape) {
		t.Fatalf("panjang melebihi maksimum: error %v", err)
	}
}

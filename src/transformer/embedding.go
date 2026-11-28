package transformer

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/adhitiad/gotorch/src/autograd"
	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/nn"
	"github.com/adhitiad/gotorch/src/tensor"
)

// TokenEmbedding memetakan token int64 menjadi vektor berdimensi Dim.
type TokenEmbedding struct {
	*nn.ModuleBase
	Weight *nn.Parameter
	Dim    int64
}

// NewTokenEmbedding membuat embedding dengan inisialisasi normal deterministik.
func NewTokenEmbedding(name string, vocabulary, dimension int64, elementType dtype.DType, rng *rand.Rand) (*TokenEmbedding, error) {
	base, err := nn.NewModule(name)
	if err != nil {
		return nil, err
	}
	if vocabulary <= 0 || dimension <= 0 || rng == nil {
		_ = base.Close()
		return nil, errs.ErrInvalidArgument
	}
	elementType = resolveDType(elementType)
	value, err := tensor.NewTensor(elementType, vocabulary, dimension)
	if err != nil {
		_ = base.Close()
		return nil, err
	}
	values := make([]float64, int(vocabulary*dimension))
	scale := 1 / math.Sqrt(float64(dimension))
	for index := range values {
		values[index] = rng.NormFloat64() * scale
	}
	if elementType == dtype.Float32 {
		converted := make([]float32, len(values))
		for index, item := range values {
			converted[index] = float32(item)
		}
		err = tensor.Write(value, converted)
	} else {
		err = tensor.Write(value, values)
	}
	if err != nil {
		_ = value.Close()
		_ = base.Close()
		return nil, err
	}
	parameter, err := nn.NewParameter("weight", value, true)
	_ = value.Close()
	if err != nil {
		_ = base.Close()
		return nil, err
	}
	if err := base.RegisterParameter("weight", parameter); err != nil {
		_ = parameter.Close()
		_ = base.Close()
		return nil, err
	}
	return &TokenEmbedding{ModuleBase: base, Weight: parameter, Dim: dimension}, nil
}

// Forward menerima token [batch, time] dan menghasilkan [batch, time, dim].
func (e *TokenEmbedding) Forward(ctx *nn.Context, tokens *autograd.Variable) (*autograd.Variable, error) {
	if e == nil || e.ModuleBase == nil {
		return nil, errs.ErrInvalidArgument
	}
	tokenTensor, err := tokens.Value()
	if err != nil {
		return nil, err
	}
	defer tokenTensor.Close()
	dims, tokenType, err := tensorDims(tokenTensor)
	if err != nil {
		return nil, err
	}
	if len(dims) != 2 || tokenType != dtype.Int64 {
		return nil, fmt.Errorf("%w: tokens must be int64 [batch, time]", errs.ErrInvalidShape)
	}
	batch, length := dims[0], dims[1]
	// nn.Reshape hanya menerima tensor float, sehingga indeks int64 diratakan
	// langsung sebagai variabel leaf tanpa operasi diferensial.
	indices, err := tensor.Read[int64](tokenTensor)
	if err != nil {
		return nil, err
	}
	flatTensor, err := tensor.NewTensor(dtype.Int64, batch*length)
	if err != nil {
		return nil, err
	}
	if err := tensor.Write(flatTensor, indices); err != nil {
		_ = flatTensor.Close()
		return nil, err
	}
	flat, err := variableFromTensor(ctx, flatTensor, false)
	if err != nil {
		return nil, err
	}
	weight, err := ctx.Variable(e.Weight)
	if err != nil {
		return nil, err
	}
	embedded, err := nn.Embedding(ctx, weight, flat)
	if err != nil {
		return nil, err
	}
	return nn.Reshape(ctx, embedded, batch, length, e.Dim)
}

// RotaryEmbedding menyimpan tabel cos/sinus untuk RoPE dan menghitung potongannya per langkah.
type RotaryEmbedding struct {
	*nn.ModuleBase
	Cosine    *nn.Buffer
	Sine      *nn.Buffer
	MaxLength int64
	HeadDim   int64
	Base      float64
}

// NewRotaryEmbedding membuat tabel RoPE untuk panjang maksimum yang ditentukan.
func NewRotaryEmbedding(name string, maxLength, headDim int64, elementType dtype.DType, base float64) (*RotaryEmbedding, error) {
	baseModule, err := nn.NewModule(name)
	if err != nil {
		return nil, err
	}
	if maxLength <= 0 || headDim <= 0 || headDim%2 != 0 {
		_ = baseModule.Close()
		return nil, fmt.Errorf("%w: RoPE needs positive even head dimension", errs.ErrInvalidShape)
	}
	if base <= 0 {
		base = 10000
	}
	elementType = resolveDType(elementType)
	pairs := headDim / 2
	cosineValues := make([]float64, int(maxLength*pairs))
	sineValues := make([]float64, len(cosineValues))
	for position := int64(0); position < maxLength; position++ {
		for pair := int64(0); pair < pairs; pair++ {
			exponent := float64(2*pair) / float64(headDim)
			angle := float64(position) / math.Pow(base, exponent)
			index := int(position*pairs + pair)
			cosineValues[index] = math.Cos(angle)
			sineValues[index] = math.Sin(angle)
		}
	}
	cosineTensor, err := floatTensor(elementType, []int64{maxLength, pairs}, cosineValues)
	if err != nil {
		_ = baseModule.Close()
		return nil, err
	}
	cosine, err := nn.NewBuffer("cosine", cosineTensor, true)
	_ = cosineTensor.Close()
	if err != nil {
		_ = baseModule.Close()
		return nil, err
	}
	if err := baseModule.RegisterBuffer("cosine", cosine); err != nil {
		_ = cosine.Close()
		_ = baseModule.Close()
		return nil, err
	}
	sineTensor, err := floatTensor(elementType, []int64{maxLength, pairs}, sineValues)
	if err != nil {
		_ = baseModule.Close()
		return nil, err
	}
	sine, err := nn.NewBuffer("sine", sineTensor, true)
	_ = sineTensor.Close()
	if err != nil {
		_ = baseModule.Close()
		return nil, err
	}
	if err := baseModule.RegisterBuffer("sine", sine); err != nil {
		_ = sine.Close()
		_ = baseModule.Close()
		return nil, err
	}
	return &RotaryEmbedding{
		ModuleBase: baseModule,
		Cosine:     cosine,
		Sine:       sine,
		MaxLength:  maxLength,
		HeadDim:    headDim,
		Base:       base,
	}, nil
}

// Forward menerapkan RoPE pada tensor [batch, heads, time, headDim].
func (r *RotaryEmbedding) Forward(ctx *nn.Context, input *autograd.Variable) (*autograd.Variable, error) {
	if r == nil || r.ModuleBase == nil || ctx == nil {
		return nil, errs.ErrInvalidArgument
	}
	value, err := input.Value()
	if err != nil {
		return nil, err
	}
	defer value.Close()
	dims, elementType, err := tensorDims(value)
	if err != nil {
		return nil, err
	}
	if len(dims) != 4 || dims[3] != r.HeadDim || dims[2] > r.MaxLength {
		return nil, fmt.Errorf("%w: RoPE input %v exceeds configured [*, *, %d, %d]", errs.ErrInvalidShape, dims, r.MaxLength, r.HeadDim)
	}
	length, pairs := dims[2], r.HeadDim/2
	cosineValues, err := sliceBuffer(r.Cosine, length)
	if err != nil {
		return nil, err
	}
	sineValues, err := sliceBuffer(r.Sine, length)
	if err != nil {
		return nil, err
	}
	cosineTensor, err := floatTensor(elementType, []int64{length, pairs}, cosineValues)
	if err != nil {
		return nil, err
	}
	sineTensor, err := floatTensor(elementType, []int64{length, pairs}, sineValues)
	if err != nil {
		_ = cosineTensor.Close()
		return nil, err
	}
	cosine, err := variableFromTensor(ctx, cosineTensor, false)
	if err != nil {
		_ = sineTensor.Close()
		return nil, err
	}
	sine, err := variableFromTensor(ctx, sineTensor, false)
	if err != nil {
		return nil, err
	}
	// nn.RoPE menghitung posisi dari indeks data multi-head, sehingga setiap
	// head harus diproses sebagai irisan satu head agar posisinya tidak bergeser.
	batch, heads := dims[0], dims[1]
	grouped, err := nn.Reshape(ctx, input, batch*heads, 1, length, r.HeadDim)
	if err != nil {
		return nil, err
	}
	rotated, err := nn.RoPE(ctx, grouped, cosine, sine)
	if err != nil {
		return nil, err
	}
	return nn.Reshape(ctx, rotated, batch, heads, length, r.HeadDim)
}

func sliceBuffer(buffer *nn.Buffer, length int64) ([]float64, error) {
	value := buffer.Value()
	if value == nil {
		return nil, errs.ErrInvalidState
	}
	metadata, err := value.Metadata()
	if err != nil {
		return nil, err
	}
	if metadata.DType() == dtype.Float64 {
		values, err := tensor.Read[float64](value)
		if err != nil || int64(len(values)) < length*metadata.Shape().Dims()[1] {
			return nil, errs.ErrInvalidState
		}
		return values[:length*metadata.Shape().Dims()[1]], nil
	}
	items, err := tensor.Read[float32](value)
	if err != nil {
		return nil, err
	}
	count := int(length * metadata.Shape().Dims()[1])
	if len(items) < count {
		return nil, errs.ErrInvalidState
	}
	values := make([]float64, count)
	for index := 0; index < count; index++ {
		values[index] = float64(items[index])
	}
	return values, nil
}

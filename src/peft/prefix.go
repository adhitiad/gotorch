package peft

import (
	"fmt"
	"math"

	"github.com/adhitiad/gotorch/src/autograd"
	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/nn"
	"github.com/adhitiad/gotorch/src/tensor"
)

// PrefixTuningConfig mengonfigurasi prototipe prefix tuning.
// RNG disuntikkan agar inisialisasi dapat direproduksi.
type PrefixTuningConfig struct {
	PrefixLength int64
	DType        dtype.DType
	RNG          interface{ NormFloat64() float64 }
}

// PrefixTuning menyimpan prefix key dan value trainable [prefixLength, dim].
// Modul ini menambahkan prefix ke key/value sebelum attention; ia belum
// mengelola mask, state adapter, atau dukungan banyak layer.
type PrefixTuning struct {
	*nn.ModuleBase
	KeyPrefix   *nn.Parameter
	ValuePrefix *nn.Parameter
	Dim         int64
}

// NewPrefixTuning membuat prototipe prefix tuning dengan dua parameter.
func NewPrefixTuning(name string, dimension int64, config PrefixTuningConfig) (*PrefixTuning, error) {
	module, err := nn.NewModule(name)
	if err != nil {
		return nil, err
	}
	if dimension <= 0 || config.PrefixLength <= 0 || config.RNG == nil {
		_ = module.Close()
		return nil, fmt.Errorf("%w: invalid prefix tuning configuration for %q", errs.ErrInvalidArgument, name)
	}
	elementType := config.DType
	if elementType == dtype.Invalid {
		elementType = dtype.Float64
	}
	if !isFloat(elementType) {
		_ = module.Close()
		return nil, errs.ErrUnsupportedDType
	}
	newPrefix := func(suffix string) (*nn.Parameter, error) {
		value, createErr := tensor.NewTensor(elementType, config.PrefixLength, dimension)
		if createErr != nil {
			return nil, createErr
		}
		scale := 1 / math.Sqrt(float64(dimension))
		values := make([]float64, int(config.PrefixLength*dimension))
		for index := range values {
			values[index] = config.RNG.NormFloat64() * scale
		}
		if writeErr := writeFloatValues(value, values); writeErr != nil {
			_ = value.Close()
			return nil, writeErr
		}
		parameter, createErr := nn.NewParameter(suffix, value, true)
		_ = value.Close()
		return parameter, createErr
	}
	keyPrefix, err := newPrefix("key")
	if err != nil {
		_ = module.Close()
		return nil, err
	}
	if err := module.RegisterParameter("key", keyPrefix); err != nil {
		_ = keyPrefix.Close()
		_ = module.Close()
		return nil, err
	}
	valuePrefix, err := newPrefix("value")
	if err != nil {
		_ = module.Close()
		return nil, err
	}
	if err := module.RegisterParameter("value", valuePrefix); err != nil {
		_ = valuePrefix.Close()
		_ = module.Close()
		return nil, err
	}
	return &PrefixTuning{
		ModuleBase:  module,
		KeyPrefix:   keyPrefix,
		ValuePrefix: valuePrefix,
		Dim:         dimension,
	}, nil
}

// Forward menambahkan prefix ke key dan value [batch, time, dim] dan
// mengembalikan pasangan [batch, prefixLength+time, dim] dengan urutan yang sama.
func (p *PrefixTuning) Forward(ctx *nn.Context, key, value *autograd.Variable) (*autograd.Variable, *autograd.Variable, error) {
	if p == nil || p.ModuleBase == nil || p.KeyPrefix == nil || p.ValuePrefix == nil || ctx == nil {
		return nil, nil, errs.ErrInvalidArgument
	}
	keyDims, keyType, err := prefixTensorLayout(key, p.Dim)
	if err != nil {
		return nil, nil, err
	}
	valueDims, valueType, err := prefixTensorLayout(value, p.Dim)
	if err != nil {
		return nil, nil, err
	}
	if !dimsEqual(keyDims, valueDims) || keyType != valueType {
		return nil, nil, errs.ErrGradientShapeMismatch
	}
	keyParameter, err := p.KeyPrefix.Value().Metadata()
	if err != nil {
		return nil, nil, err
	}
	if keyType != keyParameter.DType() {
		return nil, nil, errs.ErrUnsupportedDType
	}
	keyPrefix, err := ctx.Variable(p.KeyPrefix)
	if err != nil {
		return nil, nil, err
	}
	valuePrefix, err := ctx.Variable(p.ValuePrefix)
	if err != nil {
		return nil, nil, err
	}
	extendedKey, err := nn.ConcatTimePrefix(ctx, keyPrefix, key)
	if err != nil {
		return nil, nil, err
	}
	extendedValue, err := nn.ConcatTimePrefix(ctx, valuePrefix, value)
	if err != nil {
		return nil, nil, err
	}
	return extendedKey, extendedValue, nil
}

func prefixTensorLayout(variable *autograd.Variable, dimension int64) ([]int64, dtype.DType, error) {
	if variable == nil {
		return nil, dtype.Invalid, errs.ErrInvalidArgument
	}
	value, err := variable.Value()
	if err != nil {
		return nil, dtype.Invalid, err
	}
	defer value.Close()
	metadata, err := value.Metadata()
	if err != nil {
		return nil, dtype.Invalid, err
	}
	dims := metadata.Shape().Dims()
	if len(dims) != 3 || dims[0] <= 0 || dims[1] <= 0 || dims[2] != dimension {
		return nil, dtype.Invalid, fmt.Errorf("%w: prefix input %v, want (batch, time, %d)", errs.ErrInvalidShape, dims, dimension)
	}
	return append([]int64(nil), dims...), metadata.DType(), nil
}

func dimsEqual(left, right []int64) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}
	return true
}

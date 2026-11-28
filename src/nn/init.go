package nn

import (
	"math"
	"math/rand"

	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/tensor"
)

// NewRNG membuat generator deterministik dari seed eksplisit.
func NewRNG(seed int64) *rand.Rand {
	return rand.New(rand.NewSource(seed))
}

// Fill mengisi tensor dengan nilai dari generator yang disuntikkan.
func Fill(value *tensor.Tensor, generator func() float64) error {
	if value == nil || generator == nil {
		return errs.ErrInvalidArgument
	}
	metadata, err := value.Metadata()
	if err != nil {
		return err
	}
	if !isDifferentiable(metadata.DType()) {
		return errs.ErrUnsupportedDType
	}
	values, err := readValues(value)
	if err != nil {
		return err
	}
	for index := range values {
		values[index] = generator()
	}
	return writeValues(value, values)
}

// Zeros mengisi tensor dengan nol.
func Zeros(value *tensor.Tensor) error {
	return Fill(value, func() float64 { return 0 })
}

// Ones mengisi tensor dengan satu.
func Ones(value *tensor.Tensor) error {
	return Fill(value, func() float64 { return 1 })
}

// Uniform mengisi tensor dengan distribusi seragam [minimum, maximum).
func Uniform(rng *rand.Rand, value *tensor.Tensor, minimum, maximum float64) error {
	if rng == nil || maximum <= minimum {
		return errs.ErrInvalidArgument
	}
	return Fill(value, func() float64 { return minimum + rng.Float64()*(maximum-minimum) })
}

// Normal mengisi tensor dengan distribusi normal mean/stddev.
func Normal(rng *rand.Rand, value *tensor.Tensor, mean, stddev float64) error {
	if rng == nil || stddev <= 0 {
		return errs.ErrInvalidArgument
	}
	return Fill(value, func() float64 { return mean + rng.NormFloat64()*stddev })
}

func fanInOut(value *tensor.Tensor) (int, int, error) {
	metadata, err := value.Metadata()
	if err != nil {
		return 0, 0, err
	}
	if metadata.DType() != dtype.Float32 && metadata.DType() != dtype.Float64 {
		return 0, 0, errs.ErrUnsupportedDType
	}
	dims := metadata.Shape().Dims()
	count, err := elementCount(dims)
	if err != nil {
		return 0, 0, err
	}
	switch len(dims) {
	case 0, 1:
		return 1, 1, nil
	case 2:
		if dims[1] > int64(math.MaxInt) {
			return 0, 0, errs.ErrShapeOverflow
		}
		return int(dims[1]), int(dims[0]), nil
	default:
		if dims[len(dims)-1] > int64(math.MaxInt) || dims[0] > int64(math.MaxInt) {
			return 0, 0, errs.ErrShapeOverflow
		}
		receptive := count / int(dims[0])
		return receptive, int(dims[0]), nil
	}
}

// XavierUniform mengisi tensor dengan batas ±sqrt(6/(fanIn+fanOut)).
func XavierUniform(rng *rand.Rand, value *tensor.Tensor) error {
	if rng == nil {
		return errs.ErrInvalidArgument
	}
	fanIn, fanOut, err := fanInOut(value)
	if err != nil {
		return err
	}
	limit := math.Sqrt(6 / float64(fanIn+fanOut))
	return Uniform(rng, value, -limit, limit)
}

// XavierNormal mengisi tensor dengan standar deviasi sqrt(2/(fanIn+fanOut)).
func XavierNormal(rng *rand.Rand, value *tensor.Tensor) error {
	if rng == nil {
		return errs.ErrInvalidArgument
	}
	fanIn, fanOut, err := fanInOut(value)
	if err != nil {
		return err
	}
	return Normal(rng, value, 0, math.Sqrt(2/float64(fanIn+fanOut)))
}

// KaimingUniform mengisi tensor dengan batas ±sqrt(6/fanIn).
func KaimingUniform(rng *rand.Rand, value *tensor.Tensor) error {
	if rng == nil {
		return errs.ErrInvalidArgument
	}
	fanIn, _, err := fanInOut(value)
	if err != nil {
		return err
	}
	limit := math.Sqrt(6 / float64(fanIn))
	return Uniform(rng, value, -limit, limit)
}

// KaimingNormal mengisi tensor dengan standar deviasi sqrt(2/fanIn).
func KaimingNormal(rng *rand.Rand, value *tensor.Tensor) error {
	if rng == nil {
		return errs.ErrInvalidArgument
	}
	fanIn, _, err := fanInOut(value)
	if err != nil {
		return err
	}
	return Normal(rng, value, 0, math.Sqrt(2/float64(fanIn)))
}

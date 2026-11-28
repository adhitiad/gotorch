// Package optim menyediakan pembaruan parameter berbasis gradien di atas nn.Context.
package optim

import (
	"fmt"
	"path"
	"sort"

	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/nn"
	"github.com/adhitiad/gotorch/src/tensor"
)

// Param mengikat satu parameter dengan nama lengkapnya di dalam model.
type Param struct {
	Name      string
	Parameter *nn.Parameter
}

// Parameters mengambil seluruh parameter modul beserta nama lengkapnya.
func Parameters(module nn.Module) []Param {
	if module == nil {
		return nil
	}
	named := module.Parameters()
	result := make([]Param, 0, len(named))
	for _, item := range named {
		result = append(result, Param{Name: item.Name, Parameter: item.Parameter})
	}
	return result
}

// SelectParameters memilih parameter berdasarkan pola glob pada nama lengkap.
func SelectParameters(module nn.Module, patterns ...string) ([]Param, error) {
	if module == nil || len(patterns) == 0 {
		return nil, errs.ErrInvalidArgument
	}
	all := Parameters(module)
	result := make([]Param, 0, len(all))
	for _, item := range all {
		for _, pattern := range patterns {
			matched, err := matchPattern(pattern, item.Name)
			if err != nil {
				return nil, err
			}
			if matched {
				result = append(result, item)
				break
			}
		}
	}
	return result, nil
}

// Optimizer memperbarui parameter dan state-nya dari gradien pada Context.
type Optimizer interface {
	Name() string
	Step(ctx *nn.Context, parameters []Param) error
	SetLearningRate(learningRate float64) error
	LearningRate() float64
	MarshalState(parameters []Param) ([]byte, error)
	UnmarshalState(parameters []Param, data []byte) error
	Close() error
}

func validateParameters(parameters []Param) ([]Param, error) {
	if len(parameters) == 0 {
		return nil, errs.ErrInvalidArgument
	}
	ordered := append([]Param(nil), parameters...)
	for _, item := range ordered {
		if item.Name == "" || item.Parameter == nil {
			return nil, errs.ErrInvalidArgument
		}
		if item.Parameter.Value() == nil {
			return nil, errs.ErrInvalidState
		}
	}
	sort.Slice(ordered, func(i, j int) bool { return ordered[i].Name < ordered[j].Name })
	for index := 1; index < len(ordered); index++ {
		if ordered[index].Name == ordered[index-1].Name {
			return nil, fmt.Errorf("%w: duplicate parameter %q", errs.ErrInvalidArgument, ordered[index].Name)
		}
	}
	return ordered, nil
}

func paramCount(parameter *nn.Parameter) (int, error) {
	value := parameter.Value()
	if value == nil {
		return 0, errs.ErrInvalidState
	}
	metadata, err := value.Metadata()
	if err != nil {
		return 0, err
	}
	if metadata.DType() != dtype.Float32 && metadata.DType() != dtype.Float64 {
		return 0, errs.ErrUnsupportedDType
	}
	dims := metadata.Shape().Dims()
	count := 1
	for _, dim := range dims {
		if dim < 0 {
			return 0, errs.ErrShapeOverflow
		}
		// Dimensi nol membuat jumlah elemen nol dan harus dihindari sebelum
		// pembagian pada pemeriksaan overflow.
		if dim == 0 {
			return 0, nil
		}
		if int64(count) > int64(^uint(0)>>1)/dim {
			return 0, errs.ErrShapeOverflow
		}
		count *= int(dim)
	}
	return count, nil
}

func gradientValues(ctx *nn.Context, parameter *nn.Parameter) ([]float64, bool, error) {
	if ctx == nil || parameter == nil {
		return nil, false, errs.ErrInvalidArgument
	}
	gradient, err := ctx.Gradient(parameter)
	if err == errs.ErrNoGradient {
		return nil, false, nil
	}
	if err != nil {
		return nil, false, err
	}
	defer gradient.Close()
	values, err := readFloatValues(gradient)
	return values, true, err
}

func matchPattern(pattern, name string) (bool, error) {
	matched, err := path.Match(pattern, name)
	if err != nil {
		return false, fmt.Errorf("%w: invalid parameter pattern %q", errs.ErrInvalidArgument, pattern)
	}
	return matched, nil
}

func readFloatValues(value *tensor.Tensor) ([]float64, error) {
	metadata, err := value.Metadata()
	if err != nil {
		return nil, err
	}
	switch metadata.DType() {
	case dtype.Float64:
		return tensor.Read[float64](value)
	case dtype.Float32:
		items, err := tensor.Read[float32](value)
		if err != nil {
			return nil, err
		}
		result := make([]float64, len(items))
		for index, item := range items {
			result[index] = float64(item)
		}
		return result, nil
	default:
		return nil, errs.ErrUnsupportedDType
	}
}

func validateLearningRate(value float64) error {
	if value <= 0 || value != value || value > 1e30 {
		return fmt.Errorf("%w: learning rate %g out of range", errs.ErrInvalidArgument, value)
	}
	return nil
}

// Package peft menyediakan targeting parameter, pembekuan, dan adapter ringan.
package peft

import (
	"fmt"
	"path"
	"sort"
	"strings"

	"github.com/adhitiad/gotorch/src/autograd"
	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/nn"
	"github.com/adhitiad/gotorch/src/tensor"
)

// Projector adalah layer linear yang bisa dibungkus adapter.
type Projector interface {
	nn.Module
	Forward(*nn.Context, *autograd.Variable) (*autograd.Variable, error)
	InputDim() int64
	OutputDim() int64
}

// LinearLike menambahkan akses weight untuk merge adapter.
type LinearLike interface {
	Projector
	WeightParameter() *nn.Parameter
}

// TrainableParameters mengembalikan parameter dengan requiresGrad=true.
func TrainableParameters(module nn.Module) ([]nn.NamedParameter, error) {
	if module == nil {
		return nil, errs.ErrInvalidArgument
	}
	result := make([]nn.NamedParameter, 0)
	for _, named := range module.Parameters() {
		if named.Parameter.RequiresGrad() {
			result = append(result, named)
		}
	}
	return result, nil
}

// Freeze membekukan parameter yang cocok dengan pola glob. Tanpa pola, semua dibekukan.
func Freeze(module nn.Module, patterns ...string) error {
	return setRequiresGrad(module, false, patterns...)
}

// Unfreeze mengaktifkan kembali parameter yang cocok dengan pola glob.
func Unfreeze(module nn.Module, patterns ...string) error {
	return setRequiresGrad(module, true, patterns...)
}

func setRequiresGrad(module nn.Module, enabled bool, patterns ...string) error {
	if module == nil {
		return errs.ErrInvalidArgument
	}
	// Validasi pola dilakukan lebih awal agar modul tanpa parameter pun menolak pola rusak.
	if _, err := matches("", patterns); err != nil {
		return err
	}
	for _, named := range module.Parameters() {
		matched, err := matches(named.Name, patterns)
		if err != nil {
			return err
		}
		if !matched {
			continue
		}
		if err := named.Parameter.SetRequiresGrad(enabled); err != nil {
			return err
		}
	}
	return nil
}

func matches(name string, patterns []string) (bool, error) {
	if len(patterns) == 0 {
		return true, nil
	}
	for _, pattern := range patterns {
		matched, err := path.Match(pattern, name)
		if err != nil {
			return false, fmt.Errorf("%w: invalid parameter pattern %q", errs.ErrInvalidArgument, pattern)
		}
		if matched {
			return true, nil
		}
	}
	return false, nil
}

func readFloatValues(value *tensor.Tensor) ([]float64, error) {
	if value == nil {
		return nil, errs.ErrInvalidArgument
	}
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

func writeFloatValues(value *tensor.Tensor, values []float64) error {
	metadata, err := value.Metadata()
	if err != nil {
		return err
	}
	switch metadata.DType() {
	case dtype.Float64:
		return tensor.Write(value, values)
	case dtype.Float32:
		converted := make([]float32, len(values))
		for index, item := range values {
			converted[index] = float32(item)
		}
		return tensor.Write(value, converted)
	default:
		return errs.ErrUnsupportedDType
	}
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
	count := 1
	for _, dim := range metadata.Shape().Dims() {
		if dim < 0 || int64(count) > int64(^uint(0)>>1)/dim {
			return 0, errs.ErrShapeOverflow
		}
		count *= int(dim)
	}
	return count, nil
}

func normalizeTargets(targets []string) []string {
	seen := make(map[string]struct{}, len(targets))
	result := make([]string, 0, len(targets))
	for _, target := range targets {
		target = strings.TrimSpace(strings.ToLower(target))
		if target == "" {
			continue
		}
		if _, exists := seen[target]; exists {
			continue
		}
		seen[target] = struct{}{}
		result = append(result, target)
	}
	sort.Strings(result)
	return result
}

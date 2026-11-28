package autograd

import (
	"context"
	"fmt"
	"math"

	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/tensor"
)

// ScalarFunction builds a scalar output from one Variable on the supplied Tape.
// It is called once for the analytical pass and twice per input element for the
// central finite-difference approximation.
type ScalarFunction func(context.Context, *Tape, *Variable) (*Variable, error)

// GradientCheckOptions controls central finite differences. Zero values select
// dtype-aware defaults. MaxElements bounds the O(n) function evaluations.
type GradientCheckOptions struct {
	Step              float64
	AbsoluteTolerance float64
	RelativeTolerance float64
	MaxElements       int
}

// GradientElement contains the analytical and numerical derivative for one
// row-major logical input element.
type GradientElement struct {
	Index         int64
	Analytical    float64
	Numerical     float64
	AbsoluteError float64
	RelativeError float64
}

// GradientCheckResult summarizes a finite-difference comparison.
type GradientCheckResult struct {
	Passed           bool
	CheckedElements  int
	MaxAbsoluteError float64
	MaxRelativeError float64
	Elements         []GradientElement
}

// CheckGradient compares reverse-mode gradients with central finite differences.
// The function must return a rank-zero scalar. Only float32/float64 inputs are
// supported; max-element limits prevent unexpectedly expensive checks.
func CheckGradient(
	ctx context.Context,
	input *tensor.Tensor,
	fn ScalarFunction,
	options GradientCheckOptions,
) (GradientCheckResult, error) {
	result := GradientCheckResult{Passed: true}
	if nilContext(ctx) || input == nil || fn == nil {
		return result, errs.ErrInvalidArgument
	}
	inputValues, elementType, dimensions, err := readNumbers(input)
	if err != nil {
		return result, err
	}
	if elementType != dtype.Float32 && elementType != dtype.Float64 {
		return result, errs.ErrUnsupportedDType
	}
	metadata, err := input.Metadata()
	if err != nil {
		return result, err
	}
	numel, err := metadata.Numel()
	if err != nil {
		return result, err
	}
	if numel > int64(maxElements(options.MaxElements)) {
		return result, fmt.Errorf("%w: gradient check has %d elements; limit is %d", errs.ErrInvalidArgument, numel, maxElements(options.MaxElements))
	}
	step, absoluteTolerance, relativeTolerance, maximum := normalizedCheckOptions(elementType, options)
	if maximum <= 0 || step <= 0 || absoluteTolerance < 0 || relativeTolerance < 0 ||
		math.IsNaN(step) || math.IsInf(step, 0) ||
		math.IsNaN(absoluteTolerance) || math.IsInf(absoluteTolerance, 0) ||
		math.IsNaN(relativeTolerance) || math.IsInf(relativeTolerance, 0) {
		return result, errs.ErrInvalidArgument
	}

	tape := NewTape()
	defer tape.Close()
	leaf, err := tape.NewVariable(input, true)
	if err != nil {
		return result, err
	}
	output, err := fn(ctx, tape, leaf)
	if err != nil {
		return result, err
	}
	if output == nil || output.tape != tape {
		return result, errs.ErrInvalidArgument
	}
	outputValue, err := output.Value()
	if err != nil {
		return result, err
	}
	outputMetadata, err := outputValue.Metadata()
	if err != nil {
		_ = outputValue.Close()
		return result, err
	}
	if outputMetadata.Shape().Rank() != 0 {
		_ = outputValue.Close()
		return result, fmt.Errorf("%w: gradient check output must be a rank-zero scalar", errs.ErrInvalidShape)
	}
	if _, _, _, err := readNumbers(outputValue); err != nil {
		_ = outputValue.Close()
		return result, err
	}
	_ = outputValue.Close()
	if err := tape.Backward(ctx, output); err != nil {
		return result, err
	}
	analyticalTensor, err := leaf.Grad()
	if err != nil {
		return result, err
	}
	analyticalValues, _, _, err := readNumbers(analyticalTensor)
	_ = analyticalTensor.Close()
	if err != nil {
		return result, err
	}
	if int64(len(analyticalValues)) != numel {
		return result, errs.ErrGradientShapeMismatch
	}

	result.Elements = make([]GradientElement, 0, len(inputValues))
	for index := range inputValues {
		if err := ctx.Err(); err != nil {
			return GradientCheckResult{}, err
		}
		plusValues := append([]float64(nil), inputValues...)
		minusValues := append([]float64(nil), inputValues...)
		plusValues[index] += step
		minusValues[index] -= step
		plusTensor, err := numberTensor(elementType, plusValues, dimensions...)
		if err != nil {
			return GradientCheckResult{}, err
		}
		minusTensor, err := numberTensor(elementType, minusValues, dimensions...)
		if err != nil {
			_ = plusTensor.Close()
			return GradientCheckResult{}, err
		}
		plusOutput, plusErr := evaluateScalar(ctx, fn, plusTensor)
		minusOutput, minusErr := evaluateScalar(ctx, fn, minusTensor)
		_ = plusTensor.Close()
		_ = minusTensor.Close()
		if plusErr != nil {
			return GradientCheckResult{}, plusErr
		}
		if minusErr != nil {
			return GradientCheckResult{}, minusErr
		}
		numerical := (plusOutput - minusOutput) / (2 * step)
		analytical := analyticalValues[index]
		absoluteError := math.Abs(analytical - numerical)
		denominator := math.Max(math.Abs(analytical), math.Abs(numerical))
		relativeError := 0.0
		if denominator > 0 {
			relativeError = absoluteError / denominator
		} else if absoluteError > 0 {
			relativeError = math.Inf(1)
		}
		if absoluteError > result.MaxAbsoluteError {
			result.MaxAbsoluteError = absoluteError
		}
		if relativeError > result.MaxRelativeError {
			result.MaxRelativeError = relativeError
		}
		limit := absoluteTolerance + relativeTolerance*denominator
		if math.IsNaN(absoluteError) || absoluteError > limit {
			result.Passed = false
		}
		result.Elements = append(result.Elements, GradientElement{
			Index: int64(index), Analytical: analytical, Numerical: numerical,
			AbsoluteError: absoluteError, RelativeError: relativeError,
		})
	}
	result.CheckedElements = len(result.Elements)
	return result, nil
}

func evaluateScalar(ctx context.Context, fn ScalarFunction, input *tensor.Tensor) (float64, error) {
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	tape := NewTape()
	defer tape.Close()
	leaf, err := tape.NewVariable(input, false)
	if err != nil {
		return 0, err
	}
	output, err := fn(ctx, tape, leaf)
	if err != nil {
		return 0, err
	}
	if output == nil || output.tape != tape {
		return 0, errs.ErrInvalidArgument
	}
	value, err := output.Value()
	if err != nil {
		return 0, err
	}
	defer value.Close()
	metadata, err := value.Metadata()
	if err != nil {
		return 0, err
	}
	if metadata.Shape().Rank() != 0 {
		return 0, errs.ErrInvalidShape
	}
	values, _, _, err := readNumbers(value)
	if err != nil {
		return 0, err
	}
	if len(values) != 1 {
		return 0, errs.ErrGradientShapeMismatch
	}
	return values[0], nil
}

func maxElements(value int) int {
	if value == 0 {
		return 256
	}
	return value
}

func normalizedCheckOptions(elementType dtype.DType, options GradientCheckOptions) (float64, float64, float64, int) {
	step := options.Step
	absoluteTolerance := options.AbsoluteTolerance
	relativeTolerance := options.RelativeTolerance
	if elementType == dtype.Float32 {
		if step == 0 {
			step = 1e-3
		}
		if absoluteTolerance == 0 {
			absoluteTolerance = 1e-3
		}
		if relativeTolerance == 0 {
			relativeTolerance = 1e-2
		}
	} else {
		if step == 0 {
			step = 1e-6
		}
		if absoluteTolerance == 0 {
			absoluteTolerance = 1e-6
		}
		if relativeTolerance == 0 {
			relativeTolerance = 1e-4
		}
	}
	maximum := maxElements(options.MaxElements)
	return step, absoluteTolerance, relativeTolerance, maximum
}

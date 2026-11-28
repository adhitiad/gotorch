package optim

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/nn"
	"github.com/adhitiad/gotorch/src/tensor"
)

var (
	_ Optimizer = (*SGD)(nil)
	_ Optimizer = (*Adam)(nil)
)

// testTensor membuat tensor float64 dari nilai yang diberikan.
func testTensor(t *testing.T, values []float64, dims ...int64) *tensor.Tensor {
	t.Helper()
	value, err := tensor.NewTensor(dtype.Float64, dims...)
	if err != nil {
		t.Fatalf("NewTensor(%v): %v", dims, err)
	}
	if err := tensor.Write(value, values); err != nil {
		t.Fatalf("tensor.Write: %v", err)
	}
	return value
}

// testParameter membuat parameter float64 yang siap dipakai optimizer.
func testParameter(t *testing.T, name string, values []float64, dims ...int64) *nn.Parameter {
	t.Helper()
	value := testTensor(t, values, dims...)
	parameter, err := nn.NewParameter(name, value, true)
	if err != nil {
		t.Fatalf("NewParameter(%q): %v", name, err)
	}
	if err := value.Close(); err != nil {
		t.Fatalf("Close tensor: %v", err)
	}
	return parameter
}

// testParam mengikat satu parameter dengan nama lengkapnya.
func testParam(t *testing.T, name string, values []float64, dims ...int64) Param {
	t.Helper()
	return Param{Name: name, Parameter: testParameter(t, name, values, dims...)}
}

// valuesOf membaca seluruh nilai float64 parameter.
func valuesOf(t *testing.T, parameter *Param) []float64 {
	t.Helper()
	values, err := tensor.Read[float64](parameter.Parameter.Value())
	if err != nil {
		t.Fatalf("tensor.Read(%q): %v", parameter.Name, err)
	}
	return values
}

// assertClose membandingkan dua deretan nilai dengan toleransi absolut dan relatif.
func assertClose(t *testing.T, label string, got, want []float64, tolerance float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: got %d values, want %d: %v vs %v", label, len(got), len(want), got, want)
	}
	for index := range want {
		difference := math.Abs(got[index] - want[index])
		limit := tolerance + tolerance*math.Abs(want[index])
		if math.IsNaN(difference) || difference > limit {
			t.Fatalf("%s[%d]: got %g, want %g (diff %g)", label, index, got[index], want[index], difference)
		}
	}
}

// mathClose membandingkan dua skalar dengan toleransi absolut dan relatif.
func mathClose(got, want float64) bool {
	difference := math.Abs(got - want)
	return !math.IsNaN(difference) && difference <= 1e-12+1e-12*math.Abs(want)
}

// assertLess menolak nilai yang tidak lebih kecil dari batas.
func assertLess(t *testing.T, label string, got, limit float64) {
	t.Helper()
	if !(got < limit) {
		t.Fatalf("%s: got %g, want < %g", label, got, limit)
	}
}

// applyGradientUsingMul memasang gradien eksak pada Context lewat operasi
// perkalian dengan konstanta. Ini memakai jalur autograd yang sama dengan
// pelatihan nyata, bukan palsu state gradien.
func applyGradientUsingMul(t *testing.T, ctx *nn.Context, parameter *Param, gradient []float64) {
	t.Helper()
	value := parameter.Parameter.Value()
	metadata, err := value.Metadata()
	if err != nil {
		t.Fatalf("Metadata(%q): %v", parameter.Name, err)
	}
	dims := metadata.Shape().Dims()
	var factor *tensor.Tensor
	if metadata.DType() == dtype.Float32 {
		converted := make([]float32, len(gradient))
		for index, item := range gradient {
			converted[index] = float32(item)
		}
		factor, err = tensor.NewTensor(dtype.Float32, dims...)
		if err == nil {
			err = tensor.Write(factor, converted)
		}
	} else {
		factor, err = tensor.NewTensor(dtype.Float64, dims...)
		if err == nil {
			err = tensor.Write(factor, gradient)
		}
	}
	if err != nil {
		t.Fatalf("factor tensor for %q: %v", parameter.Name, err)
	}
	defer factor.Close()
	factorVariable, err := ctx.Tape.NewVariable(factor, false)
	if err != nil {
		t.Fatalf("NewVariable(factor): %v", err)
	}
	parameterVariable, err := ctx.Variable(parameter.Parameter)
	if err != nil {
		t.Fatalf("Variable(%q): %v", parameter.Name, err)
	}
	product, err := ctx.Tape.Mul(parameterVariable, factorVariable)
	if err != nil {
		t.Fatalf("Mul(%q): %v", parameter.Name, err)
	}
	if err := ctx.Tape.Backward(context.Background(), product); err != nil {
		t.Fatalf("Backward(%q): %v", parameter.Name, err)
	}
}

// newGradientContext membuat Tape dan Context baru untuk satu langkah.
func newGradientContext(t *testing.T) (*nn.Context, func()) {
	t.Helper()
	tape, err := nn.NewTape()
	if err != nil {
		t.Fatalf("nn.NewTape: %v", err)
	}
	ctx, err := nn.NewContext(tape)
	if err != nil {
		t.Fatalf("nn.NewContext: %v", err)
	}
	return ctx, func() {
		if err := tape.Close(); err != nil {
			t.Errorf("tape.Close: %v", err)
		}
	}
}

func TestParametersAndSelection(t *testing.T) {
	root, err := nn.NewModule("model")
	if err != nil {
		t.Fatalf("NewModule: %v", err)
	}
	defer root.Close()
	child, err := nn.NewModule("layer1")
	if err != nil {
		t.Fatalf("NewModule: %v", err)
	}
	if err := root.RegisterModule(child); err != nil {
		t.Fatalf("RegisterModule: %v", err)
	}
	if err := root.RegisterParameter("root.bias", testParameter(t, "root.bias", []float64{1})); err != nil {
		t.Fatalf("RegisterParameter: %v", err)
	}
	if err := child.RegisterParameter("weight", testParameter(t, "layer1.weight", []float64{2, 3}, 2)); err != nil {
		t.Fatalf("RegisterParameter: %v", err)
	}

	all := Parameters(root)
	if len(all) != 2 || all[0].Name != "model.layer1.weight" || all[1].Name != "model.root.bias" {
		t.Fatalf("Parameters: got %+v", all)
	}
	if Parameters(nil) != nil {
		t.Fatal("Parameters(nil) harus nil")
	}

	selected, err := SelectParameters(root, "model.layer1.*")
	if err != nil {
		t.Fatalf("SelectParameters: %v", err)
	}
	if len(selected) != 1 || selected[0].Name != "model.layer1.weight" {
		t.Fatalf("SelectParameters: got %+v", selected)
	}
	selected, err = SelectParameters(root, "*.bias", "model.layer*.weight")
	if err != nil {
		t.Fatalf("SelectParameters: %v", err)
	}
	if len(selected) != 2 {
		t.Fatalf("SelectParameters dengan dua pola: got %+v", selected)
	}
	if _, err := SelectParameters(root); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("SelectParameters tanpa pola: got %v", err)
	}
	if _, err := SelectParameters(root, "["); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("SelectParameters pola rusak: got %v", err)
	}
}

func TestValidateParametersRejectsInvalidInput(t *testing.T) {
	valid := testParam(t, "weight", []float64{1}, 1)
	defer valid.Parameter.Close()

	cases := []struct {
		name       string
		parameters []Param
		want       error
	}{
		{"empty", nil, errs.ErrInvalidArgument},
		{"nil parameter", []Param{{Name: "x"}}, errs.ErrInvalidArgument},
		{"empty name", []Param{{Name: "", Parameter: valid.Parameter}}, errs.ErrInvalidArgument},
		{"duplicate name", []Param{valid, valid}, errs.ErrInvalidArgument},
	}
	for _, item := range cases {
		t.Run(item.name, func(t *testing.T) {
			if _, err := validateParameters(item.parameters); !errors.Is(err, item.want) {
				t.Fatalf("validateParameters: got %v, want %v", err, item.want)
			}
		})
	}

	closed := testParameter(t, "closed", []float64{1}, 1)
	if err := closed.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if _, err := validateParameters([]Param{{Name: "closed", Parameter: closed}}); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("parameter tertutup: got %v", err)
	}

	integerValue, err := tensor.NewTensor(dtype.Int64, 1)
	if err != nil {
		t.Fatalf("NewTensor: %v", err)
	}
	defer integerValue.Close()
	integerParameter, err := nn.NewParameter("integer", integerValue, false)
	if err != nil {
		t.Fatalf("NewParameter: %v", err)
	}
	defer integerParameter.Close()
	if _, err := paramCount(integerParameter); !errors.Is(err, errs.ErrUnsupportedDType) {
		t.Fatalf("paramCount dtype tidak didukung: got %v", err)
	}
}

func TestOptimizerConfigurationValidation(t *testing.T) {
	nan := math.NaN()
	for _, rate := range []float64{0, -0.1, nan, 1e31} {
		if _, err := NewSGD(SGDConfig{LearningRate: rate}); !errors.Is(err, errs.ErrInvalidArgument) {
			t.Fatalf("NewSGD lr %g: got %v", rate, err)
		}
		if _, err := NewAdam(AdamConfig{LearningRate: rate}); !errors.Is(err, errs.ErrInvalidArgument) {
			t.Fatalf("NewAdam lr %g: got %v", rate, err)
		}
	}
	for _, config := range []SGDConfig{
		{LearningRate: 0.1, Momentum: 1},
		{LearningRate: 0.1, Momentum: -0.1},
		{LearningRate: 0.1, WeightDecay: -1},
	} {
		if _, err := NewSGD(config); !errors.Is(err, errs.ErrInvalidArgument) {
			t.Fatalf("NewSGD %+v: got %v", config, err)
		}
	}
	for _, config := range []AdamConfig{
		{LearningRate: 0.1, Beta1: 1},
		{LearningRate: 0.1, Beta2: -0.1},
		{LearningRate: 0.1, Epsilon: -1},
		{LearningRate: 0.1, WeightDecay: -1},
	} {
		if _, err := NewAdam(config); !errors.Is(err, errs.ErrInvalidArgument) {
			t.Fatalf("NewAdam %+v: got %v", config, err)
		}
	}

	sgd, err := NewSGD(SGDConfig{LearningRate: 0.1})
	if err != nil {
		t.Fatalf("NewSGD: %v", err)
	}
	defer sgd.Close()
	adam, err := NewAdam(AdamConfig{LearningRate: 0.1})
	if err != nil {
		t.Fatalf("NewAdam: %v", err)
	}
	defer adam.Close()
	if sgd.Name() != "sgd" || adam.Name() != "adam" {
		t.Fatalf("Name: got %q and %q", sgd.Name(), adam.Name())
	}
	if err := sgd.SetLearningRate(0); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("SetLearningRate(0): got %v", err)
	}
	if err := adam.SetLearningRate(math.Inf(1)); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("SetLearningRate(Inf): got %v", err)
	}
	if err := sgd.SetLearningRate(0.25); err != nil || sgd.LearningRate() != 0.25 {
		t.Fatalf("SetLearningRate: got rate %g, err %v", sgd.LearningRate(), err)
	}
	if err := adam.SetLearningRate(0.35); err != nil || adam.LearningRate() != 0.35 {
		t.Fatalf("SetLearningRate: got rate %g, err %v", adam.LearningRate(), err)
	}

	adamW, err := NewAdam(AdamConfig{LearningRate: 0.1, AdamW: true})
	if err != nil {
		t.Fatalf("NewAdam(AdamW): %v", err)
	}
	defer adamW.Close()
	if adamW.Name() != "adamw" {
		t.Fatalf("AdamW Name: got %q", adamW.Name())
	}
}

func TestOptimizerStepRejectsInvalidInput(t *testing.T) {
	parameter := testParam(t, "weight", []float64{1, 2}, 2)
	defer parameter.Parameter.Close()
	ctx, release := newGradientContext(t)
	defer release()
	applyGradientUsingMul(t, ctx, &parameter, []float64{0.5, -0.5})

	sgd, err := NewSGD(SGDConfig{LearningRate: 0.1})
	if err != nil {
		t.Fatalf("NewSGD: %v", err)
	}
	defer sgd.Close()
	adam, err := NewAdam(AdamConfig{LearningRate: 0.1})
	if err != nil {
		t.Fatalf("NewAdam: %v", err)
	}
	defer adam.Close()

	for name, step := range map[string]func(*nn.Context, []Param) error{
		"sgd":  sgd.Step,
		"adam": adam.Step,
	} {
		t.Run(name, func(t *testing.T) {
			if err := step(ctx, nil); !errors.Is(err, errs.ErrInvalidArgument) {
				t.Fatalf("Step tanpa parameter: got %v", err)
			}
			if err := step(nil, []Param{parameter}); !errors.Is(err, errs.ErrInvalidArgument) {
				t.Fatalf("Step tanpa Context: got %v", err)
			}
		})
	}
}

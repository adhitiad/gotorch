package nn

import (
	"math"
	"testing"

	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/tensor"
)

func TestModuleCompositionAndMode(t *testing.T) {
	root, err := NewModule("root")
	if err != nil {
		t.Fatalf("NewModule: %v", err)
	}
	defer root.Close()
	child, err := NewModule("child")
	if err != nil {
		t.Fatalf("NewModule: %v", err)
	}
	if err := root.RegisterModule(child); err != nil {
		t.Fatalf("RegisterModule: %v", err)
	}
	weight := newTestTensor(t, dtype.Float64, []int64{1}, []float64{1})
	parameter, err := NewParameter("weight", weight, true)
	if err != nil {
		t.Fatalf("NewParameter: %v", err)
	}
	if err := child.RegisterParameter("weight", parameter); err != nil {
		t.Fatalf("RegisterParameter: %v", err)
	}
	running := newTestTensor(t, dtype.Float64, []int64{1}, []float64{2})
	buffer, err := NewBuffer("running", running, true)
	if err != nil {
		t.Fatalf("NewBuffer: %v", err)
	}
	if err := root.RegisterBuffer("running", buffer); err != nil {
		t.Fatalf("RegisterBuffer: %v", err)
	}
	transient := newTestTensor(t, dtype.Float64, []int64{1}, []float64{3})
	temporary, err := NewBuffer("temporary", transient, false)
	if err != nil {
		t.Fatalf("NewBuffer: %v", err)
	}
	if err := root.RegisterBuffer("temporary", temporary); err != nil {
		t.Fatalf("RegisterBuffer: %v", err)
	}

	parameters := root.Parameters()
	if len(parameters) != 1 || parameters[0].Name != "root.child.weight" {
		t.Fatalf("Parameters: %+v", parameters)
	}
	buffers := root.Buffers()
	if len(buffers) != 1 || buffers[0].Name != "root.running" {
		t.Fatalf("Buffers: %+v", buffers)
	}
	root.Train(false)
	if root.IsTraining() || child.IsTraining() {
		t.Fatal("Train(false) tidak merambat ke anak")
	}
	root.Train(true)
	if !root.IsTraining() || !child.IsTraining() {
		t.Fatal("Train(true) tidak merambat ke anak")
	}
}

func stateModule(t *testing.T, weightValue, bufferValue float64) *ModuleBase {
	t.Helper()
	module, err := NewModule("state")
	if err != nil {
		t.Fatalf("NewModule: %v", err)
	}
	weight := newTestTensor(t, dtype.Float64, []int64{1}, []float64{weightValue})
	parameter, err := NewParameter("weight", weight, true)
	if err != nil {
		t.Fatalf("NewParameter: %v", err)
	}
	if err := module.RegisterParameter("weight", parameter); err != nil {
		t.Fatalf("RegisterParameter: %v", err)
	}
	running := newTestTensor(t, dtype.Float64, []int64{1}, []float64{bufferValue})
	buffer, err := NewBuffer("running", running, true)
	if err != nil {
		t.Fatalf("NewBuffer: %v", err)
	}
	if err := module.RegisterBuffer("running", buffer); err != nil {
		t.Fatalf("RegisterBuffer: %v", err)
	}
	return module
}

func TestStateDictBinaryRoundTrip(t *testing.T) {
	source := stateModule(t, 1.5, 2.5)
	defer source.Close()
	state, err := StateDictOf(source)
	if err != nil {
		t.Fatalf("StateDictOf: %v", err)
	}
	defer state.Close()
	if got := state.Names(); len(got) != 2 || got[0] != "state.running" || got[1] != "state.weight" {
		t.Fatalf("Names: %v", got)
	}
	encoded, err := state.MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary: %v", err)
	}
	decoded := NewStateDict()
	defer decoded.Close()
	if err := decoded.UnmarshalBinary(encoded); err != nil {
		t.Fatalf("UnmarshalBinary: %v", err)
	}
	if got := decoded.Names(); len(got) != 2 || got[0] != "state.running" || got[1] != "state.weight" {
		t.Fatalf("decoded Names: %v", got)
	}

	target := stateModule(t, 9, 9)
	defer target.Close()
	if err := LoadStateDict(target, decoded); err != nil {
		t.Fatalf("LoadStateDict: %v", err)
	}
	if got := tensorValues(t, target.Parameters()[0].Parameter.Value()); len(got) != 1 || math.Abs(got[0]-1.5) > 1e-12 {
		t.Fatalf("loaded weight: %v", got)
	}
	if got := tensorValues(t, target.Buffers()[0].Buffer.Value()); len(got) != 1 || math.Abs(got[0]-2.5) > 1e-12 {
		t.Fatalf("loaded buffer: %v", got)
	}
}

func TestContextBindsParameterOnce(t *testing.T) {
	tape, ctx := newTestContext(t)
	defer tape.Close()
	parameter, err := NewParameter("weight", newTestTensor(t, dtype.Float64, []int64{1}, []float64{1}), true)
	if err != nil {
		t.Fatalf("NewParameter: %v", err)
	}
	defer parameter.Close()
	first, err := ctx.Variable(parameter)
	if err != nil {
		t.Fatalf("Variable: %v", err)
	}
	second, err := ctx.Variable(parameter)
	if err != nil {
		t.Fatalf("Variable: %v", err)
	}
	if first != second {
		t.Fatal("Context.Variable membuat leaf ganda untuk parameter yang sama")
	}
}

func TestInitializersDeterministic(t *testing.T) {
	first := newTestTensor(t, dtype.Float64, []int64{2, 3}, make([]float64, 6))
	second := newTestTensor(t, dtype.Float64, []int64{2, 3}, make([]float64, 6))
	if err := Uniform(NewRNG(7), first, -1, 1); err != nil {
		t.Fatalf("Uniform: %v", err)
	}
	if err := Uniform(NewRNG(7), second, -1, 1); err != nil {
		t.Fatalf("Uniform: %v", err)
	}
	left, right := tensorValues(t, first), tensorValues(t, second)
	for index := range left {
		if left[index] != right[index] {
			t.Fatalf("seed berbeda pada indeks %d: %g vs %g", index, left[index], right[index])
		}
		if left[index] < -1 || left[index] >= 1 {
			t.Fatalf("nilai di luar rentang: %g", left[index])
		}
	}
	if err := Zeros(first); err != nil {
		t.Fatalf("Zeros: %v", err)
	}
	if err := Ones(second); err != nil {
		t.Fatalf("Ones: %v", err)
	}
	for index, value := range tensorValues(t, first) {
		if value != 0 || tensorValues(t, second)[index] != 1 {
			t.Fatalf("Zeros/Ones gagal pada indeks %d", index)
		}
	}
	for _, initializer := range []struct {
		name string
		fn   func(*tensor.Tensor) error
	}{
		{"xavier_uniform", func(value *tensor.Tensor) error { return XavierUniform(NewRNG(11), value) }},
		{"xavier_normal", func(value *tensor.Tensor) error { return XavierNormal(NewRNG(11), value) }},
		{"kaiming_uniform", func(value *tensor.Tensor) error { return KaimingUniform(NewRNG(11), value) }},
		{"kaiming_normal", func(value *tensor.Tensor) error { return KaimingNormal(NewRNG(11), value) }},
	} {
		value := newTestTensor(t, dtype.Float64, []int64{4, 6}, make([]float64, 24))
		if err := initializer.fn(value); err != nil {
			t.Fatalf("%s: %v", initializer.name, err)
		}
		for _, item := range tensorValues(t, value) {
			if math.IsNaN(item) || math.IsInf(item, 0) {
				t.Fatalf("%s menghasilkan %g", initializer.name, item)
			}
		}
	}
}

func TestDropoutMode(t *testing.T) {
	tape, ctx := newTestContext(t)
	defer tape.Close()
	dropout, err := NewDropout("dropout", DropoutConfig{Probability: 0.5, RNG: NewRNG(3)})
	if err != nil {
		t.Fatalf("NewDropout: %v", err)
	}
	defer dropout.Close()
	input := variableOf(t, tape, newTestTensor(t, dtype.Float64, []int64{4}, []float64{1, 1, 1, 1}), false)
	dropout.Train(false)
	evaluated, err := dropout.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if evaluated != input {
		t.Fatal("Dropout eval mode tidak mengembalikan input")
	}
	dropout.Train(true)
	trained, err := dropout.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if trained == input {
		t.Fatal("Dropout train mode tidak menghasilkan variabel baru")
	}
}

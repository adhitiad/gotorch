package peft

import (
	"math/rand"
	"testing"

	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/nn"
	"github.com/adhitiad/gotorch/src/tensor"
)

func newLoRA(t *testing.T, name string, base Projector, rank int64, alpha float64) *LoRALinear {
	t.Helper()
	wrapper, err := NewLoRALinear(name, base, LoRAConfig{
		Rank: rank, Alpha: alpha, DType: dtype.Float64, RNG: rand.New(rand.NewSource(5)),
	})
	if err != nil {
		t.Fatalf("NewLoRALinear(%q): %v", name, err)
	}
	t.Cleanup(func() {
		if err := wrapper.Close(); err != nil {
			t.Errorf("wrapper.Close(%q): %v", name, err)
		}
	})
	return wrapper
}

func TestLoRAConstructionValidationAndDeterministicInit(t *testing.T) {
	base := newLinearWithWeight(t, "base", 2, 3, []float64{1, 0, 0, 1, 0.5, -0.5})
	first := newLoRA(t, "lora", base, 2, 4)
	if first.Scale != 2 {
		t.Fatalf("Scale = %g, diharapkan 2", first.Scale)
	}
	if first.InputDim() != 2 || first.OutputDim() != 3 {
		t.Fatalf("dimensi = (%d, %d), diharapkan (2, 3)", first.InputDim(), first.OutputDim())
	}
	for _, item := range []struct {
		name      string
		parameter *nn.Parameter
		wantDims  []int64
	}{
		{name: "A", parameter: first.A, wantDims: []int64{2, 2}},
		{name: "B", parameter: first.B, wantDims: []int64{3, 2}},
	} {
		metadata, err := item.parameter.Value().Metadata()
		if err != nil {
			t.Fatalf("Metadata(%s): %v", item.name, err)
		}
		if !equalInt64(metadata.Shape().Dims(), item.wantDims) {
			t.Fatalf("dimensi %s = %v, diharapkan %v", item.name, metadata.Shape().Dims(), item.wantDims)
		}
		if !item.parameter.RequiresGrad() {
			t.Fatalf("%s seharusnya trainable", item.name)
		}
	}

	second := newLoRA(t, "lora_lain", base, 2, 4)
	assertClose(t, "inisialisasi A deterministik", parameterValues(t, first.A), parameterValues(t, second.A), 0)
	assertClose(t, "inisialisasi B deterministik", parameterValues(t, first.B), parameterValues(t, second.B), 0)

	cases := []struct {
		label  string
		name   string
		base   Projector
		config LoRAConfig
		want   error
	}{
		{label: "nama kosong", name: "", base: base, want: errs.ErrInvalidArgument},
		{label: "base nil", name: "x", base: nil, want: errs.ErrInvalidArgument},
		{label: "rank nol", name: "x", base: base, want: errs.ErrInvalidArgument},
		{label: "RNG nil", name: "x", base: base, want: errs.ErrInvalidArgument},
		{label: "dtype tidak diferensial", name: "x", base: base, want: errs.ErrUnsupportedDType},
	}
	configs := []LoRAConfig{
		{Rank: 2, Alpha: 2, DType: dtype.Float64, RNG: rand.New(rand.NewSource(5))},
		{Rank: 2, Alpha: 2, DType: dtype.Float64, RNG: rand.New(rand.NewSource(5))},
		{Rank: 0, Alpha: 2, DType: dtype.Float64, RNG: rand.New(rand.NewSource(5))},
		{Rank: 2, Alpha: 2, DType: dtype.Float64},
		{Rank: 2, Alpha: 2, DType: dtype.Int64, RNG: rand.New(rand.NewSource(5))},
	}
	for index, item := range cases {
		t.Run(item.label, func(t *testing.T) {
			config := configs[index]
			if item.base == nil {
				config.RNG = rand.New(rand.NewSource(5))
			}
			_, err := NewLoRALinear(item.name, item.base, config)
			assertErrorIs(t, err, item.want)
		})
	}
	_, err := NewLoRALinear("rank_besar", base, LoRAConfig{Rank: 4, Alpha: 4, DType: dtype.Float64, RNG: rand.New(rand.NewSource(5))})
	assertErrorIs(t, err, errs.ErrInvalidArgument)
}

func runLoRAForward(t *testing.T, wrapper *LoRALinear, inputValues []float64) []float64 {
	t.Helper()
	_, ctx := newTestTape(t)
	tape := ctx.Tape
	input := newVariable(t, tape, newFloatTensor(t, []int64{2, 2}, inputValues), false)
	output, err := wrapper.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	value, err := output.Value()
	if err != nil {
		t.Fatalf("output.Value: %v", err)
	}
	defer func() { _ = value.Close() }()
	return floatValues(t, value)
}

func prepareLoRAFixture(t *testing.T) (*LoRALinear, []float64) {
	t.Helper()
	base := newLinearWithWeight(t, "base", 2, 2, []float64{0.5, -0.25, 1.5, 0.75})
	if err := Freeze(base); err != nil {
		t.Fatalf("Freeze(base): %v", err)
	}
	wrapper := newLoRA(t, "lora", base, 1, 2)
	setParameterValues(t, wrapper.A, []float64{0.5, -1})
	setParameterValues(t, wrapper.B, []float64{2, -3})
	return wrapper, []float64{1, 2, 3, 4}
}

func TestLoRAForwardFrozenBaseAndGradients(t *testing.T) {
	wrapper, inputValues := prepareLoRAFixture(t)
	got := runLoRAForward(t, wrapper, inputValues)
	assertClose(t, "forward LoRA", got, []float64{-6, 12, -9.5, 22.5}, 1e-12)

	tape, ctx := newTestTape(t)
	input := newVariable(t, tape, newFloatTensor(t, []int64{2, 2}, inputValues), false)
	output, err := wrapper.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	backwardSum(t, tape, output)
	gradientA, err := gradientValues(t, ctx, wrapper.A)
	if err != nil {
		t.Fatalf("gradient A: %v", err)
	}
	gradientB, err := gradientValues(t, ctx, wrapper.B)
	if err != nil {
		t.Fatalf("gradient B: %v", err)
	}
	assertClose(t, "gradien A", gradientA, []float64{-8, -12}, 1e-12)
	assertClose(t, "gradien B", gradientB, []float64{-8, -8}, 1e-12)
	if wrapper.Base.(*nn.Linear).Weight.RequiresGrad() {
		t.Fatal("base seharusnya tetap beku")
	}
	assertNoGradient(t, ctx, wrapper.Base.(*nn.Linear).Weight)
}

func TestLoRAGradientWithTrainableBase(t *testing.T) {
	wrapper, inputValues := prepareLoRAFixture(t)
	base := wrapper.Base.(*nn.Linear)
	if err := Unfreeze(base); err != nil {
		t.Fatalf("Unfreeze(base): %v", err)
	}
	tape, ctx := newTestTape(t)
	input := newVariable(t, tape, newFloatTensor(t, []int64{2, 2}, inputValues), false)
	output, err := wrapper.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	backwardSum(t, tape, output)
	gradientWeight, err := gradientValues(t, ctx, base.Weight)
	if err != nil {
		t.Fatalf("gradien weight: %v", err)
	}
	assertClose(t, "gradien weight base", gradientWeight, []float64{4, 6, 4, 6}, 1e-12)
}

func TestLoRAMergeUnmergeAndIdempotence(t *testing.T) {
	wrapper, inputValues := prepareLoRAFixture(t)
	base := wrapper.Base.(*nn.Linear)
	original := parameterValues(t, base.Weight)
	before := runLoRAForward(t, wrapper, inputValues)

	if err := wrapper.Merge(); err != nil {
		t.Fatalf("Merge: %v", err)
	}
	if !wrapper.IsMerged() {
		t.Fatal("Merge seharusnya menandai wrapper sebagai tergabung")
	}
	assertClose(t, "weight setelah merge", parameterValues(t, base.Weight), []float64{2.5, -4.25, -1.5, 6.75}, 1e-12)
	assertClose(t, "forward setelah merge", runLoRAForward(t, wrapper, inputValues), before, 1e-12)
	if err := wrapper.Merge(); err != nil {
		t.Fatalf("Merge kedua: %v", err)
	}
	assertClose(t, "weight setelah merge kedua", parameterValues(t, base.Weight), []float64{2.5, -4.25, -1.5, 6.75}, 1e-12)

	tape, ctx := newTestTape(t)
	input := newVariable(t, tape, newFloatTensor(t, []int64{2, 2}, inputValues), false)
	output, err := wrapper.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward tergabung: %v", err)
	}
	if output.RequiresGrad() {
		t.Fatal("forward tergabung tidak boleh membangun jalur gradien adapter")
	}
	for _, parameter := range ctx.Parameters() {
		if parameter == wrapper.A || parameter == wrapper.B {
			t.Fatalf("parameter %q tidak boleh terikat ketika adapter tergabung", parameter.Name())
		}
	}

	if err := wrapper.Unmerge(); err != nil {
		t.Fatalf("Unmerge: %v", err)
	}
	if wrapper.IsMerged() {
		t.Fatal("Unmerge seharusnya menandai wrapper sebagai terpisah")
	}
	assertClose(t, "weight setelah unmerge", parameterValues(t, base.Weight), original, 1e-12)
	assertClose(t, "forward setelah unmerge", runLoRAForward(t, wrapper, inputValues), before, 1e-12)
	if err := wrapper.Unmerge(); err != nil {
		t.Fatalf("Unmerge kedua: %v", err)
	}
	assertClose(t, "weight setelah unmerge kedua", parameterValues(t, base.Weight), original, 1e-12)
}

func TestLoRAMergeRequiresLinearLikeBase(t *testing.T) {
	opaque := newOpaqueProjector(t, "opaque", 2, 2)
	wrapper := newLoRA(t, "lora_opaque", opaque, 1, 2)
	err := wrapper.Merge()
	assertErrorIs(t, err, errs.ErrUnsupportedOperation)
	if wrapper.IsMerged() {
		t.Fatal("merge gagal seharusnya tidak mengubah status")
	}
}

func equalInt64(left, right []int64) bool {
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

func assertShape(t *testing.T, value *tensor.Tensor, want []int64) {
	t.Helper()
	metadata, err := value.Metadata()
	if err != nil {
		t.Fatalf("Metadata: %v", err)
	}
	if !equalInt64(metadata.Shape().Dims(), want) {
		t.Fatalf("dimensi = %v, diharapkan %v", metadata.Shape().Dims(), want)
	}
}

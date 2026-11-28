package peft

import (
	"math/rand"
	"testing"

	"github.com/adhitiad/gotorch/src/autograd"
	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/nn"
)

type opaqueProjector struct {
	*nn.ModuleBase
	inputDim  int64
	outputDim int64
}

func newOpaqueProjector(t *testing.T, name string, in, out int64) *opaqueProjector {
	t.Helper()
	base, err := nn.NewModule(name)
	if err != nil {
		t.Fatalf("nn.NewModule(%q): %v", name, err)
	}
	t.Cleanup(func() {
		if err := base.Close(); err != nil {
			t.Errorf("base.Close(%q): %v", name, err)
		}
	})
	return &opaqueProjector{ModuleBase: base, inputDim: in, outputDim: out}
}

func (p *opaqueProjector) Forward(_ *nn.Context, input *autograd.Variable) (*autograd.Variable, error) {
	return input, nil
}

func (p *opaqueProjector) InputDim() int64  { return p.inputDim }
func (p *opaqueProjector) OutputDim() int64 { return p.outputDim }

func newAdapterFixture(t *testing.T) (*Adapter, *nn.Linear, *nn.Linear) {
	t.Helper()
	adapter, err := NewAdapter("adapter")
	if err != nil {
		t.Fatalf("NewAdapter: %v", err)
	}
	t.Cleanup(func() {
		if err := adapter.Close(); err != nil {
			t.Errorf("adapter.Close: %v", err)
		}
	})
	firstBase := newLinearWithWeight(t, "basis_satu", 2, 2, []float64{0.5, -0.25, 1.5, 0.75})
	secondBase := newLinearWithWeight(t, "basis_dua", 2, 2, []float64{-1, 0.25, 2, -0.5})
	for _, base := range []*nn.Linear{firstBase, secondBase} {
		if err := Freeze(base); err != nil {
			t.Fatalf("Freeze(%q): %v", base.Name(), err)
		}
	}
	config := LoRAConfig{Rank: 1, Alpha: 2, DType: dtype.Float64, RNG: rand.New(rand.NewSource(5))}
	if _, err := adapter.WrapLinear("a_lora", firstBase, config); err != nil {
		t.Fatalf("WrapLinear(a_lora): %v", err)
	}
	if _, err := adapter.WrapLinear("z_lora", secondBase, config); err != nil {
		t.Fatalf("WrapLinear(z_lora): %v", err)
	}
	return adapter, firstBase, secondBase
}

func fillAdapterValues(t *testing.T, adapter *Adapter) {
	t.Helper()
	wrappers := adapter.Wrappers()
	setParameterValues(t, wrappers[0].A, []float64{0.5, -1})
	setParameterValues(t, wrappers[0].B, []float64{2, -3})
	setParameterValues(t, wrappers[1].A, []float64{1, 0.5})
	setParameterValues(t, wrappers[1].B, []float64{-1, 2})
}

func TestAdapterWrapLinearNamedEnumerationAndValidation(t *testing.T) {
	adapter, _, base := newAdapterFixture(t)
	wrappers := adapter.Wrappers()
	gotNames := make([]string, 0, len(wrappers))
	for _, wrapper := range wrappers {
		gotNames = append(gotNames, wrapper.Name())
	}
	wantNames := []string{"a_lora", "z_lora"}
	if !equalStrings(gotNames, wantNames) {
		t.Fatalf("nama wrapper = %v, diharapkan %v", gotNames, wantNames)
	}
	wrappers[0] = nil
	if adapter.Wrappers()[0] == nil {
		t.Fatal("Wrappers seharusnya mengembalikan salinan")
	}

	names := trainableNames(t, adapter)
	want := []string{"adapter.a_lora.a", "adapter.a_lora.b", "adapter.z_lora.a", "adapter.z_lora.b"}
	if !equalStrings(names, want) {
		t.Fatalf("parameter trainable = %v, diharapkan %v", names, want)
	}

	config := LoRAConfig{Rank: 1, Alpha: 2, DType: dtype.Float64, RNG: rand.New(rand.NewSource(5))}
	_, err := adapter.WrapLinear("a_lora", base, config)
	assertErrorIs(t, err, errs.ErrInvalidArgument)
	var nilAdapter *Adapter
	_, err = nilAdapter.WrapLinear("x", base, config)
	assertErrorIs(t, err, errs.ErrInvalidArgument)
	if nilAdapter.Wrappers() != nil {
		t.Fatal("Wrappers pada adapter nil seharusnya nil")
	}
}

func TestAdapterStateRoundTrip(t *testing.T) {
	adapter, _, _ := newAdapterFixture(t)
	fillAdapterValues(t, adapter)
	wantFirstA := parameterValues(t, adapter.Wrappers()[0].A)
	wantSecondB := parameterValues(t, adapter.Wrappers()[1].B)

	data, err := adapter.MarshalState()
	if err != nil {
		t.Fatalf("MarshalState: %v", err)
	}
	decoded := nn.NewStateDict()
	defer func() { _ = decoded.Close() }()
	if err := decoded.UnmarshalBinary(data); err != nil {
		t.Fatalf("UnmarshalBinary: %v", err)
	}
	foundFlag := false
	for _, name := range decoded.Names() {
		if name == mergedStateKey {
			foundFlag = true
		}
	}
	if !foundFlag {
		t.Fatalf("state tidak memuat penanda %q: %v", mergedStateKey, decoded.Names())
	}

	for _, wrapper := range adapter.Wrappers() {
		setParameterValues(t, wrapper.A, []float64{0, 0})
		setParameterValues(t, wrapper.B, []float64{0, 0})
	}
	if err := adapter.LoadState(data); err != nil {
		t.Fatalf("LoadState: %v", err)
	}
	assertClose(t, "A wrapper pertama", parameterValues(t, adapter.Wrappers()[0].A), wantFirstA, 0)
	assertClose(t, "B wrapper kedua", parameterValues(t, adapter.Wrappers()[1].B), wantSecondB, 0)
	for _, wrapper := range adapter.Wrappers() {
		if wrapper.IsMerged() {
			t.Fatal("status merge harusnya salah setelah state tidak tergabung")
		}
	}
}

func TestAdapterStateRestoresMergeFlagWithoutTouchingBase(t *testing.T) {
	adapter, firstBase, secondBase := newAdapterFixture(t)
	fillAdapterValues(t, adapter)
	originalFirst := parameterValues(t, firstBase.Weight)
	originalSecond := parameterValues(t, secondBase.Weight)
	if err := adapter.Merge(); err != nil {
		t.Fatalf("Merge: %v", err)
	}
	data, err := adapter.MarshalState()
	if err != nil {
		t.Fatalf("MarshalState: %v", err)
	}
	if err := adapter.Unmerge(); err != nil {
		t.Fatalf("Unmerge: %v", err)
	}
	if err := adapter.LoadState(data); err != nil {
		t.Fatalf("LoadState: %v", err)
	}
	for _, wrapper := range adapter.Wrappers() {
		if !wrapper.IsMerged() {
			t.Fatalf("wrapper %q seharusnya tergabung setelah LoadState", wrapper.Name())
		}
	}
	assertClose(t, "base pertama tidak diubah LoadState", parameterValues(t, firstBase.Weight), originalFirst, 1e-12)
	assertClose(t, "base kedua tidak diubah LoadState", parameterValues(t, secondBase.Weight), originalSecond, 1e-12)
}

func marshalState(t *testing.T, state *nn.StateDict) []byte {
	t.Helper()
	data, err := state.MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary: %v", err)
	}
	return data
}

func TestAdapterLoadStateRejectsInvalidData(t *testing.T) {
	adapter, _, _ := newAdapterFixture(t)
	good, err := adapter.MarshalState()
	if err != nil {
		t.Fatalf("MarshalState: %v", err)
	}
	corrupted := append([]byte(nil), good...)
	corrupted[0] = 'X'

	missingFlag := nn.NewStateDict()
	defer func() { _ = missingFlag.Close() }()
	if err := missingFlag.Put("bukan_penanda", newFloatTensor(t, nil, []float64{1})); err != nil {
		t.Fatalf("Put: %v", err)
	}

	invalidFlag := nn.NewStateDict()
	defer func() { _ = invalidFlag.Close() }()
	if err := invalidFlag.Put(mergedStateKey, intTensor(t, nil, []int64{2})); err != nil {
		t.Fatalf("Put(flag): %v", err)
	}

	unknown := nn.NewStateDict()
	defer func() { _ = unknown.Close() }()
	state, err := nn.StateDictOf(adapter)
	if err != nil {
		t.Fatalf("StateDictOf: %v", err)
	}
	defer func() { _ = state.Close() }()
	for _, name := range state.Names() {
		value, err := state.Get(name)
		if err != nil {
			t.Fatalf("Get(%q): %v", name, err)
		}
		if err := unknown.Put(name, value); err != nil {
			t.Fatalf("Put(%q): %v", name, err)
		}
		_ = value.Close()
	}
	if err := unknown.Put(mergedStateKey, intTensor(t, nil, []int64{0})); err != nil {
		t.Fatalf("Put(flag): %v", err)
	}
	if err := unknown.Put("entri_asing", newFloatTensor(t, nil, []float64{1})); err != nil {
		t.Fatalf("Put(entri_asing): %v", err)
	}

	cases := []struct {
		label string
		data  []byte
		want  error
	}{
		{label: "data kosong", data: nil, want: errs.ErrInvalidArgument},
		{label: "magic rusak", data: corrupted, want: errs.ErrInvalidArgument},
		{label: "penanda hilang", data: marshalState(t, missingFlag), want: errs.ErrInvalidArgument},
		{label: "nilai penanda salah", data: marshalState(t, invalidFlag), want: errs.ErrInvalidSerialization},
		{label: "entri asing", data: marshalState(t, unknown), want: errs.ErrInvalidArgument},
	}
	for _, item := range cases {
		t.Run(item.label, func(t *testing.T) {
			assertErrorIs(t, adapter.LoadState(item.data), item.want)
		})
	}
	var nilAdapter *Adapter
	assertErrorIs(t, nilAdapter.LoadState(good), errs.ErrInvalidArgument)
	_, err = nilAdapter.MarshalState()
	assertErrorIs(t, err, errs.ErrInvalidArgument)
}

func TestAdapterMergeRollbackWhenSecondWrapperFails(t *testing.T) {
	adapter, err := NewAdapter("adapter")
	if err != nil {
		t.Fatalf("NewAdapter: %v", err)
	}
	defer func() { _ = adapter.Close() }()
	goodBase := newLinearWithWeight(t, "basis_baik", 2, 2, []float64{0.5, -0.25, 1.5, 0.75})
	if err := Freeze(goodBase); err != nil {
		t.Fatalf("Freeze: %v", err)
	}
	config := LoRAConfig{Rank: 1, Alpha: 2, DType: dtype.Float64, RNG: rand.New(rand.NewSource(5))}
	good, err := adapter.WrapLinear("a_baik", goodBase, config)
	if err != nil {
		t.Fatalf("WrapLinear(a_baik): %v", err)
	}
	setParameterValues(t, good.A, []float64{0.5, -1})
	setParameterValues(t, good.B, []float64{2, -3})
	original := parameterValues(t, goodBase.Weight)
	if _, err := adapter.WrapLinear("z_gagal", newOpaqueProjector(t, "opaque", 2, 2), config); err != nil {
		t.Fatalf("WrapLinear(z_gagal): %v", err)
	}

	assertErrorIs(t, adapter.Merge(), errs.ErrUnsupportedOperation)
	if good.IsMerged() {
		t.Fatal("wrapper pertama seharusnya dikembalikan ke keadaan terpisah")
	}
	assertClose(t, "weight setelah rollback merge", parameterValues(t, goodBase.Weight), original, 1e-12)
}

func TestAdapterUnmergeRollbackWhenSecondWrapperFails(t *testing.T) {
	adapter, err := NewAdapter("adapter")
	if err != nil {
		t.Fatalf("NewAdapter: %v", err)
	}
	defer func() { _ = adapter.Close() }()
	goodBase := newLinearWithWeight(t, "basis_baik", 2, 2, []float64{0.5, -0.25, 1.5, 0.75})
	if err := Freeze(goodBase); err != nil {
		t.Fatalf("Freeze: %v", err)
	}
	config := LoRAConfig{Rank: 1, Alpha: 2, DType: dtype.Float64, RNG: rand.New(rand.NewSource(5))}
	good, err := adapter.WrapLinear("a_baik", goodBase, config)
	if err != nil {
		t.Fatalf("WrapLinear(a_baik): %v", err)
	}
	setParameterValues(t, good.A, []float64{0.5, -1})
	setParameterValues(t, good.B, []float64{2, -3})
	if err := good.Merge(); err != nil {
		t.Fatalf("Merge(good): %v", err)
	}
	mergedWeight := parameterValues(t, goodBase.Weight)
	bad, err := adapter.WrapLinear("z_gagal", newOpaqueProjector(t, "opaque", 2, 2), config)
	if err != nil {
		t.Fatalf("WrapLinear(z_gagal): %v", err)
	}
	bad.mu.Lock()
	bad.merged = true
	bad.mu.Unlock()

	assertErrorIs(t, adapter.Unmerge(), errs.ErrUnsupportedOperation)
	if !good.IsMerged() {
		t.Fatal("wrapper pertama seharusnya dikembalikan ke keadaan tergabung")
	}
	assertClose(t, "weight setelah rollback unmerge", parameterValues(t, goodBase.Weight), mergedWeight, 1e-12)
}

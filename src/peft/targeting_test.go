package peft

import (
	"testing"

	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/nn"
)

func newTargetTree(t *testing.T) *nn.ModuleBase {
	t.Helper()
	root, err := nn.NewModule("model")
	if err != nil {
		t.Fatalf("nn.NewModule: %v", err)
	}
	t.Cleanup(func() {
		if err := root.Close(); err != nil {
			t.Errorf("root.Close: %v", err)
		}
	})
	attention := newLinearWithWeight(t, "attn", 2, 2, []float64{1, 0, 0, 1})
	mlp := newLinearWithWeight(t, "mlp", 2, 3, []float64{1, 0, 0, 1, 0, 1})
	for _, child := range []nn.Module{attention, mlp} {
		if err := root.RegisterModule(child); err != nil {
			t.Fatalf("RegisterModule(%q): %v", child.Name(), err)
		}
	}
	return root
}

func trainableNames(t *testing.T, module nn.Module) []string {
	t.Helper()
	trainable, err := TrainableParameters(module)
	if err != nil {
		t.Fatalf("TrainableParameters: %v", err)
	}
	names := make([]string, 0, len(trainable))
	for _, named := range trainable {
		names = append(names, named.Name)
	}
	return names
}

func TestNamedTargetingFreezeUnfreezeAndEnumeration(t *testing.T) {
	root := newTargetTree(t)
	all := []string{"model.attn.bias", "model.attn.weight", "model.mlp.bias", "model.mlp.weight"}
	if got := trainableNames(t, root); !equalStrings(got, all) {
		t.Fatalf("parameter awal = %v, diharapkan %v", got, all)
	}

	if err := Freeze(root, "model.attn.*"); err != nil {
		t.Fatalf("Freeze(attn): %v", err)
	}
	want := []string{"model.mlp.bias", "model.mlp.weight"}
	if got := trainableNames(t, root); !equalStrings(got, want) {
		t.Fatalf("setelah Freeze(attn) = %v, diharapkan %v", got, want)
	}

	if err := Unfreeze(root, "*.bias"); err != nil {
		t.Fatalf("Unfreeze(bias): %v", err)
	}
	want = []string{"model.attn.bias", "model.mlp.bias", "model.mlp.weight"}
	if got := trainableNames(t, root); !equalStrings(got, want) {
		t.Fatalf("setelah Unfreeze(bias) = %v, diharapkan %v", got, want)
	}

	if err := Freeze(root); err != nil {
		t.Fatalf("Freeze(semua): %v", err)
	}
	if got := trainableNames(t, root); len(got) != 0 {
		t.Fatalf("setelah Freeze(semua) = %v, diharapkan kosong", got)
	}

	if err := Unfreeze(root, "model.mlp.weight"); err != nil {
		t.Fatalf("Unfreeze(weight): %v", err)
	}
	want = []string{"model.mlp.weight"}
	if got := trainableNames(t, root); !equalStrings(got, want) {
		t.Fatalf("setelah Unfreeze(weight) = %v, diharapkan %v", got, want)
	}
}

func TestTargetingArgumentValidation(t *testing.T) {
	root := newTargetTree(t)
	assertErrorIs(t, Freeze(nil), errs.ErrInvalidArgument)
	assertErrorIs(t, Unfreeze(nil), errs.ErrInvalidArgument)
	_, err := TrainableParameters(nil)
	assertErrorIs(t, err, errs.ErrInvalidArgument)
	assertErrorIs(t, Freeze(root, "["), errs.ErrInvalidArgument)
	assertErrorIs(t, Unfreeze(root, "["), errs.ErrInvalidArgument)
}

func TestFreezeRejectsInvalidPatternWithoutParameters(t *testing.T) {
	empty, err := nn.NewModule("empty")
	if err != nil {
		t.Fatalf("nn.NewModule: %v", err)
	}
	defer func() { _ = empty.Close() }()
	assertErrorIs(t, Freeze(empty, "["), errs.ErrInvalidArgument)
	assertErrorIs(t, Unfreeze(empty, "["), errs.ErrInvalidArgument)
}

func equalStrings(left, right []string) bool {
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

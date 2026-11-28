package data

import (
	"errors"
	"reflect"
	"testing"

	"github.com/adhitiad/gotorch/src/errs"
)

func TestInstructionSampleNewAndSampleText(t *testing.T) {
	s, err := NewInstructionSample("Tulis cerita", "", "Era sekali")
	if err != nil {
		t.Fatalf("NewInstructionSample: %v", err)
	}
	if s.Instruction != "Tulis cerita" || s.Input != "" || s.Output != "Era sekali" {
		t.Fatalf("field tidak sesuai: %+v", s)
	}
	text := s.SampleText()
	if text != "Instruksi: Tulis cerita\nBalasan: Era sekali" {
		t.Fatalf("SampleText = %q, want %q", text, "Instruksi: Tulis cerita\nBalasan: Era sekali")
	}

	s2, err := NewInstructionSample("Tulis cerita", "Masukan contoh", "Era sekali")
	if err != nil {
		t.Fatalf("NewInstructionSample with input: %v", err)
	}
	text2 := s2.SampleText()
	if text2 != "Instruksi: Tulis cerita\nMasukan: Masukan contoh\nBalasan: Era sekali" {
		t.Fatalf("SampleText with input = %q", text2)
	}
}

func TestInstructionSampleValidation(t *testing.T) {
	if _, err := NewInstructionSample("", "", "output"); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("empty instruction accepted: %v", err)
	}
	if _, err := NewInstructionSample("instr", "", ""); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("empty output accepted: %v", err)
	}
	if _, err := NewInstructionSample("   ", "", "output"); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("whitespace instruction accepted: %v", err)
	}
}

func TestPreferenceSampleNewAndTexts(t *testing.T) {
	s, err := NewPreferenceSample("Prompt", "Chosen respon", "Rejected respon")
	if err != nil {
		t.Fatalf("NewPreferenceSample: %v", err)
	}
	if s.Prompt != "Prompt" || s.Chosen != "Chosen respon" || s.Rejected != "Rejected respon" {
		t.Fatalf("field tidak sesuai: %+v", s)
	}
	if s.ChosenText() != "PromptChosen respon" {
		t.Fatalf("ChosenText = %q", s.ChosenText())
	}
	if s.RejectedText() != "PromptRejected respon" {
		t.Fatalf("RejectedText = %q", s.RejectedText())
	}
}

func TestPreferenceSampleValidation(t *testing.T) {
	if _, err := NewPreferenceSample("", "chosen", "rejected"); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("empty prompt accepted: %v", err)
	}
	if _, err := NewPreferenceSample("prompt", "", "rejected"); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("empty chosen accepted: %v", err)
	}
	if _, err := NewPreferenceSample("prompt", "chosen", ""); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("empty rejected accepted: %v", err)
	}
}

func TestDatasetSchema(t *testing.T) {
	if SchemaInstruction.IsValid() != true || SchemaPreference.IsValid() != true {
		t.Fatal("known schemas should be valid")
	}
	if DatasetSchema("unknown").IsValid() != false {
		t.Fatal("unknown schema should be invalid")
	}
	if SchemaInstruction.String() != "instruction" || SchemaPreference.String() != "preference" {
		t.Fatal("String() mismatch")
	}
}

func TestTruncationPolicy(t *testing.T) {
	if TruncateEnd.IsValid() != true || TruncateStart.IsValid() != true || TruncateNone.IsValid() != true {
		t.Fatal("known policies should be valid")
	}
	if TruncationPolicy("middle").IsValid() != false {
		t.Fatal("unknown policy should be invalid")
	}
}

func TestMaskType(t *testing.T) {
	if MaskCausal.IsValid() != true || MaskBidirectional.IsValid() != true || MaskLoss.IsValid() != true {
		t.Fatal("known mask types should be valid")
	}
	if MaskType("attention").IsValid() != false {
		t.Fatal("unknown mask type should be invalid")
	}
}

func TestTruncate(t *testing.T) {
	ids := []int64{1, 2, 3, 4, 5}

	// No truncation needed.
	out, err := Truncate(ids, 10, TruncateEnd)
	if err != nil {
		t.Fatalf("Truncate(10): %v", err)
	}
	if !reflect.DeepEqual(out, ids) {
		t.Fatalf("Truncate(10) = %v, want %v", out, ids)
	}

	// Truncate end.
	out, err = Truncate(ids, 3, TruncateEnd)
	if err != nil {
		t.Fatalf("Truncate end: %v", err)
	}
	if !reflect.DeepEqual(out, []int64{1, 2, 3}) {
		t.Fatalf("Truncate end = %v, want [1 2 3]", out)
	}

	// Truncate start.
	out, err = Truncate(ids, 3, TruncateStart)
	if err != nil {
		t.Fatalf("Truncate start: %v", err)
	}
	if !reflect.DeepEqual(out, []int64{3, 4, 5}) {
		t.Fatalf("Truncate start = %v, want [3 4 5]", out)
	}

	// Truncate none with overflow.
	if _, err := Truncate(ids, 3, TruncateNone); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("TruncateNone overflow accepted: %v", err)
	}

	// Negative max.
	if _, err := Truncate(ids, -1, TruncateEnd); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("negative max accepted: %v", err)
	}

	// Unknown policy.
	if _, err := Truncate(ids, 3, TruncationPolicy("middle")); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("unknown policy accepted: %v", err)
	}

	// Original slice not modified.
	_, _ = Truncate(ids, 3, TruncateEnd)
	if !reflect.DeepEqual(ids, []int64{1, 2, 3, 4, 5}) {
		t.Fatal("original slice modified")
	}
}

func TestCausalMask(t *testing.T) {
	mask := CausalMask(4)
	if len(mask) != 4 || len(mask[0]) != 4 {
		t.Fatalf("causal mask shape wrong: %d x %d", len(mask), len(mask[0]))
	}
	want := [][]bool{
		{true, false, false, false},
		{true, true, false, false},
		{true, true, true, false},
		{true, true, true, true},
	}
	if !reflect.DeepEqual(mask, want) {
		t.Fatalf("causal mask = %v, want %v", mask, want)
	}
	if CausalMask(0) != nil {
		t.Fatal("CausalMask(0) should be nil")
	}
}

func TestBidirectionalMask(t *testing.T) {
	mask := BidirectionalMask(3)
	if len(mask) != 3 || len(mask[0]) != 3 {
		t.Fatalf("bidir mask shape wrong")
	}
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if !mask[i][j] {
				t.Fatalf("bidir mask[%d][%d] should be true", i, j)
			}
		}
	}
}

func TestLossMask(t *testing.T) {
	mask := LossMask(5)
	want := []bool{true, true, true, true, true}
	if !reflect.DeepEqual(mask, want) {
		t.Fatalf("loss mask = %v, want %v", mask, want)
	}

	mask = LossMask(5, 1, 3)
	want = []bool{true, false, true, false, true}
	if !reflect.DeepEqual(mask, want) {
		t.Fatalf("loss mask with exclude = %v, want %v", mask, want)
	}

	if len(LossMask(0)) != 0 {
		t.Fatal("LossMask(0) should be empty")
	}
}

func TestSortIndices(t *testing.T) {
	indices := []int64{3, 1, 4, 1, 5}
	out := SortIndices(indices)
	want := []int64{1, 1, 3, 4, 5}
	if !reflect.DeepEqual(out, want) {
		t.Fatalf("SortIndices = %v, want %v", out, want)
	}
	if !reflect.DeepEqual(indices, []int64{3, 1, 4, 1, 5}) {
		t.Fatal("original slice modified")
	}
}

func TestValidateSchema(t *testing.T) {
	// Valid instruction.
	sSample := mustSample(t, map[string]any{"instruction": "instr", "input": "input", "output": "output"})
	if err := ValidateSchema(SchemaInstruction, sSample); err != nil {
		t.Fatalf("ValidateSchema(instruction, valid): %v", err)
	}

	// Missing instruction.
	s2 := Sample{}
	s2.fields = map[string]any{"output": "out"}
	if err := ValidateSchema(SchemaInstruction, s2); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("missing instruction accepted: %v", err)
	}

	// Missing output.
	s3 := Sample{}
	s3.fields = map[string]any{"instruction": "instr"}
	if err := ValidateSchema(SchemaInstruction, s3); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("missing output accepted: %v", err)
	}

	// Valid preference.
	s4Sample := mustSample(t, map[string]any{"prompt": "prompt", "chosen": "chosen", "rejected": "rejected"})
	if err := ValidateSchema(SchemaPreference, s4Sample); err != nil {
		t.Fatalf("ValidateSchema(preference, valid): %v", err)
	}

	// Missing chosen.
	s5 := Sample{}
	s5.fields = map[string]any{"prompt": "p", "rejected": "r"}
	if err := ValidateSchema(SchemaPreference, s5); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("missing chosen accepted: %v", err)
	}

	// Unknown schema.
	if err := ValidateSchema(DatasetSchema("unknown"), Sample{}); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("unknown schema accepted: %v", err)
	}
}

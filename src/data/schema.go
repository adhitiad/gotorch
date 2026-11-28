// Package data menyediakan skema dataset instruksi dan preferensi, collator
// SFT/DPO, masker attention/loss, dan kebijakan pemotongan untuk P6.
// Semua konstruktor memvalidasi masukan dan tidak pernah panic.
// Komentar dalam bahasa Indonesia; string error dalam bahasa Inggris.
package data

import (
	"fmt"
	"sort"
	"strings"

	"github.com/adhitiad/gotorch/src/errs"
)

// InstructionSample adalah satu rekaman data pelatihan instruksi (SFT).
// Field:
//   - Instruction: teks instruksi/tugas.
//   - Input: teks masukan tambahan (bisa kosong).
//   - Output: teks target yang diharapkan model hasilkan.
//
// Sistem yang mengonsumsi skema ini bertanggung jawab memetakan field-field
// ini ke dalam template prompt sebelum di-encode oleh Tokenizer.
type InstructionSample struct {
	Instruction string
	Input       string
	Output      string
}

// NewInstructionSample membentuk InstructionSample dan memvalidasi bahwa
// instruction dan output tidak kosong. Input boleh kosong.
func NewInstructionSample(instruction, input, output string) (*InstructionSample, error) {
	if strings.TrimSpace(instruction) == "" {
		return nil, fmt.Errorf("%w: instruction must not be empty", errs.ErrInvalidArgument)
	}
	if strings.TrimSpace(output) == "" {
		return nil, fmt.Errorf("%w: output must not be empty", errs.ErrInvalidArgument)
	}
	return &InstructionSample{
		Instruction: instruction,
		Input:       input,
		Output:      output,
	}, nil
}

// SampleText mengembalikan teks gabungan instruksi, input, dan output yang
// siap di-encode. Format: "Instruksi: <instruction>\nMasukan: <input>\nBalasan: <output>".
// Jika input kosong, baris masukan dihilangkan.
func (s *InstructionSample) SampleText() string {
	var b strings.Builder
	b.WriteString("Instruksi: ")
	b.WriteString(s.Instruction)
	if strings.TrimSpace(s.Input) != "" {
		b.WriteString("\nMasukan: ")
		b.WriteString(s.Input)
	}
	b.WriteString("\nBalasan: ")
	b.WriteString(s.Output)
	return b.String()
}

// PreferenceSample adalah satu rekaman data preferensi (DPO/PPO).
// Field:
//   - Prompt: teks prompt yang diberikan ke model.
//   - Chosen: respons yang disukai (label positif).
//   - Rejected: respons yang tidak disukai (label negatif).
type PreferenceSample struct {
	Prompt   string
	Chosen   string
	Rejected string
}

// NewPreferenceSample membentuk PreferenceSample dan memvalidasi bahwa prompt,
// chosen, dan rejected tidak kosong.
func NewPreferenceSample(prompt, chosen, rejected string) (*PreferenceSample, error) {
	if strings.TrimSpace(prompt) == "" {
		return nil, fmt.Errorf("%w: prompt must not be empty", errs.ErrInvalidArgument)
	}
	if strings.TrimSpace(chosen) == "" {
		return nil, fmt.Errorf("%w: chosen must not be empty", errs.ErrInvalidArgument)
	}
	if strings.TrimSpace(rejected) == "" {
		return nil, fmt.Errorf("%w: rejected must not be empty", errs.ErrInvalidArgument)
	}
	return &PreferenceSample{
		Prompt:   prompt,
		Chosen:   chosen,
		Rejected: rejected,
	}, nil
}

// ChosenText mengembalikan gabungan prompt dan chosen yang siap di-encode.
func (s *PreferenceSample) ChosenText() string {
	return s.Prompt + s.Chosen
}

// RejectedText mengembalikan gabungan prompt dan rejected yang siap di-encode.
func (s *PreferenceSample) RejectedText() string {
	return s.Prompt + s.Rejected
}

func (s *PreferenceSample) prompt() string   { return s.Prompt }
func (s *PreferenceSample) chosen() string   { return s.Chosen }
func (s *PreferenceSample) rejected() string { return s.Rejected }

// DatasetSchema adalah tipe skema dataset yang didukung.
type DatasetSchema string

const (
	// SchemaInstruction untuk data pelatihan instruksi (SFT).
	SchemaInstruction DatasetSchema = "instruction"
	// SchemaPreference untuk data pelatihan preferensi (DPO/PPO).
	SchemaPreference DatasetSchema = "preference"
)

// IsValid melaporkan apakah skema merupakan salah satu yang dikenal.
func (s DatasetSchema) IsValid() bool {
	switch s {
	case SchemaInstruction, SchemaPreference:
		return true
	default:
		return false
	}
}

// String mengembalikan representasi string skema.
func (s DatasetSchema) String() string {
	return string(s)
}

// TruncationPolicy menentukan bagaimana urutan yang terlalu panjang dipotong.
type TruncationPolicy string

const (
	// TruncateEnd memotong dari akhir urutan (default).
	TruncateEnd TruncationPolicy = "end"
	// TruncateStart memotong dari awal urutan.
	TruncateStart TruncationPolicy = "start"
	// TruncateNone menolak urutan yang melebihi batas dengan error.
	TruncateNone TruncationPolicy = "none"
)

// IsValid melaporkan apakah kebijakan pemotongan dikenal.
func (p TruncationPolicy) IsValid() bool {
	switch p {
	case TruncateEnd, TruncateStart, TruncateNone:
		return true
	default:
		return false
	}
}

// MaskType menentukan tipe masker yang dihasilkan.
type MaskType string

const (
	// MaskCausal untuk attention causal (mula-mulai ke akhir).
	MaskCausal MaskType = "causal"
	// MaskBidirectional untuk attention bidirectional.
	MaskBidirectional MaskType = "bidirectional"
	// MaskLoss untuk loss mask (1 = hitung loss, 0 = abaikan).
	MaskLoss MaskType = "loss"
)

// IsValid melaporkan apakah tipe masker dikenal.
func (m MaskType) IsValid() bool {
	switch m {
	case MaskCausal, MaskBidirectional, MaskLoss:
		return true
	default:
		return false
	}
}

// Truncate memotong urutan ID token ke panjang maksimal sesuai kebijakan.
// Jika kebijakan TruncateNone dan panjang melebihi max, mengembalikan error.
// Slice yang dikembalikan adalah salinan baru.
func Truncate(ids []int64, max int, policy TruncationPolicy) ([]int64, error) {
	if max < 0 {
		return nil, fmt.Errorf("%w: max length must be non-negative, got %d", errs.ErrInvalidArgument, max)
	}
	if !policy.IsValid() {
		return nil, fmt.Errorf("%w: unknown truncation policy %q", errs.ErrInvalidArgument, policy)
	}
	if len(ids) <= max {
		out := make([]int64, len(ids))
		copy(out, ids)
		return out, nil
	}
	if policy == TruncateNone {
		return nil, fmt.Errorf("%w: sequence length %d exceeds max %d", errs.ErrInvalidArgument, len(ids), max)
	}
	out := make([]int64, max)
	if policy == TruncateStart {
		copy(out, ids[len(ids)-max:])
	} else {
		copy(out, ids[:max])
	}
	return out, nil
}

// CausalMask menghasilkan masker attention causal untuk panjang n.
// Masker[i][j] = true jika j <= i (token i dapat melihat token j yang sama atau sebelumnya).
// Hasil adalah slice berdimensi dua yang sudah dialokasi.
func CausalMask(n int) [][]bool {
	if n <= 0 {
		return nil
	}
	mask := make([][]bool, n)
	for i := 0; i < n; i++ {
		row := make([]bool, n)
		for j := 0; j <= i; j++ {
			row[j] = true
		}
		mask[i] = row
	}
	return mask
}

// BidirectionalMask menghasilkan masker attention bidirectional untuk panjang n.
// Semua elemen true.
func BidirectionalMask(n int) [][]bool {
	if n <= 0 {
		return nil
	}
	mask := make([][]bool, n)
	for i := 0; i < n; i++ {
		row := make([]bool, n)
		for j := 0; j < n; j++ {
			row[j] = true
		}
		mask[i] = row
	}
	return mask
}

// LossMask menghasilkan masker loss untuk panjang n.
// Semua elemen true (semua token dihitung loss) kecuali yang disebutkan dalam
// excludeIndices.
func LossMask(n int, excludeIndices ...int) []bool {
	mask := make([]bool, n)
	for i := 0; i < n; i++ {
		mask[i] = true
	}
	exclude := make(map[int]struct{}, len(excludeIndices))
	for _, idx := range excludeIndices {
		if idx >= 0 && idx < n {
			exclude[idx] = struct{}{}
		}
	}
	for idx := range exclude {
		mask[idx] = false
	}
	return mask
}

// SortIndices mengembalikan salinan indices yang diurutkan menaik. Berguna
// untuk memastikan urutan deterministik saat membandingkan batch.
func SortIndices(indices []int64) []int64 {
	out := make([]int64, len(indices))
	copy(out, indices)
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

// ValidateSchema memeriksa apakah sample memiliki field yang diperlukan
// untuk skema yang diberikan. Instruction membutuhkan instruction dan output;
// Preference membutuhkan prompt, chosen, dan rejected.
func ValidateSchema(schema DatasetSchema, sample Sample) error {
	if !schema.IsValid() {
		return fmt.Errorf("%w: unknown dataset schema %q", errs.ErrInvalidArgument, schema)
	}
	switch schema {
	case SchemaInstruction:
		instruction, ok := sample.Value("instruction")
		if !ok || instruction == nil || strings.TrimSpace(fmt.Sprint(instruction)) == "" {
			return fmt.Errorf("%w: instruction field missing or empty", errs.ErrInvalidArgument)
		}
		output, ok := sample.Value("output")
		if !ok || output == nil || strings.TrimSpace(fmt.Sprint(output)) == "" {
			return fmt.Errorf("%w: output field missing or empty", errs.ErrInvalidArgument)
		}
	case SchemaPreference:
		prompt, ok := sample.Value("prompt")
		if !ok || prompt == nil || strings.TrimSpace(fmt.Sprint(prompt)) == "" {
			return fmt.Errorf("%w: prompt field missing or empty", errs.ErrInvalidArgument)
		}
		chosen, ok := sample.Value("chosen")
		if !ok || chosen == nil || strings.TrimSpace(fmt.Sprint(chosen)) == "" {
			return fmt.Errorf("%w: chosen field missing or empty", errs.ErrInvalidArgument)
		}
		rejected, ok := sample.Value("rejected")
		if !ok || rejected == nil || strings.TrimSpace(fmt.Sprint(rejected)) == "" {
			return fmt.Errorf("%w: rejected field missing or empty", errs.ErrInvalidArgument)
		}
	}
	return nil
}

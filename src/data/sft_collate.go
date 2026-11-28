package data

import (
	"context"
	"fmt"

	"github.com/adhitiad/gotorch/src/errs"
)

// InstructionCollate adalah collator untuk data instruksi (SFT).
// Mengencode instruksi+input sebagai konteks dan output sebagai target.
// Menghasilkan Batch dengan field tambahan melalui collate custom.
// Untuk penggunaan langsung dengan Batcher, gunakan CollateFunc yang dihasilkan
// oleh NewInstructionCollateFunc.
type InstructionCollate struct {
	tokenizer Tokenizer
	maxLen    int
	truncate  TruncationPolicy
}

// NewInstructionCollate membentuk collator SFT. tokenizer harus non-nil,
// maxLen harus positif, dan truncate harus kebijakan yang valid.
func NewInstructionCollate(tokenizer Tokenizer, maxLen int, truncate TruncationPolicy) (*InstructionCollate, error) {
	if isNil(tokenizer) {
		return nil, fmt.Errorf("%w: nil tokenizer", errs.ErrInvalidState)
	}
	if maxLen <= 0 {
		return nil, fmt.Errorf("%w: max length must be positive, got %d", errs.ErrInvalidArgument, maxLen)
	}
	if !truncate.IsValid() {
		return nil, fmt.Errorf("%w: unknown truncation policy %q", errs.ErrInvalidArgument, truncate)
	}
	return &InstructionCollate{tokenizer: tokenizer, maxLen: maxLen, truncate: truncate}, nil
}

// Collate mengubah sekumpulan Sample instruksi menjadi Batch.
// Setiap sample diharapkan memiliki field "instruction", "input", dan "output".
// Hasil berisi encoding gabungan dan loss mask yang hanya aktif pada token output.
func (c *InstructionCollate) Collate(ctx context.Context, samples []Sample, indices []int64) (*Batch, error) {
	if err := checkContext(ctx); err != nil {
		return nil, err
	}
	if c == nil || c.tokenizer == nil {
		return nil, fmt.Errorf("%w: nil instruction collate", errs.ErrInvalidState)
	}
	if len(samples) == 0 {
		return nil, fmt.Errorf("%w: empty samples", errs.ErrInvalidArgument)
	}
	if len(indices) != len(samples) {
		return nil, fmt.Errorf("%w: %d samples but %d indices", errs.ErrInvalidArgument, len(samples), len(indices))
	}

	type encoded struct {
		ids      []int64
		lossMask []bool
	}

	encodedList := make([]encoded, 0, len(samples))
	maxEncoded := 0
	for position, sample := range samples {
		instruction, ok := sample.Value("instruction")
		if !ok || instruction == nil {
			return nil, fmt.Errorf("%w: sample %d missing instruction field", errs.ErrInvalidArgument, position)
		}
		input, _ := sample.Value("input")
		output, ok := sample.Value("output")
		if !ok || output == nil {
			return nil, fmt.Errorf("%w: sample %d missing output field", errs.ErrInvalidArgument, position)
		}

		instructionStr, ok := instruction.(string)
		if !ok {
			return nil, fmt.Errorf("%w: sample %d instruction is not string", errs.ErrInvalidArgument, position)
		}
		inputStr, _ := input.(string)
		outputStr, ok := output.(string)
		if !ok {
			return nil, fmt.Errorf("%w: sample %d output is not string", errs.ErrInvalidArgument, position)
		}

		sampleObj, err := NewInstructionSample(instructionStr, inputStr, outputStr)
		if err != nil {
			return nil, fmt.Errorf("%w: sample %d", err, position)
		}

		fullText := sampleObj.SampleText()
		enc, err := c.tokenizer.Encode(ctx, fullText)
		if err != nil {
			return nil, fmt.Errorf("%w: encode sample %d", err, position)
		}
		ids, err := Truncate(enc.IDs(), c.maxLen, c.truncate)
		if err != nil {
			return nil, fmt.Errorf("%w: truncate sample %d", err, position)
		}
		lossMask := c.buildLossMask(enc, ids, sampleObj)
		encodedList = append(encodedList, encoded{ids: ids, lossMask: lossMask})
		if len(ids) > maxEncoded {
			maxEncoded = len(ids)
		}
	}

	// Pad semua ke panjang sama dengan PAD ID.
	padID, hasPAD := c.tokenizer.SpecialTokens()[PADName]
	if !hasPAD {
		padID = 0
	}
	batchSamples := make([]Sample, 0, len(encodedList))
	batchIndices := make([]int64, 0, len(encodedList))
	for position, enc := range encodedList {
		padded := make([]int64, maxEncoded)
		copy(padded, enc.ids)
		for i := len(enc.ids); i < maxEncoded; i++ {
			padded[i] = padID
		}
		paddedLoss := make([]bool, maxEncoded)
		copy(paddedLoss, enc.lossMask)
		fields := map[string]any{
			"input_ids": padded,
			"loss_mask": paddedLoss,
			"indices":   indices[position],
		}
		sample, err := NewSample(fields)
		if err != nil {
			return nil, fmt.Errorf("%w: build sample %d", err, position)
		}
		batchSamples = append(batchSamples, sample)
		batchIndices = append(batchIndices, indices[position])
	}
	return NewBatch(batchSamples, batchIndices)
}

// buildLossMask menghasilkan loss mask: true hanya pada token yang berasal dari
// output ( setelah "Balasan: "), false pada instruksi, input, padding, dan
// special token.
func (c *InstructionCollate) buildLossMask(enc *Encoding, ids []int64, sample *InstructionSample) []bool {
	mask := make([]bool, len(ids))
	special := enc.SpecialTokensMask()
	outputText := sample.Output

	// Cari posisi awal output dalam encoded tokens dengan mencocokkan token output
	// terhadap encoded tokens dari akhir ke awal (heuristik sederhana).
	outputTokens := c.tokenizerEncodeTokens(outputText)
	if len(outputTokens) == 0 {
		return mask
	}

	// Tandai token yang cocok dengan output sebagai loss=true, kecuali special token.
	outputIdx := len(outputTokens) - 1
	for i := len(ids) - 1; i >= 0 && outputIdx >= 0; i-- {
		if special[i] {
			continue
		}
		if ids[i] == outputTokens[outputIdx] {
			mask[i] = true
			outputIdx--
		}
	}
	return mask
}

// tokenizerEncodeTokens mengencode teks dan mengembalikan ID token.
// Digunakan internal untuk mencari posisi output dalam loss mask.
func (c *InstructionCollate) tokenizerEncodeTokens(text string) []int64 {
	enc, err := c.tokenizer.Encode(context.Background(), text)
	if err != nil {
		return nil
	}
	return enc.IDs()
}

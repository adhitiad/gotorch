package data

import (
	"context"
	"fmt"

	"github.com/adhitiad/gotorch/src/errs"
)

// DPOCollate adalah collator untuk data preferensi (DPO).
// Mengencode chosen dan rejected terpisah untuk setiap sample.
// Hasil berisi encoding chosen dan rejected dengan loss mask masing-masing.
type DPOCollate struct {
	tokenizer Tokenizer
	maxLen    int
	truncate  TruncationPolicy
}

// NewDPOCollate membentuk collator DPO. tokenizer harus non-nil,
// maxLen harus positif, dan truncate harus kebijakan yang valid.
func NewDPOCollate(tokenizer Tokenizer, maxLen int, truncate TruncationPolicy) (*DPOCollate, error) {
	if isNil(tokenizer) {
		return nil, fmt.Errorf("%w: nil tokenizer", errs.ErrInvalidState)
	}
	if maxLen <= 0 {
		return nil, fmt.Errorf("%w: max length must be positive, got %d", errs.ErrInvalidArgument, maxLen)
	}
	if !truncate.IsValid() {
		return nil, fmt.Errorf("%w: unknown truncation policy %q", errs.ErrInvalidArgument, truncate)
	}
	return &DPOCollate{tokenizer: tokenizer, maxLen: maxLen, truncate: truncate}, nil
}

// Collate mengubah sekumpulan Sample preferensi menjadi Batch.
// Setiap sample diharapkan memiliki field "prompt", "chosen", dan "rejected".
// Hasil berisi encoding chosen dan rejected dengan loss mask masing-masing.
func (c *DPOCollate) Collate(ctx context.Context, samples []Sample, indices []int64) (*Batch, error) {
	if err := checkContext(ctx); err != nil {
		return nil, err
	}
	if c == nil || c.tokenizer == nil {
		return nil, fmt.Errorf("%w: nil DPO collate", errs.ErrInvalidState)
	}
	if len(samples) == 0 {
		return nil, fmt.Errorf("%w: empty samples", errs.ErrInvalidArgument)
	}
	if len(indices) != len(samples) {
		return nil, fmt.Errorf("%w: %d samples but %d indices", errs.ErrInvalidArgument, len(samples), len(indices))
	}

	type encodedPair struct {
		chosenIDs    []int64
		chosenMask   []bool
		rejectedIDs  []int64
		rejectedMask []bool
	}

	encodedList := make([]encodedPair, 0, len(samples))
	maxLen := 0
	for position, sample := range samples {
		prompt, ok := sample.Value("prompt")
		if !ok || prompt == nil {
			return nil, fmt.Errorf("%w: sample %d missing prompt field", errs.ErrInvalidArgument, position)
		}
		chosen, ok := sample.Value("chosen")
		if !ok || chosen == nil {
			return nil, fmt.Errorf("%w: sample %d missing chosen field", errs.ErrInvalidArgument, position)
		}
		rejected, ok := sample.Value("rejected")
		if !ok || rejected == nil {
			return nil, fmt.Errorf("%w: sample %d missing rejected field", errs.ErrInvalidArgument, position)
		}

		promptStr, ok := prompt.(string)
		if !ok {
			return nil, fmt.Errorf("%w: sample %d prompt is not string", errs.ErrInvalidArgument, position)
		}
		chosenStr, ok := chosen.(string)
		if !ok {
			return nil, fmt.Errorf("%w: sample %d chosen is not string", errs.ErrInvalidArgument, position)
		}
		rejectedStr, ok := rejected.(string)
		if !ok {
			return nil, fmt.Errorf("%w: sample %d rejected is not string", errs.ErrInvalidArgument, position)
		}

		sampleObj, err := NewPreferenceSample(promptStr, chosenStr, rejectedStr)
		if err != nil {
			return nil, fmt.Errorf("%w: sample %d", err, position)
		}

		chosenEnc, err := c.tokenizer.Encode(ctx, sampleObj.ChosenText())
		if err != nil {
			return nil, fmt.Errorf("%w: encode chosen sample %d", err, position)
		}
		chosenIDs, err := Truncate(chosenEnc.IDs(), c.maxLen, c.truncate)
		if err != nil {
			return nil, fmt.Errorf("%w: truncate chosen sample %d", err, position)
		}
		chosenMask := c.buildDPOLossMask(chosenEnc, chosenIDs, chosenStr)

		rejectedEnc, err := c.tokenizer.Encode(ctx, sampleObj.RejectedText())
		if err != nil {
			return nil, fmt.Errorf("%w: encode rejected sample %d", err, position)
		}
		rejectedIDs, err := Truncate(rejectedEnc.IDs(), c.maxLen, c.truncate)
		if err != nil {
			return nil, fmt.Errorf("%w: truncate rejected sample %d", err, position)
		}
		rejectedMask := c.buildDPOLossMask(rejectedEnc, rejectedIDs, rejectedStr)

		if len(chosenIDs) > maxLen {
			maxLen = len(chosenIDs)
		}
		if len(rejectedIDs) > maxLen {
			maxLen = len(rejectedIDs)
		}

		encodedList = append(encodedList, encodedPair{
			chosenIDs:    chosenIDs,
			chosenMask:   chosenMask,
			rejectedIDs:  rejectedIDs,
			rejectedMask: rejectedMask,
		})
	}

	padID, _ := c.tokenizer.SpecialTokens()[PADName]
	batchSamples := make([]Sample, 0, len(encodedList))
	batchIndices := make([]int64, 0, len(encodedList))
	for position, pair := range encodedList {
		paddedChosen := padTo(pair.chosenIDs, maxLen, padID)
		paddedChosenMask := padBoolTo(pair.chosenMask, maxLen, false)
		paddedRejected := padTo(pair.rejectedIDs, maxLen, padID)
		paddedRejectedMask := padBoolTo(pair.rejectedMask, maxLen, false)
		fields := map[string]any{
			"chosen_input_ids":   paddedChosen,
			"chosen_loss_mask":   paddedChosenMask,
			"rejected_input_ids": paddedRejected,
			"rejected_loss_mask": paddedRejectedMask,
			"indices":            indices[position],
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

func (c *DPOCollate) buildDPOLossMask(enc *Encoding, ids []int64, targetText string) []bool {
	mask := make([]bool, len(ids))
	special := enc.SpecialTokensMask()
	targetTokens := c.tokenizerEncodeTokens(targetText)
	if len(targetTokens) == 0 {
		return mask
	}
	targetIdx := len(targetTokens) - 1
	for i := len(ids) - 1; i >= 0 && targetIdx >= 0; i-- {
		if special[i] {
			continue
		}
		if ids[i] == targetTokens[targetIdx] {
			mask[i] = true
			targetIdx--
		}
	}
	return mask
}

// tokenizerEncodeTokens mengencode teks dan mengembalikan ID token.
// Digunakan internal untuk mencari posisi target dalam loss mask.
func (c *DPOCollate) tokenizerEncodeTokens(text string) []int64 {
	enc, err := c.tokenizer.Encode(context.Background(), text)
	if err != nil {
		return nil
	}
	return enc.IDs()
}

func padTo(ids []int64, max int, pad int64) []int64 {
	out := make([]int64, max)
	copy(out, ids)
	for i := len(ids); i < max; i++ {
		out[i] = pad
	}
	return out
}

func padBoolTo(mask []bool, max int, val bool) []bool {
	out := make([]bool, max)
	copy(out, mask)
	for i := len(mask); i < max; i++ {
		out[i] = val
	}
	return out
}

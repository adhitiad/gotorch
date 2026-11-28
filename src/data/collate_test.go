package data

import (
	"context"
	"errors"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/adhitiad/gotorch/src/errs"
)

// mockTokenizer adalah tokenizer sederhana untuk tes collate.
type mockTokenizer struct {
	vocab   map[string]int64
	idToTok map[int64]string
}

func newMockTokenizer() *mockTokenizer {
	v := map[string]int64{
		"[BOS]": 0, "[EOS]": 1, "[UNK]": 2, "[PAD]": 3,
		"a": 4, "b": 5, "c": 6, "d": 7, "e": 8, "f": 9, "g": 10,
		"h": 11, "i": 12, "j": 13, "k": 14, "l": 15, "m": 16,
		"n": 17, "o": 18, "p": 19, "q": 20, "r": 21, "s": 22,
		"t": 23, "u": 24, "v": 25, "w": 26, "x": 27, "y": 28, "z": 29,
		" ": 30, "\n": 31,
	}
	rev := make(map[int64]string, len(v))
	for tok, id := range v {
		rev[id] = tok
	}
	return &mockTokenizer{vocab: v, idToTok: rev}
}

func (m *mockTokenizer) Encode(ctx context.Context, text string) (*Encoding, error) {
	if ctx == nil {
		return nil, errs.ErrInvalidArgument
	}
	var ids []int64
	var toks []string
	var offs [][2]int64
	var mask []bool
	pos := int64(0)
	for _, ch := range text {
		tok := string(ch)
		id, ok := m.vocab[tok]
		if !ok {
			id = m.vocab["[UNK]"]
			tok = "[UNK]"
		}
		ids = append(ids, id)
		toks = append(toks, tok)
		offs = append(offs, [2]int64{pos, pos + 1})
		mask = append(mask, false)
		pos += int64(utf8.RuneLen(ch))
	}
	ids = append([]int64{m.vocab["[BOS]"]}, ids...)
	toks = append([]string{"[BOS]"}, toks...)
	offs = append([][2]int64{{0, 0}}, offs...)
	mask = append([]bool{true}, mask...)
	ids = append(ids, m.vocab["[EOS]"])
	toks = append(toks, "[EOS]")
	offs = append(offs, [2]int64{pos, pos})
	mask = append(mask, true)
	return &Encoding{ids: ids, tokens: toks, offsets: offs, specialMask: mask}, nil
}

func (m *mockTokenizer) EncodeBatch(ctx context.Context, texts []string) ([]*Encoding, error) {
	out := make([]*Encoding, 0, len(texts))
	for _, t := range texts {
		enc, err := m.Encode(ctx, t)
		if err != nil {
			return nil, err
		}
		out = append(out, enc)
	}
	return out, nil
}

func (m *mockTokenizer) Decode(ctx context.Context, ids []int64) (string, error) {
	if ctx == nil {
		return "", errs.ErrInvalidArgument
	}
	var b strings.Builder
	for _, id := range ids {
		if id == m.vocab["[BOS]"] || id == m.vocab["[EOS]"] || id == m.vocab["[PAD]"] || id == m.vocab["[UNK]"] {
			continue
		}
		b.WriteString(m.idToTok[id])
	}
	return b.String(), nil
}

func (m *mockTokenizer) VocabSize() int64 { return int64(len(m.vocab)) }

func (m *mockTokenizer) SpecialTokens() map[string]int64 {
	out := make(map[string]int64, len(m.vocab))
	for k, v := range m.vocab {
		if strings.HasPrefix(k, "[") && strings.HasSuffix(k, "]") {
			out[k] = v
		}
	}
	return out
}

func (m *mockTokenizer) ID(tok string) int64 {
	if id, ok := m.vocab[tok]; ok {
		return id
	}
	return -1
}

func (m *mockTokenizer) Token(id int64) string {
	if t, ok := m.idToTok[id]; ok {
		return t
	}
	return ""
}

func TestInstructionCollate(t *testing.T) {
	tok := newMockTokenizer()
	collate, err := NewInstructionCollate(tok, 16, TruncateEnd)
	if err != nil {
		t.Fatalf("NewInstructionCollate: %v", err)
	}

	samples := []Sample{
		mustSample(t, map[string]any{"instruction": "hi", "input": "", "output": "ho"}),
		mustSample(t, map[string]any{"instruction": "hello", "input": " world", "output": "bye"}),
	}
	indices := []int64{10, 20}

	batch, err := collate.Collate(context.Background(), samples, indices)
	if err != nil {
		t.Fatalf("Collate: %v", err)
	}
	if batch.Len() != 2 {
		t.Fatalf("batch len = %d, want 2", batch.Len())
	}

	// Periksa semua sample memiliki input_ids dan loss_mask.
	for i := 0; i < batch.Len(); i++ {
		sample, idx, err := batch.At(i)
		if err != nil {
			t.Fatalf("At(%d): %v", i, err)
		}
		if idx != indices[i] {
			t.Fatalf("index = %d, want %d", idx, indices[i])
		}
		inputIDs, ok := sample.Value("input_ids")
		if !ok {
			t.Fatal("missing input_ids")
		}
		lossMask, ok := sample.Value("loss_mask")
		if !ok {
			t.Fatal("missing loss_mask")
		}
		ids, ok := inputIDs.([]int64)
		if !ok {
			t.Fatalf("input_ids not []int64: %T", inputIDs)
		}
		mask, ok := lossMask.([]bool)
		if !ok {
			t.Fatalf("loss_mask not []bool: %T", lossMask)
		}
		if len(ids) != len(mask) {
			t.Fatalf("len mismatch: %d vs %d", len(ids), len(mask))
		}
		if len(ids) > collate.maxLen {
			t.Fatalf("exceeds maxLen: %d > %d", len(ids), collate.maxLen)
		}
	}
}

func TestInstructionCollateValidation(t *testing.T) {
	tok := newMockTokenizer()
	collate, err := NewInstructionCollate(tok, 16, TruncateEnd)
	if err != nil {
		t.Fatalf("NewInstructionCollate: %v", err)
	}

	// Missing instruction.
	if _, err := collate.Collate(context.Background(), []Sample{mustSample(t, map[string]any{"output": "out"})}, []int64{0}); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("missing instruction accepted: %v", err)
	}
	// Missing output.
	if _, err := collate.Collate(context.Background(), []Sample{mustSample(t, map[string]any{"instruction": "in"})}, []int64{0}); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("missing output accepted: %v", err)
	}
	// Empty samples.
	if _, err := collate.Collate(context.Background(), nil, nil); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("empty samples accepted: %v", err)
	}
	// Mismatched lengths.
	if _, err := collate.Collate(context.Background(), []Sample{mustSample(t, map[string]any{"instruction": "in", "output": "out"})}, []int64{0, 1}); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("length mismatch accepted: %v", err)
	}
	// Nil context.
	if _, err := collate.Collate(nil, []Sample{mustSample(t, map[string]any{"instruction": "in", "output": "out"})}, []int64{0}); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("nil context accepted: %v", err)
	}
	// Nil collate.
	var nilCollate *InstructionCollate
	if _, err := nilCollate.Collate(context.Background(), nil, nil); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("nil collate accepted: %v", err)
	}
}

func TestInstructionCollateTruncation(t *testing.T) {
	tok := newMockTokenizer()
	// maxLen kecil untuk memaksa truncation.
	collate, err := NewInstructionCollate(tok, 6, TruncateEnd)
	if err != nil {
		t.Fatalf("NewInstructionCollate: %v", err)
	}

	samples := []Sample{
		mustSample(t, map[string]any{"instruction": "hello world", "input": "", "output": "goodbye world"}),
	}
	batch, err := collate.Collate(context.Background(), samples, []int64{0})
	if err != nil {
		t.Fatalf("Collate: %v", err)
	}
	sample, _, err := batch.At(0)
	if err != nil {
		t.Fatalf("At(0): %v", err)
	}
	inputIDs, _ := sample.Value("input_ids")
	ids, ok := inputIDs.([]int64)
	if !ok {
		t.Fatalf("input_ids not []int64: %T", inputIDs)
	}
	if len(ids) > 6 {
		t.Fatalf("not truncated: len=%d > 6", len(ids))
	}
}

func mustSample(t *testing.T, fields map[string]any) Sample {
	t.Helper()
	s, err := NewSample(fields)
	if err != nil {
		t.Fatalf("NewSample: %v", err)
	}
	return s
}

func TestDPOCollate(t *testing.T) {
	tok := newMockTokenizer()
	collate, err := NewDPOCollate(tok, 16, TruncateEnd)
	if err != nil {
		t.Fatalf("NewDPOCollate: %v", err)
	}

	samples := []Sample{
		mustSample(t, map[string]any{"prompt": "p1", "chosen": "c1", "rejected": "r1"}),
		mustSample(t, map[string]any{"prompt": "p2", "chosen": "c2", "rejected": "r2"}),
	}
	indices := []int64{100, 200}

	batch, err := collate.Collate(context.Background(), samples, indices)
	if err != nil {
		t.Fatalf("Collate: %v", err)
	}
	if batch.Len() != 2 {
		t.Fatalf("batch len = %d, want 2", batch.Len())
	}

	for i := 0; i < batch.Len(); i++ {
		sample, idx, err := batch.At(i)
		if err != nil {
			t.Fatalf("At(%d): %v", i, err)
		}
		if idx != indices[i] {
			t.Fatalf("index = %d, want %d", idx, indices[i])
		}
		for _, field := range []string{"chosen_input_ids", "chosen_loss_mask", "rejected_input_ids", "rejected_loss_mask"} {
			if _, ok := sample.Value(field); !ok {
				t.Fatalf("missing field %s", field)
			}
		}
	}
}

func TestDPOCollateValidation(t *testing.T) {
	tok := newMockTokenizer()
	collate, err := NewDPOCollate(tok, 16, TruncateEnd)
	if err != nil {
		t.Fatalf("NewDPOCollate: %v", err)
	}

	if _, err := collate.Collate(context.Background(), []Sample{mustSample(t, map[string]any{"chosen": "c", "rejected": "r"})}, []int64{0}); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("missing prompt accepted: %v", err)
	}
	if _, err := collate.Collate(context.Background(), []Sample{mustSample(t, map[string]any{"prompt": "p", "rejected": "r"})}, []int64{0}); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("missing chosen accepted: %v", err)
	}
	if _, err := collate.Collate(context.Background(), []Sample{mustSample(t, map[string]any{"prompt": "p", "chosen": "c"})}, []int64{0}); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("missing rejected accepted: %v", err)
	}
	if _, err := collate.Collate(context.Background(), nil, nil); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("empty samples accepted: %v", err)
	}
	var nilCollate *DPOCollate
	if _, err := nilCollate.Collate(context.Background(), nil, nil); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("nil collate accepted: %v", err)
	}
}

func TestNewCollateValidation(t *testing.T) {
	tok := newMockTokenizer()

	if _, err := NewInstructionCollate(nil, 16, TruncateEnd); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("nil tokenizer accepted: %v", err)
	}
	if _, err := NewInstructionCollate(tok, 0, TruncateEnd); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("zero maxLen accepted: %v", err)
	}
	if _, err := NewInstructionCollate(tok, 16, TruncationPolicy("bad")); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("bad policy accepted: %v", err)
	}

	if _, err := NewDPOCollate(nil, 16, TruncateEnd); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("nil tokenizer accepted: %v", err)
	}
	if _, err := NewDPOCollate(tok, -1, TruncateEnd); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("negative maxLen accepted: %v", err)
	}
}

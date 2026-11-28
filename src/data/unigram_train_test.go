package data

import (
	"context"
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/adhitiad/gotorch/src/errs"
)

func TestUnigramValidation(t *testing.T) {
	tokenizer := NewUnigramTokenizer(false, DefaultSpecialIDs())
	canceled, cancel := context.WithCancel(context.Background())
	cancel()

	if _, err := tokenizer.Encode(context.Background(), "a"); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("encode sebelum train: %v", err)
	}
	if _, err := tokenizer.Decode(context.Background(), []int64{1}); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("decode sebelum train: %v", err)
	}

	if err := tokenizer.Train([]string{"a"}, 200); err != nil {
		t.Fatalf("Train: %v", err)
	}
	if _, err := tokenizer.Encode(nil, "a"); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("encode context nil: %v", err)
	}
	if _, err := tokenizer.Encode(canceled, "a"); !errors.Is(err, context.Canceled) {
		t.Fatalf("encode context dibatalkan: %v", err)
	}
	if _, err := tokenizer.Decode(nil, []int64{1}); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("decode context nil: %v", err)
	}

	var nilTokenizer *UnigramTokenizer
	if _, err := nilTokenizer.Encode(context.Background(), "a"); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("nil encode: %v", err)
	}
	if _, err := nilTokenizer.Decode(context.Background(), []int64{1}); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("nil decode: %v", err)
	}
	if nilTokenizer.VocabSize() != 0 {
		t.Fatalf("nil VocabSize = %d", nilTokenizer.VocabSize())
	}
	if nilTokenizer.SpecialTokens() != nil {
		t.Fatalf("nil SpecialTokens = %v", nilTokenizer.SpecialTokens())
	}
	if nilTokenizer.ID("a") != -1 {
		t.Fatalf("nil ID = %d", nilTokenizer.ID("a"))
	}
	if nilTokenizer.Token(1) != "" {
		t.Fatalf("nil Token = %q", nilTokenizer.Token(1))
	}
	if nilTokenizer.LogProb(1) != math.Inf(-1) {
		t.Fatalf("nil LogProb = %v", nilTokenizer.LogProb(1))
	}

	if err := tokenizer.Train([]string{"a"}, 1); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("vocab size 1 diterima: %v", err)
	}
}

func TestUnigramUnknownID(t *testing.T) {
	tokenizer := NewUnigramTokenizer(false, DefaultSpecialIDs())
	if err := tokenizer.Train([]string{"hello"}, 200); err != nil {
		t.Fatalf("Train: %v", err)
	}
	if _, err := tokenizer.Decode(context.Background(), []int64{99999}); !errors.Is(err, errs.ErrOutOfBounds) {
		t.Fatalf("ID tidak dikenal diterima: %v", err)
	}
}

func TestUnigramEncodeBatch(t *testing.T) {
	tokenizer := NewUnigramTokenizer(false, DefaultSpecialIDs())
	if err := tokenizer.Train([]string{"hello", "world"}, 200); err != nil {
		t.Fatalf("Train: %v", err)
	}

	texts := []string{"hello", "world", "hello world"}
	encodings, err := tokenizer.EncodeBatch(context.Background(), texts)
	if err != nil {
		t.Fatalf("EncodeBatch: %v", err)
	}
	if len(encodings) != 3 {
		t.Fatalf("jumlah encoding = %d, diharapkan 3", len(encodings))
	}
	for position, text := range texts {
		decoded, err := tokenizer.Decode(context.Background(), encodings[position].IDs())
		if err != nil {
			t.Fatalf("Decode batch %d: %v", position, err)
		}
		if decoded != text {
			t.Fatalf("decode batch %d = %q, diharapkan %q", position, decoded, text)
		}
	}
}

func TestUnigramEncodeBatchValidation(t *testing.T) {
	tokenizer := NewUnigramTokenizer(false, DefaultSpecialIDs())
	if err := tokenizer.Train([]string{"a"}, 200); err != nil {
		t.Fatalf("Train: %v", err)
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := tokenizer.EncodeBatch(nil, []string{"a"}); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("encode batch context nil: %v", err)
	}
	if _, err := tokenizer.EncodeBatch(canceled, []string{"a"}); !errors.Is(err, context.Canceled) {
		t.Fatalf("encode batch context dibatalkan: %v", err)
	}
	var nilTokenizer *UnigramTokenizer
	if _, err := nilTokenizer.EncodeBatch(context.Background(), []string{"a"}); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("nil encode batch: %v", err)
	}
}

func TestUnigramSpecialTokens(t *testing.T) {
	special := DefaultSpecialIDs()
	tokenizer := NewUnigramTokenizer(false, special)
	if err := tokenizer.Train([]string{"hello world"}, 200); err != nil {
		t.Fatalf("Train: %v", err)
	}

	got := tokenizer.SpecialTokens()
	if len(got) != len(special) {
		t.Fatalf("jumlah special token = %d, diharapkan %d", len(got), len(special))
	}
	got[BOSName] = 99
	if tokenizer.SpecialTokens()[BOSName] != 0 {
		t.Fatal("SpecialTokens tidak mengembalikan salinan baru")
	}

	for name, id := range special {
		if gotID := tokenizer.ID(name); gotID != id {
			t.Fatalf("ID(%q) = %d, diharapkan %d", name, gotID, id)
		}
		if gotToken := tokenizer.Token(id); gotToken != name {
			t.Fatalf("Token(%d) = %q, diharapkan %q", id, gotToken, name)
		}
	}
}

func TestUnigramTrainingEmpty(t *testing.T) {
	tokenizer := NewUnigramTokenizer(false, DefaultSpecialIDs())
	if err := tokenizer.Train([]string{}, 200); err != nil {
		t.Fatalf("Train kosong: %v", err)
	}
	if !tokenizer.trained {
		t.Fatal("harus ditandai sebagai dilatih meski kosong")
	}
	if tokenizer.VocabSize() != int64(len(DefaultSpecialIDs())+1) {
		t.Fatalf("VocabSize = %d, diharapkan %d (khusus + spasi)", tokenizer.VocabSize(), len(DefaultSpecialIDs())+1)
	}

	encoding, err := tokenizer.Encode(context.Background(), "anything")
	if err != nil {
		t.Fatalf("Encode setelah train kosong: %v", err)
	}
	// Tidak ada karakter di kosakata, setiap karakter menjadi UNK.
	hasUNK := false
	for _, token := range encoding.Tokens() {
		if token == UNKName {
			hasUNK = true
			break
		}
	}
	if !hasUNK {
		t.Fatalf("harus mengandung UNK untuk karakter yang tidak dikenal, dapat %v", encoding.Tokens())
	}
}

func TestUnigramEncodingStructure(t *testing.T) {
	tokenizer := NewUnigramTokenizer(false, DefaultSpecialIDs())
	if err := tokenizer.Train([]string{"ab"}, 200); err != nil {
		t.Fatalf("Train: %v", err)
	}

	encoding, err := tokenizer.Encode(context.Background(), "ab")
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if len(encoding.IDs()) != len(encoding.Tokens()) {
		t.Fatal("panjang IDs dan Tokens harus sama")
	}
	if len(encoding.IDs()) != len(encoding.Offsets()) {
		t.Fatal("panjang IDs dan Offsets harus sama")
	}
	if len(encoding.IDs()) != len(encoding.SpecialTokensMask()) {
		t.Fatal("panjang IDs dan SpecialTokensMask harus sama")
	}
	if encoding.Len() != len(encoding.IDs()) {
		t.Fatalf("Len = %d, tidak sama dengan len(IDs) = %d", encoding.Len(), len(encoding.IDs()))
	}

	cloned := encoding.Clone()
	cloned.IDs()[0] = 99
	if encoding.IDs()[0] == 99 {
		t.Fatal("Clone tidak independen untuk IDs")
	}
}

func TestUnigramWithVocab(t *testing.T) {
	vocab := map[string]int64{
		"[BOS]": 0, "[EOS]": 1, "[UNK]": 2, "[PAD]": 3,
		"a": 4, "b": 5, "ab": 6,
	}
	logProb := map[int64]float64{
		0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0,
		4: math.Log(0.1), 5: math.Log(0.1), 6: math.Log(0.3),
	}
	special := DefaultSpecialIDs()
	tokenizer, err := NewUnigramTokenizerWithVocab(vocab, logProb, special)
	if err != nil {
		t.Fatalf("NewUnigramTokenizerWithVocab: %v", err)
	}

	if tokenizer.VocabSize() != 7 {
		t.Fatalf("VocabSize = %d, diharapkan 7", tokenizer.VocabSize())
	}

	encoding, err := tokenizer.Encode(context.Background(), "ab")
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if !reflect.DeepEqual(encoding.IDs(), []int64{6}) {
		t.Fatalf("IDs = %v, diharapkan [6]", encoding.IDs())
	}
	decoded, err := tokenizer.Decode(context.Background(), encoding.IDs())
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if decoded != "ab" {
		t.Fatalf("decode = %q, diharapkan ab", decoded)
	}
}

func TestUnigramSortedTokens(t *testing.T) {
	tokenizer := NewUnigramTokenizer(false, DefaultSpecialIDs())
	if err := tokenizer.Train([]string{"hello world"}, 200); err != nil {
		t.Fatalf("Train: %v", err)
	}

	tokens := tokenizer.sortedTokens()
	for i := 1; i < len(tokens); i++ {
		if tokens[i-1] > tokens[i] {
			t.Fatalf("token tidak terurut: %q > %q", tokens[i-1], tokens[i])
		}
	}
}

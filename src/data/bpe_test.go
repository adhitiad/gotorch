package data

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/adhitiad/gotorch/src/errs"
)

func TestBPETokenizerTrainAndRoundTrip(t *testing.T) {
	tokenizer := NewCharLevelBPETokenizer(100, DefaultSpecialIDs())
	texts := []string{
		"hello world",
		"hello there",
		"world of code",
		"hello code world",
	}
	if err := tokenizer.Train(texts, 100); err != nil {
		t.Fatalf("Train: %v", err)
	}

	if tokenizer.VocabSize() <= int64(len(DefaultSpecialIDs())) {
		t.Fatalf("VocabSize = %d, diharapkan lebih dari jumlah special token", tokenizer.VocabSize())
	}

	// Round-trip: encode lalu decode harus menghasilkan teks asli.
	original := "hello world"
	encoding, err := tokenizer.Encode(context.Background(), original)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	decoded, err := tokenizer.Decode(context.Background(), encoding.IDs())
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if decoded != original {
		t.Fatalf("decode = %q, diharapkan %q", decoded, original)
	}
}

func TestBPETokenizerDeterministic(t *testing.T) {
	texts := []string{"hello world", "hello there", "world of code"}
	t1 := NewCharLevelBPETokenizer(100, DefaultSpecialIDs())
	t2 := NewCharLevelBPETokenizer(100, DefaultSpecialIDs())
	if err := t1.Train(texts, 100); err != nil {
		t.Fatalf("Train t1: %v", err)
	}
	if err := t2.Train(texts, 100); err != nil {
		t.Fatalf("Train t2: %v", err)
	}

	text := "hello world"
	e1, err := t1.Encode(context.Background(), text)
	if err != nil {
		t.Fatalf("Encode t1: %v", err)
	}
	e2, err := t2.Encode(context.Background(), text)
	if err != nil {
		t.Fatalf("Encode t2: %v", err)
	}
	if !reflect.DeepEqual(e1.IDs(), e2.IDs()) {
		t.Fatalf("tidak deterministik: %v vs %v", e1.IDs(), e2.IDs())
	}
	d1, _ := t1.Decode(context.Background(), e1.IDs())
	d2, _ := t2.Decode(context.Background(), e2.IDs())
	if d1 != d2 {
		t.Fatalf("decode tidak deterministik: %q vs %q", d1, d2)
	}
}

func TestBPETokenizerMergeRankOrder(t *testing.T) {
	// "aaaa" dengan vocab kecil harus membentuk merge "aa" lalu "aaaa".
	tokenizer := NewCharLevelBPETokenizer(10, DefaultSpecialIDs())
	if err := tokenizer.Train([]string{"aaaa aaaa aaaa"}, 10); err != nil {
		t.Fatalf("Train: %v", err)
	}

	// Harus memiliki setidaknya satu aturan penggabungan.
	if len(tokenizer.merges) == 0 {
		t.Fatal("harus memiliki aturan penggabungan")
	}

	// Encode "aaaa" harus menghasilkan token yang digabungkan.
	encoding, err := tokenizer.Encode(context.Background(), "aaaa")
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if encoding.Len() != 1 {
		t.Fatalf("Len = %d, diharapkan 1 (semua 'a' digabung)", encoding.Len())
	}
	decoded, err := tokenizer.Decode(context.Background(), encoding.IDs())
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if decoded != "aaaa" {
		t.Fatalf("decode = %q, diharapkan aaaa", decoded)
	}
}

func TestBPETokenizerByteLevel(t *testing.T) {
	tokenizer := NewByteLevelBPETokenizer(300, DefaultSpecialIDs())
	texts := []string{"hello", "world", "hello world", "café"}
	if err := tokenizer.Train(texts, 300); err != nil {
		t.Fatalf("Train: %v", err)
	}

	original := "hello world"
	encoding, err := tokenizer.Encode(context.Background(), original)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	decoded, err := tokenizer.Decode(context.Background(), encoding.IDs())
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if decoded != original {
		t.Fatalf("decode = %q, diharapkan %q", decoded, original)
	}
}

func TestBPETokenizerUnicodeEdgeCases(t *testing.T) {
	tokenizer := NewCharLevelBPETokenizer(500, DefaultSpecialIDs())
	texts := []string{
		"café",
		"naïve",
		"こんにちは",
		"😀😁😂",
		"e\u0301", // e + combining acute accent
		"日本語",
		"👨‍👩‍👧‍👦",         // family emoji (ZWJ sequence)
		"a\u0301b\u0301c", // multiple combining marks
	}
	if err := tokenizer.Train(texts, 500); err != nil {
		t.Fatalf("Train: %v", err)
	}

	for _, original := range texts {
		encoding, err := tokenizer.Encode(context.Background(), original)
		if err != nil {
			t.Fatalf("Encode(%q): %v", original, err)
		}
		decoded, err := tokenizer.Decode(context.Background(), encoding.IDs())
		if err != nil {
			t.Fatalf("Decode(%q): %v", original, err)
		}
		if decoded != original {
			t.Fatalf("round-trip(%q) = %q, diharapkan %q", original, decoded, original)
		}
	}
}

func TestBPETokenizerEmptyText(t *testing.T) {
	tokenizer := NewCharLevelBPETokenizer(100, DefaultSpecialIDs())
	if err := tokenizer.Train([]string{"hello world"}, 100); err != nil {
		t.Fatalf("Train: %v", err)
	}

	encoding, err := tokenizer.Encode(context.Background(), "")
	if err != nil {
		t.Fatalf("Encode empty: %v", err)
	}
	if encoding.Len() != 0 {
		t.Fatalf("Len = %d, diharapkan 0", encoding.Len())
	}
	decoded, err := tokenizer.Decode(context.Background(), []int64{})
	if err != nil {
		t.Fatalf("Decode empty: %v", err)
	}
	if decoded != "" {
		t.Fatalf("decode empty = %q", decoded)
	}
}

func TestBPETokenizerSingleCharacter(t *testing.T) {
	tokenizer := NewCharLevelBPETokenizer(100, DefaultSpecialIDs())
	if err := tokenizer.Train([]string{"a b c"}, 100); err != nil {
		t.Fatalf("Train: %v", err)
	}

	encoding, err := tokenizer.Encode(context.Background(), "a")
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if encoding.Len() != 1 {
		t.Fatalf("Len = %d, diharapkan 1", encoding.Len())
	}
	decoded, err := tokenizer.Decode(context.Background(), encoding.IDs())
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if decoded != "a" {
		t.Fatalf("decode = %q, diharapkan a", decoded)
	}
}

func TestBPETokenizerValidation(t *testing.T) {
	tokenizer := NewCharLevelBPETokenizer(100, DefaultSpecialIDs())
	canceled, cancel := context.WithCancel(context.Background())
	cancel()

	// Tidak dilatih.
	if _, err := tokenizer.Encode(context.Background(), "a"); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("encode sebelum train: %v", err)
	}
	if _, err := tokenizer.Decode(context.Background(), []int64{1}); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("decode sebelum train: %v", err)
	}

	// Konteks tidak valid.
	if err := tokenizer.Train([]string{"a"}, 100); err != nil {
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

	// Receiver nil.
	var nilTokenizer *BPETokenizer
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

	// Vocab size terlalu kecil.
	if err := tokenizer.Train([]string{"a"}, 1); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("vocab size 1 diterima: %v", err)
	}
}

func TestBPETokenizerUnknownID(t *testing.T) {
	tokenizer := NewCharLevelBPETokenizer(100, DefaultSpecialIDs())
	if err := tokenizer.Train([]string{"hello"}, 100); err != nil {
		t.Fatalf("Train: %v", err)
	}
	if _, err := tokenizer.Decode(context.Background(), []int64{99999}); !errors.Is(err, errs.ErrOutOfBounds) {
		t.Fatalf("ID tidak dikenal diterima: %v", err)
	}
}

func TestBPETokenizerEncodeBatch(t *testing.T) {
	tokenizer := NewCharLevelBPETokenizer(100, DefaultSpecialIDs())
	if err := tokenizer.Train([]string{"hello", "world"}, 100); err != nil {
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

func TestBPETokenizerEncodeBatchValidation(t *testing.T) {
	tokenizer := NewCharLevelBPETokenizer(100, DefaultSpecialIDs())
	if err := tokenizer.Train([]string{"a"}, 100); err != nil {
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
	var nilTokenizer *BPETokenizer
	if _, err := nilTokenizer.EncodeBatch(context.Background(), []string{"a"}); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("nil encode batch: %v", err)
	}
}

func TestBPETokenizerSpecialTokens(t *testing.T) {
	special := DefaultSpecialIDs()
	tokenizer := NewCharLevelBPETokenizer(100, special)
	if err := tokenizer.Train([]string{"hello world"}, 100); err != nil {
		t.Fatalf("Train: %v", err)
	}

	got := tokenizer.SpecialTokens()
	if len(got) != len(special) {
		t.Fatalf("jumlah special token = %d, diharapkan %d", len(got), len(special))
	}
	// Salinan defensif.
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

func TestBPETokenizerTrainingEmpty(t *testing.T) {
	tokenizer := NewCharLevelBPETokenizer(100, DefaultSpecialIDs())
	if err := tokenizer.Train([]string{}, 100); err != nil {
		t.Fatalf("Train kosong: %v", err)
	}
	if !tokenizer.trained {
		t.Fatal("harus ditandai sebagai dilatih meski kosong")
	}
	if tokenizer.VocabSize() != int64(len(DefaultSpecialIDs())) {
		t.Fatalf("VocabSize = %d, diharapkan %d", tokenizer.VocabSize(), len(DefaultSpecialIDs()))
	}

	encoding, err := tokenizer.Encode(context.Background(), "")
	if err != nil {
		t.Fatalf("Encode setelah train kosong: %v", err)
	}
	// Tidak ada kata, encoding string kosong harus kosong.
	if encoding.Len() != 0 {
		t.Fatalf("Len = %d, diharapkan 0", encoding.Len())
	}
	// Teks non-kosong setelah train kosong menghasilkan UNK per karakter.
	unkEncoding, err := tokenizer.Encode(context.Background(), "ab")
	if err != nil {
		t.Fatalf("Encode non-kosong setelah train kosong: %v", err)
	}
	unkID, ok := tokenizer.SpecialTokens()[UNKName]
	if !ok {
		t.Fatal("UNK tidak ada dalam special tokens")
	}
	for _, id := range unkEncoding.IDs() {
		if id != unkID {
			t.Fatalf("ID = %d, diharapkan UNK %d", id, unkID)
		}
	}
}

func TestBPETokenizerEncodingStructure(t *testing.T) {
	tokenizer := NewCharLevelBPETokenizer(100, DefaultSpecialIDs())
	if err := tokenizer.Train([]string{"ab"}, 100); err != nil {
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

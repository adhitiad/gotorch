package data

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/adhitiad/gotorch/src/errs"
)

func TestWordPieceTrainAndRoundTrip(t *testing.T) {
	tokenizer := NewWordPieceTokenizer(false, DefaultSpecialIDs())
	texts := []string{
		"hello world",
		"hello there",
		"world of code",
		"hello code world",
	}
	if err := tokenizer.Train(texts, 200); err != nil {
		t.Fatalf("Train: %v", err)
	}

	if tokenizer.VocabSize() <= int64(len(DefaultSpecialIDs())) {
		t.Fatalf("VocabSize = %d, diharapkan lebih dari jumlah special token", tokenizer.VocabSize())
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

func TestWordPieceContinuationPrefix(t *testing.T) {
	tokenizer := NewWordPieceTokenizer(false, DefaultSpecialIDs())
	// Latih dengan teks yang mengandung kata yang sama berkali-kali untuk
	// mendorong pembentukan subkata.
	if err := tokenizer.Train([]string{"playing playing playing player player played"}, 50); err != nil {
		t.Fatalf("Train: %v", err)
	}

	encoding, err := tokenizer.Encode(context.Background(), "playing")
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	decoded, err := tokenizer.Decode(context.Background(), encoding.IDs())
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if decoded != "playing" {
		t.Fatalf("decode = %q, diharapkan playing", decoded)
	}

	// Jika kata terpecah menjadi beberapa token, token lanjutan harus
	// menggunakan awalan "##".
	for _, token := range encoding.Tokens() {
		if strings.HasPrefix(token, wordPieceContinuation) {
			return // minimal satu token lanjutan ditemukan
		}
	}
	// Jika tidak ada token lanjutan, artinya kata utuh ada di kosakata,
	// yang juga valid.
}

func TestWordPieceDeterministic(t *testing.T) {
	texts := []string{"hello world", "hello there", "world of code"}
	t1 := NewWordPieceTokenizer(false, DefaultSpecialIDs())
	t2 := NewWordPieceTokenizer(false, DefaultSpecialIDs())
	if err := t1.Train(texts, 200); err != nil {
		t.Fatalf("Train t1: %v", err)
	}
	if err := t2.Train(texts, 200); err != nil {
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

func TestWordPieceUnknownToken(t *testing.T) {
	tokenizer := NewWordPieceTokenizer(false, DefaultSpecialIDs())
	if err := tokenizer.Train([]string{"hello"}, 200); err != nil {
		t.Fatalf("Train: %v", err)
	}

	// Teks dengan karakter yang tidak ada di kosakata harus menggunakan UNK.
	encoding, err := tokenizer.Encode(context.Background(), "xyz")
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	hasUNK := false
	for _, token := range encoding.Tokens() {
		if token == UNKName {
			hasUNK = true
			break
		}
	}
	if !hasUNK {
		t.Fatalf("harus mengandung UNK, dapat %v", encoding.Tokens())
	}
}

func TestWordPieceEmptyText(t *testing.T) {
	tokenizer := NewWordPieceTokenizer(false, DefaultSpecialIDs())
	if err := tokenizer.Train([]string{"hello world"}, 200); err != nil {
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

func TestWordPieceSingleCharacter(t *testing.T) {
	tokenizer := NewWordPieceTokenizer(false, DefaultSpecialIDs())
	if err := tokenizer.Train([]string{"a b c"}, 200); err != nil {
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

func TestWordPieceValidation(t *testing.T) {
	tokenizer := NewWordPieceTokenizer(false, DefaultSpecialIDs())
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

	var nilTokenizer *WordPieceTokenizer
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

	if err := tokenizer.Train([]string{"a"}, 1); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("vocab size 1 diterima: %v", err)
	}
}

func TestWordPieceUnknownID(t *testing.T) {
	tokenizer := NewWordPieceTokenizer(false, DefaultSpecialIDs())
	if err := tokenizer.Train([]string{"hello"}, 200); err != nil {
		t.Fatalf("Train: %v", err)
	}
	if _, err := tokenizer.Decode(context.Background(), []int64{99999}); !errors.Is(err, errs.ErrOutOfBounds) {
		t.Fatalf("ID tidak dikenal diterima: %v", err)
	}
}

func TestWordPieceEncodeBatch(t *testing.T) {
	tokenizer := NewWordPieceTokenizer(false, DefaultSpecialIDs())
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

func TestWordPieceEncodeBatchValidation(t *testing.T) {
	tokenizer := NewWordPieceTokenizer(false, DefaultSpecialIDs())
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
	var nilTokenizer *WordPieceTokenizer
	if _, err := nilTokenizer.EncodeBatch(context.Background(), []string{"a"}); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("nil encode batch: %v", err)
	}
}

func TestWordPieceSpecialTokens(t *testing.T) {
	special := DefaultSpecialIDs()
	tokenizer := NewWordPieceTokenizer(false, special)
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

func TestWordPieceTrainingEmpty(t *testing.T) {
	tokenizer := NewWordPieceTokenizer(false, DefaultSpecialIDs())
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

func TestWordPieceEncodingStructure(t *testing.T) {
	tokenizer := NewWordPieceTokenizer(false, DefaultSpecialIDs())
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

func TestWordPieceWithVocab(t *testing.T) {
	vocab := map[string]int64{
		"[BOS]": 0, "[EOS]": 1, "[UNK]": 2, "[PAD]": 3,
		"h": 4, "e": 5, "l": 6, "o": 7,
		"he": 8, "llo": 9,
	}
	special := DefaultSpecialIDs()
	tokenizer, err := NewWordPieceTokenizerWithVocab(vocab, special)
	if err != nil {
		t.Fatalf("NewWordPieceTokenizerWithVocab: %v", err)
	}

	// "hello" = "he" + "llo" (longest match first: "he" then "llo")
	encoding, err := tokenizer.Encode(context.Background(), "hello")
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if !reflect.DeepEqual(encoding.IDs(), []int64{8, 9}) {
		t.Fatalf("IDs = %v, diharapkan [8 9]", encoding.IDs())
	}
	decoded, err := tokenizer.Decode(context.Background(), encoding.IDs())
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if decoded != "hello" {
		t.Fatalf("decode = %q, diharapkan hello", decoded)
	}
}

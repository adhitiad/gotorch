package data

import (
	"context"
	"math"
	"reflect"
	"strings"
	"testing"
)

func TestUnigramTrainAndRoundTrip(t *testing.T) {
	tokenizer := NewUnigramTokenizer(false, DefaultSpecialIDs())
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

func TestUnigramWordStartPrefix(t *testing.T) {
	tokenizer := NewUnigramTokenizer(false, DefaultSpecialIDs())
	if err := tokenizer.Train([]string{"playing player played"}, 200); err != nil {
		t.Fatalf("Train: %v", err)
	}

	encoding, err := tokenizer.Encode(context.Background(), "playing")
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	// Potongan pertama harus diawali "_" (word start), lanjutan juga dalam gaya SentencePiece.
	foundWordStart := false
	for _, token := range encoding.Tokens() {
		if strings.HasPrefix(token, unigramWordStart) {
			foundWordStart = true
			break
		}
	}
	if !foundWordStart {
		t.Fatalf("harus mengandung token dengan awalan %q, dapat %v", unigramWordStart, encoding.Tokens())
	}

	decoded, err := tokenizer.Decode(context.Background(), encoding.IDs())
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if decoded != "playing" {
		t.Fatalf("decode = %q, diharapkan playing", decoded)
	}
}

func TestUnigramDeterministic(t *testing.T) {
	texts := []string{"hello world", "hello there", "world of code"}
	t1 := NewUnigramTokenizer(false, DefaultSpecialIDs())
	t2 := NewUnigramTokenizer(false, DefaultSpecialIDs())
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

func TestUnigramDPMaximizesLogProb(t *testing.T) {
	// Buat vocab dengan "ab" sebagai satu potongan yang lebih mungkin dari "a"+"b".
	vocab := map[string]int64{
		"[BOS]": 0, "[EOS]": 1, "[UNK]": 2, "[PAD]": 3,
		"a": 4, "b": 5, "ab": 6,
	}
	logProb := map[int64]float64{
		0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0,
		4: math.Log(0.1),
		5: math.Log(0.1),
		6: math.Log(0.8),
	}
	special := DefaultSpecialIDs()
	tokenizer, err := NewUnigramTokenizerWithVocab(vocab, logProb, special)
	if err != nil {
		t.Fatalf("NewUnigramTokenizerWithVocab: %v", err)
	}

	encoding, err := tokenizer.Encode(context.Background(), "ab")
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	// Harus memilih "ab" (satu potongan) daripada "a"+"b" karena log prob lebih tinggi.
	if !reflect.DeepEqual(encoding.IDs(), []int64{6}) {
		t.Fatalf("IDs = %v, diharapkan [6] (pilih ab karena probabilitas lebih tinggi)", encoding.IDs())
	}
	decoded, err := tokenizer.Decode(context.Background(), encoding.IDs())
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if decoded != "ab" {
		t.Fatalf("decode = %q, diharapkan ab", decoded)
	}
}

func TestUnigramUnknownToken(t *testing.T) {
	tokenizer := NewUnigramTokenizer(false, DefaultSpecialIDs())
	if err := tokenizer.Train([]string{"hello"}, 200); err != nil {
		t.Fatalf("Train: %v", err)
	}

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

func TestUnigramEmptyText(t *testing.T) {
	tokenizer := NewUnigramTokenizer(false, DefaultSpecialIDs())
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

func TestUnigramSingleCharacter(t *testing.T) {
	tokenizer := NewUnigramTokenizer(false, DefaultSpecialIDs())
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

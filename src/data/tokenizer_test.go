package data

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/adhitiad/gotorch/src/errs"
)

func TestDefaultSpecialIDs(t *testing.T) {
	first := DefaultSpecialIDs()
	second := DefaultSpecialIDs()
	if len(first) != 7 {
		t.Fatalf("DefaultSpecialIDs length = %d, diharapkan 7", len(first))
	}
	want := map[string]int64{
		BOSName:  0,
		EOSName:  1,
		UNKName:  2,
		PADName:  3,
		CLSName:  4,
		SEPName:  5,
		MASKName: 6,
	}
	if !reflect.DeepEqual(first, want) {
		t.Fatalf("DefaultSpecialIDs = %v, diharapkan %v", first, want)
	}
	// Salinan baru setiap pemanggilan.
	first[BOSName] = 99
	if second[BOSName] != 0 {
		t.Fatal("DefaultSpecialIDs tidak mengembalikan salinan baru")
	}
}

func TestNewEncodingValidation(t *testing.T) {
	if _, err := NewEncoding(nil, nil, nil, nil); err != nil {
		t.Fatalf("empty encoding rejected: %v", err)
	}
	if _, err := NewEncoding([]int64{1}, []string{"a"}, [][2]int64{{0, 1}}, []bool{false}); err != nil {
		t.Fatalf("valid encoding rejected: %v", err)
	}
	if _, err := NewEncoding([]int64{1}, []string{"a", "b"}, [][2]int64{{0, 1}}, []bool{false}); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("mismatched slices accepted: %v", err)
	}
}

func TestEncodingDefensiveCopies(t *testing.T) {
	encoding, err := NewEncoding([]int64{0, 7, 1}, []string{"[BOS]", "a", "[EOS]"}, [][2]int64{{0, 0}, {0, 1}, {1, 1}}, []bool{true, false, true})
	if err != nil {
		t.Fatalf("NewEncoding: %v", err)
	}
	if encoding.Len() != 3 {
		t.Fatalf("Len = %d, diharapkan 3", encoding.Len())
	}
	if !encoding.HasSpecial() {
		t.Fatal("encoding harus mengandung special token")
	}

	ids := encoding.IDs()
	ids[0] = 99
	if got := encoding.IDs()[0]; got != 0 {
		t.Fatalf("IDs salinan tidak defensif: %d", got)
	}

	tokens := encoding.Tokens()
	tokens[0] = "changed"
	if got := encoding.Tokens()[0]; got != "[BOS]" {
		t.Fatalf("Tokens salinan tidak defensif: %q", got)
	}

	offsets := encoding.Offsets()
	offsets[0][0] = 99
	if got := encoding.Offsets()[0][0]; got != 0 {
		t.Fatalf("Offsets salinan tidak defensif: %d", got)
	}

	mask := encoding.SpecialTokensMask()
	mask[0] = false
	if got := encoding.SpecialTokensMask()[0]; !got {
		t.Fatalf("SpecialTokensMask salinan tidak defensif: %v", got)
	}

	cloned := encoding.Clone()
	cloned.IDs()[0] = 99
	if got := encoding.IDs()[0]; got != 0 {
		t.Fatalf("Clone tidak independen: %d", got)
	}
}

func TestEncodingNilReceiver(t *testing.T) {
	var nilEncoding *Encoding
	if nilEncoding.Len() != 0 {
		t.Fatalf("nil Len = %d", nilEncoding.Len())
	}
	if nilEncoding.IDs() != nil || len(nilEncoding.IDs()) != 0 {
		t.Fatalf("nil IDs = %v", nilEncoding.IDs())
	}
	if nilEncoding.Tokens() != nil || len(nilEncoding.Tokens()) != 0 {
		t.Fatalf("nil Tokens = %v", nilEncoding.Tokens())
	}
	if nilEncoding.Offsets() != nil || len(nilEncoding.Offsets()) != 0 {
		t.Fatalf("nil Offsets = %v", nilEncoding.Offsets())
	}
	if nilEncoding.SpecialTokensMask() != nil || len(nilEncoding.SpecialTokensMask()) != 0 {
		t.Fatalf("nil SpecialTokensMask = %v", nilEncoding.SpecialTokensMask())
	}
	if nilEncoding.HasSpecial() {
		t.Fatal("nil HasSpecial harus false")
	}
	if nilEncoding.Clone() != nil {
		t.Fatal("nil Clone harus nil")
	}
}

func TestByteTokenizerRoundTrip(t *testing.T) {
	tokenizer := NewByteTokenizer()
	if tokenizer.VocabSize() != 263 {
		t.Fatalf("VocabSize = %d, diharapkan 263", tokenizer.VocabSize())
	}

	text := "hello"
	encoding, err := tokenizer.Encode(context.Background(), text)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	wantIDs := []int64{0, 104 + 7, 101 + 7, 108 + 7, 108 + 7, 111 + 7, 1}
	if !reflect.DeepEqual(encoding.IDs(), wantIDs) {
		t.Fatalf("IDs = %v, diharapkan %v", encoding.IDs(), wantIDs)
	}
	if !encoding.HasSpecial() {
		t.Fatal("encoded teks harus mengandung BOS/EOS")
	}
	if mask := encoding.SpecialTokensMask(); !mask[0] || !mask[len(mask)-1] {
		t.Fatalf("BOS/EOS harus ditandai special: %v", mask)
	}

	decoded, err := tokenizer.Decode(context.Background(), encoding.IDs())
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if decoded != text {
		t.Fatalf("decode = %q, diharapkan %q", decoded, text)
	}
}

func TestByteTokenizerEmptyText(t *testing.T) {
	tokenizer := NewByteTokenizer()
	encoding, err := tokenizer.Encode(context.Background(), "")
	if err != nil {
		t.Fatalf("Encode empty: %v", err)
	}
	if encoding.Len() != 2 {
		t.Fatalf("Len = %d, diharapkan 2 (BOS+EOS)", encoding.Len())
	}
	if !reflect.DeepEqual(encoding.IDs(), []int64{0, 1}) {
		t.Fatalf("IDs = %v, diharapkan [0 1]", encoding.IDs())
	}
	decoded, err := tokenizer.Decode(context.Background(), encoding.IDs())
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if decoded != "" {
		t.Fatalf("decode empty = %q", decoded)
	}
}

func TestByteTokenizerSpecialTokens(t *testing.T) {
	tokenizer := NewByteTokenizer()
	special := tokenizer.SpecialTokens()
	if len(special) != 7 {
		t.Fatalf("special token count = %d, diharapkan 7", len(special))
	}
	// Salinan defensif.
	special[BOSName] = 99
	if tokenizer.SpecialTokens()[BOSName] != 0 {
		t.Fatal("SpecialTokens tidak mengembalikan salinan baru")
	}

	if got := tokenizer.ID(BOSName); got != 0 {
		t.Fatalf("ID([BOS]) = %d, diharapkan 0", got)
	}
	if got := tokenizer.Token(0); got != BOSName {
		t.Fatalf("Token(0) = %q, diharapkan %q", got, BOSName)
	}
	if got := tokenizer.ID("tidak-ada"); got != -1 {
		t.Fatalf("ID(unknown) = %d, diharapkan -1", got)
	}
	if got := tokenizer.Token(999); got != "" {
		t.Fatalf("Token(999) = %q, diharapkan kosong", got)
	}
}

func TestByteTokenizerValidation(t *testing.T) {
	tokenizer := NewByteTokenizer()
	canceled, cancel := context.WithCancel(context.Background())
	cancel()

	if _, err := tokenizer.Encode(nil, "a"); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("Encode context nil: %v", err)
	}
	if _, err := tokenizer.Encode(canceled, "a"); !errors.Is(err, context.Canceled) {
		t.Fatalf("Encode context dibatalkan: %v", err)
	}
	if _, err := tokenizer.Decode(nil, []int64{7}); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("Decode context nil: %v", err)
	}
	if _, err := tokenizer.Decode(canceled, []int64{7}); !errors.Is(err, context.Canceled) {
		t.Fatalf("Decode context dibatalkan: %v", err)
	}

	var nilTokenizer *ByteTokenizer
	if _, err := nilTokenizer.Encode(context.Background(), "a"); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("nil Encode: %v", err)
	}
	if _, err := nilTokenizer.Decode(context.Background(), []int64{7}); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("nil Decode: %v", err)
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
	if nilTokenizer.Token(7) != "" {
		t.Fatalf("nil Token = %q", nilTokenizer.Token(7))
	}
}

func TestByteTokenizerUnknownID(t *testing.T) {
	tokenizer := NewByteTokenizer()
	if _, err := tokenizer.Decode(context.Background(), []int64{7, 999}); !errors.Is(err, errs.ErrOutOfBounds) {
		t.Fatalf("unknown ID accepted: %v", err)
	}
	if _, err := tokenizer.Decode(context.Background(), []int64{263}); !errors.Is(err, errs.ErrOutOfBounds) {
		t.Fatalf("overflow ID accepted: %v", err)
	}
}

func TestByteTokenizerDeterminism(t *testing.T) {
	first := NewByteTokenizer()
	second := NewByteTokenizer()
	text := "deterministic tokenization test"
	enc1, err := first.Encode(context.Background(), text)
	if err != nil {
		t.Fatalf("Encode first: %v", err)
	}
	enc2, err := second.Encode(context.Background(), text)
	if err != nil {
		t.Fatalf("Encode second: %v", err)
	}
	if !reflect.DeepEqual(enc1.IDs(), enc2.IDs()) {
		t.Fatalf("tidak deterministik: %v vs %v", enc1.IDs(), enc2.IDs())
	}
	dec1, _ := first.Decode(context.Background(), enc1.IDs())
	dec2, _ := second.Decode(context.Background(), enc2.IDs())
	if dec1 != dec2 {
		t.Fatalf("decode tidak deterministik: %q vs %q", dec1, dec2)
	}
}

func TestByteTokenizerEncodeBatch(t *testing.T) {
	tokenizer := NewByteTokenizer()
	texts := []string{"a", "bb", "ccc"}
	encodings, err := tokenizer.EncodeBatch(context.Background(), texts)
	if err != nil {
		t.Fatalf("EncodeBatch: %v", err)
	}
	if len(encodings) != 3 {
		t.Fatalf("jumlah encoding = %d, diharapkan 3", len(encodings))
	}
	for position, text := range texts {
		if encodings[position].Len() != len(text)+2 {
			t.Fatalf("encoding %d Len = %d, diharapkan %d", position, encodings[position].Len(), len(text)+2)
		}
		decoded, err := tokenizer.Decode(context.Background(), encodings[position].IDs())
		if err != nil {
			t.Fatalf("Decode batch %d: %v", position, err)
		}
		if decoded != text {
			t.Fatalf("decode batch %d = %q, diharapkan %q", position, decoded, text)
		}
	}
}

func TestByteTokenizerEncodeBatchValidation(t *testing.T) {
	tokenizer := NewByteTokenizer()
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := tokenizer.EncodeBatch(nil, []string{"a"}); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("EncodeBatch context nil: %v", err)
	}
	if _, err := tokenizer.EncodeBatch(canceled, []string{"a"}); !errors.Is(err, context.Canceled) {
		t.Fatalf("EncodeBatch context dibatalkan: %v", err)
	}
	var nilTokenizer *ByteTokenizer
	if _, err := nilTokenizer.EncodeBatch(context.Background(), []string{"a"}); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("nil EncodeBatch: %v", err)
	}
}

func TestTokenizerFunc(t *testing.T) {
	encodeFunc := func(ctx context.Context, text string) (*Encoding, error) {
		if err := checkContext(ctx); err != nil {
			return nil, err
		}
		return NewEncoding([]int64{int64(len(text))}, []string{text}, [][2]int64{{0, int64(len(text))}}, []bool{false})
	}
	decodeFunc := func(ctx context.Context, ids []int64) (string, error) {
		if err := checkContext(ctx); err != nil {
			return "", err
		}
		if len(ids) == 0 {
			return "", nil
		}
		return string(rune(ids[0])), nil
	}
	tokenizer, err := NewTokenizerFunc(encodeFunc, decodeFunc, 10, DefaultSpecialIDs())
	if err != nil {
		t.Fatalf("NewTokenizerFunc: %v", err)
	}
	if tokenizer.VocabSize() != 10 {
		t.Fatalf("VocabSize = %d, diharapkan 10", tokenizer.VocabSize())
	}
	encoding, err := tokenizer.Encode(context.Background(), "hi")
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if encoding.Len() != 1 || encoding.IDs()[0] != 2 {
		t.Fatalf("unexpected encoding: %v", encoding.IDs())
	}
	decoded, err := tokenizer.Decode(context.Background(), []int64{65})
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if decoded != "A" {
		t.Fatalf("decode = %q, diharapkan A", decoded)
	}

	// Validasi konstruksi.
	if _, err := NewTokenizerFunc(nil, decodeFunc, 10, nil); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatal("nil encodeFunc accepted")
	}
	if _, err := NewTokenizerFunc(encodeFunc, nil, 10, nil); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatal("nil decodeFunc accepted")
	}
	if _, err := NewTokenizerFunc(encodeFunc, decodeFunc, 0, nil); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatal("zero vocabSize accepted")
	}
	if _, err := NewTokenizerFunc(encodeFunc, decodeFunc, 10, map[string]int64{"[BOS]": 10}); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatal("special ID outside vocab accepted")
	}
	if _, err := NewTokenizerFunc(encodeFunc, decodeFunc, 10, map[string]int64{"[BOS]": 0, "[EOS]": 0}); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatal("duplicate special ID accepted")
	}
}

func TestTokenizerFuncNilReceiver(t *testing.T) {
	var nilTokenizer *TokenizerFunc
	if _, err := nilTokenizer.Encode(context.Background(), "a"); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("nil Encode: %v", err)
	}
	if _, err := nilTokenizer.Decode(context.Background(), []int64{1}); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("nil Decode: %v", err)
	}
	if _, err := nilTokenizer.EncodeBatch(context.Background(), []string{"a"}); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("nil EncodeBatch: %v", err)
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
}

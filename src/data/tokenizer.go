package data

import (
	"context"
	"fmt"

	"github.com/adhitiad/gotorch/src/errs"
)

// Tokenizer mengubah teks menjadi ID token dan kembali. Encode dan Decode
// deterministik untuk konfigurasi tokenizer yang sama dan mendukung pembatalan
// konteks. Implementasi aman untuk panggilan bersamaan.
type Tokenizer interface {
	// Encode mengubah teks menjadi Encoding.
	Encode(ctx context.Context, text string) (*Encoding, error)
	// EncodeBatch mengubah beberapa teks menjadi Encoding.
	EncodeBatch(ctx context.Context, texts []string) ([]*Encoding, error)
	// Decode mengubah ID token kembali menjadi teks.
	Decode(ctx context.Context, ids []int64) (string, error)
	// VocabSize mengembalikan ukuran kosakata.
	VocabSize() int64
	// SpecialTokens mengembalikan salinan pemetaan nama special token ke ID.
	SpecialTokens() map[string]int64
	// ID mengembalikan ID untuk string token, atau -1 jika tidak ditemukan.
	ID(token string) int64
	// Token mengembalikan string token untuk ID, atau "" jika tidak ditemukan.
	Token(id int64) string
}

// TokenizerFunc menyesuaikan fungsi biasa menjadi Tokenizer. Semua metode
// mendelegasikan ke fungsi yang dibungkus kecuali yang ditandai di bawah.
type TokenizerFunc struct {
	encodeFunc func(context.Context, string) (*Encoding, error)
	decodeFunc func(context.Context, []int64) (string, error)
	vocabSize  int64
	special    map[string]int64
	reverse    map[int64]string
}

// NewTokenizerFunc membentuk TokenizerFunc. Fungsi encode dan decode harus
// non-nil. ukuran kosakata harus positif. ID khusus tidak boleh tumpang tindih
// dengan ID reguler (0 hingga vocabSize-1).
func NewTokenizerFunc(
	encodeFunc func(context.Context, string) (*Encoding, error),
	decodeFunc func(context.Context, []int64) (string, error),
	vocabSize int64,
	special map[string]int64,
) (*TokenizerFunc, error) {
	if encodeFunc == nil {
		return nil, fmt.Errorf("%w: nil encode function", errs.ErrInvalidArgument)
	}
	if decodeFunc == nil {
		return nil, fmt.Errorf("%w: nil decode function", errs.ErrInvalidArgument)
	}
	if vocabSize <= 0 {
		return nil, fmt.Errorf("%w: vocabulary size must be positive, got %d", errs.ErrInvalidArgument, vocabSize)
	}
	reverse := make(map[int64]string, len(special))
	for name, id := range special {
		if name == "" {
			return nil, fmt.Errorf("%w: empty special token name", errs.ErrInvalidArgument)
		}
		if id < 0 || id >= vocabSize {
			return nil, fmt.Errorf("%w: special token %q ID %d outside [0,%d)",
				errs.ErrInvalidArgument, name, id, vocabSize)
		}
		if _, exists := reverse[id]; exists {
			return nil, fmt.Errorf("%w: duplicate special token ID %d", errs.ErrInvalidArgument, id)
		}
		reverse[id] = name
	}
	return &TokenizerFunc{
		encodeFunc: encodeFunc,
		decodeFunc: decodeFunc,
		vocabSize:  vocabSize,
		special:    appendMap(nil, special),
		reverse:    reverse,
	}, nil
}

// Encode implements Tokenizer.
func (t *TokenizerFunc) Encode(ctx context.Context, text string) (*Encoding, error) {
	if t == nil || t.encodeFunc == nil {
		return nil, fmt.Errorf("%w: nil tokenizer", errs.ErrInvalidState)
	}
	return t.encodeFunc(ctx, text)
}

// EncodeBatch implements Tokenizer.
func (t *TokenizerFunc) EncodeBatch(ctx context.Context, texts []string) ([]*Encoding, error) {
	if t == nil || t.encodeFunc == nil {
		return nil, fmt.Errorf("%w: nil tokenizer", errs.ErrInvalidState)
	}
	if err := checkContext(ctx); err != nil {
		return nil, err
	}
	encodings := make([]*Encoding, 0, len(texts))
	for position, text := range texts {
		if err := checkContext(ctx); err != nil {
			return nil, err
		}
		encoding, err := t.encodeFunc(ctx, text)
		if err != nil {
			return nil, fmt.Errorf("%w: text %d", err, position)
		}
		encodings = append(encodings, encoding)
	}
	return encodings, nil
}

// Decode implements Tokenizer.
func (t *TokenizerFunc) Decode(ctx context.Context, ids []int64) (string, error) {
	if t == nil || t.decodeFunc == nil {
		return "", fmt.Errorf("%w: nil tokenizer", errs.ErrInvalidState)
	}
	return t.decodeFunc(ctx, ids)
}

// VocabSize implements Tokenizer.
func (t *TokenizerFunc) VocabSize() int64 {
	if t == nil {
		return 0
	}
	return t.vocabSize
}

// SpecialTokens implements Tokenizer.
func (t *TokenizerFunc) SpecialTokens() map[string]int64 {
	if t == nil {
		return nil
	}
	return appendMap(nil, t.special)
}

// ID implements Tokenizer.
func (t *TokenizerFunc) ID(token string) int64 {
	if t == nil {
		return -1
	}
	if id, ok := t.special[token]; ok {
		return id
	}
	// TokenizerFunc tidak menyimpan kosakata reguler; implementasi spesifik
	// harus menangani pencarian reguler sendiri.
	return -1
}

// Token implements Tokenizer.
func (t *TokenizerFunc) Token(id int64) string {
	if t == nil {
		return ""
	}
	if name, ok := t.reverse[id]; ok {
		return name
	}
	return ""
}

// appendMap menyalin peta ke peta baru.
func appendMap(dst map[string]int64, src map[string]int64) map[string]int64 {
	if dst == nil {
		dst = make(map[string]int64, len(src))
	}
	for name, id := range src {
		dst[name] = id
	}
	return dst
}

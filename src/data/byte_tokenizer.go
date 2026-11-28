package data

import (
	"context"
	"fmt"

	"github.com/adhitiad/gotorch/src/errs"
)

// ByteTokenizer adalah implementasi referensi deterministik yang memetakan setiap
// byte ke ID token. Special token menempati ID 0..6 dan nilai byte menempati ID
// 7..262. Encode menambahkan BOS di depan dan EOS di belakang.
type ByteTokenizer struct {
	vocabSize int64
	special   map[string]int64
	reverse   map[int64]string
}

var _ Tokenizer = (*ByteTokenizer)(nil)

// NewByteTokenizer membentuk tokenizer level byte.
func NewByteTokenizer() *ByteTokenizer {
	special := DefaultSpecialIDs()
	reverse := make(map[int64]string, len(special)+256)
	for name, id := range special {
		reverse[id] = name
	}
	for b := 0; b < 256; b++ {
		id := int64(b) + 7
		reverse[id] = string([]byte{byte(b)})
	}
	return &ByteTokenizer{
		vocabSize: 263,
		special:   special,
		reverse:   reverse,
	}
}

// Encode implements Tokenizer.
func (t *ByteTokenizer) Encode(ctx context.Context, text string) (*Encoding, error) {
	if err := checkContext(ctx); err != nil {
		return nil, err
	}
	if t == nil {
		return nil, fmt.Errorf("%w: nil tokenizer", errs.ErrInvalidState)
	}
	ids := make([]int64, 0, len(text)+2)
	tokens := make([]string, 0, len(text)+2)
	offsets := make([][2]int64, 0, len(text)+2)
	specialMask := make([]bool, 0, len(text)+2)

	ids = append(ids, 0) // BOS
	tokens = append(tokens, BOSName)
	offsets = append(offsets, [2]int64{0, 0})
	specialMask = append(specialMask, true)

	pos := int64(0)
	for i := 0; i < len(text); i++ {
		id := int64(text[i]) + 7
		ids = append(ids, id)
		tokens = append(tokens, string(text[i]))
		offsets = append(offsets, [2]int64{pos, pos + 1})
		specialMask = append(specialMask, false)
		pos++
	}

	ids = append(ids, 1) // EOS
	tokens = append(tokens, EOSName)
	offsets = append(offsets, [2]int64{pos, pos})
	specialMask = append(specialMask, true)

	return &Encoding{
		ids:         ids,
		tokens:      tokens,
		offsets:     offsets,
		specialMask: specialMask,
	}, nil
}

// EncodeBatch implements Tokenizer.
func (t *ByteTokenizer) EncodeBatch(ctx context.Context, texts []string) ([]*Encoding, error) {
	if err := checkContext(ctx); err != nil {
		return nil, err
	}
	if t == nil {
		return nil, fmt.Errorf("%w: nil tokenizer", errs.ErrInvalidState)
	}
	encodings := make([]*Encoding, 0, len(texts))
	for position, text := range texts {
		if err := checkContext(ctx); err != nil {
			return nil, err
		}
		encoding, err := t.Encode(ctx, text)
		if err != nil {
			return nil, fmt.Errorf("%w: text %d", err, position)
		}
		encodings = append(encodings, encoding)
	}
	return encodings, nil
}

// Decode implements Tokenizer. BOS, EOS, PAD, dan UNK diabaikan; ID byte
// dikembalikan menjadi byte asli. ID di luar jangkauan menghasilkan error.
func (t *ByteTokenizer) Decode(ctx context.Context, ids []int64) (string, error) {
	if err := checkContext(ctx); err != nil {
		return "", err
	}
	if t == nil {
		return "", fmt.Errorf("%w: nil tokenizer", errs.ErrInvalidState)
	}
	var bytes []byte
	for position, id := range ids {
		switch id {
		case 0: // BOS
		case 1: // EOS
		case 3: // PAD
		case 2: // UNK
		default:
			if id < 7 || id > 262 {
				return "", fmt.Errorf("%w: unknown token ID %d at position %d", errs.ErrOutOfBounds, id, position)
			}
			bytes = append(bytes, byte(id-7))
		}
	}
	return string(bytes), nil
}

// VocabSize implements Tokenizer.
func (t *ByteTokenizer) VocabSize() int64 {
	if t == nil {
		return 0
	}
	return t.vocabSize
}

// SpecialTokens implements Tokenizer.
func (t *ByteTokenizer) SpecialTokens() map[string]int64 {
	if t == nil {
		return nil
	}
	return appendMap(nil, t.special)
}

// ID implements Tokenizer.
func (t *ByteTokenizer) ID(token string) int64 {
	if t == nil {
		return -1
	}
	if id, ok := t.special[token]; ok {
		return id
	}
	if len(token) == 1 {
		return int64(token[0]) + 7
	}
	return -1
}

// Token implements Tokenizer.
func (t *ByteTokenizer) Token(id int64) string {
	if t == nil {
		return ""
	}
	if name, ok := t.reverse[id]; ok {
		return name
	}
	return ""
}

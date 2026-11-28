package data

import (
	"context"
	"fmt"
	"strings"

	"github.com/adhitiad/gotorch/src/errs"
)

// unigramWordStart adalah awalan untuk potongan yang dimulai kata baru
// dalam gaya SentencePiece/Unigram.
const unigramWordStart = "_"

// UnigramTokenizer mengimplementasikan tokenisasi gaya SentencePiece/Unigram.
// Menggunakan model bahasa unigram dengan probabilitas log untuk setiap potongan
// dan pemotongan melalui pemrograman dinamis untuk memaksimalkan jumlah
// probabilitas log. Implementasi ini independen dari BPE dan WordPiece.
type UnigramTokenizer struct {
	vocab     map[string]int64  // potongan string -> id
	idToToken map[int64]string  // id -> potongan string
	logProb   map[int64]float64 // id -> log probabilitas
	special   map[string]int64
	byteLevel bool
	trained   bool
}

var _ Tokenizer = (*UnigramTokenizer)(nil)

// NewUnigramTokenizer membentuk tokenizer Unigram.
// byteLevel=true bekerja pada byte UTF-8; false bekerja pada rune Unicode.
// ID khusus disisipkan di awal kosakata.
func NewUnigramTokenizer(byteLevel bool, special map[string]int64) *UnigramTokenizer {
	spec := appendMap(nil, special)
	return &UnigramTokenizer{
		vocab:     make(map[string]int64),
		idToToken: make(map[int64]string),
		logProb:   make(map[int64]float64),
		special:   spec,
		byteLevel: byteLevel,
	}
}

// Encode implements Tokenizer. Menggunakan pemrograman dinamis untuk mencari
// segmentasi yang memaksimalkan jumlah log probabilitas. Potongan awal kata
// diawali "_" (gaya SentencePiece).
func (t *UnigramTokenizer) Encode(ctx context.Context, text string) (*Encoding, error) {
	if err := checkContext(ctx); err != nil {
		return nil, err
	}
	if t == nil {
		return nil, fmt.Errorf("%w: nil tokenizer", errs.ErrInvalidState)
	}
	if !t.trained {
		return nil, fmt.Errorf("%w: tokenizer not trained", errs.ErrInvalidState)
	}

	ids := make([]int64, 0)
	tokens := make([]string, 0)
	offsets := make([][2]int64, 0)
	specialMask := make([]bool, 0)

	unkID, hasUNK := t.special[UNKName]

	words := strings.Fields(text)
	for wordIndex, word := range words {
		if wordIndex > 0 {
			if spaceID, ok := t.vocab[" "]; ok {
				ids = append(ids, spaceID)
				tokens = append(tokens, " ")
				offsets = append(offsets, [2]int64{0, 0})
				specialMask = append(specialMask, false)
			}
		}

		pieces := t.segmentWord(word, unkID, hasUNK)
		for _, piece := range pieces {
			lookupKey := strings.TrimPrefix(piece, unigramWordStart)
			id, ok := t.vocab[lookupKey]
			if !ok {
				id = unkID
				piece = UNKName
			}
			ids = append(ids, id)
			tokens = append(tokens, piece)
			offsets = append(offsets, [2]int64{0, 0})
			if _, special := t.special[piece]; special {
				specialMask = append(specialMask, true)
			} else {
				specialMask = append(specialMask, false)
			}
		}
	}

	return &Encoding{
		ids:         ids,
		tokens:      tokens,
		offsets:     offsets,
		specialMask: specialMask,
	}, nil
}

// EncodeBatch implements Tokenizer.
func (t *UnigramTokenizer) EncodeBatch(ctx context.Context, texts []string) ([]*Encoding, error) {
	if err := checkContext(ctx); err != nil {
		return nil, err
	}
	if t == nil {
		return nil, fmt.Errorf("%w: nil tokenizer", errs.ErrInvalidState)
	}
	if !t.trained {
		return nil, fmt.Errorf("%w: tokenizer not trained", errs.ErrInvalidState)
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

// Decode implements Tokenizer. Menggabungkan string potongan, menghapus awalan
// "_" dan melewati special token.
func (t *UnigramTokenizer) Decode(ctx context.Context, ids []int64) (string, error) {
	if err := checkContext(ctx); err != nil {
		return "", err
	}
	if t == nil {
		return "", fmt.Errorf("%w: nil tokenizer", errs.ErrInvalidState)
	}
	if !t.trained {
		return "", fmt.Errorf("%w: tokenizer not trained", errs.ErrInvalidState)
	}

	var parts []string
	for position, id := range ids {
		token, ok := t.idToToken[id]
		if !ok {
			return "", fmt.Errorf("%w: unknown token ID %d at position %d", errs.ErrOutOfBounds, id, position)
		}
		if _, special := t.special[token]; special {
			continue
		}
		if strings.HasPrefix(token, unigramWordStart) {
			parts = append(parts, strings.TrimPrefix(token, unigramWordStart))
		} else {
			parts = append(parts, token)
		}
	}
	return strings.Join(parts, ""), nil
}

// VocabSize implements Tokenizer.
func (t *UnigramTokenizer) VocabSize() int64 {
	if t == nil {
		return 0
	}
	return int64(len(t.vocab))
}

// SpecialTokens implements Tokenizer.
func (t *UnigramTokenizer) SpecialTokens() map[string]int64 {
	if t == nil {
		return nil
	}
	return appendMap(nil, t.special)
}

// ID implements Tokenizer.
func (t *UnigramTokenizer) ID(token string) int64 {
	if t == nil {
		return -1
	}
	if id, ok := t.vocab[token]; ok {
		return id
	}
	return -1
}

// Token implements Tokenizer.
func (t *UnigramTokenizer) Token(id int64) string {
	if t == nil {
		return ""
	}
	if tok, ok := t.idToToken[id]; ok {
		return tok
	}
	return ""
}

package data

import (
	"context"
	"fmt"
	"strings"
	"unicode/utf8"

	"github.com/adhitiad/gotorch/src/errs"
)

// mergeRule adalah aturan penggabungan BPE: pasangan token (kiri, kanan) yang
// digabungkan menjadi satu token baru dengan ID tertentu pada peringkat tertentu.
type mergeRule struct {
	left  int64
	right int64
	id    int64
	rank  int
}

// BPETokenizer mengimplementasikan tokenisasi Byte Pair Encoding (BPE).
// Dukungan baik untuk BPE tingkat karakter maupun tingkat byte. Kosakata dan
// aturan penggabungan dibentuk melalui Train; Encode menerapkan aturan dalam
// urutan peringkat hingga tidak ada penggabungan lagi yang mungkin.
// Implementasi deterministik dan aman untuk panggilan bersamaan setelah dilatih.
type BPETokenizer struct {
	vocab     map[string]int64 // token string -> id
	idToToken map[int64]string // id -> token string
	merges    []mergeRule      // diurutkan menurut rank
	mergeIdx  map[[2]int64]int // pasangan (kiri, kanan) -> indeks dalam merges
	special   map[string]int64
	revSpec   map[int64]string
	byteLevel bool
	trained   bool
}

var _ Tokenizer = (*BPETokenizer)(nil)

// NewBPETokenizer membentuk tokenizer BPE dengan ukuran kosakata yang diinginkan.
// byteLevel=true menggunakan byte UTF-8 sebagai unit dasar; false menggunakan
// rune Unicode. ID khusus disisipkan di awal kosakata.
func NewBPETokenizer(vocabSize int, byteLevel bool, special map[string]int64) *BPETokenizer {
	if vocabSize < 0 {
		vocabSize = 0
	}
	spec := appendMap(nil, special)
	rev := make(map[int64]string, len(spec))
	for name, id := range spec {
		rev[id] = name
	}
	return &BPETokenizer{
		vocab:     make(map[string]int64, vocabSize),
		idToToken: make(map[int64]string, vocabSize),
		special:   spec,
		revSpec:   rev,
		byteLevel: byteLevel,
	}
}

// Encode implements Tokenizer.
func (t *BPETokenizer) Encode(ctx context.Context, text string) (*Encoding, error) {
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

	words := strings.Fields(text)
	spaceID, hasSpace := t.vocab[" "]

	for wordIndex, word := range words {
		if wordIndex > 0 {
			if hasSpace {
				ids = append(ids, spaceID)
				tokens = append(tokens, " ")
				offsets = append(offsets, [2]int64{0, 0})
				specialMask = append(specialMask, false)
			}
		}

		unitIDs := t.wordToUnitIDs(word)
		merged := t.applyMerges(unitIDs)

		for _, id := range merged {
			token := t.idToToken[id]
			ids = append(ids, id)
			tokens = append(tokens, token)
			offsets = append(offsets, [2]int64{0, 0})
			if _, special := t.special[token]; special {
				specialMask = append(specialMask, true)
			} else {
				specialMask = append(specialMask, false)
			}
		}
	}

	if len(ids) == 0 && len(text) > 0 {
		// Teks hanya spasi; gunakan UNK jika tersedia.
		if unkID, ok := t.special[UNKName]; ok {
			ids = append(ids, unkID)
			tokens = append(tokens, UNKName)
			offsets = append(offsets, [2]int64{0, 0})
			specialMask = append(specialMask, true)
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
func (t *BPETokenizer) EncodeBatch(ctx context.Context, texts []string) ([]*Encoding, error) {
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

// Decode implements Tokenizer. Menggabungkan string token dan, untuk byte-level,
// menafsirkan hasil sebagai UTF-8.
func (t *BPETokenizer) Decode(ctx context.Context, ids []int64) (string, error) {
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
		parts = append(parts, token)
	}

	joined := strings.Join(parts, "")
	if t.byteLevel {
		if !utf8.ValidString(joined) {
			return "", fmt.Errorf("%w: decoded bytes are not valid UTF-8", errs.ErrInvalidState)
		}
	}
	return joined, nil
}

// VocabSize implements Tokenizer.
func (t *BPETokenizer) VocabSize() int64 {
	if t == nil {
		return 0
	}
	return int64(len(t.vocab))
}

// SpecialTokens implements Tokenizer.
func (t *BPETokenizer) SpecialTokens() map[string]int64 {
	if t == nil {
		return nil
	}
	return appendMap(nil, t.special)
}

// ID implements Tokenizer.
func (t *BPETokenizer) ID(token string) int64 {
	if t == nil {
		return -1
	}
	if id, ok := t.vocab[token]; ok {
		return id
	}
	return -1
}

// Token implements Tokenizer.
func (t *BPETokenizer) Token(id int64) string {
	if t == nil {
		return ""
	}
	if tok, ok := t.idToToken[id]; ok {
		return tok
	}
	return ""
}

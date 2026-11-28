package data

import (
	"context"
	"fmt"
	"sort"
	"strings"

	"github.com/adhitiad/gotorch/src/errs"
)

// wordPieceContinuation adalah awalan untuk subkata yang bukan awal kata
// dalam WordPiece (mirip BERT).
const wordPieceContinuation = "##"

// WordPieceTokenizer mengimplementasikan tokenisasi WordPiece (mirip BERT).
// Menggunakan algoritma pencocokan terpanjang terlebih dahulu (longest-match-first)
// dengan awalan "##" untuk subkata lanjutan. Implementasi ini independen dari
// BPE dan tidak berbagi kode pelatihan atau segmentasi dengan BPETokenizer.
type WordPieceTokenizer struct {
	vocab     map[string]int64 // token string -> id
	idToToken map[int64]string // id -> token string
	special   map[string]int64
	byteLevel bool
	trained   bool
}

var _ Tokenizer = (*WordPieceTokenizer)(nil)

// NewWordPieceTokenizer membentuk tokenizer WordPiece.
// byteLevel=true bekerja pada byte UTF-8; false bekerja pada rune Unicode.
// ID khusus disisipkan di awal kosakata.
func NewWordPieceTokenizer(byteLevel bool, special map[string]int64) *WordPieceTokenizer {
	spec := appendMap(nil, special)
	return &WordPieceTokenizer{
		vocab:     make(map[string]int64),
		idToToken: make(map[int64]string),
		special:   spec,
		byteLevel: byteLevel,
	}
}

// Train melatih tokenizer WordPiece pada kumpulan teks. Membentuk kosakata
// dari semua subkata yang ditemukan. Panggilan berikutnya menggantikan hasil
// pelatihan sebelumnya.
func (t *WordPieceTokenizer) Train(texts []string, vocabSize int) error {
	if t == nil {
		return fmt.Errorf("%w: nil tokenizer", errs.ErrInvalidState)
	}
	if vocabSize <= len(t.special) {
		return fmt.Errorf("%w: vocab size %d must exceed special token count %d", errs.ErrInvalidArgument, vocabSize, len(t.special))
	}

	t.vocab = appendMap(nil, t.special)
	t.idToToken = make(map[int64]string, vocabSize)
	for name, id := range t.special {
		t.idToToken[id] = name
	}
	nextID := int64(len(t.special))

	// Kumpulkan frekuensi kata.
	wordFreq := make(map[string]int)
	for _, text := range texts {
		for _, word := range strings.Fields(text) {
			wordFreq[word]++
		}
	}

	// Kumpulkan semua unit unik dan urutkan untuk ID karakter yang deterministik.
	unitSet := make(map[string]bool)
	for word := range wordFreq {
		for _, unit := range t.wordUnits(word) {
			unitSet[unit] = true
		}
	}
	units := make([]string, 0, len(unitSet))
	for unit := range unitSet {
		units = append(units, unit)
	}
	sort.Strings(units)

	// Inisialisasi kosakata dengan unit dasar yang diurutkan.
	for _, unit := range units {
		if _, ok := t.vocab[unit]; !ok {
			id := nextID
			nextID++
			t.vocab[unit] = id
			t.idToToken[id] = unit
		}
	}
	// Spasi sebagai pemisah antar kata.
	if _, ok := t.vocab[" "]; !ok {
		id := nextID
		nextID++
		t.vocab[" "] = id
		t.idToToken[id] = " "
	}

	// Ulangi: tokenisasi semua kata dengan kosakata saat ini, hitung pasangan
	// terdekat, dan gabungkan pasangan yang paling sering hingga ukuran
	// kosakata tercapai.
	type tokenizedWord struct {
		tokens []string
		freq   int
	}
	tokenized := make([]tokenizedWord, 0, len(wordFreq))
	for word, freq := range wordFreq {
		tokens := t.tokenizeWordBasic(word)
		tokenized = append(tokenized, tokenizedWord{tokens: tokens, freq: freq})
	}

	for len(t.vocab) < vocabSize {
		pairFreq := make(map[[2]string]int)
		for _, tw := range tokenized {
			for i := 0; i+1 < len(tw.tokens); i++ {
				pair := [2]string{tw.tokens[i], tw.tokens[i+1]}
				pairFreq[pair] += tw.freq
			}
		}
		if len(pairFreq) == 0 {
			break
		}

		// Pilih pasangan dengan frekuensi tertinggi; tali tambahan menggunakan
		// perbandingan lexikografis lengkap (kiri lalu kanan) untuk determinisme.
		var bestLeft, bestRight string
		var bestFreq int
		found := false
		for pair, freq := range pairFreq {
			if !found || freq > bestFreq ||
				(freq == bestFreq && (pair[0] < bestLeft || (pair[0] == bestLeft && pair[1] < bestRight))) {
				bestLeft, bestRight = pair[0], pair[1]
				bestFreq = freq
				found = true
			}
		}
		if !found {
			break
		}

		merged := bestLeft + bestRight
		if _, ok := t.vocab[merged]; ok {
			// Sudah ada, tidak bisa menambahkan lagi, hentikan pelatihan.
			break
		}
		id := nextID
		nextID++
		t.vocab[merged] = id
		t.idToToken[id] = merged

		// Re-tokenisasi semua kata dengan kosakata yang diperbarui.
		for i := range tokenized {
			tokenized[i].tokens = t.tokenizeWordBasic(strings.Join(tokenized[i].tokens, ""))
		}
	}

	t.trained = true
	return nil
}

// tokenizeWordBasic memotong kata menggunakan kosakata saat ini, tanpa
// mempertimbangkan UNK (digunakan selama training).
func (t *WordPieceTokenizer) tokenizeWordBasic(word string) []string {
	if len(word) == 0 {
		return nil
	}

	var result []string
	start := 0
	for start < len(word) {
		end := len(word)
		found := false
		for end > start {
			candidate := word[start:end]
			lookup := candidate
			if start > 0 {
				lookup = word[start:end]
			}
			if _, ok := t.vocab[lookup]; ok {
				result = append(result, lookup)
				start = end
				found = true
				break
			}
			end--
		}
		if !found {
			// Tidak ada cocokan; ambil satu karakter sebagai fallback.
			result = append(result, word[start:start+1])
			start++
		}
	}
	return result
}

// wordUnits mengembalikan unit dasar (karakter atau byte) dari suatu kata.
func (t *WordPieceTokenizer) wordUnits(word string) []string {
	units := make([]string, 0, len(word))
	if t.byteLevel {
		for _, b := range []byte(word) {
			units = append(units, string([]byte{b}))
		}
	} else {
		for _, r := range word {
			units = append(units, string(r))
		}
	}
	return units
}

// Encode implements Tokenizer. Menggunakan pencocokan terpanjang terlebih dahulu:
// untuk setiap kata, cari cocokan terpanjang di kosakata dari awal, tambahkan,
// lalu lanjutkan dengan sisa yang diawali "##". Jika tidak ada cocokan sama sekali,
// gunakan [UNK].
func (t *WordPieceTokenizer) Encode(ctx context.Context, text string) (*Encoding, error) {
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

		subwords := t.tokenizeWord(word, unkID, hasUNK)
		for _, sw := range subwords {
			lookupKey := sw
			if strings.HasPrefix(sw, wordPieceContinuation) {
				lookupKey = strings.TrimPrefix(sw, wordPieceContinuation)
			}
			id, ok := t.vocab[lookupKey]
			if !ok {
				id = unkID
				sw = UNKName
			}
			ids = append(ids, id)
			tokens = append(tokens, sw)
			offsets = append(offsets, [2]int64{0, 0})
			if _, special := t.special[sw]; special {
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

// tokenizeWord memotong satu kata menjadi subkata WordPiece.
func (t *WordPieceTokenizer) tokenizeWord(word string, unkID int64, hasUNK bool) []string {
	if len(word) == 0 {
		return nil
	}

	var result []string
	start := 0
	for start < len(word) {
		// Cari cocokan terpanjang dari posisi start.
		end := len(word)
		found := false
		for end > start {
			candidate := word[start:end]
			lookup := candidate
			if start > 0 {
				candidate = wordPieceContinuation + candidate
				lookup = word[start:end]
			}
			if _, ok := t.vocab[lookup]; ok {
				result = append(result, candidate)
				start = end
				found = true
				break
			}
			end--
		}
		if !found {
			if hasUNK {
				result = append(result, UNKName)
			} else {
				result = append(result, word[start:])
			}
			break
		}
	}
	return result
}

// EncodeBatch implements Tokenizer.
func (t *WordPieceTokenizer) EncodeBatch(ctx context.Context, texts []string) ([]*Encoding, error) {
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

// Decode implements Tokenizer. Menggabungkan string token, menghapus awalan
// "##" dan melewati special token.
func (t *WordPieceTokenizer) Decode(ctx context.Context, ids []int64) (string, error) {
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
		if strings.HasPrefix(token, wordPieceContinuation) {
			parts = append(parts, strings.TrimPrefix(token, wordPieceContinuation))
		} else {
			parts = append(parts, token)
		}
	}
	return strings.Join(parts, ""), nil
}

// VocabSize implements Tokenizer.
func (t *WordPieceTokenizer) VocabSize() int64 {
	if t == nil {
		return 0
	}
	return int64(len(t.vocab))
}

// SpecialTokens implements Tokenizer.
func (t *WordPieceTokenizer) SpecialTokens() map[string]int64 {
	if t == nil {
		return nil
	}
	return appendMap(nil, t.special)
}

// ID implements Tokenizer.
func (t *WordPieceTokenizer) ID(token string) int64 {
	if t == nil {
		return -1
	}
	if id, ok := t.vocab[token]; ok {
		return id
	}
	return -1
}

// Token implements Tokenizer.
func (t *WordPieceTokenizer) Token(id int64) string {
	if t == nil {
		return ""
	}
	if tok, ok := t.idToToken[id]; ok {
		return tok
	}
	return ""
}

// NewWordPieceTokenizerWithVocab membentuk WordPieceTokenizer dengan kosakata
// yang ditentukan secara langsung (untuk pengujian kompatibilitas).
func NewWordPieceTokenizerWithVocab(vocab map[string]int64, special map[string]int64) (*WordPieceTokenizer, error) {
	if len(vocab) == 0 {
		return nil, fmt.Errorf("%w: empty vocabulary", errs.ErrInvalidArgument)
	}
	spec := appendMap(nil, special)
	idToToken := make(map[int64]string, len(vocab))
	for token, id := range vocab {
		idToToken[id] = token
	}
	return &WordPieceTokenizer{
		vocab:     appendMap(nil, vocab),
		idToToken: idToToken,
		special:   spec,
		trained:   true,
	}, nil
}

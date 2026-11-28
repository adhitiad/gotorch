package data

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/adhitiad/gotorch/src/errs"
)

// LogProb mengembalikan log probabilitas untuk ID potongan, atau math.Inf(-1)
// jika ID tidak ditemukan.
func (t *UnigramTokenizer) LogProb(id int64) float64 {
	if t == nil {
		return math.Inf(-1)
	}
	if lp, ok := t.logProb[id]; ok {
		return lp
	}
	return math.Inf(-1)
}

// NewUnigramTokenizerWithVocab membentuk UnigramTokenizer dengan kosakata dan
// probabilitas yang ditentukan secara langsung (untuk pengujian kompatibilitas).
func NewUnigramTokenizerWithVocab(vocab map[string]int64, logProb map[int64]float64, special map[string]int64) (*UnigramTokenizer, error) {
	if len(vocab) == 0 {
		return nil, fmt.Errorf("%w: empty vocabulary", errs.ErrInvalidArgument)
	}
	spec := appendMap(nil, special)
	idToToken := make(map[int64]string, len(vocab))
	probs := make(map[int64]float64, len(vocab))
	for token, id := range vocab {
		idToToken[id] = token
		if lp, ok := logProb[id]; ok {
			probs[id] = lp
		} else {
			probs[id] = 0.0
		}
	}
	return &UnigramTokenizer{
		vocab:     appendMap(nil, vocab),
		idToToken: idToToken,
		logProb:   probs,
		special:   spec,
		trained:   true,
	}, nil
}

// sortedTokens mengembalikan token dalam urutan leksikografis (untuk uji
// determinisme dan kompatibilitas fixture).
func (t *UnigramTokenizer) sortedTokens() []string {
	tokens := make([]string, 0, len(t.vocab))
	for token := range t.vocab {
		tokens = append(tokens, token)
	}
	sort.Strings(tokens)
	return tokens
}

// Train melatih tokenizer Unigram pada kumpulan teks. Membentuk kosakata
// potongan dengan probabilitas log berdasarkan frekuensi. Panggilan berikutnya
// menggantikan hasil pelatihan sebelumnya.
func (t *UnigramTokenizer) Train(texts []string, vocabSize int) error {
	if t == nil {
		return fmt.Errorf("%w: nil tokenizer", errs.ErrInvalidState)
	}
	if vocabSize <= len(t.special) {
		return fmt.Errorf("%w: vocab size %d must exceed special token count %d", errs.ErrInvalidArgument, vocabSize, len(t.special))
	}

	t.vocab = appendMap(nil, t.special)
	t.idToToken = make(map[int64]string, vocabSize)
	t.logProb = make(map[int64]float64, vocabSize)
	for name, id := range t.special {
		t.idToToken[id] = name
		t.logProb[id] = 0.0
	}
	nextID := int64(len(t.special))

	// Kumpulkan frekuensi kata.
	wordFreq := make(map[string]int)
	totalWords := 0
	for _, text := range texts {
		for _, word := range strings.Fields(text) {
			wordFreq[word]++
			totalWords++
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
			t.logProb[id] = math.Log(float64(wordFreq[unit]) / float64(totalWords))
		}
	}
	// Tambahkan kata-kata lengkap sebagai potongan tambahan, diurutkan untuk determinisme.
	words := make([]string, 0, len(wordFreq))
	for word := range wordFreq {
		words = append(words, word)
	}
	sort.Strings(words)
	for _, word := range words {
		if _, ok := t.vocab[word]; !ok && len(t.vocab) < vocabSize {
			id := nextID
			nextID++
			t.vocab[word] = id
			t.idToToken[id] = word
			t.logProb[id] = math.Log(float64(wordFreq[word]) / float64(totalWords))
		}
	}
	// Spasi sebagai pemisah antar kata.
	if _, ok := t.vocab[" "]; !ok {
		id := nextID
		nextID++
		t.vocab[" "] = id
		t.idToToken[id] = " "
		t.logProb[id] = math.Log(0.5)
	}

	// Tambahkan potongan gabungan paling sering hingga ukuran kosakata tercapai.
	for len(t.vocab) < vocabSize {
		pairFreq := make(map[[2]string]int)
		for word, freq := range wordFreq {
			units := t.wordUnits(word)
			for i := 0; i+1 < len(units); i++ {
				pair := [2]string{units[i], units[i+1]}
				pairFreq[pair] += freq
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
		t.logProb[id] = math.Log(float64(bestFreq) / float64(totalWords))
	}

	t.trained = true
	return nil
}

// wordUnits mengembalikan unit dasar (karakter atau byte) dari suatu kata.
func (t *UnigramTokenizer) wordUnits(word string) []string {
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

// segmentWord memotong kata menggunakan DP untuk memaksimalkan log prob.
func (t *UnigramTokenizer) segmentWord(word string, unkID int64, hasUNK bool) []string {
	if len(word) == 0 {
		return nil
	}

	n := len(word)
	// dp[i] = skor terbaik untuk awalan panjang i.
	dp := make([]float64, n+1)
	for i := 1; i <= n; i++ {
		dp[i] = math.Inf(-1)
	}
	parent := make([]int, n+1)
	pieceAt := make([]string, n+1)

	dp[0] = 0

	for i := 0; i < n; i++ {
		if dp[i] == math.Inf(-1) {
			continue
		}
		// Coba semua potongan yang dimulai di posisi i. Kosakata menyimpan
		// token tanpa awalan "_"; awalan ditambahkan saat penyimpanan encoding.
		for j := i + 1; j <= n; j++ {
			raw := word[i:j]
			piece := unigramWordStart + raw
			id, ok := t.vocab[raw]
			if !ok {
				continue
			}
			score := dp[i] + t.logProb[id]
			if score > dp[j] {
				dp[j] = score
				parent[j] = i
				pieceAt[j] = piece
			}
		}
	}

	if dp[n] == math.Inf(-1) {
		if hasUNK {
			return []string{UNKName}
		}
		return []string{word}
	}

	// Rekonstruksi potongan dari belakang.
	var result []string
	pos := n
	for pos > 0 {
		result = append(result, pieceAt[pos])
		pos = parent[pos]
	}
	// Balik urutan.
	for i, j := 0, len(result)-1; i < j; i, j = i+1, j-1 {
		result[i], result[j] = result[j], result[i]
	}
	return result
}

package data

import (
	"fmt"
	"sort"
	"strings"

	"github.com/adhitiad/gotorch/src/errs"
)

// Train melatih tokenizer pada kumpulan teks. Membentuk kosakata dan aturan
// penggabungan secara deterministik. Panggilan berikutnya menggantikan hasil
// pelatihan sebelumnya.
func (t *BPETokenizer) Train(texts []string, vocabSize int) error {
	if t == nil {
		return fmt.Errorf("%w: nil tokenizer", errs.ErrInvalidState)
	}
	if vocabSize <= len(t.special) {
		return fmt.Errorf("%w: vocab size %d must exceed special token count %d", errs.ErrInvalidArgument, vocabSize, len(t.special))
	}

	// Inisialisasi kosakata dengan special token.
	t.vocab = appendMap(nil, t.special)
	t.idToToken = make(map[int64]string, vocabSize)
	for name, id := range t.special {
		t.idToToken[id] = name
	}
	nextID := int64(len(t.special))

	// Kumpulkan unit dasar dan kata-kata dari semua teks.
	type wordUnits struct {
		units []int64 // unit dasar sebagai ID
		raw   []string
	}
	var words []wordUnits

	for _, text := range texts {
		for _, word := range strings.Fields(text) {
			var units []int64
			var raw []string
			if t.byteLevel {
				for _, b := range []byte(word) {
					s := string([]byte{b})
					units = append(units, t.unitID(s, &nextID))
					raw = append(raw, s)
				}
			} else {
				for _, r := range word {
					s := string(r)
					units = append(units, t.unitID(s, &nextID))
					raw = append(raw, s)
				}
			}
			if len(units) > 0 {
				words = append(words, wordUnits{units: units, raw: raw})
			}
		}
		// Spasi adalah unit dasar yang memisahkan kata-kata.
		if t.byteLevel || !t.byteLevel {
			s := " "
			t.unitID(s, &nextID)
		}
	}

	// Jika tidak ada kata, hanya special token yang tersedia.
	if len(words) == 0 {
		t.merges = nil
		t.mergeIdx = nil
		t.trained = true
		return nil
	}

	// Latih aturan penggabungan hingga ukuran kosakata tercapai.
	merges := []mergeRule{}
	mergeIdx := map[[2]int64]int{}
	currentWords := make([][]int64, len(words))
	for i, w := range words {
		currentWords[i] = append([]int64(nil), w.units...)
	}

	rank := 0
	for len(merges) < vocabSize-len(t.special) {
		pairCount := countPairs(currentWords)
		if len(pairCount) == 0 {
			break
		}
		// Pilih pasangan dengan frekuensi tertinggi; tali tambahan menggunakan
		// urutan leksikografis gabungan token untuk determinisme.
		var bestLeft, bestRight int64
		var bestCount int
		var bestKey string
		found := false
		for pair, count := range pairCount {
			key := t.pairKey(pair[0], pair[1])
			if !found || count > bestCount || (count == bestCount && key < bestKey) {
				bestLeft, bestRight = pair[0], pair[1]
				bestCount = count
				bestKey = key
				found = true
			}
		}
		if !found {
			break
		}

		newID := nextID
		nextID++
		leftToken := t.idToToken[bestLeft]
		rightToken := t.idToToken[bestRight]
		mergedToken := leftToken + rightToken
		t.vocab[mergedToken] = newID
		t.idToToken[newID] = mergedToken

		rule := mergeRule{left: bestLeft, right: bestRight, id: newID, rank: rank}
		merges = append(merges, rule)
		mergeIdx[[2]int64{bestLeft, bestRight}] = rank
		rank++

		// Terapkan penggabungan ke semua kata.
		for i := range currentWords {
			currentWords[i] = applyMerge(currentWords[i], bestLeft, bestRight, newID)
		}
	}

	sort.Slice(merges, func(i, j int) bool { return merges[i].rank < merges[j].rank })
	t.merges = merges
	t.mergeIdx = mergeIdx
	t.trained = true
	return nil
}

// unitID mengembalikan ID untuk unit dasar, membuat entri baru jika diperlukan.
func (t *BPETokenizer) unitID(s string, nextID *int64) int64 {
	if id, ok := t.vocab[s]; ok {
		return id
	}
	id := *nextID
	*nextID++
	t.vocab[s] = id
	t.idToToken[id] = s
	return id
}

// pairKey mengembalikan kunci leksikografis untuk pasangan ID.
func (t *BPETokenizer) pairKey(left, right int64) string {
	return t.idToToken[left] + "\x00" + t.idToToken[right]
}

// countPairs menghitung frekuensi pasangan berseberangan di semua kata.
func countPairs(words [][]int64) map[[2]int64]int {
	counts := map[[2]int64]int{}
	for _, w := range words {
		for i := 0; i+1 < len(w); i++ {
			pair := [2]int64{w[i], w[i+1]}
			counts[pair]++
		}
	}
	return counts
}

// applyMerge mengganti semua kemunculan pasangan (kiri, kanan) dengan id baru
// dalam satu urutan token, secara greedy dari kiri ke kanan.
func applyMerge(seq []int64, left, right, newID int64) []int64 {
	if len(seq) < 2 {
		return seq
	}
	result := make([]int64, 0, len(seq))
	for i := 0; i < len(seq); i++ {
		if i+1 < len(seq) && seq[i] == left && seq[i+1] == right {
			result = append(result, newID)
			i++ // lewati kanan karena sudah digabung
		} else {
			result = append(result, seq[i])
		}
	}
	return result
}

// wordToUnitIDs mengubah kata menjadi urutan ID unit dasar (karakter atau byte).
func (t *BPETokenizer) wordToUnitIDs(word string) []int64 {
	units := make([]int64, 0, len(word))
	if t.byteLevel {
		for _, b := range []byte(word) {
			s := string([]byte{b})
			if id, ok := t.vocab[s]; ok {
				units = append(units, id)
			} else if unkID, ok := t.special[UNKName]; ok {
				units = append(units, unkID)
			}
		}
	} else {
		for _, r := range word {
			s := string(r)
			if id, ok := t.vocab[s]; ok {
				units = append(units, id)
			} else if unkID, ok := t.special[UNKName]; ok {
				units = append(units, unkID)
			}
		}
	}
	return units
}

// applyMerges menerapkan semua aturan penggabungan dalam urutan peringkat pada
// urutan token, secara greedily hingga tidak ada penggabungan lagi.
func (t *BPETokenizer) applyMerges(ids []int64) []int64 {
	if len(ids) < 2 || len(t.merges) == 0 {
		return ids
	}
	current := append([]int64(nil), ids...)
	for _, rule := range t.merges {
		current = applyMerge(current, rule.left, rule.right, rule.id)
	}
	return current
}

// NewByteLevelBPETokenizer membentuk tokenizer BPE tingkat byte dengan ukuran
// kosakata dan special token yang diberikan.
func NewByteLevelBPETokenizer(vocabSize int, special map[string]int64) *BPETokenizer {
	return NewBPETokenizer(vocabSize, true, special)
}

// NewCharLevelBPETokenizer membentuk tokenizer BPE tingkat karakter dengan ukuran
// kosakata dan special token yang diberikan.
func NewCharLevelBPETokenizer(vocabSize int, special map[string]int64) *BPETokenizer {
	return NewBPETokenizer(vocabSize, false, special)
}

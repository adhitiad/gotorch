// Package data menyediakan parser aset dan konfigurasi tokenizer untuk P6.
// Parser membaca format aset tokenizer umum (Hugging Face kompatibel) langsung
// di Go, termasuk vocabulary, aturan merge BPE, added tokens, dan special-token
// map, tanpa dependensi pada pustaka eksternal.
package data

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"

	"github.com/adhitiad/gotorch/src/errs"
)

// Nama file aset tokenizer yang dikenal. Semua opsional; parser akan melewati
// file yang tidak ada dan hanya memuat yang tersedia.
const (
	// VocabJSON adalah berkas kosakata dalam format JSON (token string -> ID).
	VocabJSON = "vocab.json"
	// VocabTXT adalah berkas kosakata dalam format teks (satu token per baris;
	// ID = nomor baris dimulai dari 0).
	VocabTXT = "vocab.txt"
	// MergesTXT adalah berkas aturan merge BPE (pasangan token per baris).
	MergesTXT = "merges.txt"
	// AddedTokensJSON adalah berkas token tambahan (daftar objek dengan id dan
	// content, atau peta token -> ID).
	AddedTokensJSON = "added_tokens.json"
	// SpecialTokensMapJSON adalah berkas pemetaan nama special token ke string
	// token.
	SpecialTokensMapJSON = "special_tokens_map.json"
	// TokenizerConfigJSON adalah berkas konfigurasi lengkap tokenizer yang
	// dapat mencakup special_tokens_map di dalamnya.
	TokenizerConfigJSON = "tokenizer_config.json"
)

// TokenizerConfig adalah hasil pemuaian aset tokenizer. Semua bidang opsional;
// bidang yang nil berarti aset tersebut tidak tersedia. Konfigurasi immutable
// setelah dibangun dan aman dipakai bersamaan.
type TokenizerConfig struct {
	// Vocab memetakan token string ke ID. Nil jika tidak ada berkas kosakata.
	Vocab map[string]int64
	// IDToVocab adalah invers dari Vocab (ID -> token string). Dibangun otomatis.
	IDToVocab map[int64]string
	// Merges adalah daftar aturan merge BPE diurutkan menurut peringkat. Nil jika
	// tidak ada berkas merges.
	Merges []MergeRule
	// MergeIndex memetakan pasangan (kiri, kanan) ke indeks dalam Merges.
	MergeIndex map[[2]int64]int
	// AddedTokens adalah daftar token tambahan. Nil jika tidak ada berkas added
	// tokens.
	AddedTokens []AddedToken
	// SpecialTokens memetakan nama special token (mis. "bos_token") ke string
	// token (mis. "[BOS]"). Nil jika tidak ada berkas peta special token.
	SpecialTokens map[string]string
	// Source adalah direktori tempat aset dimuat. Kosong jika konfigurasi
	// dibangun dari berkas individu.
	Source string
}

// MergeRule adalah satu aturan penggabungan BPE: pasangan token (kiri, kanan)
// yang digabungkan menjadi token baru pada peringkat tertentu.
type MergeRule struct {
	Left  string // token kiri
	Right string // token kanan
	Rank  int    // peringkat penggabungan (semakin kecil, semakin awal dipakai)
	ID    int64  // ID token hasil penggabungan (jika ada di vocab)
}

// AddedToken adalah satu token tambahan beserta metadata-nya.
type AddedToken struct {
	ID         int64  `json:"id"`
	Content    string `json:"content"`
	SingleWord bool   `json:"single_word"`
	Lstrip     bool   `json:"lstrip"`
	Rstrip     bool   `json:"rstrip"`
	Normalized bool   `json:"normalized"`
}

// NewTokenizerConfig membentuk konfigurasi kosong. Semua bidang nil.
func NewTokenizerConfig() *TokenizerConfig {
	return &TokenizerConfig{}
}

// Len mengembalikan jumlah entri vocabulary. Konfigurasi nil memiliki panjang nol.
func (c *TokenizerConfig) Len() int {
	if c == nil || c.Vocab == nil {
		return 0
	}
	return len(c.Vocab)
}

// VocabSize mengembalikan ukuran vocabulary (sama dengan Len). Konfigurasi nil
// mengembalikan nol.
func (c *TokenizerConfig) VocabSize() int64 {
	if c == nil || c.Vocab == nil {
		return 0
	}
	return int64(len(c.Vocab))
}

// HasMerges melaporkan apakah konfigurasi memiliki aturan merge BPE.
func (c *TokenizerConfig) HasMerges() bool {
	return c != nil && len(c.Merges) > 0
}

// HasAddedTokens melaporkan apakah konfigurasi memiliki token tambahan.
func (c *TokenizerConfig) HasAddedTokens() bool {
	return c != nil && len(c.AddedTokens) > 0
}

// HasSpecialTokens melaporkan apakah konfigurasi memiliki peta special token.
func (c *TokenizerConfig) HasSpecialTokens() bool {
	return c != nil && len(c.SpecialTokens) > 0
}

// FromDirectory memuat semua aset tokenizer yang tersedia dari direktori.
// Berkas yang tidak ada akan diabaikan; setidaknya satu berkas kosakata
// (vocab.json atau vocab.txt) harus tersedia, jika tidak konfigurasi akan kosong
// tetapi tidak error. Error hanya terjadi pada kesalahan pembacaan atau parsing
// berkas yang ada.
func FromDirectory(dir string) (*TokenizerConfig, error) {
	if dir == "" {
		return nil, fmt.Errorf("%w: empty directory", errs.ErrInvalidArgument)
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("%w: cannot read directory %q: %w", errs.ErrInvalidArgument, dir, err)
	}

	cfg := NewTokenizerConfig()
	cfg.Source = dir

	hasVocab := false
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		switch name {
		case VocabJSON:
			if err := cfg.loadVocabJSON(filepath.Join(dir, name)); err != nil {
				return nil, err
			}
			hasVocab = true
		case VocabTXT:
			if err := cfg.loadVocabTXT(filepath.Join(dir, name)); err != nil {
				return nil, err
			}
			hasVocab = true
		case MergesTXT:
			if err := cfg.loadMergesTXT(filepath.Join(dir, name)); err != nil {
				return nil, err
			}
		case AddedTokensJSON:
			if err := cfg.loadAddedTokensJSON(filepath.Join(dir, name)); err != nil {
				return nil, err
			}
		case SpecialTokensMapJSON:
			if err := cfg.loadSpecialTokensMapJSON(filepath.Join(dir, name)); err != nil {
				return nil, err
			}
		case TokenizerConfigJSON:
			if err := cfg.loadTokenizerConfigJSON(filepath.Join(dir, name)); err != nil {
				return nil, err
			}
		}
	}

	// Jika tidak ada berkas kosakata, konfigurasi tetap valid tetapi kosong.
	if !hasVocab {
		return cfg, nil
	}

	// Bangun invers ID -> token setelah semua vocab dimuat.
	cfg.buildInverse()
	return cfg, nil
}

// FromFiles membentuk konfigurasi dari berkas individu yang diberikan. Setiap
// path kosong akan diabaikan. Minimal satu berkas kosakata harus diberikan.
func FromFiles(vocabJSON, vocabTXT, mergesTXT, addedTokensJSON, specialTokensMapJSON string) (*TokenizerConfig, error) {
	cfg := NewTokenizerConfig()
	hasVocab := false

	if vocabJSON != "" {
		if err := cfg.loadVocabJSON(vocabJSON); err != nil {
			return nil, err
		}
		hasVocab = true
	}
	if vocabTXT != "" {
		if err := cfg.loadVocabTXT(vocabTXT); err != nil {
			return nil, err
		}
		hasVocab = true
	}
	if mergesTXT != "" {
		if err := cfg.loadMergesTXT(mergesTXT); err != nil {
			return nil, err
		}
	}
	if addedTokensJSON != "" {
		if err := cfg.loadAddedTokensJSON(addedTokensJSON); err != nil {
			return nil, err
		}
	}
	if specialTokensMapJSON != "" {
		if err := cfg.loadSpecialTokensMapJSON(specialTokensMapJSON); err != nil {
			return nil, err
		}
	}

	if !hasVocab {
		return nil, fmt.Errorf("%w: no vocabulary file provided", errs.ErrInvalidArgument)
	}

	cfg.buildInverse()
	return cfg, nil
}

// buildInverse membangun peta ID -> token dari Vocab. Dipanggil setelah semua
// berkas kosakata dimuat.
func (c *TokenizerConfig) buildInverse() {
	if c.Vocab == nil {
		return
	}
	c.IDToVocab = make(map[int64]string, len(c.Vocab))
	for token, id := range c.Vocab {
		c.IDToVocab[id] = token
	}
}

// buildMergeIndex membangun indeks pasangan (kiri, kanan) ke indeks dalam Merges.
func (c *TokenizerConfig) buildMergeIndex() {
	if len(c.Merges) == 0 {
		return
	}
	c.MergeIndex = make(map[[2]int64]int, len(c.Merges))
	for i, rule := range c.Merges {
		var leftID, rightID int64 = -1, -1
		if c.Vocab != nil {
			if id, ok := c.Vocab[rule.Left]; ok {
				leftID = id
			}
			if id, ok := c.Vocab[rule.Right]; ok {
				rightID = id
			}
		}
		c.MergeIndex[[2]int64{leftID, rightID}] = i
	}
}

// SpecialTokenID mengembalikan ID untuk special token dengan nama yang diberikan,
// atau -1 jika tidak ditemukan.
func (c *TokenizerConfig) SpecialTokenID(name string) int64 {
	if c == nil || c.SpecialTokens == nil || c.Vocab == nil {
		return -1
	}
	token, ok := c.SpecialTokens[name]
	if !ok {
		return -1
	}
	id, ok := c.Vocab[token]
	if !ok {
		return -1
	}
	return id
}

// TokenForID mengembalikan string token untuk ID yang diberikan, atau string
// kosong jika tidak ditemukan.
func (c *TokenizerConfig) TokenForID(id int64) string {
	if c == nil || c.IDToVocab == nil {
		return ""
	}
	token, ok := c.IDToVocab[id]
	if !ok {
		return ""
	}
	return token
}

// IDForToken mengembalikan ID untuk token string yang diberikan, atau -1 jika
// tidak ditemukan.
func (c *TokenizerConfig) IDForToken(token string) int64 {
	if c == nil || c.Vocab == nil {
		return -1
	}
	id, ok := c.Vocab[token]
	if !ok {
		return -1
	}
	return id
}

// SortedVocab mengembalikan daftar token dalam urutan leksikografis. Berguna
// untuk uji determinisme dan kompatibilitas fixture.
func (c *TokenizerConfig) SortedVocab() []string {
	if c == nil || c.Vocab == nil {
		return nil
	}
	tokens := make([]string, 0, len(c.Vocab))
	for token := range c.Vocab {
		tokens = append(tokens, token)
	}
	sort.Strings(tokens)
	return tokens
}

// Validate memeriksa konsistensi konfigurasi: semua ID dalam Merges harus
// merujuk ke token yang ada di Vocab, dan semua special token harus ada di
// Vocab. Mengembalikan error jika konfigurasi tidak konsisten.
func (c *TokenizerConfig) Validate(ctx context.Context) error {
	if err := checkContext(ctx); err != nil {
		return err
	}
	if c == nil {
		return fmt.Errorf("%w: nil tokenizer config", errs.ErrInvalidState)
	}

	// Validasi special token.
	for name, token := range c.SpecialTokens {
		if _, ok := c.Vocab[token]; !ok {
			return fmt.Errorf("%w: special token %q (%q) not in vocabulary", errs.ErrInvalidArgument, name, token)
		}
	}

	// Validasi added tokens.
	for _, tok := range c.AddedTokens {
		if _, ok := c.Vocab[tok.Content]; !ok {
			return fmt.Errorf("%w: added token %q not in vocabulary", errs.ErrInvalidArgument, tok.Content)
		}
	}

	// Validasi merges: ID hasil penggabungan harus ada di vocab.
	for _, rule := range c.Merges {
		if rule.ID >= 0 {
			if _, ok := c.IDToVocab[rule.ID]; !ok {
				return fmt.Errorf("%w: merge rule %q+%q -> ID %d not in vocabulary", errs.ErrInvalidArgument, rule.Left, rule.Right, rule.ID)
			}
		}
	}
	return nil
}

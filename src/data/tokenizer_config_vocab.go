package data

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/adhitiad/gotorch/src/errs"
)

// loadVocabJSON memuat berkas kosakata dalam format JSON. Format yang didukung:
// peta sederhana {"token": id, ...} atau objek dengan kunci "vocab" yang berisi
// peta token -> ID.
func (c *TokenizerConfig) loadVocabJSON(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("%w: cannot read %q: %w", errs.ErrInvalidArgument, path, err)
	}

	var raw map[string]int64
	if err := json.Unmarshal(data, &raw); err != nil {
		// Coba format dengan kunci "vocab".
		var wrapper struct {
			Vocab map[string]int64 `json:"vocab"`
		}
		if err2 := json.Unmarshal(data, &wrapper); err2 != nil {
			return fmt.Errorf("%w: cannot parse vocab JSON %q: %w", errs.ErrInvalidArgument, path, err)
		}
		raw = wrapper.Vocab
	}

	if c.Vocab == nil {
		c.Vocab = make(map[string]int64, len(raw))
	}
	for token, id := range raw {
		if token == "" {
			return fmt.Errorf("%w: empty token in vocab %q", errs.ErrInvalidArgument, path)
		}
		if id < 0 {
			return fmt.Errorf("%w: negative ID %d for token %q in %q", errs.ErrInvalidArgument, id, token, path)
		}
		c.Vocab[token] = id
	}
	return nil
}

// loadVocabTXT memuat berkas kosakata dalam format teks. Satu token per baris;
// ID ditetapkan dari nomor baris dimulai dari 0. Baris kosong dan baris yang
// dimulai dengan '#' diabaikan.
func (c *TokenizerConfig) loadVocabTXT(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("%w: cannot read %q: %w", errs.ErrInvalidArgument, path, err)
	}

	lines := strings.Split(string(data), "\n")
	if c.Vocab == nil {
		c.Vocab = make(map[string]int64, len(lines))
	}
	id := int64(0)
	for _, line := range lines {
		line = strings.TrimRight(line, "\r")
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		token := line
		if c.Vocab != nil {
			if _, exists := c.Vocab[token]; exists {
				// ID sudah ditetapkan, lewati duplikat.
				continue
			}
		}
		c.Vocab[token] = id
		id++
	}
	return nil
}

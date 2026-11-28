package data

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/adhitiad/gotorch/src/errs"
)

// loadAddedTokensJSON memuat berkas token tambahan. Format yang didukung:
// daftar objek [{"id": 1, "content": "<token>", ...}, ...] atau peta
// {"<token>": id, ...}.
func (c *TokenizerConfig) loadAddedTokensJSON(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("%w: cannot read %q: %w", errs.ErrInvalidArgument, path, err)
	}

	// Coba format daftar objek terlebih dahulu.
	var list []AddedToken
	if err := json.Unmarshal(data, &list); err == nil {
		c.AddedTokens = append(c.AddedTokens, list...)
		return nil
	}

	// Coba format peta token -> ID.
	var raw map[string]int64
	if err2 := json.Unmarshal(data, &raw); err2 != nil {
		return fmt.Errorf("%w: cannot parse added tokens JSON %q: %w", errs.ErrInvalidArgument, path, err)
	}
	for content, id := range raw {
		if content == "" {
			return fmt.Errorf("%w: empty token in added tokens %q", errs.ErrInvalidArgument, path)
		}
		if id < 0 {
			return fmt.Errorf("%w: negative ID %d for token %q in %q", errs.ErrInvalidArgument, id, content, path)
		}
		c.AddedTokens = append(c.AddedTokens, AddedToken{ID: id, Content: content})
	}
	return nil
}

// loadSpecialTokensMapJSON memuat berkas peta special token. Format: peta nama
// special token -> string token, mis. {"bos_token": "[BOS]", "eos_token": "[EOS]"}.
func (c *TokenizerConfig) loadSpecialTokensMapJSON(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("%w: cannot read %q: %w", errs.ErrInvalidArgument, path, err)
	}

	var raw map[string]string
	if err := json.Unmarshal(data, &raw); err != nil {
		return fmt.Errorf("%w: cannot parse special tokens map JSON %q: %w", errs.ErrInvalidArgument, path, err)
	}

	if c.SpecialTokens == nil {
		c.SpecialTokens = make(map[string]string, len(raw))
	}
	for name, token := range raw {
		if name == "" {
			return fmt.Errorf("%w: empty special token name in %q", errs.ErrInvalidArgument, path)
		}
		if token == "" {
			return fmt.Errorf("%w: empty special token value for %q in %q", errs.ErrInvalidArgument, name, path)
		}
		c.SpecialTokens[name] = token
	}
	return nil
}

// loadTokenizerConfigJSON memuat berkas konfigurasi lengkap tokenizer. Dari sini
// diekstrak special_tokens_map jika tersedia. Bidang lainnya diabaikan untuk
// saat ini (boleh diperluas nanti).
func (c *TokenizerConfig) loadTokenizerConfigJSON(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("%w: cannot read %q: %w", errs.ErrInvalidArgument, path, err)
	}

	var raw struct {
		SpecialTokensMap map[string]string `json:"special_tokens_map"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return fmt.Errorf("%w: cannot parse tokenizer config JSON %q: %w", errs.ErrInvalidArgument, path, err)
	}

	if len(raw.SpecialTokensMap) > 0 {
		if c.SpecialTokens == nil {
			c.SpecialTokens = make(map[string]string, len(raw.SpecialTokensMap))
		}
		for name, token := range raw.SpecialTokensMap {
			c.SpecialTokens[name] = token
		}
	}
	return nil
}

package data

import (
	"errors"
	"testing"

	"github.com/adhitiad/gotorch/src/errs"
)

func TestTokenizerConfigVocabJSONValidation(t *testing.T) {
	dir := t.TempDir()

	// Empty token.
	path := writeFile(t, dir, "bad.json", `{"":0,"a":1}`)
	cfg := NewTokenizerConfig()
	if err := cfg.loadVocabJSON(path); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("expected ErrInvalidArgument for empty token, got %v", err)
	}

	// Negative ID.
	path = writeFile(t, dir, "bad2.json", `{"a":-1}`)
	cfg = NewTokenizerConfig()
	if err := cfg.loadVocabJSON(path); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("expected ErrInvalidArgument for negative ID, got %v", err)
	}

	// Invalid JSON.
	path = writeFile(t, dir, "bad3.json", `not json`)
	cfg = NewTokenizerConfig()
	if err := cfg.loadVocabJSON(path); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("expected ErrInvalidArgument for invalid JSON, got %v", err)
	}

	// Wrapper format with "vocab" key.
	path = writeFile(t, dir, "wrapper.json", `{"vocab":{"x":5,"y":6}}`)
	cfg = NewTokenizerConfig()
	if err := cfg.loadVocabJSON(path); err != nil {
		t.Fatalf("loadVocabJSON(wrapper): %v", err)
	}
	if cfg.VocabSize() != 2 || cfg.IDForToken("x") != 5 || cfg.IDForToken("y") != 6 {
		t.Fatalf("wrapper vocab not parsed correctly: %v", cfg.Vocab)
	}
}

func TestTokenizerConfigVocabTXT(t *testing.T) {
	dir := t.TempDir()
	path := writeFile(t, dir, "vocab.txt", "# comment\n\nhello\nworld\n\nfoo\r\n")
	cfg := NewTokenizerConfig()
	if err := cfg.loadVocabTXT(path); err != nil {
		t.Fatalf("loadVocabTXT: %v", err)
	}
	if cfg.VocabSize() != 3 {
		t.Fatalf("VocabSize = %d, want 3", cfg.VocabSize())
	}
	if cfg.IDForToken("hello") != 0 || cfg.IDForToken("world") != 1 || cfg.IDForToken("foo") != 2 {
		t.Fatalf("vocab TXT IDs = %v", cfg.Vocab)
	}
}

func TestTokenizerConfigMergesTXT(t *testing.T) {
	dir := t.TempDir()
	path := writeFile(t, dir, "merges.txt", "#version: 0.0.2\n\na b\nb c\n# comment\nc d\n")
	cfg := NewTokenizerConfig()
	cfg.Vocab = map[string]int64{"a": 0, "b": 1, "c": 2, "d": 3, "ab": 4, "bc": 5, "cd": 6}
	if err := cfg.loadMergesTXT(path); err != nil {
		t.Fatalf("loadMergesTXT: %v", err)
	}
	if len(cfg.Merges) != 3 {
		t.Fatalf("len(Merges) = %d, want 3", len(cfg.Merges))
	}
	if cfg.Merges[0] != (MergeRule{Left: "a", Right: "b", Rank: 0, ID: 4}) {
		t.Fatalf("merge[0] = %+v", cfg.Merges[0])
	}
	if cfg.Merges[1] != (MergeRule{Left: "b", Right: "c", Rank: 1, ID: 5}) {
		t.Fatalf("merge[1] = %+v", cfg.Merges[1])
	}
	if cfg.Merges[2] != (MergeRule{Left: "c", Right: "d", Rank: 2, ID: 6}) {
		t.Fatalf("merge[2] = %+v", cfg.Merges[2])
	}
	if idx := cfg.MergeIndex[[2]int64{0, 1}]; idx != 0 {
		t.Fatalf("MergeIndex[a,b] = %d, want 0", idx)
	}

	// Invalid merge rule (single token).
	path2 := writeFile(t, dir, "bad.txt", "onlyone\n")
	cfg2 := NewTokenizerConfig()
	if err := cfg2.loadMergesTXT(path2); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("expected ErrInvalidArgument for single-token merge, got %v", err)
	}
}

func TestTokenizerConfigAddedTokensJSON(t *testing.T) {
	dir := t.TempDir()

	// List format.
	path := writeFile(t, dir, "added.json", `[{"id":10,"content":"<extra>","single_word":false},{"id":11,"content":"<mask>"}]`)
	cfg := NewTokenizerConfig()
	if err := cfg.loadAddedTokensJSON(path); err != nil {
		t.Fatalf("loadAddedTokensJSON(list): %v", err)
	}
	if len(cfg.AddedTokens) != 2 {
		t.Fatalf("len(AddedTokens) = %d, want 2", len(cfg.AddedTokens))
	}
	if cfg.AddedTokens[0].ID != 10 || cfg.AddedTokens[0].Content != "<extra>" {
		t.Fatalf("added[0] = %+v", cfg.AddedTokens[0])
	}

	// Map format.
	path2 := writeFile(t, dir, "added2.json", `{"<custom>":20,"<another>":21}`)
	cfg2 := NewTokenizerConfig()
	if err := cfg2.loadAddedTokensJSON(path2); err != nil {
		t.Fatalf("loadAddedTokensJSON(map): %v", err)
	}
	if len(cfg2.AddedTokens) != 2 {
		t.Fatalf("len(AddedTokens) = %d, want 2", len(cfg2.AddedTokens))
	}

	// Empty token.
	path3 := writeFile(t, dir, "bad.json", `{"":5}`)
	cfg3 := NewTokenizerConfig()
	if err := cfg3.loadAddedTokensJSON(path3); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("expected ErrInvalidArgument for empty token, got %v", err)
	}

	// Negative ID.
	path4 := writeFile(t, dir, "bad2.json", `{"a":-1}`)
	cfg4 := NewTokenizerConfig()
	if err := cfg4.loadAddedTokensJSON(path4); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("expected ErrInvalidArgument for negative ID, got %v", err)
	}

	// Invalid JSON.
	path5 := writeFile(t, dir, "bad3.json", `not json`)
	cfg5 := NewTokenizerConfig()
	if err := cfg5.loadAddedTokensJSON(path5); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("expected ErrInvalidArgument for invalid JSON, got %v", err)
	}
}

func TestTokenizerConfigSpecialTokensMapJSON(t *testing.T) {
	dir := t.TempDir()
	path := writeFile(t, dir, "special.json", `{"bos_token":"[BOS]","eos_token":"[EOS]","unk_token":"[UNK]"}`)
	cfg := NewTokenizerConfig()
	if err := cfg.loadSpecialTokensMapJSON(path); err != nil {
		t.Fatalf("loadSpecialTokensMapJSON: %v", err)
	}
	if len(cfg.SpecialTokens) != 3 {
		t.Fatalf("len(SpecialTokens) = %d, want 3", len(cfg.SpecialTokens))
	}
	if cfg.SpecialTokens["bos_token"] != "[BOS]" || cfg.SpecialTokens["eos_token"] != "[EOS]" {
		t.Fatalf("special tokens = %v", cfg.SpecialTokens)
	}

	// Empty name.
	path2 := writeFile(t, dir, "bad.json", `{"":"[BOS]"}`)
	cfg2 := NewTokenizerConfig()
	if err := cfg2.loadSpecialTokensMapJSON(path2); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("expected ErrInvalidArgument for empty name, got %v", err)
	}

	// Empty value.
	path3 := writeFile(t, dir, "bad2.json", `{"bos_token":""}`)
	cfg3 := NewTokenizerConfig()
	if err := cfg3.loadSpecialTokensMapJSON(path3); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("expected ErrInvalidArgument for empty value, got %v", err)
	}
}

func TestTokenizerConfigTokenizerConfigJSON(t *testing.T) {
	dir := t.TempDir()
	path := writeFile(t, dir, "tokenizer_config.json", `{"special_tokens_map":{"bos_token":"[BOS]","eos_token":"[EOS]"},"model_max_length":512}`)
	cfg := NewTokenizerConfig()
	if err := cfg.loadTokenizerConfigJSON(path); err != nil {
		t.Fatalf("loadTokenizerConfigJSON: %v", err)
	}
	if len(cfg.SpecialTokens) != 2 {
		t.Fatalf("len(SpecialTokens) = %d, want 2", len(cfg.SpecialTokens))
	}
	if cfg.SpecialTokens["bos_token"] != "[BOS]" || cfg.SpecialTokens["eos_token"] != "[EOS]" {
		t.Fatalf("special tokens = %v", cfg.SpecialTokens)
	}

	// Invalid JSON.
	path2 := writeFile(t, dir, "bad.json", `not json`)
	cfg2 := NewTokenizerConfig()
	if err := cfg2.loadTokenizerConfigJSON(path2); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("expected ErrInvalidArgument for invalid JSON, got %v", err)
	}
}

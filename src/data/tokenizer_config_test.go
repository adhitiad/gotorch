package data

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/adhitiad/gotorch/src/errs"
)

func writeFile(t *testing.T, dir, name, content string) string {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("writeFile(%q): %v", path, err)
	}
	return path
}

func TestTokenizerConfigLenAndVocabSize(t *testing.T) {
	var nilCfg *TokenizerConfig
	if nilCfg.Len() != 0 || nilCfg.VocabSize() != 0 {
		t.Fatal("nil config should have zero length")
	}

	cfg := NewTokenizerConfig()
	if cfg.Len() != 0 || cfg.VocabSize() != 0 {
		t.Fatalf("empty config should have zero length, got %d/%d", cfg.Len(), cfg.VocabSize())
	}

	cfg.Vocab = map[string]int64{"a": 0, "b": 1, "c": 2}
	if cfg.Len() != 3 || cfg.VocabSize() != 3 {
		t.Fatalf("Len/VocabSize = %d/%d, want 3/3", cfg.Len(), cfg.VocabSize())
	}
}

func TestTokenizerConfigHasMethods(t *testing.T) {
	var nilCfg *TokenizerConfig
	if nilCfg.HasMerges() || nilCfg.HasAddedTokens() || nilCfg.HasSpecialTokens() {
		t.Fatal("nil config should report false for all Has methods")
	}

	cfg := NewTokenizerConfig()
	if cfg.HasMerges() || cfg.HasAddedTokens() || cfg.HasSpecialTokens() {
		t.Fatal("empty config should report false for all Has methods")
	}

	cfg.Merges = []MergeRule{{Left: "a", Right: "b", Rank: 0}}
	if !cfg.HasMerges() {
		t.Fatal("config with merges should report true for HasMerges")
	}
	cfg.AddedTokens = []AddedToken{{ID: 10, Content: "<extra>"}}
	if !cfg.HasAddedTokens() {
		t.Fatal("config with added tokens should report true for HasAddedTokens")
	}
	cfg.SpecialTokens = map[string]string{"bos_token": "[BOS]"}
	if !cfg.HasSpecialTokens() {
		t.Fatal("config with special tokens should report true for HasSpecialTokens")
	}
}

func TestTokenizerConfigFromDirectory(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, VocabJSON, `{"[BOS]":0,"[EOS]":1,"hello":2,"world":3}`)
	writeFile(t, dir, MergesTXT, `#version: 0.0.2
hello world
`)
	writeFile(t, dir, AddedTokensJSON, `[{"id":10,"content":"<extra>","single_word":false}]`)
	writeFile(t, dir, SpecialTokensMapJSON, `{"bos_token":"[BOS]","eos_token":"[EOS]","unk_token":"[UNK]"}`)

	cfg, err := FromDirectory(dir)
	if err != nil {
		t.Fatalf("FromDirectory: %v", err)
	}
	if cfg == nil {
		t.Fatal("FromDirectory returned nil config")
	}
	if cfg.Source != dir {
		t.Fatalf("Source = %q, want %q", cfg.Source, dir)
	}
	if cfg.VocabSize() != 4 {
		t.Fatalf("VocabSize = %d, want 4", cfg.VocabSize())
	}
	if id := cfg.IDForToken("hello"); id != 2 {
		t.Fatalf("IDForToken(hello) = %d, want 2", id)
	}
	if token := cfg.TokenForID(3); token != "world" {
		t.Fatalf("TokenForID(3) = %q, want world", token)
	}
	if !cfg.HasMerges() {
		t.Fatal("expected merges to be loaded")
	}
	if len(cfg.Merges) != 1 {
		t.Fatalf("len(Merges) = %d, want 1", len(cfg.Merges))
	}
	if cfg.Merges[0].Left != "hello" || cfg.Merges[0].Right != "world" {
		t.Fatalf("merge rule = %+v, want hello+world", cfg.Merges[0])
	}
	if !cfg.HasAddedTokens() {
		t.Fatal("expected added tokens to be loaded")
	}
	if len(cfg.AddedTokens) != 1 || cfg.AddedTokens[0].Content != "<extra>" {
		t.Fatalf("added tokens = %+v, want one <extra>", cfg.AddedTokens)
	}
	if !cfg.HasSpecialTokens() {
		t.Fatal("expected special tokens to be loaded")
	}
	if cfg.SpecialTokens["bos_token"] != "[BOS]" {
		t.Fatalf("special bos_token = %q, want [BOS]", cfg.SpecialTokens["bos_token"])
	}
	if id := cfg.SpecialTokenID("bos_token"); id != 0 {
		t.Fatalf("SpecialTokenID(bos_token) = %d, want 0", id)
	}
}

func TestTokenizerConfigFromDirectoryEmpty(t *testing.T) {
	dir := t.TempDir()
	cfg, err := FromDirectory(dir)
	if err != nil {
		t.Fatalf("FromDirectory(empty): %v", err)
	}
	if cfg == nil {
		t.Fatal("FromDirectory(empty) returned nil config")
	}
	if cfg.VocabSize() != 0 {
		t.Fatalf("VocabSize = %d, want 0", cfg.VocabSize())
	}
}

func TestTokenizerConfigFromDirectoryNonexistent(t *testing.T) {
	_, err := FromDirectory(filepath.Join(t.TempDir(), "does-not-exist"))
	if !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("expected ErrInvalidArgument, got %v", err)
	}
}

func TestTokenizerConfigFromFiles(t *testing.T) {
	dir := t.TempDir()
	vocabPath := writeFile(t, dir, "vocab.json", `{"a":0,"b":1,"c":2}`)
	mergesPath := writeFile(t, dir, "merges.txt", "a b\nb c\n")
	specialPath := writeFile(t, dir, "special_tokens_map.json", `{"bos_token":"a"}`)

	cfg, err := FromFiles(vocabPath, "", mergesPath, "", specialPath)
	if err != nil {
		t.Fatalf("FromFiles: %v", err)
	}
	if cfg.VocabSize() != 3 {
		t.Fatalf("VocabSize = %d, want 3", cfg.VocabSize())
	}
	if !cfg.HasMerges() || len(cfg.Merges) != 2 {
		t.Fatalf("len(Merges) = %d, want 2", len(cfg.Merges))
	}
	if !cfg.HasSpecialTokens() || cfg.SpecialTokens["bos_token"] != "a" {
		t.Fatalf("special tokens = %v", cfg.SpecialTokens)
	}

	// No vocabulary file should error.
	if _, err := FromFiles("", "", "", "", ""); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("expected ErrInvalidArgument for no vocab, got %v", err)
	}
}

func TestTokenizerConfigFromFilesEmptyDir(t *testing.T) {
	_, err := FromFiles("", "", "", "", "")
	if !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("expected ErrInvalidArgument, got %v", err)
	}
}

func TestTokenizerConfigValidate(t *testing.T) {
	ctx := context.Background()

	// Valid config.
	cfg := NewTokenizerConfig()
	cfg.Vocab = map[string]int64{"[BOS]": 0, "a": 1, "b": 2}
	cfg.IDToVocab = map[int64]string{0: "[BOS]", 1: "a", 2: "b"}
	cfg.SpecialTokens = map[string]string{"bos_token": "[BOS]"}
	cfg.Merges = []MergeRule{{Left: "a", Right: "b", Rank: 0, ID: 2}}
	if err := cfg.Validate(ctx); err != nil {
		t.Fatalf("Validate(valid): %v", err)
	}

	// Special token not in vocab.
	cfg2 := NewTokenizerConfig()
	cfg2.Vocab = map[string]int64{"a": 0}
	cfg2.IDToVocab = map[int64]string{0: "a"}
	cfg2.SpecialTokens = map[string]string{"bos_token": "[BOS]"}
	if err := cfg2.Validate(ctx); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("expected ErrInvalidArgument for missing special token, got %v", err)
	}

	// Merge rule ID not in vocab.
	cfg3 := NewTokenizerConfig()
	cfg3.Vocab = map[string]int64{"a": 0, "b": 1}
	cfg3.IDToVocab = map[int64]string{0: "a", 1: "b"}
	cfg3.Merges = []MergeRule{{Left: "a", Right: "b", Rank: 0, ID: 99}}
	if err := cfg3.Validate(ctx); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("expected ErrInvalidArgument for invalid merge ID, got %v", err)
	}

	// Nil context.
	if err := cfg.Validate(nil); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("expected ErrInvalidArgument for nil context, got %v", err)
	}

	// Nil config.
	var nilCfg *TokenizerConfig
	if err := nilCfg.Validate(ctx); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("expected ErrInvalidState for nil config, got %v", err)
	}
}

func TestTokenizerConfigSortedVocab(t *testing.T) {
	var nilCfg *TokenizerConfig
	if nilCfg.SortedVocab() != nil {
		t.Fatal("nil config SortedVocab should return nil")
	}

	cfg := NewTokenizerConfig()
	cfg.Vocab = map[string]int64{"b": 1, "a": 0, "c": 2}
	tokens := cfg.SortedVocab()
	want := []string{"a", "b", "c"}
	if !reflect.DeepEqual(tokens, want) {
		t.Fatalf("SortedVocab = %v, want %v", tokens, want)
	}
}

func TestTokenizerConfigSpecialTokenIDAndLookups(t *testing.T) {
	var nilCfg *TokenizerConfig
	if nilCfg.SpecialTokenID("bos_token") != -1 {
		t.Fatal("nil config SpecialTokenID should return -1")
	}
	if nilCfg.TokenForID(0) != "" {
		t.Fatal("nil config TokenForID should return empty")
	}
	if nilCfg.IDForToken("a") != -1 {
		t.Fatal("nil config IDForToken should return -1")
	}

	cfg := NewTokenizerConfig()
	cfg.Vocab = map[string]int64{"[BOS]": 0, "[EOS]": 1, "hello": 2}
	cfg.IDToVocab = map[int64]string{0: "[BOS]", 1: "[EOS]", 2: "hello"}
	cfg.SpecialTokens = map[string]string{"bos_token": "[BOS]", "eos_token": "[EOS]"}

	if id := cfg.SpecialTokenID("bos_token"); id != 0 {
		t.Fatalf("SpecialTokenID(bos_token) = %d, want 0", id)
	}
	if id := cfg.SpecialTokenID("unk_token"); id != -1 {
		t.Fatalf("SpecialTokenID(unk_token) = %d, want -1", id)
	}
	if token := cfg.TokenForID(1); token != "[EOS]" {
		t.Fatalf("TokenForID(1) = %q, want [EOS]", token)
	}
	if token := cfg.TokenForID(99); token != "" {
		t.Fatalf("TokenForID(99) = %q, want empty", token)
	}
	if id := cfg.IDForToken("hello"); id != 2 {
		t.Fatalf("IDForToken(hello) = %d, want 2", id)
	}
	if id := cfg.IDForToken("missing"); id != -1 {
		t.Fatalf("IDForToken(missing) = %d, want -1", id)
	}
}

func TestTokenizerConfigFromDirectoryMergesOnly(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, MergesTXT, "a b\n")
	cfg, err := FromDirectory(dir)
	if err != nil {
		t.Fatalf("FromDirectory(merges only): %v", err)
	}
	if !cfg.HasMerges() || len(cfg.Merges) != 1 {
		t.Fatalf("len(Merges) = %d, want 1", len(cfg.Merges))
	}
	if cfg.VocabSize() != 0 {
		t.Fatalf("VocabSize = %d, want 0", cfg.VocabSize())
	}
}

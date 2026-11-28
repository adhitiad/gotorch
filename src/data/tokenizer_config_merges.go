package data

import (
	"fmt"
	"os"
	"strings"

	"github.com/adhitiad/gotorch/src/errs"
)

// loadMergesTXT memuat berkas aturan merge BPE. Format: "token1 token2" per baris.
// Baris yang dimulai dengan '#' (termasuk baris versi #version:) diabaikan.
// Peringkat ditetapkan dari urutan baris (semakin awal, semakin rendah).
func (c *TokenizerConfig) loadMergesTXT(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("%w: cannot read %q: %w", errs.ErrInvalidArgument, path, err)
	}

	lines := strings.Split(string(data), "\n")
	merges := make([]MergeRule, 0, len(lines))
	rank := 0
	for _, line := range lines {
		line = strings.TrimRight(line, "\r")
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.Fields(line)
		if len(parts) < 2 {
			return fmt.Errorf("%w: invalid merge rule %q in %q: expected 'left right'", errs.ErrInvalidArgument, line, path)
		}
		left, right := parts[0], parts[1]
		var id int64 = -1
		if c.Vocab != nil {
			if mergedID, ok := c.Vocab[left+right]; ok {
				id = mergedID
			}
		}
		merges = append(merges, MergeRule{Left: left, Right: right, Rank: rank, ID: id})
		rank++
	}
	c.Merges = merges
	c.buildMergeIndex()
	return nil
}

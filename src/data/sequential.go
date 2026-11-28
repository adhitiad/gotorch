package data

import (
	"context"
	"fmt"

	"github.com/adhitiad/gotorch/src/errs"
)

// SequentialSampler menghasilkan indeks dataset dalam urutan alami. Sampler ini
// tidak memiliki cursor yang dapat berubah, sehingga posisi yang sama dapat
// diminta berulang kali atau oleh worker bersamaan tanpa mengubah hasil.
type SequentialSampler struct {
	length int64
}

var _ Sampler = (*SequentialSampler)(nil)

// NewSequentialSampler membentuk sampler dengan jumlah posisi epoch tertentu.
// Panjang harus non-negatif; panjang nol valid dan tidak menghasilkan posisi.
func NewSequentialSampler(length int64) (*SequentialSampler, error) {
	if length < 0 {
		return nil, fmt.Errorf("%w: negative sampler length %d", errs.ErrInvalidArgument, length)
	}
	return &SequentialSampler{length: length}, nil
}

// Len mengembalikan jumlah posisi epoch. Sampler nil memiliki panjang nol.
func (s *SequentialSampler) Len() int64 {
	if s == nil {
		return 0
	}
	return s.length
}

// Index mengembalikan indeks dataset yang sama dengan posisi. Pembatalan,
// context nil, dan posisi tidak valid mengikuti kontrak Sampler.
func (s *SequentialSampler) Index(ctx context.Context, position int64) (int64, error) {
	if err := checkContext(ctx); err != nil {
		return 0, err
	}
	if s == nil {
		return 0, fmt.Errorf("%w: nil sequential sampler", errs.ErrInvalidState)
	}
	if err := checkPosition(s.length, position, "sampler position"); err != nil {
		return 0, err
	}
	return position, nil
}

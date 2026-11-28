package data

import (
	"context"
	"fmt"
	"math/rand"

	"github.com/adhitiad/gotorch/src/errs"
)

// RandomSampler menghasilkan permutasi [0, length) berseed. Permutasi dibentuk
// sekali saat konstruksi dan tidak pernah diubah, sehingga Index deterministik
// dan aman untuk panggilan bersamaan. Panjang menentukan alokasi berukuran int;
// pemanggil harus memilih ukuran dataset yang sesuai anggaran memori.
type RandomSampler struct {
	length int64
	seed   int64
	order  []int
}

var _ Sampler = (*RandomSampler)(nil)

// NewRandomSampler membentuk sampler dengan permutasi berseed sepanjang length.
// Seed bersifat eksplisit; sumber global math/rand tidak pernah dipakai.
// Determinisme dijamin untuk seed yang sama dalam satu toolchain Go;
// reproduksibilitas lintas versi memerlukan fixture referensi.
func NewRandomSampler(length, seed int64) (*RandomSampler, error) {
	if length < 0 {
		return nil, fmt.Errorf("%w: negative sampler length %d", errs.ErrInvalidArgument, length)
	}
	if length > int64(^uint(0)>>1) {
		return nil, fmt.Errorf("%w: sampler length %d exceeds int range", errs.ErrShapeOverflow, length)
	}
	order := rand.New(rand.NewSource(seed)).Perm(int(length))
	return &RandomSampler{length: length, seed: seed, order: order}, nil
}

// Len mengembalikan jumlah posisi epoch. Sampler nil memiliki panjang nol.
func (s *RandomSampler) Len() int64 {
	if s == nil {
		return 0
	}
	return s.length
}

// Seed mengembalikan seed yang diberikan saat konstruksi. Sampler nil
// mengembalikan nol.
func (s *RandomSampler) Seed() int64 {
	if s == nil {
		return 0
	}
	return s.seed
}

// Index mengembalikan indeks dataset pada posisi permutasi tertentu.
// Pembatalan, context nil, dan posisi tidak valid mengikuti kontrak Sampler.
func (s *RandomSampler) Index(ctx context.Context, position int64) (int64, error) {
	if err := checkContext(ctx); err != nil {
		return 0, err
	}
	if s == nil {
		return 0, fmt.Errorf("%w: nil random sampler", errs.ErrInvalidState)
	}
	if err := checkPosition(s.length, position, "sampler position"); err != nil {
		return 0, err
	}
	return int64(s.order[position]), nil
}

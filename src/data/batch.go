package data

import (
	"fmt"

	"github.com/adhitiad/gotorch/src/errs"
)

// Batch adalah kelompok sampel berurut beserta indeks dataset asal tiap sampel.
// Indeks dan sampel memiliki panjang yang sama sehingga loader dapat menelusuri
// hasil collate kembali ke sumbernya tanpa kanal pencatatan terpisah.
//
// NewBatch menyalin kedua slice. Nilai field di dalam sampel tetap dipinjam dan
// harus tetap valid sampai batch dikonsumsi. Batch bersifat immutable lewat API
// publiknya serta tidak memiliki atau menutup nilai sampel.
type Batch struct {
	samples []Sample
	indices []int64
}

// NewBatch membentuk batch non-kosong. Indeks sumber harus non-negatif; batas
// atasnya ditentukan oleh dataset yang menghasilkan indeks tersebut.
func NewBatch(samples []Sample, indices []int64) (*Batch, error) {
	if len(samples) == 0 {
		return nil, fmt.Errorf("%w: empty batch", errs.ErrInvalidArgument)
	}
	if len(indices) != len(samples) {
		return nil, fmt.Errorf("%w: %d samples but %d indices", errs.ErrInvalidArgument, len(samples), len(indices))
	}
	for position, index := range indices {
		if index < 0 {
			return nil, fmt.Errorf("%w: negative source index %d at position %d", errs.ErrOutOfBounds, index, position)
		}
	}
	return &Batch{
		samples: append([]Sample(nil), samples...),
		indices: append([]int64(nil), indices...),
	}, nil
}

// Len mengembalikan jumlah sampel dalam batch. Batch nil memiliki panjang nol.
func (b *Batch) Len() int {
	if b == nil {
		return 0
	}
	return len(b.samples)
}

// At mengembalikan sampel dan indeks dataset asal pada posisi tertentu. Posisi
// tidak valid dan batch nil berturut-turut membungkus errs.ErrOutOfBounds dan
// errs.ErrInvalidState.
func (b *Batch) At(position int) (Sample, int64, error) {
	if b == nil {
		return Sample{}, 0, fmt.Errorf("%w: nil batch", errs.ErrInvalidState)
	}
	if position < 0 || position >= len(b.samples) {
		return Sample{}, 0, fmt.Errorf("%w: batch position %d outside [0,%d)", errs.ErrOutOfBounds, position, len(b.samples))
	}
	return b.samples[position].Clone(), b.indices[position], nil
}

// Samples mengembalikan salinan slice sampel.
func (b *Batch) Samples() []Sample {
	if b == nil {
		return nil
	}
	return append([]Sample(nil), b.samples...)
}

// Indices mengembalikan salinan slice indeks sumber.
func (b *Batch) Indices() []int64 {
	if b == nil {
		return nil
	}
	return append([]int64(nil), b.indices...)
}

package data

import (
	"context"
	"fmt"

	"github.com/adhitiad/gotorch/src/errs"
)

// Transform mengubah satu Sample menjadi Sample lain. Implementasi tidak boleh
// memodifikasi masukan, harus menghormati ctx, dan aman dipanggil bersamaan.
type Transform interface {
	Apply(ctx context.Context, sample Sample) (Sample, error)
}

// TransformFunc menyesuaikan fungsi biasa menjadi Transform.
type TransformFunc func(ctx context.Context, sample Sample) (Sample, error)

// Apply menjalankan fungsi yang dibungkus. Fungsi nil mengembalikan error yang
// membungkus errs.ErrInvalidState.
func (f TransformFunc) Apply(ctx context.Context, sample Sample) (Sample, error) {
	if f == nil {
		return Sample{}, fmt.Errorf("%w: nil transform function", errs.ErrInvalidState)
	}
	return f(ctx, sample)
}

// TransformChain menjalankan beberapa Transform secara berurutan. Rantai bersifat
// immutable setelah dibentuk sehingga aman dipakai ulang oleh worker bersamaan.
type TransformChain struct {
	transforms []Transform
}

var _ Transform = (*TransformChain)(nil)

// NewTransformChain membentuk rantai transform. Semua elemen harus non-nil;
// urutan pemanggilan mengikuti urutan argumen. Slice argumen disalin.
func NewTransformChain(transforms ...Transform) (*TransformChain, error) {
	for position, transform := range transforms {
		if isNil(transform) {
			return nil, fmt.Errorf("%w: nil transform at position %d", errs.ErrInvalidArgument, position)
		}
	}
	return &TransformChain{transforms: append([]Transform(nil), transforms...)}, nil
}

// Len mengembalikan jumlah transform dalam rantai. Rantai nil memiliki panjang nol.
func (c *TransformChain) Len() int {
	if c == nil {
		return 0
	}
	return len(c.transforms)
}

// Apply menjalankan tiap transform pada salinan sample masukan. Context diperiksa
// sebelum setiap langkah agar pembatalan tidak menjalankan transform berikutnya.
// Sample masukan tidak diubah, baik oleh rantai maupun oleh transform.
func (c *TransformChain) Apply(ctx context.Context, sample Sample) (Sample, error) {
	if err := checkContext(ctx); err != nil {
		return Sample{}, err
	}
	if c == nil {
		return Sample{}, fmt.Errorf("%w: nil transform chain", errs.ErrInvalidState)
	}
	current := sample.Clone()
	for position, transform := range c.transforms {
		if err := checkContext(ctx); err != nil {
			return Sample{}, err
		}
		if isNil(transform) {
			return Sample{}, fmt.Errorf("%w: nil transform at position %d", errs.ErrInvalidState, position)
		}
		next, err := transform.Apply(ctx, current)
		if err != nil {
			return Sample{}, fmt.Errorf("%w: transform at position %d", err, position)
		}
		current = next.Clone()
	}
	return current, nil
}

// RequireFields membentuk transform validasi yang memeriksa keberadaan field
// bernama. Nama tidak boleh kosong dan duplikat diperbolehkan.
func RequireFields(names ...string) (Transform, error) {
	if len(names) == 0 {
		return nil, fmt.Errorf("%w: no required field names", errs.ErrInvalidArgument)
	}
	required := append([]string(nil), names...)
	for position, name := range required {
		if name == "" {
			return nil, fmt.Errorf("%w: empty field name at position %d", errs.ErrInvalidArgument, position)
		}
	}
	return TransformFunc(func(ctx context.Context, sample Sample) (Sample, error) {
		if err := checkContext(ctx); err != nil {
			return Sample{}, err
		}
		for _, name := range required {
			value, ok := sample.Value(name)
			if !ok || value == nil {
				return Sample{}, fmt.Errorf("%w: missing required field %q", errs.ErrInvalidArgument, name)
			}
		}
		return sample.Clone(), nil
	}), nil
}

// CollateWithTransform membungkus collate dasar dengan transform yang diterapkan
// ke tiap sampel sebelum penggabungan. Indeks sumber tidak diubah dan salinan
// indeks diberikan ke collate dasar.
func CollateWithTransform(transform Transform, base CollateFunc) (CollateFunc, error) {
	if isNil(transform) {
		return nil, fmt.Errorf("%w: nil transform", errs.ErrInvalidArgument)
	}
	if base == nil {
		return nil, fmt.Errorf("%w: nil collate", errs.ErrInvalidArgument)
	}
	return func(ctx context.Context, samples []Sample, indices []int64) (*Batch, error) {
		if err := checkContext(ctx); err != nil {
			return nil, err
		}
		transformed := make([]Sample, len(samples))
		for position, sample := range samples {
			next, err := transform.Apply(ctx, sample)
			if err != nil {
				return nil, fmt.Errorf("%w: sample %d", err, position)
			}
			transformed[position] = next
		}
		return base(ctx, transformed, append([]int64(nil), indices...))
	}, nil
}

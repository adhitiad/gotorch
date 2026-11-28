package data

import (
	"context"
	"fmt"

	"github.com/adhitiad/gotorch/src/errs"
)

// LastBatchPolicy menentukan perlakuan terhadap batch terakhir yang partial.
type LastBatchPolicy string

// Kebijakan batch terakhir yang didukung.
const (
	KeepLast LastBatchPolicy = "keep"
	DropLast LastBatchPolicy = "drop"
)

// CollateFunc menggabungkan sampel dan indeks sumber menjadi satu Batch.
// Implementasi harus menghormati ctx dan tidak boleh memodifikasi masukan.
type CollateFunc func(ctx context.Context, samples []Sample, indices []int64) (*Batch, error)

// Batcher mengelompokkan posisi sampler ke dalam batch berukuran tetap.
// Batcher tidak menyimpan dataset maupun sampler dan tidak memiliki cursor,
// sehingga satu instance dapat dipakai ulang untuk beberapa epoch.
type Batcher struct {
	batchSize int64
	policy    LastBatchPolicy
	collate   CollateFunc
}

// NewBatcher membentuk batcher dengan ukuran, kebijakan batch terakhir, dan
// fungsi collate eksplisit. Semua masukan divalidasi sebelum alokasi.
func NewBatcher(batchSize int64, policy LastBatchPolicy, collate CollateFunc) (*Batcher, error) {
	if batchSize <= 0 {
		return nil, fmt.Errorf("%w: batch size must be positive, got %d", errs.ErrInvalidArgument, batchSize)
	}
	if policy != KeepLast && policy != DropLast {
		return nil, fmt.Errorf("%w: unknown last batch policy %q", errs.ErrInvalidArgument, policy)
	}
	if collate == nil {
		return nil, fmt.Errorf("%w: nil collate", errs.ErrInvalidArgument)
	}
	return &Batcher{batchSize: batchSize, policy: policy, collate: collate}, nil
}

// DefaultCollate membentuk Batch dari sampel dan indeks tanpa transformasi.
func DefaultCollate(ctx context.Context, samples []Sample, indices []int64) (*Batch, error) {
	if err := checkContext(ctx); err != nil {
		return nil, err
	}
	return NewBatch(samples, indices)
}

// BatchSize mengembalikan ukuran batch penuh. Batcher nil memiliki ukuran nol.
func (b *Batcher) BatchSize() int64 {
	if b == nil {
		return 0
	}
	return b.batchSize
}

// Policy mengembalikan kebijakan batch terakhir. Batcher nil mengembalikan string kosong.
func (b *Batcher) Policy() LastBatchPolicy {
	if b == nil {
		return ""
	}
	return b.policy
}

// BatchCount menghitung jumlah batch untuk panjang epoch tertentu. Batch partial
// ikut dihitung hanya ketika kebijakan KeepLast dipakai.
func (b *Batcher) BatchCount(samplerLen int64) int64 {
	if b == nil || samplerLen <= 0 {
		return 0
	}
	count := samplerLen / b.batchSize
	if samplerLen%b.batchSize != 0 && b.policy == KeepLast {
		count++
	}
	return count
}

// Batch memuat satu batch dari dataset melalui sampler lalu menerapkannya ke
// collate. Posisi, indeks dataset, context, dan hasil collate divalidasi.
func (b *Batcher) Batch(ctx context.Context, dataset Dataset, sampler Sampler, batchIndex int64) (*Batch, error) {
	if err := checkContext(ctx); err != nil {
		return nil, err
	}
	if b == nil {
		return nil, fmt.Errorf("%w: nil batcher", errs.ErrInvalidState)
	}
	if isNil(dataset) || isNil(sampler) {
		return nil, fmt.Errorf("%w: nil dataset or sampler", errs.ErrInvalidArgument)
	}
	length := sampler.Len()
	count := b.BatchCount(length)
	if err := checkPosition(count, batchIndex, "batch index"); err != nil {
		return nil, err
	}
	start := batchIndex * b.batchSize
	take := b.batchSize
	if remaining := length - start; take > remaining {
		take = remaining
	}
	if take > int64(^uint(0)>>1) {
		return nil, fmt.Errorf("%w: batch size %d exceeds int range", errs.ErrShapeOverflow, take)
	}
	samples := make([]Sample, 0, int(take))
	indices := make([]int64, 0, int(take))
	for position := start; position < start+take; position++ {
		if err := checkContext(ctx); err != nil {
			return nil, err
		}
		index, err := sampler.Index(ctx, position)
		if err != nil {
			return nil, stepError(ctx, err, "%w: sampler position %d", position)
		}
		if err := checkPosition(dataset.Len(), index, "dataset index"); err != nil {
			return nil, err
		}
		sample, err := dataset.At(ctx, index)
		if err != nil {
			return nil, stepError(ctx, err, "%w: dataset index %d", index)
		}
		samples = append(samples, sample)
		indices = append(indices, index)
	}
	batch, err := b.collate(ctx, samples, indices)
	if err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return nil, ctxErr
		}
		return nil, err
	}
	if batch == nil {
		return nil, fmt.Errorf("%w: collate returned nil batch", errs.ErrInvalidState)
	}
	return batch, nil
}

// stepError lebih mengutamakan ctx.Err() ketika pembatalan terjadi selama satu
// langkah. Tanpa pembatalan, error langkah dibungkus bersama informasi posisi.
func stepError(ctx context.Context, err error, format string, args ...any) error {
	if ctxErr := ctx.Err(); ctxErr != nil {
		return ctxErr
	}
	parameters := append([]any{err}, args...)
	return fmt.Errorf(format, parameters...)
}

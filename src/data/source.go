package data

import (
	"context"
	"errors"
	"fmt"
	"io"
	"sync"

	"github.com/adhitiad/gotorch/src/errs"
)

// Source membaca sampel secara berurutan tanpa akses acak. Next mengembalikan
// io.EOF setelah sampel terakhir. Source bukan pengganti Dataset: pemakaian
// bersamaan harus diatur oleh adapter yang memiliki batasnya sendiri.
type Source interface {
	Next(ctx context.Context) (Sample, error)
	Close() error
}

// SliceSource membaca sampel dari slice yang disalin saat konstruksi. SliceSource
// menyimpan posisi sehingga Next hanya boleh dipanggil oleh satu konsumen.
type SliceSource struct {
	samples  []Sample
	position int
	closed   bool
}

var _ Source = (*SliceSource)(nil)

// NewSliceSource membentuk source di atas salinan slice sampel.
func NewSliceSource(samples []Sample) *SliceSource {
	return &SliceSource{samples: append([]Sample(nil), samples...)}
}

// Len mengembalikan jumlah sampel yang belum dibaca. Source nil memiliki panjang nol.
func (s *SliceSource) Len() int64 {
	if s == nil {
		return 0
	}
	return int64(len(s.samples) - s.position)
}

// Next mengembalikan sampel berikutnya atau io.EOF. Context dibatalkan, source
// nil, dan source yang sudah ditutup mengembalikan error.
func (s *SliceSource) Next(ctx context.Context) (Sample, error) {
	if err := checkContext(ctx); err != nil {
		return Sample{}, err
	}
	if s == nil {
		return Sample{}, fmt.Errorf("%w: nil source", errs.ErrInvalidState)
	}
	if s.closed {
		return Sample{}, fmt.Errorf("%w: source closed", errs.ErrInvalidState)
	}
	if s.position >= len(s.samples) {
		return Sample{}, io.EOF
	}
	sample := s.samples[s.position].Clone()
	s.position++
	return sample, nil
}

// Close menandai source sebagai selesai. Pemanggilan berulang tidak melakukan apa-apa.
func (s *SliceSource) Close() error {
	if s == nil {
		return fmt.Errorf("%w: nil source", errs.ErrInvalidState)
	}
	s.closed = true
	return nil
}

type boundedDataset struct {
	mu      sync.RWMutex
	samples []Sample
	closed  bool
}

var _ Dataset = (*boundedDataset)(nil)

func (d *boundedDataset) Len() int64 {
	if d == nil {
		return 0
	}
	d.mu.RLock()
	defer d.mu.RUnlock()
	return int64(len(d.samples))
}

func (d *boundedDataset) At(ctx context.Context, index int64) (Sample, error) {
	if err := checkContext(ctx); err != nil {
		return Sample{}, err
	}
	if d == nil {
		return Sample{}, fmt.Errorf("%w: nil dataset", errs.ErrInvalidState)
	}
	d.mu.RLock()
	defer d.mu.RUnlock()
	if d.closed {
		return Sample{}, fmt.Errorf("%w: dataset closed", errs.ErrInvalidState)
	}
	if err := checkPosition(int64(len(d.samples)), index, "dataset index"); err != nil {
		return Sample{}, err
	}
	return d.samples[index].Clone(), nil
}

func (d *boundedDataset) Close() error {
	if d == nil {
		return fmt.Errorf("%w: nil dataset", errs.ErrInvalidState)
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	d.closed = true
	return nil
}

// NewSourceDataset membaca paling banyak maxSamples sampel dari source menjadi
// Dataset berbatas memori yang dapat dipakai Batcher dan DataLoader. Batas harus
// positif dan harus masuk ke dalam jangkauan int. Sisa sampel yang tidak dibaca
// tetap berada di source; pemanggil memiliki source dan tidak ditutup oleh fungsi
// ini. Pembatalan konteks diambil alih oleh Source.Next.
func NewSourceDataset(ctx context.Context, source Source, maxSamples int64) (Dataset, error) {
	if err := checkContext(ctx); err != nil {
		return nil, err
	}
	if isNil(source) {
		return nil, fmt.Errorf("%w: nil source", errs.ErrInvalidArgument)
	}
	if maxSamples <= 0 {
		return nil, fmt.Errorf("%w: non-positive sample limit %d", errs.ErrInvalidArgument, maxSamples)
	}
	if maxSamples > int64(^uint(0)>>1) {
		return nil, fmt.Errorf("%w: sample limit %d exceeds int range", errs.ErrShapeOverflow, maxSamples)
	}
	initial := int(maxSamples)
	if initial > 1024 {
		initial = 1024
	}
	samples := make([]Sample, 0, initial)
	for index := int64(0); index < maxSamples; index++ {
		if err := checkContext(ctx); err != nil {
			return nil, err
		}
		sample, err := source.Next(ctx)
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("%w: source sample %d", err, index)
		}
		samples = append(samples, sample.Clone())
	}
	return &boundedDataset{samples: samples}, nil
}

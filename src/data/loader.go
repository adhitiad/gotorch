package data

import (
	"context"
	"fmt"
	"io"
	"sync"

	"github.com/adhitiad/gotorch/src/errs"
)

const (
	// MaxWorkers membatasi jumlah worker prefetch per iterator.
	MaxWorkers = 64
	// MaxPrefetch membatasi kedalaman antrean batch.
	MaxPrefetch = 1024
)

type batchResult struct {
	index int64
	batch *Batch
	err   error
}

// DataLoader merangkai Dataset, Sampler, dan Batcher menjadi aliran batch dengan
// worker dan antrean terbatas. DataLoader tidak memiliki goroutine atau channel
// sebelum Iterate dipanggil, tidak menyimpan cursor, dan tidak menutup dataset
// masukan. Satu nilai dapat dipakai ulang untuk beberapa epoch.
type DataLoader struct {
	batcher  *Batcher
	dataset  Dataset
	sampler  Sampler
	workers  int
	prefetch int
}

// NewDataLoader membentuk loader. Jumlah worker harus dalam [1, MaxWorkers] dan
// kedalaman prefetch dalam [0, MaxPrefetch]. Batas ini menjaga goroutine, channel,
// dan memori tetap terukur.
func NewDataLoader(batcher *Batcher, dataset Dataset, sampler Sampler, workers, prefetch int) (*DataLoader, error) {
	if isNil(batcher) || isNil(dataset) || isNil(sampler) {
		return nil, fmt.Errorf("%w: nil batcher, dataset or sampler", errs.ErrInvalidArgument)
	}
	if workers < 1 || workers > MaxWorkers {
		return nil, fmt.Errorf("%w: worker count %d outside [1,%d]", errs.ErrInvalidArgument, workers, MaxWorkers)
	}
	if prefetch < 0 || prefetch > MaxPrefetch {
		return nil, fmt.Errorf("%w: prefetch %d outside [0,%d]", errs.ErrInvalidArgument, prefetch, MaxPrefetch)
	}
	return &DataLoader{batcher: batcher, dataset: dataset, sampler: sampler, workers: workers, prefetch: prefetch}, nil
}

// Len mengembalikan jumlah batch yang dihasilkan sampler. Loader nil memiliki panjang nol.
func (l *DataLoader) Len() int64 {
	if l == nil {
		return 0
	}
	return l.batcher.BatchCount(l.sampler.Len())
}

// Workers mengembalikan jumlah worker yang dikonfigurasi. Loader nil mengembalikan nol.
func (l *DataLoader) Workers() int {
	if l == nil {
		return 0
	}
	return l.workers
}

// Prefetch mengembalikan kedalaman antrean yang dikonfigurasi. Loader nil mengembalikan nol.
func (l *DataLoader) Prefetch() int {
	if l == nil {
		return 0
	}
	return l.prefetch
}

// Iterate memulai pipeline dan mengembalikan iterator yang harus ditutup oleh
// pemanggil. Batch dihasilkan dalam urutan indeks batch meski worker selesai di
// luar urutan. Pipeline berhenti pada pembatalan, error worker, atau akhir epoch.
func (l *DataLoader) Iterate(ctx context.Context) (*BatchIterator, error) {
	if err := checkContext(ctx); err != nil {
		return nil, err
	}
	if l == nil {
		return nil, fmt.Errorf("%w: nil data loader", errs.ErrInvalidState)
	}
	runCtx, cancel := context.WithCancel(ctx)
	limit := l.workers + l.prefetch
	iterator := &BatchIterator{
		ctx:     runCtx,
		cancel:  cancel,
		jobs:    make(chan int64, l.prefetch),
		results: make(chan batchResult, l.workers),
		ordered: make(chan batchResult, limit),
		count:   l.Len(),
		limit:   limit,
		pending: make(map[int64]batchResult, limit),
	}
	iterator.wg.Add(1)
	go iterator.dispatch(l.batcher, l.dataset, l.sampler)
	for worker := 0; worker < l.workers; worker++ {
		iterator.wg.Add(1)
		go iterator.work(l.batcher, l.dataset, l.sampler)
	}
	return iterator, nil
}

// BatchIterator membaca batch hasil pipeline. Satu iterator hanya boleh dipakai
// oleh satu konsumen; Next dan Close tidak boleh dipanggil bersamaan.
type BatchIterator struct {
	ctx       context.Context
	cancel    context.CancelFunc
	jobs      chan int64
	results   chan batchResult
	ordered   chan batchResult
	count     int64
	limit     int
	submitted int64
	pending   map[int64]batchResult
	mu        sync.Mutex
	err       error
	once      sync.Once
	wg        sync.WaitGroup
}

func (it *BatchIterator) work(batcher *Batcher, dataset Dataset, sampler Sampler) {
	defer it.wg.Done()
	for {
		select {
		case <-it.ctx.Done():
			return
		case index, ok := <-it.jobs:
			if !ok {
				return
			}
			batch, err := batcher.Batch(it.ctx, dataset, sampler, index)
			select {
			case it.results <- batchResult{index: index, batch: batch, err: err}:
			case <-it.ctx.Done():
				return
			}
		}
	}
}

func (it *BatchIterator) dispatch(batcher *Batcher, dataset Dataset, sampler Sampler) {
	defer it.wg.Done()
	defer close(it.ordered)
	emitted := int64(0)
	inFlight := 0
	for emitted < it.count || inFlight > 0 {
		for it.submitted < it.count && inFlight < it.limit {
			select {
			case it.jobs <- it.submitted:
				it.submitted++
				inFlight++
			case <-it.ctx.Done():
				return
			}
		}
		if result, ok := it.pending[emitted]; ok {
			delete(it.pending, emitted)
			emitted++
			select {
			case it.ordered <- result:
			case <-it.ctx.Done():
				return
			}
			continue
		}
		select {
		case result := <-it.results:
			inFlight--
			if result.err != nil {
				it.fail(result.err)
				return
			}
			it.pending[result.index] = result
		case <-it.ctx.Done():
			return
		}
	}
}

// Next mengembalikan batch berikutnya dalam urutan, io.EOF pada akhir epoch,
// ctx.Err() pada pembatalan, atau error worker pertama yang gagal.
func (it *BatchIterator) Next(ctx context.Context) (*Batch, error) {
	if err := checkContext(ctx); err != nil {
		return nil, err
	}
	if it == nil {
		return nil, fmt.Errorf("%w: nil batch iterator", errs.ErrInvalidState)
	}
	if err := it.loadError(); err != nil {
		return nil, err
	}
	select {
	case result, ok := <-it.ordered:
		if !ok {
			it.cleanup()
			if err := it.loadError(); err != nil {
				return nil, err
			}
			return nil, io.EOF
		}
		return result.batch, nil
	case <-ctx.Done():
		it.fail(ctx.Err())
		it.cleanup()
		return nil, ctx.Err()
	case <-it.ctx.Done():
		it.cleanup()
		if err := it.loadError(); err != nil {
			return nil, err
		}
		if err := ctx.Err(); err != nil {
			it.fail(err)
			return nil, err
		}
		err := it.ctx.Err()
		it.fail(err)
		return nil, err
	}
}

// Close menghentikan pipeline dan menunggu semua worker keluar. Pemanggilan
// berulang tidak melakukan apa-apa dan tidak mengembalikan error.
func (it *BatchIterator) Close() error {
	if it == nil {
		return fmt.Errorf("%w: nil batch iterator", errs.ErrInvalidState)
	}
	it.cleanup()
	return nil
}

func (it *BatchIterator) fail(err error) {
	it.mu.Lock()
	if it.err == nil {
		it.err = fmt.Errorf("data loader stopped: %w", err)
	}
	it.mu.Unlock()
	it.cancel()
}

func (it *BatchIterator) loadError() error {
	it.mu.Lock()
	defer it.mu.Unlock()
	return it.err
}

func (it *BatchIterator) cleanup() {
	it.once.Do(func() {
		it.cancel()
		it.wg.Wait()
	})
}

package memory

import (
	"context"
	"fmt"
	"io"
	"reflect"
	"sync"

	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/storage"
)

const maxInt64 = int64(^uint64(0) >> 1)

// PoolConfig membatasi cache dan total byte yang dikelola pool.
// Nilai nol pada MaxCachedBytes atau MaxTotalBytes berarti tanpa batas untuk dimensi itu.
type PoolConfig struct {
	MaxCachedBytes int64
	MaxTotalBytes  int64
}

// DefaultPoolConfig memakai cache 64 MiB dan tanpa batas total alokasi.
func DefaultPoolConfig() PoolConfig {
	return PoolConfig{MaxCachedBytes: 64 << 20}
}

// Validate menolak anggaran negatif.
func (c PoolConfig) Validate() error {
	if c.MaxCachedBytes < 0 || c.MaxTotalBytes < 0 {
		return fmt.Errorf("%w: pool byte limits must be non-negative", errs.ErrInvalidArgument)
	}
	return nil
}

// Pool menggunakan kembali buffer berdasar ukuran persis dan menghitung byte
// active, cached, allocated, dan peak. Pool bukan model kepemilikan: handle
// storage tetap harus dilepaskan oleh pemiliknya.
type Pool struct {
	mu             sync.Mutex
	config         PoolConfig
	closed         bool
	cached         map[int64][]*poolAllocation
	cachedBytes    int64
	activeBytes    int64
	allocatedBytes int64
	peakBytes      int64
	activeBuffers  int64
	cachedBuffers  int64
	allocations    uint64
	reuses         uint64
	releases       uint64
	rejected       uint64
}

type poolAllocation struct {
	mu       sync.RWMutex
	data     []byte
	size     int64
	refs     int64
	released bool
	pool     *Pool
}

type poolStorage struct {
	mu         sync.RWMutex
	allocation *poolAllocation
	released   bool
}

// NewPool membuat pool dengan konfigurasi tertentu.
func NewPool(config PoolConfig) (*Pool, error) {
	if err := config.Validate(); err != nil {
		return nil, err
	}
	return &Pool{config: config, cached: make(map[int64][]*poolAllocation)}, nil
}

// Allocate mengembalikan storage handle zero-initialized. Buffer yang dilepaskan
// dikembalikan ke cache bila masih memenuhi anggaran.
func (p *Pool) Allocate(ctx context.Context, size int64) (storage.Storage, error) {
	if p == nil || nilLike(ctx) {
		return nil, errs.ErrInvalidArgument
	}
	if size < 0 {
		return nil, errs.ErrInvalidArgument
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if uint64(size) > uint64(^uint(0)>>1) {
		return nil, errs.ErrShapeOverflow
	}

	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closed {
		return nil, ErrPoolClosed
	}
	if size > maxInt64-p.allocatedBytes {
		p.rejected++
		return nil, ErrMemoryLimit
	}
	if p.config.MaxTotalBytes > 0 && size > p.config.MaxTotalBytes-p.allocatedBytes {
		p.rejected++
		return nil, ErrMemoryLimit
	}

	allocation, reused := p.takeLocked(size)
	if !reused {
		allocation = &poolAllocation{
			data: make([]byte, int(size)),
			size: size,
			refs: 1,
			pool: p,
		}
		p.allocations++
		p.allocatedBytes += size
		if p.allocatedBytes > p.peakBytes {
			p.peakBytes = p.allocatedBytes
		}
	} else {
		allocation.mu.Lock()
		allocation.refs = 1
		allocation.released = false
		allocation.mu.Unlock()
		p.reuses++
		p.cachedBytes -= size
		p.cachedBuffers--
	}
	p.activeBytes += size
	p.activeBuffers++
	return &poolStorage{allocation: allocation}, nil
}

// takeLocked mengambil buffer dari cache atau menandai bahwa alokasi baru perlu dibuat.
func (p *Pool) takeLocked(size int64) (*poolAllocation, bool) {
	items := p.cached[size]
	if len(items) == 0 {
		return nil, false
	}
	allocation := items[len(items)-1]
	p.cached[size] = items[:len(items)-1]
	if len(p.cached[size]) == 0 {
		delete(p.cached, size)
	}
	return allocation, true
}

// Close melepaskan seluruh buffer cache. Handle aktif tetap valid hingga dilepaskan.
func (p *Pool) Close() error {
	if p == nil {
		return errs.ErrInvalidArgument
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closed {
		return nil
	}
	p.closed = true
	for size, items := range p.cached {
		for _, allocation := range items {
			clear(allocation.data)
			allocation.data = nil
		}
		delete(p.cached, size)
	}
	p.allocatedBytes -= p.cachedBytes
	p.cachedBytes = 0
	p.cachedBuffers = 0
	return nil
}

// Stats mengembalikan snapshot statistik byte dan operasi pool.
func (p *Pool) Stats() PoolStats {
	if p == nil {
		return PoolStats{}
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	return PoolStats{
		ActiveBytes:    p.activeBytes,
		AllocatedBytes: p.allocatedBytes,
		CachedBytes:    p.cachedBytes,
		PeakBytes:      p.peakBytes,
		ActiveBuffers:  p.activeBuffers,
		CachedBuffers:  p.cachedBuffers,
		Allocations:    p.allocations,
		Reuses:         p.reuses,
		Releases:       p.releases,
		Rejected:       p.rejected,
		Closed:         p.closed,
	}
}

// PoolStats adalah snapshot penggunaan memori pool.
type PoolStats struct {
	ActiveBytes    int64
	AllocatedBytes int64
	CachedBytes    int64
	PeakBytes      int64
	ActiveBuffers  int64
	CachedBuffers  int64
	Allocations    uint64
	Reuses         uint64
	Releases       uint64
	Rejected       uint64
	Closed         bool
}

// Size mengembalikan ukuran aktif atau nol bila handle sudah dilepaskan.
func (s *poolStorage) Size() int64 {
	if s == nil {
		return 0
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.released || s.allocation == nil {
		return 0
	}
	return s.allocation.size
}

// Retain membuat handle independen yang berbagi alokasi yang sama.
func (s *poolStorage) Retain() (storage.Storage, error) {
	if s == nil {
		return nil, errs.ErrStorageReleased
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.released || s.allocation == nil {
		return nil, errs.ErrStorageReleased
	}
	a := s.allocation
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.released || a.refs <= 0 {
		return nil, errs.ErrStorageReleased
	}
	if a.refs == maxInt64 {
		return nil, errs.ErrShapeOverflow
	}
	a.refs++
	return &poolStorage{allocation: a}, nil
}

// Release melepaskan hanya handle ini; pelepasan terakhir mengembalikan buffer ke pool.
func (s *poolStorage) Release() error {
	if s == nil {
		return errs.ErrStorageReleased
	}
	s.mu.Lock()
	if s.released || s.allocation == nil {
		s.mu.Unlock()
		return errs.ErrStorageReleased
	}
	a := s.allocation
	s.released = true
	s.allocation = nil
	s.mu.Unlock()

	a.mu.Lock()
	if a.refs <= 0 {
		a.mu.Unlock()
		return errs.ErrInvalidState
	}
	a.refs--
	if a.refs > 0 {
		a.mu.Unlock()
		return nil
	}
	clear(a.data)
	a.released = true
	a.mu.Unlock()
	a.pool.release(a)
	return nil
}

// ReadAt mengikuti semantik io.ReaderAt dan tidak mengekspos slice backing.
func (s *poolStorage) ReadAt(p []byte, off int64) (int, error) {
	if s == nil || off < 0 {
		return 0, errs.ErrInvalidArgument
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.released || s.allocation == nil {
		return 0, errs.ErrStorageReleased
	}
	a := s.allocation
	a.mu.RLock()
	defer a.mu.RUnlock()
	if a.released {
		return 0, errs.ErrStorageReleased
	}
	if off > a.size {
		return 0, io.EOF
	}
	if len(p) == 0 {
		return 0, nil
	}
	if off == a.size {
		return 0, io.EOF
	}
	n := copy(p, a.data[int(off):])
	if n != len(p) {
		return n, io.EOF
	}
	return n, nil
}

// WriteAt mengikuti semantik io.WriterAt dan menyalin data ke alokasi.
func (s *poolStorage) WriteAt(p []byte, off int64) (int, error) {
	if s == nil || off < 0 {
		return 0, errs.ErrInvalidArgument
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.released || s.allocation == nil {
		return 0, errs.ErrStorageReleased
	}
	a := s.allocation
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.released {
		return 0, errs.ErrStorageReleased
	}
	if off > a.size {
		return 0, io.ErrShortWrite
	}
	if len(p) == 0 {
		return 0, nil
	}
	if off == a.size {
		return 0, io.ErrShortWrite
	}
	n := copy(a.data[int(off):], p)
	if n != len(p) {
		return n, io.ErrShortWrite
	}
	return n, nil
}

func (p *Pool) release(a *poolAllocation) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.activeBytes -= a.size
	p.activeBuffers--
	p.releases++
	limit := p.config.MaxCachedBytes
	overLimit := limit > 0 && (a.size > limit || a.size > limit-p.cachedBytes)
	if p.closed || overLimit {
		p.allocatedBytes -= a.size
		a.data = nil
		return
	}
	p.cached[a.size] = append(p.cached[a.size], a)
	p.cachedBytes += a.size
	p.cachedBuffers++
}

func nilLike(value any) bool {
	if value == nil {
		return true
	}
	v := reflect.ValueOf(value)
	switch v.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return v.IsNil()
	default:
		return false
	}
}

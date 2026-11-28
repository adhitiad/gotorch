package memory

import (
	"context"
	"io"
	"sync"

	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/storage"
)

// Arena adalah alokator bump dengan kapasitas tetap. Reset hanya sah setelah
// seluruh handle yang berasal dari arena dilepaskan.
type Arena struct {
	mu              sync.Mutex
	data            []byte
	capacity        int64
	offset          int64
	active          int64
	activeBytes     int64
	peakActiveBytes int64
	allocations     uint64
	resets          uint64
	closed          bool
}

type arenaAllocation struct {
	mu       sync.RWMutex
	arena    *Arena
	offset   int64
	size     int64
	refs     int64
	released bool
}

type arenaStorage struct {
	mu         sync.RWMutex
	allocation *arenaAllocation
	released   bool
}

// NewArena membuat arena dengan kapasitas byte tertentu.
func NewArena(capacity int64) (*Arena, error) {
	if capacity < 0 {
		return nil, errs.ErrInvalidArgument
	}
	if uint64(capacity) > uint64(^uint(0)>>1) {
		return nil, errs.ErrShapeOverflow
	}
	return &Arena{data: make([]byte, int(capacity)), capacity: capacity}, nil
}

// Allocate menyisihkan rentang selaras 8 byte dari arena.
func (a *Arena) Allocate(ctx context.Context, size int64) (storage.Storage, error) {
	if a == nil || nilLike(ctx) {
		return nil, errs.ErrInvalidArgument
	}
	if size < 0 {
		return nil, errs.ErrInvalidArgument
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	a.mu.Lock()
	defer a.mu.Unlock()
	if a.closed {
		return nil, ErrArenaClosed
	}
	aligned := alignUp8(a.offset)
	if aligned < a.offset || aligned > a.capacity || size > a.capacity-aligned {
		return nil, ErrArenaFull
	}
	a.offset = aligned + size
	a.active++
	a.activeBytes += size
	if a.activeBytes > a.peakActiveBytes {
		a.peakActiveBytes = a.activeBytes
	}
	a.allocations++
	return &arenaStorage{allocation: &arenaAllocation{
		arena:  a,
		offset: aligned,
		size:   size,
		refs:   1,
	}}, nil
}

// Reset mengosongkan arena dan menolak operasi bila masih ada handle hidup.
func (a *Arena) Reset() error {
	if a == nil {
		return errs.ErrInvalidArgument
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.closed {
		return ErrArenaClosed
	}
	if a.active > 0 {
		return ErrArenaLive
	}
	clear(a.data)
	a.offset = 0
	a.resets++
	return nil
}

// Close menutup arena setelah seluruh handle dilepaskan.
func (a *Arena) Close() error {
	if a == nil {
		return errs.ErrInvalidArgument
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.closed {
		return nil
	}
	if a.active > 0 {
		return ErrArenaLive
	}
	a.closed = true
	a.data = nil
	return nil
}

// Stats mengembalikan snapshot kapasitas, pemakaian, dan alokasi hidup.
func (a *Arena) Stats() ArenaStats {
	if a == nil {
		return ArenaStats{}
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	return ArenaStats{
		Capacity:        a.capacity,
		Used:            a.offset,
		ActiveBytes:     a.activeBytes,
		PeakActiveBytes: a.peakActiveBytes,
		Active:          a.active,
		Allocations:     a.allocations,
		Resets:          a.resets,
		Closed:          a.closed,
	}
}

// ArenaStats adalah snapshot penggunaan arena.
type ArenaStats struct {
	Capacity        int64
	Used            int64
	ActiveBytes     int64
	PeakActiveBytes int64
	Active          int64
	Allocations     uint64
	Resets          uint64
	Closed          bool
}

// Size mengembalikan ukuran aktif atau nol bila handle sudah dilepaskan.
func (s *arenaStorage) Size() int64 {
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

// Retain membuat handle independen yang berbagi alokasi arena yang sama.
func (s *arenaStorage) Retain() (storage.Storage, error) {
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
	return &arenaStorage{allocation: a}, nil
}

// Release melepaskan handle ini; pelepasan terakhir mengosongkan slot arena.
func (s *arenaStorage) Release() error {
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
	a.released = true
	a.mu.Unlock()

	a.arena.mu.Lock()
	a.arena.active--
	a.arena.activeBytes -= a.size
	a.arena.mu.Unlock()
	return nil
}

// ReadAt mengikuti semantik io.ReaderAt untuk rentang arena yang dipinjam.
func (s *arenaStorage) ReadAt(p []byte, off int64) (int, error) {
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
	start := a.offset + off
	available := a.size - off
	count := int64(len(p))
	if count > available {
		count = available
	}
	n := copy(p[:int(count)], a.arena.data[start:start+count])
	if n != len(p) {
		return n, io.EOF
	}
	return n, nil
}

// WriteAt mengikuti semantik io.WriterAt untuk rentang arena yang dipinjam.
func (s *arenaStorage) WriteAt(p []byte, off int64) (int, error) {
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
		return 0, io.ErrShortWrite
	}
	if len(p) == 0 {
		return 0, nil
	}
	if off == a.size {
		return 0, io.ErrShortWrite
	}
	start := a.offset + off
	available := a.size - off
	count := int64(len(p))
	if count > available {
		count = available
	}
	n := copy(a.arena.data[start:start+count], p[:int(count)])
	if n != len(p) {
		return n, io.ErrShortWrite
	}
	return n, nil
}

func alignUp8(offset int64) int64 {
	remainder := offset % 8
	if remainder == 0 {
		return offset
	}
	return offset + (8 - remainder)
}

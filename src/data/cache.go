package data

import (
	"container/list"
	"context"
	"fmt"
	"sync"

	"github.com/adhitiad/gotorch/src/errs"
)

// Cache menyimpan nilai menurut kunci untuk dipakai ulang lintas epoch.
// Implementasi harus aman dipanggil bersamaan dan membatasi memorinya sendiri.
type Cache[V any] interface {
	Get(ctx context.Context, key string) (V, bool, error)
	Put(ctx context.Context, key string, value V) error
	Delete(ctx context.Context, key string) error
	Len() int
	Close() error
}

type cacheEntry[V any] struct {
	key   string
	value V
}

// MemoryCache adalah Cache LRU dengan batas jumlah entry. Nilai disimpan apa
// adanya: untuk Sample, peta field disalin oleh Sample.Clone sebelum disimpan.
type MemoryCache[V any] struct {
	mu       sync.Mutex
	items    map[string]*list.Element
	order    *list.List
	capacity int
	closed   bool
}

var _ Cache[any] = (*MemoryCache[any])(nil)

// NewMemoryCache membentuk cache dengan kapasitas entry positif. Kapasitas
// menentukan batas jumlah entry dan tidak dapat diubah setelah konstruksi.
func NewMemoryCache[V any](capacity int) (*MemoryCache[V], error) {
	if capacity <= 0 {
		return nil, fmt.Errorf("%w: non-positive cache capacity %d", errs.ErrInvalidArgument, capacity)
	}
	return &MemoryCache[V]{
		items:    make(map[string]*list.Element, capacity),
		order:    list.New(),
		capacity: capacity,
	}, nil
}

// Get mengembalikan nilai dan status keberadaan. Entry yang dibaca menjadi yang
// terbaru. Kunci kosong dan context tidak valid mengembalikan error.
func (c *MemoryCache[V]) Get(ctx context.Context, key string) (V, bool, error) {
	var zero V
	if err := checkContext(ctx); err != nil {
		return zero, false, err
	}
	if c == nil {
		return zero, false, fmt.Errorf("%w: nil cache", errs.ErrInvalidState)
	}
	if key == "" {
		return zero, false, fmt.Errorf("%w: empty cache key", errs.ErrInvalidArgument)
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return zero, false, fmt.Errorf("%w: cache closed", errs.ErrInvalidState)
	}
	element, ok := c.items[key]
	if !ok {
		return zero, false, nil
	}
	c.order.MoveToFront(element)
	return element.Value.(*cacheEntry[V]).value, true, nil
}

// Put menyimpan atau memperbarui nilai. Entry terlama dikeluarkan ketika
// kapasitas terlampaui.
func (c *MemoryCache[V]) Put(ctx context.Context, key string, value V) error {
	if err := checkContext(ctx); err != nil {
		return err
	}
	if c == nil {
		return fmt.Errorf("%w: nil cache", errs.ErrInvalidState)
	}
	if key == "" {
		return fmt.Errorf("%w: empty cache key", errs.ErrInvalidArgument)
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return fmt.Errorf("%w: cache closed", errs.ErrInvalidState)
	}
	if element, ok := c.items[key]; ok {
		element.Value.(*cacheEntry[V]).value = value
		c.order.MoveToFront(element)
		return nil
	}
	c.items[key] = c.order.PushFront(&cacheEntry[V]{key: key, value: value})
	if c.order.Len() > c.capacity {
		oldest := c.order.Back()
		if oldest != nil {
			c.order.Remove(oldest)
			delete(c.items, oldest.Value.(*cacheEntry[V]).key)
		}
	}
	return nil
}

// Delete menghapus kunci. Menghapus kunci yang tidak ada bukan error.
func (c *MemoryCache[V]) Delete(ctx context.Context, key string) error {
	if err := checkContext(ctx); err != nil {
		return err
	}
	if c == nil {
		return fmt.Errorf("%w: nil cache", errs.ErrInvalidState)
	}
	if key == "" {
		return fmt.Errorf("%w: empty cache key", errs.ErrInvalidArgument)
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return fmt.Errorf("%w: cache closed", errs.ErrInvalidState)
	}
	if element, ok := c.items[key]; ok {
		c.order.Remove(element)
		delete(c.items, key)
	}
	return nil
}

// Len mengembalikan jumlah entry yang tersimpan.
func (c *MemoryCache[V]) Len() int {
	if c == nil {
		return 0
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.items)
}

// Capacity mengembalikan batas entry yang ditetapkan saat konstruksi.
func (c *MemoryCache[V]) Capacity() int {
	if c == nil {
		return 0
	}
	return c.capacity
}

// Close mengosongkan cache dan bersifat idempoten. Operasi setelah Close
// mengembalikan error yang membungkus errs.ErrInvalidState.
func (c *MemoryCache[V]) Close() error {
	if c == nil {
		return fmt.Errorf("%w: nil cache", errs.ErrInvalidState)
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return nil
	}
	c.closed = true
	c.items = make(map[string]*list.Element)
	c.order.Init()
	return nil
}

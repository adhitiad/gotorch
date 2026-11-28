package data

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"

	"github.com/adhitiad/gotorch/src/errs"
)

func TestMemoryCacheLRU(t *testing.T) {
	cache, err := NewMemoryCache[string](2)
	if err != nil {
		t.Fatalf("NewMemoryCache: %v", err)
	}
	if cache.Capacity() != 2 || cache.Len() != 0 {
		t.Fatalf("kapasitas/panjang = %d/%d", cache.Capacity(), cache.Len())
	}
	ctx := context.Background()
	for _, key := range []string{"a", "b"} {
		if err := cache.Put(ctx, key, "value-"+key); err != nil {
			t.Fatalf("Put(%q): %v", key, err)
		}
	}
	if value, ok, err := cache.Get(ctx, "a"); err != nil || !ok || value != "value-a" {
		t.Fatalf("Get(a) = %q, %v, %v", value, ok, err)
	}
	if err := cache.Put(ctx, "c", "value-c"); err != nil {
		t.Fatalf("Put(c): %v", err)
	}
	if cache.Len() != 2 {
		t.Fatalf("Len = %d, diharapkan 2", cache.Len())
	}
	if _, ok, err := cache.Get(ctx, "b"); err != nil || ok {
		t.Fatalf("b tidak dikeluarkan: %v, %v", ok, err)
	}
	if value, ok, err := cache.Get(ctx, "a"); err != nil || !ok || value != "value-a" {
		t.Fatalf("Get(a) setelah eviksi = %q, %v, %v", value, ok, err)
	}
	if err := cache.Put(ctx, "a", "updated"); err != nil {
		t.Fatalf("Put(a): %v", err)
	}
	if value, _, _ := cache.Get(ctx, "a"); value != "updated" {
		t.Fatalf("nilai tidak diperbarui: %q", value)
	}
	if cache.Len() != 2 {
		t.Fatalf("Len setelah update = %d", cache.Len())
	}
	if err := cache.Delete(ctx, "c"); err != nil {
		t.Fatalf("Delete(c): %v", err)
	}
	if err := cache.Delete(ctx, "c"); err != nil {
		t.Fatalf("Delete(c) kedua: %v", err)
	}
	if _, ok, _ := cache.Get(ctx, "c"); ok {
		t.Fatal("kunci masih ada setelah dihapus")
	}
}

func TestMemoryCacheValidation(t *testing.T) {
	if _, err := NewMemoryCache[int](0); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("kapasitas nol diterima: %v", err)
	}
	if _, err := NewMemoryCache[int](-1); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("kapasitas negatif diterima: %v", err)
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	cache, err := NewMemoryCache[int](1)
	if err != nil {
		t.Fatalf("NewMemoryCache: %v", err)
	}
	if _, _, err := cache.Get(nil, "a"); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("Get context nil = %v", err)
	}
	if err := cache.Put(canceled, "a", 1); !errors.Is(err, context.Canceled) {
		t.Fatalf("Put context dibatalkan = %v", err)
	}
	if err := cache.Delete(nil, "a"); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("Delete context nil = %v", err)
	}
	if _, _, err := cache.Get(context.Background(), ""); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("Get kunci kosong = %v", err)
	}
	if err := cache.Put(context.Background(), "", 1); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("Put kunci kosong = %v", err)
	}
	if err := cache.Delete(context.Background(), ""); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("Delete kunci kosong = %v", err)
	}
	var nilCache *MemoryCache[int]
	if nilCache.Len() != 0 || nilCache.Capacity() != 0 {
		t.Fatal("cache nil tidak bersifat kosong")
	}
	if _, _, err := nilCache.Get(context.Background(), "a"); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("Get cache nil = %v", err)
	}
	if err := nilCache.Put(context.Background(), "a", 1); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("Put cache nil = %v", err)
	}
	if err := nilCache.Close(); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("Close cache nil = %v", err)
	}
}

func TestMemoryCacheClose(t *testing.T) {
	cache, err := NewMemoryCache[int](2)
	if err != nil {
		t.Fatalf("NewMemoryCache: %v", err)
	}
	if err := cache.Put(context.Background(), "a", 1); err != nil {
		t.Fatalf("Put: %v", err)
	}
	if err := cache.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := cache.Close(); err != nil {
		t.Fatalf("Close kedua: %v", err)
	}
	if cache.Len() != 0 {
		t.Fatalf("Len setelah Close = %d", cache.Len())
	}
	ctx := context.Background()
	if _, _, err := cache.Get(ctx, "a"); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("Get setelah Close = %v", err)
	}
	if err := cache.Put(ctx, "b", 2); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("Put setelah Close = %v", err)
	}
	if err := cache.Delete(ctx, "a"); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("Delete setelah Close = %v", err)
	}
}

func TestMemoryCacheConcurrent(t *testing.T) {
	const workers, keys = 8, 8
	cache, err := NewMemoryCache[int](workers * keys)
	if err != nil {
		t.Fatalf("NewMemoryCache: %v", err)
	}
	var group sync.WaitGroup
	for worker := 0; worker < workers; worker++ {
		group.Add(1)
		go func(worker int) {
			defer group.Done()
			for index := 0; index < keys; index++ {
				key := fmt.Sprintf("w%d-k%d", worker, index)
				if err := cache.Put(context.Background(), key, worker*keys+index); err != nil {
					t.Errorf("Put(%q): %v", key, err)
					return
				}
				if value, ok, err := cache.Get(context.Background(), key); err != nil || !ok || value != worker*keys+index {
					t.Errorf("Get(%q) = %d, %v, %v", key, value, ok, err)
					return
				}
			}
		}(worker)
	}
	group.Wait()
	if cache.Len() != workers*keys {
		t.Fatalf("Len = %d, diharapkan %d", cache.Len(), workers*keys)
	}
}

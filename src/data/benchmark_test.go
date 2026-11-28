package data

import (
	"context"
	"errors"
	"fmt"
	"io"
	"testing"

	"github.com/adhitiad/gotorch/src/errs"
)

func benchDataset(length int64) *sliceDataset {
	samples := make([]Sample, length)
	for index := range samples {
		sample, err := NewSample(map[string]any{"id": int64(index), "value": float64(index)})
		if err != nil {
			panic(err)
		}
		samples[index] = sample
	}
	return &sliceDataset{samples: samples}
}

func BenchmarkSequentialSamplerIndex(b *testing.B) {
	sampler, err := NewSequentialSampler(1000)
	if err != nil {
		b.Fatalf("NewSequentialSampler: %v", err)
	}
	position := int64(0)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		index, err := sampler.Index(context.Background(), position)
		if err != nil {
			b.Fatalf("Index(%d): %v", position, err)
		}
		if index != position {
			b.Fatalf("Index(%d) = %d", position, index)
		}
		position = (position + 1) % sampler.Len()
		b.SetBytes(1)
	}
}

func BenchmarkRandomSamplerIndex(b *testing.B) {
	sampler, err := NewRandomSampler(1000, 7)
	if err != nil {
		b.Fatalf("NewRandomSampler: %v", err)
	}
	position := int64(0)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		index, err := sampler.Index(context.Background(), position)
		if err != nil {
			b.Fatalf("Index(%d): %v", position, err)
		}
		if index < 0 || index >= sampler.Len() {
			b.Fatalf("Index(%d) = %d", position, index)
		}
		position = (position + 1) % sampler.Len()
		b.SetBytes(1)
	}
}

func BenchmarkBatchConstruction(b *testing.B) {
	samples := make([]Sample, 0, 32)
	for index := 0; index < 32; index++ {
		sample, err := NewSample(map[string]any{"id": int64(index), "text": "record"})
		if err != nil {
			b.Fatalf("NewSample(%d): %v", index, err)
		}
		samples = append(samples, sample)
	}
	indices := make([]int64, 32)
	for index := range indices {
		indices[index] = int64(index)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		batch, err := NewBatch(samples, indices)
		if err != nil {
			b.Fatalf("NewBatch: %v", err)
		}
		if batch.Len() != 32 {
			b.Fatalf("Len = %d", batch.Len())
		}
		b.SetBytes(32)
	}
}

func BenchmarkBatchAt(b *testing.B) {
	samples := make([]Sample, 32)
	for index := range samples {
		sample, err := NewSample(map[string]any{"id": int64(index)})
		if err != nil {
			b.Fatalf("NewSample(%d): %v", index, err)
		}
		samples[index] = sample
	}
	batch, err := NewBatch(samples, make([]int64, 32))
	if err != nil {
		b.Fatalf("NewBatch: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for position := 0; position < batch.Len(); position++ {
			if _, _, err := batch.At(position); err != nil {
				b.Fatalf("At(%d): %v", position, err)
			}
		}
		b.SetBytes(32)
	}
}

func BenchmarkTransformChainApply(b *testing.B) {
	required, err := RequireFields("id")
	if err != nil {
		b.Fatalf("RequireFields: %v", err)
	}
	chain, err := NewTransformChain(
		required,
		appendTransform{token: "x"},
		appendTransform{token: "y"},
	)
	if err != nil {
		b.Fatalf("NewTransformChain: %v", err)
	}
	sample := benchTraceSample()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result, err := chain.Apply(context.Background(), sample)
		if err != nil {
			b.Fatalf("Apply: %v", err)
		}
		if result.Len() != 2 {
			b.Fatalf("Len = %d", result.Len())
		}
		b.SetBytes(int64(result.Len()))
	}
}

func benchTraceSample() Sample {
	sample, err := NewSample(map[string]any{"trace": "", "id": int64(1)})
	if err != nil {
		panic(err)
	}
	return sample
}

func BenchmarkCacheGet(b *testing.B) {
	cache, err := NewMemoryCache[int64](2048)
	if err != nil {
		b.Fatalf("NewMemoryCache: %v", err)
	}
	for index := 0; index < 1024; index++ {
		if err := cache.Put(context.Background(), fmt.Sprintf("key-%d", index), int64(index)); err != nil {
			b.Fatalf("Put: %v", err)
		}
	}
	key := 0
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		value, ok, err := cache.Get(context.Background(), fmt.Sprintf("key-%d", key))
		if err != nil {
			b.Fatalf("Get: %v", err)
		}
		if !ok || value != int64(key) {
			b.Fatalf("Get(%d) = %d, %v", key, value, ok)
		}
		key = (key + 1) % 1024
		b.SetBytes(1)
	}
}

func BenchmarkCachePut(b *testing.B) {
	cache, err := NewMemoryCache[int64](1024)
	if err != nil {
		b.Fatalf("NewMemoryCache: %v", err)
	}
	key := 0
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := cache.Put(context.Background(), string(rune('a'+key%26)), int64(key)); err != nil {
			b.Fatalf("Put: %v", err)
		}
		key = (key + 1) % 26
		b.SetBytes(1)
	}
}

func BenchmarkBatcherPipeline(b *testing.B) {
	dataset := benchDataset(1024)
	sampler := fixedSampler{indices: make([]int64, 1024)}
	for index := range sampler.indices {
		sampler.indices[index] = int64(index)
	}
	batcher, err := NewBatcher(32, KeepLast, DefaultCollate)
	if err != nil {
		b.Fatalf("NewBatcher: %v", err)
	}
	batchIndex := int64(0)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		batch, err := batcher.Batch(context.Background(), dataset, sampler, batchIndex)
		if err != nil {
			if !errors.Is(err, errs.ErrOutOfBounds) {
				b.Fatalf("Batch(%d): %v", batchIndex, err)
			}
			batchIndex = 0
			batch, err = batcher.Batch(context.Background(), dataset, sampler, batchIndex)
			if err != nil {
				b.Fatalf("Batch(%d): %v", batchIndex, err)
			}
		}
		if batch.Len() == 0 {
			b.Fatal("empty batch")
		}
		batchIndex++
		b.SetBytes(int64(batch.Len()))
	}
}

func BenchmarkDataLoaderThroughput(b *testing.B) {
	dataset := benchDataset(2048)
	sampler := fixedSampler{indices: make([]int64, 2048)}
	for index := range sampler.indices {
		sampler.indices[index] = int64(index)
	}
	batcher, err := NewBatcher(32, KeepLast, DefaultCollate)
	if err != nil {
		b.Fatalf("NewBatcher: %v", err)
	}
	loader, err := NewDataLoader(batcher, dataset, sampler, 4, 8)
	if err != nil {
		b.Fatalf("NewDataLoader: %v", err)
	}
	iterator, err := loader.Iterate(context.Background())
	if err != nil {
		b.Fatalf("Iterate: %v", err)
	}
	samples := 0
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		batch, err := iterator.Next(context.Background())
		if err != nil {
			if errors.Is(err, io.EOF) {
				if err := iterator.Close(); err != nil {
					b.Fatalf("Close: %v", err)
				}
				iterator, err = loader.Iterate(context.Background())
				if err != nil {
					b.Fatalf("Iterate restart: %v", err)
				}
				batch, err = iterator.Next(context.Background())
				if err != nil {
					b.Fatalf("Next restart: %v", err)
				}
			} else {
				b.Fatalf("Next: %v", err)
			}
		}
		samples += batch.Len()
		b.SetBytes(int64(batch.Len()))
	}
	if err := iterator.Close(); err != nil {
		b.Fatalf("Close: %v", err)
	}
	b.StopTimer()
	_ = samples
}

func BenchmarkSampleClone(b *testing.B) {
	sample := benchTraceSample()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		clone := sample.Clone()
		if clone.Len() != sample.Len() {
			b.Fatalf("clone Len = %d", clone.Len())
		}
		b.SetBytes(int64(clone.Len()))
	}
}

func BenchmarkNewSample(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sample, err := NewSample(map[string]any{"id": int64(i), "value": float64(i)})
		if err != nil {
			b.Fatalf("NewSample(%d): %v", i, err)
		}
		if sample.Len() != 2 {
			b.Fatalf("Len = %d", sample.Len())
		}
		b.SetBytes(2)
	}
}

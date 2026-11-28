package data_test

import (
	"context"
	"fmt"
	"strings"

	"github.com/adhitiad/gotorch/src/data"
)

type exampleDataset struct {
	values []int64
}

func (d exampleDataset) Len() int64 { return int64(len(d.values)) }

func (d exampleDataset) At(ctx context.Context, index int64) (data.Sample, error) {
	if err := ctx.Err(); err != nil {
		return data.Sample{}, err
	}
	if index < 0 || index >= d.Len() {
		return data.Sample{}, fmt.Errorf("index out of range")
	}
	return data.NewSample(map[string]any{"id": d.values[index]})
}

func (d exampleDataset) Close() error { return nil }

func ExampleNewSample() {
	sample, err := data.NewSample(map[string]any{"id": int64(3), "text": "hello"})
	if err != nil {
		panic(err)
	}
	fmt.Println(sample.Names())
	value, _ := sample.Value("text")
	fmt.Println(value)
	// Output:
	// [id text]
	// hello
}

func ExampleNewBatch() {
	first, _ := data.NewSample(map[string]any{"id": int64(0)})
	second, _ := data.NewSample(map[string]any{"id": int64(1)})
	batch, err := data.NewBatch([]data.Sample{first, second}, []int64{1, 0})
	if err != nil {
		panic(err)
	}
	sample, index, err := batch.At(1)
	if err != nil {
		panic(err)
	}
	value, _ := sample.Value("id")
	fmt.Println(index, value)
	// Output: 0 1
}

func ExampleNewBatcher() {
	dataset := exampleDataset{values: []int64{10, 20, 30}}
	sampler, err := data.NewSequentialSampler(dataset.Len())
	if err != nil {
		panic(err)
	}
	batcher, err := data.NewBatcher(2, data.KeepLast, data.DefaultCollate)
	if err != nil {
		panic(err)
	}
	for index := int64(0); index < batcher.BatchCount(sampler.Len()); index++ {
		batch, err := batcher.Batch(context.Background(), dataset, sampler, index)
		if err != nil {
			panic(err)
		}
		fmt.Println(batch.Indices())
	}
	// Output:
	// [0 1]
	// [2]
}

func ExampleNewTransformChain() {
	upper := data.TransformFunc(func(ctx context.Context, sample data.Sample) (data.Sample, error) {
		if err := ctx.Err(); err != nil {
			return data.Sample{}, err
		}
		value, _ := sample.Value("text")
		return data.NewSample(map[string]any{"text": strings.ToUpper(value.(string))})
	})
	required, err := data.RequireFields("text")
	if err != nil {
		panic(err)
	}
	chain, err := data.NewTransformChain(required, upper)
	if err != nil {
		panic(err)
	}
	sample, err := data.NewSample(map[string]any{"text": "hello"})
	if err != nil {
		panic(err)
	}
	result, err := chain.Apply(context.Background(), sample)
	if err != nil {
		panic(err)
	}
	value, _ := result.Value("text")
	fmt.Println(value)
	// Output: HELLO
}

func ExampleNewCharLevelBPETokenizer() {
	tokenizer := data.NewCharLevelBPETokenizer(100, data.DefaultSpecialIDs())
	if err := tokenizer.Train([]string{"hello world", "hello there"}, 100); err != nil {
		panic(err)
	}
	encoding, err := tokenizer.Encode(context.Background(), "hello")
	if err != nil {
		panic(err)
	}
	decoded, err := tokenizer.Decode(context.Background(), encoding.IDs())
	if err != nil {
		panic(err)
	}
	fmt.Println(decoded)
	// Output: hello
}

func ExampleNewByteLevelBPETokenizer() {
	tokenizer := data.NewByteLevelBPETokenizer(300, data.DefaultSpecialIDs())
	if err := tokenizer.Train([]string{"café hello"}, 300); err != nil {
		panic(err)
	}
	encoding, err := tokenizer.Encode(context.Background(), "café")
	if err != nil {
		panic(err)
	}
	decoded, err := tokenizer.Decode(context.Background(), encoding.IDs())
	if err != nil {
		panic(err)
	}
	fmt.Println(decoded)
	// Output: café
}

func ExampleNewWordPieceTokenizer() {
	tokenizer := data.NewWordPieceTokenizer(false, data.DefaultSpecialIDs())
	if err := tokenizer.Train([]string{"hello world", "hello there"}, 200); err != nil {
		panic(err)
	}
	encoding, err := tokenizer.Encode(context.Background(), "hello")
	if err != nil {
		panic(err)
	}
	decoded, err := tokenizer.Decode(context.Background(), encoding.IDs())
	if err != nil {
		panic(err)
	}
	fmt.Println(decoded)
	// Output: hello
}

func ExampleNewUnigramTokenizer() {
	tokenizer := data.NewUnigramTokenizer(false, data.DefaultSpecialIDs())
	if err := tokenizer.Train([]string{"hello world", "hello there"}, 200); err != nil {
		panic(err)
	}
	encoding, err := tokenizer.Encode(context.Background(), "hello")
	if err != nil {
		panic(err)
	}
	decoded, err := tokenizer.Decode(context.Background(), encoding.IDs())
	if err != nil {
		panic(err)
	}
	fmt.Println(decoded)
	// Output: hello
}

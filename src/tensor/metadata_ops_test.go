package tensor

import (
	"math/rand"
	"testing"

	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/shape"
)

func TestMetadataIndexOffsetsAndLogicalIndices(t *testing.T) {
	dimensions, _ := shape.New(2, 3, 4)
	metadata, err := NewMetadata(dtype.Float32, dimensions)
	if err != nil {
		t.Fatal(err)
	}
	if got, err := metadata.ElementOffset(1, 2, 3); err != nil || got != 23 {
		t.Fatalf("ElementOffset = %d, %v; want 23", got, err)
	}
	if got, err := metadata.LinearIndex(1, 2, 3); err != nil || got != 23 {
		t.Fatalf("LinearIndex = %d, %v; want 23", got, err)
	}

	strided, err := NewStridedMetadata(dtype.Float32, dimensions, []int64{20, 5, 1})
	if err != nil {
		t.Fatal(err)
	}
	if got, err := strided.ElementOffset(1, 2, 3); err != nil || got != 33 {
		t.Fatalf("strided offset = %d, %v; want 33", got, err)
	}
	if got, err := strided.LinearIndex(1, 2, 3); err != nil || got != 23 {
		t.Fatalf("strided logical index = %d, %v; want 23", got, err)
	}
}

func TestMetadataIndexValidation(t *testing.T) {
	dimensions, _ := shape.New(2, 3)
	metadata, _ := NewMetadata(dtype.Float32, dimensions)
	cases := [][]int64{{-1, 0}, {2, 0}, {0, 3}, {0}, {0, 1, 2}}
	for _, indices := range cases {
		if _, err := metadata.ElementOffset(indices...); err == nil {
			t.Errorf("ElementOffset(%v) unexpectedly succeeded", indices)
		}
		if _, err := metadata.LinearIndex(indices...); err == nil {
			t.Errorf("LinearIndex(%v) unexpectedly succeeded", indices)
		}
	}
	scalar, _ := NewMetadata(dtype.Int64, shape.Scalar())
	if got, err := scalar.ElementOffset(); err != nil || got != 0 {
		t.Fatalf("scalar offset = %d, %v", got, err)
	}
	if got, err := scalar.LinearIndex(); err != nil || got != 0 {
		t.Fatalf("scalar logical index = %d, %v", got, err)
	}
}

func TestMetadataSliceSelectTransposeAndReshape(t *testing.T) {
	dimensions, _ := shape.New(2, 4)
	metadata, _ := NewMetadata(dtype.Int32, dimensions)

	sliced, err := metadata.Slice(1, 1, 4, 2)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := sliced.Shape().Dims(), []int64{2, 2}; !equal(got, want) {
		t.Fatalf("slice dims = %v, want %v", got, want)
	}
	if got, want := sliced.Strides(), []int64{4, 2}; !equal(got, want) {
		t.Fatalf("slice strides = %v, want %v", got, want)
	}

	selected, err := metadata.Select(0, 1)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := selected.Shape().Dims(), []int64{4}; !equal(got, want) {
		t.Fatalf("select dims = %v, want %v", got, want)
	}

	transposed, err := metadata.Transpose()
	if err != nil {
		t.Fatal(err)
	}
	if got, want := transposed.Shape().Dims(), []int64{4, 2}; !equal(got, want) {
		t.Fatalf("transpose dims = %v, want %v", got, want)
	}
	if got, want := transposed.Strides(), []int64{1, 4}; !equal(got, want) {
		t.Fatalf("transpose strides = %v, want %v", got, want)
	}
	if transposed.IsContiguous() {
		t.Fatal("transpose reported contiguous")
	}

	reshaped, err := metadata.Reshape(4, 2)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := reshaped.Strides(), []int64{2, 1}; !equal(got, want) {
		t.Fatalf("reshape strides = %v, want %v", got, want)
	}
	if _, err := transposed.Reshape(8); err != errs.ErrUnsupportedOperation {
		t.Fatalf("non-contiguous reshape error = %v", err)
	}
}

func TestMetadataBroadcastInferenceAndValidation(t *testing.T) {
	a, _ := shape.New(2, 1, 3)
	b, _ := shape.New(1, 4, 1)
	got, err := BroadcastShapes(a, b)
	if err != nil {
		t.Fatal(err)
	}
	if want := []int64{2, 4, 3}; !equal(got.Dims(), want) {
		t.Fatalf("broadcast dims = %v, want %v", got.Dims(), want)
	}

	emptyA, _ := shape.New(0, 1)
	emptyB, _ := shape.New(1, 5)
	empty, err := BroadcastShapes(emptyA, emptyB)
	if err != nil {
		t.Fatal(err)
	}
	if want := []int64{0, 5}; !equal(empty.Dims(), want) {
		t.Fatalf("empty broadcast dims = %v, want %v", empty.Dims(), want)
	}

	incompatible, _ := shape.New(4, 3, 3)
	if _, err := BroadcastShapes(a, incompatible); err == nil {
		t.Fatal("incompatible broadcast accepted")
	}
}

func TestMetadataBroadcastToZeroStrides(t *testing.T) {
	sourceShape, _ := shape.New(1, 3)
	targetShape, _ := shape.New(4, 3)
	source, _ := NewMetadata(dtype.Float32, sourceShape)
	broadcast, err := source.BroadcastTo(targetShape)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := broadcast.Strides(), []int64{0, 1}; !equal(got, want) {
		t.Fatalf("broadcast strides = %v, want %v", got, want)
	}
	if !broadcast.hasOverlappingElements() {
		t.Fatal("broadcast should report overlapping elements")
	}
}

func TestPropertyMetadataIndexAndStrideOffsets(t *testing.T) {
	random := rand.New(rand.NewSource(0x60_70_72))
	for iteration := 0; iteration < 150; iteration++ {
		rank := random.Intn(4) + 1
		dimensions := make([]int64, rank)
		strides := make([]int64, rank)
		contiguous := int64(1)
		for axis := rank - 1; axis >= 0; axis-- {
			dimensions[axis] = int64(random.Intn(5) + 1)
			strides[axis] = contiguous * int64(random.Intn(3)+1)
			contiguous *= dimensions[axis]
		}
		shapeValue, err := shape.New(dimensions...)
		if err != nil {
			t.Fatal(err)
		}
		metadata, err := NewStridedMetadata(dtype.Uint8, shapeValue, strides)
		if err != nil {
			t.Fatal(err)
		}
		numel, _ := metadata.Numel()
		for linear := int64(0); linear < numel; linear++ {
			remaining := linear
			indices := make([]int64, rank)
			var wantOffset int64
			for axis := rank - 1; axis >= 0; axis-- {
				indices[axis] = remaining % dimensions[axis]
				remaining /= dimensions[axis]
				wantOffset += indices[axis] * strides[axis]
			}
			gotOffset, err := metadata.ElementOffset(indices...)
			if err != nil || gotOffset != wantOffset {
				t.Fatalf("iteration %d index %v offset = %d, %v; want %d", iteration, indices, gotOffset, err, wantOffset)
			}
			gotLinear, err := metadata.LinearIndex(indices...)
			if err != nil || gotLinear != linear {
				t.Fatalf("iteration %d index %v linear = %d, %v; want %d", iteration, indices, gotLinear, err, linear)
			}
		}
	}
}

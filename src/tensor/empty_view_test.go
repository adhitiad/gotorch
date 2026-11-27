package tensor

import "testing"

func TestEmptyViewsAtEndOfStridedLayout(t *testing.T) {
	base, err := From([]int32{0, 1, 2, 3, 4, 5}, 6)
	if err != nil {
		t.Fatal(err)
	}
	defer base.Close()

	gapped, err := base.Slice(0, 0, 6, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer gapped.Close()

	emptySlice, err := gapped.Slice(0, 3, 3, 1)
	if err != nil {
		t.Fatalf("empty slice at end of gapped view: %v", err)
	}
	defer emptySlice.Close()
	emptyMetadata, err := emptySlice.Metadata()
	if err != nil {
		t.Fatal(err)
	}
	if got, err := emptyMetadata.Numel(); err != nil || got != 0 {
		t.Fatalf("empty slice numel = %d, %v; want 0", got, err)
	}
	if got, err := Read[int32](emptySlice); err != nil || len(got) != 0 {
		t.Fatalf("empty slice read = %v, %v; want empty", got, err)
	}
	if err := Write[int32](emptySlice, nil); err != nil {
		t.Fatalf("empty slice write: %v", err)
	}

	emptyNarrow, err := gapped.Narrow(0, 3, 0)
	if err != nil {
		t.Fatalf("empty narrow at end of gapped view: %v", err)
	}
	defer emptyNarrow.Close()
	narrowMetadata, err := emptyNarrow.Metadata()
	if err != nil {
		t.Fatal(err)
	}
	if got, err := narrowMetadata.Numel(); err != nil || got != 0 {
		t.Fatalf("empty narrow numel = %d, %v; want 0", got, err)
	}
}

func TestEmptyViewPastAllocationEndIsAddressable(t *testing.T) {
	base, err := From([]int32{0, 1, 2, 3, 4}, 5)
	if err != nil {
		t.Fatal(err)
	}
	defer base.Close()

	// The last reachable element is at index 3; the end position is index 6,
	// which lies past the five-element allocation.
	gapped, err := base.Slice(0, 0, 5, 3)
	if err != nil {
		t.Fatal(err)
	}
	defer gapped.Close()

	empty, err := gapped.Slice(0, 2, 2, 1)
	if err != nil {
		t.Fatalf("empty slice past allocation end: %v", err)
	}
	defer empty.Close()
	metadata, err := empty.Metadata()
	if err != nil {
		t.Fatal(err)
	}
	if got, err := metadata.Numel(); err != nil || got != 0 {
		t.Fatalf("empty slice numel = %d, %v; want 0", got, err)
	}
	if got, err := Read[int32](empty); err != nil || len(got) != 0 {
		t.Fatalf("empty slice read = %v, %v; want empty", got, err)
	}
	reshaped, err := empty.Reshape(0)
	if err != nil {
		t.Fatalf("empty slice reshape: %v", err)
	}
	defer reshaped.Close()
	if got, err := Read[int32](reshaped); err != nil || len(got) != 0 {
		t.Fatalf("reshaped empty slice read = %v, %v; want empty", got, err)
	}
}

package tensor

import (
	"errors"
	"reflect"
	"testing"

	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/errs"
)

func TestTensorBinaryRoundTripFromStridedView(t *testing.T) {
	source, err := From([]float32{0, 1, 2, 3, 4, 5}, 2, 3)
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	view, err := source.Transpose()
	if err != nil {
		t.Fatal(err)
	}
	defer view.Close()
	encoded, err := view.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}

	var decoded Tensor
	if err := decoded.UnmarshalBinary(encoded); err != nil {
		t.Fatal(err)
	}
	defer decoded.Close()
	got, err := Read[float32](&decoded)
	if err != nil || !reflect.DeepEqual(got, []float32{0, 3, 1, 4, 2, 5}) {
		t.Fatalf("decoded values = %v, %v", got, err)
	}
	metadata, err := decoded.Metadata()
	if err != nil {
		t.Fatal(err)
	}
	if !metadata.IsContiguous() {
		t.Fatal("decoded tensor should use canonical contiguous storage")
	}
}

func TestTensorBinarySerializationRejectsMalformedInput(t *testing.T) {
	value, err := From([]bool{true, false}, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer value.Close()
	encoded, err := value.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}

	cases := []struct {
		name string
		edit func([]byte)
		want error
	}{
		{"version", func(data []byte) { data[4] = 99 }, errs.ErrUnsupportedSerializationVersion},
		{"dtype", func(data []byte) { data[5] = 255 }, errs.ErrInvalidSerialization},
		{"payload-length", func(data []byte) { data[16] = 3 }, errs.ErrInvalidSerialization},
		{"non-canonical-bool", func(data []byte) { data[24] = 2 }, errs.ErrInvalidSerialization},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			candidate := append([]byte(nil), encoded...)
			test.edit(candidate)
			var target Tensor
			if err := target.UnmarshalBinary(candidate); !errors.Is(err, test.want) {
				t.Fatalf("UnmarshalBinary error = %v, want %v", err, test.want)
			}
			if target.Valid() {
				t.Fatal("failed unmarshal mutated zero Tensor")
			}
		})
	}
	var truncated Tensor
	if err := truncated.UnmarshalBinary(encoded[:len(encoded)-1]); !errors.Is(err, errs.ErrInvalidSerialization) {
		t.Fatalf("truncated payload error = %v", err)
	}
}

func TestTensorUnmarshalRequiresInactiveReceiverAndPreservesIt(t *testing.T) {
	source, err := From([]int32{1, 2, 3}, 3)
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	encoded, err := source.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}

	active, err := From([]int32{9}, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer active.Close()
	if err := active.UnmarshalBinary(encoded); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("active receiver UnmarshalBinary error = %v", err)
	}
	if got, err := Read[int32](active); err != nil || !reflect.DeepEqual(got, []int32{9}) {
		t.Fatalf("active receiver changed after rejected unmarshal: %v, %v", got, err)
	}

	closed, err := From([]int32{0}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := closed.Close(); err != nil {
		t.Fatal(err)
	}
	if err := closed.UnmarshalBinary(encoded); err != nil {
		t.Fatal(err)
	}
	defer closed.Close()
	if got, err := Read[int32](closed); err != nil || !reflect.DeepEqual(got, []int32{1, 2, 3}) {
		t.Fatalf("closed receiver after unmarshal = %v, %v", got, err)
	}
}

func TestTensorSerializationSupportsEmptyAndScalar(t *testing.T) {
	empty, err := NewTensor(dtype.Float64, 0, 3)
	if err != nil {
		t.Fatal(err)
	}
	defer empty.Close()
	scalar, err := From([]float64{3.5})
	if err != nil {
		t.Fatal(err)
	}
	defer scalar.Close()
	for _, value := range []*Tensor{empty, scalar} {
		encoded, err := value.MarshalBinary()
		if err != nil {
			t.Fatal(err)
		}
		var decoded Tensor
		if err := decoded.UnmarshalBinary(encoded); err != nil {
			t.Fatal(err)
		}
		if err := decoded.Close(); err != nil {
			t.Fatal(err)
		}
	}
}

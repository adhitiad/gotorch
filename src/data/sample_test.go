package data

import (
	"errors"
	"testing"

	"github.com/adhitiad/gotorch/src/errs"
)

func TestSampleCopiesFieldsAndSortsNames(t *testing.T) {
	fields := map[string]any{"label": "cat", "id": int64(7)}
	sample, err := NewSample(fields)
	if err != nil {
		t.Fatalf("NewSample: %v", err)
	}
	fields["id"] = int64(99)
	fields["extra"] = true
	if names := sample.Names(); len(names) != 2 || names[0] != "id" || names[1] != "label" {
		t.Fatalf("Names = %v", names)
	}
	value, ok := sample.Value("id")
	if !ok || value != int64(7) {
		t.Fatalf("id = %v, found = %t", value, ok)
	}
	if _, ok := sample.Value("extra"); ok {
		t.Fatal("field dari map asli ikut tersalin")
	}
	if sample.Len() != 2 {
		t.Fatalf("Len = %d, diharapkan 2", sample.Len())
	}
}

func TestSampleValidation(t *testing.T) {
	cases := []map[string]any{
		{"": "value"},
		{"empty": nil},
	}
	for _, fields := range cases {
		if _, err := NewSample(fields); !errors.Is(err, errs.ErrInvalidArgument) {
			t.Fatalf("NewSample(%v) error = %v", fields, err)
		}
	}
}

func TestSampleZeroValueAndClone(t *testing.T) {
	var zero Sample
	if zero.Len() != 0 || len(zero.Names()) != 0 {
		t.Fatalf("zero value memiliki field: %+v", zero)
	}
	if _, ok := zero.Value("missing"); ok {
		t.Fatal("zero value mengembalikan field")
	}
	sample, err := NewSample(map[string]any{"id": int64(1)})
	if err != nil {
		t.Fatalf("NewSample: %v", err)
	}
	clone := sample.Clone()
	names := clone.Names()
	names[0] = "changed"
	if clone.Names()[0] != "id" {
		t.Fatal("Clone berbagi slice Names")
	}
	value, ok := clone.Value("id")
	if !ok || value != int64(1) {
		t.Fatalf("clone id = %v, found = %t", value, ok)
	}
}

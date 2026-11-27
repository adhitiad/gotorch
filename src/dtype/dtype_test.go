package dtype

import "testing"

func TestDTypeSize(t *testing.T) {
	cases := []struct {
		dtype DType
		size  int64
	}{{Bool, 1}, {Int16, 2}, {Float32, 4}, {Complex64, 8}, {Complex128, 16}}
	for _, tc := range cases {
		got, err := tc.dtype.Size()
		if err != nil || got != tc.size {
			t.Fatalf("%s.Size() = %d, %v; want %d", tc.dtype, got, err, tc.size)
		}
	}
}

func TestInvalidDType(t *testing.T) {
	if Invalid.Valid() {
		t.Fatal("Invalid.Valid() = true")
	}
	if _, err := Invalid.Size(); err == nil {
		t.Fatal("Invalid.Size() succeeded")
	}
}

func TestDTypeCategories(t *testing.T) {
	if !Int32.IsInteger() || !Float16.IsFloat() || !Complex128.IsComplex() || !Complex64.IsNumeric() {
		t.Fatal("dtype category classification failed")
	}
	if Bool.IsNumeric() || Bool.IsInteger() {
		t.Fatal("Bool must not be classified as numeric or integer")
	}
}

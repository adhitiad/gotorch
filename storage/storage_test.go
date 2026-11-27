package storage

import (
	"testing"

	"github.com/adhitiad/gotorch/errs"
)

func TestValidateRange(t *testing.T) {
	if err := ValidateRange(16, 8, 8); err != nil {
		t.Fatal(err)
	}
	if err := ValidateRange(16, 17, 0); err != errs.ErrOutOfBounds {
		t.Fatalf("got %v", err)
	}
	if err := ValidateRange(16, 8, 9); err != errs.ErrOutOfBounds {
		t.Fatalf("got %v", err)
	}
	if err := ValidateRange(16, -1, 1); err == nil {
		t.Fatal("negative offset accepted")
	}
}

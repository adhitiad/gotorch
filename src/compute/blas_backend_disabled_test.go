//go:build !gotorch_blas || (!linux && !windows)

package compute

import (
	"errors"
	"testing"

	"github.com/adhitiad/gotorch/src/errs"
)

func TestOpenBLASRequiresOptInBuildTag(t *testing.T) {
	backend, err := NewOpenBLASCPUBackend(OpenBLASConfig{})
	if err != nil || backend != nil {
		t.Fatalf("empty config: backend=%v err=%v, want disabled", backend, err)
	}
	_, err = NewOpenBLASCPUBackend(OpenBLASConfig{LibraryPath: "openblas"})
	if !errors.Is(err, errs.ErrUnsupportedBackend) {
		t.Fatalf("explicit path without build tag: %v", err)
	}
}

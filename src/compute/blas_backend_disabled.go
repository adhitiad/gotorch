//go:build !gotorch_blas || (!linux && !windows)

package compute

import (
	"fmt"

	"github.com/adhitiad/gotorch/src/errs"
)

// NewOpenBLASCPUBackend is unavailable unless built with -tags=gotorch_blas
// on Linux or Windows. An empty path leaves the optional backend disabled.
func NewOpenBLASCPUBackend(config OpenBLASConfig) (Backend, error) {
	if config.LibraryPath == "" {
		return nil, nil
	}
	return nil, fmt.Errorf("%w: OpenBLAS requires -tags=gotorch_blas on Linux or Windows", errs.ErrUnsupportedBackend)
}

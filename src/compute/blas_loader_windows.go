//go:build gotorch_blas && windows

package compute

import (
	"fmt"

	"github.com/ebitengine/purego"
	"golang.org/x/sys/windows"
)

func loadOpenBLAS(path string) (*openBLASAPI, error) {
	handle, err := windows.LoadLibrary(path)
	if err != nil {
		return nil, err
	}
	api := &openBLASAPI{}
	register := func(name string, target any) error {
		address, err := windows.GetProcAddress(handle, name)
		if err != nil {
			return fmt.Errorf("resolve %s: %w", name, err)
		}
		purego.RegisterFunc(target, uintptr(address))
		return nil
	}
	for _, symbol := range []struct {
		name   string
		target any
	}{
		{"cblas_sgemm", &api.sgemm},
		{"cblas_dgemm", &api.dgemm},
		{"openblas_get_num_threads", &api.getThreads},
		{"openblas_set_num_threads", &api.setThreads},
	} {
		if err := register(symbol.name, symbol.target); err != nil {
			_ = windows.FreeLibrary(handle)
			return nil, err
		}
	}
	api.close = func() error { return windows.FreeLibrary(handle) }
	return api, nil
}

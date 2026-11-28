//go:build gotorch_blas && linux

package compute

import (
	"fmt"

	"github.com/ebitengine/purego"
)

func loadOpenBLAS(path string) (*openBLASAPI, error) {
	handle, err := purego.Dlopen(path, purego.RTLD_NOW|purego.RTLD_LOCAL)
	if err != nil {
		return nil, err
	}
	api := &openBLASAPI{}
	register := func(name string, target any) error {
		address, err := purego.Dlsym(handle, name)
		if err != nil {
			return fmt.Errorf("resolve %s: %w", name, err)
		}
		purego.RegisterFunc(target, address)
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
			_ = purego.Dlclose(handle)
			return nil, err
		}
	}
	api.close = func() error { return purego.Dlclose(handle) }
	return api, nil
}

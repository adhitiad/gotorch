//go:build !goexperiment.simd || (!amd64 && !arm64)

package compute

import "testing"

func TestSIMDBackendDisabledBuildUsesScalarFallback(t *testing.T) {
	if got := NewSIMDCPUBackend(); got != nil {
		t.Fatalf("SIMD backend=%v, want nil without supported build configuration", got)
	}
	registry, err := NewDefaultCPURegistry()
	if err != nil {
		t.Fatal(err)
	}
	if names := registry.Backends(); len(names) != 1 || names[0] != "purego-reference" {
		t.Fatalf("backends=%v, want scalar reference only", names)
	}
}

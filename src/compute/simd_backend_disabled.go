//go:build !goexperiment.simd || (!amd64 && !arm64)

package compute

// NewSIMDCPUBackend returns nil when the experimental SIMD API is not enabled
// or the target architecture has no GoTorch SIMD kernel.
func NewSIMDCPUBackend() Backend { return nil }

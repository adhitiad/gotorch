package compute

// NewDefaultCPURegistry registers explicitly configured OpenBLAS and available
// architecture-specific SIMD backends before the mandatory scalar reference backend.
// The reference backend remains the portable fallback for every unsupported kernel.
func NewDefaultCPURegistry() (*Registry, error) {
	registry := NewRegistry()
	blas, err := NewOpenBLASCPUBackend(DefaultOpenBLASConfig())
	if err != nil {
		return nil, err
	}
	if blas != nil {
		if err := registry.Register(blas); err != nil {
			return nil, err
		}
	}
	if backend := NewSIMDCPUBackend(); backend != nil {
		if err := registry.Register(backend); err != nil {
			return nil, err
		}
	}
	if err := registry.Register(NewReferenceCPUBackend()); err != nil {
		return nil, err
	}
	return registry, nil
}

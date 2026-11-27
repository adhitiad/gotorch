package compute

// NewDefaultCPURegistry registers an available architecture-specific SIMD backend
// before the mandatory scalar reference backend. The reference backend is always
// present and handles unsupported operations, layouts, dtypes, small workloads,
// and machines without a supported SIMD feature.
func NewDefaultCPURegistry() (*Registry, error) {
	registry := NewRegistry()
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

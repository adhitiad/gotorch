package compute

import (
	"runtime"

	"golang.org/x/sys/cpu"
)

// CPUFeatures is an immutable snapshot of OS-usable instruction-set capabilities.
// Initialized is false only when no trustworthy feature information is available;
// ARM64's mandatory ASIMD baseline is sufficient to establish a usable snapshot.
type CPUFeatures struct {
	Architecture string
	Initialized  bool
	AVX2         bool
	ASIMD        bool
	AES          bool
}

// DetectCPUFeatures snapshots the current process CPU capabilities. x/sys/cpu
// accounts for OS support required to safely use extended x86 register state.
// On ARM64, x/sys may populate HWCAP or /proc/cpuinfo feature flags while leaving
// cpu.Initialized false; a positively reported ASIMD baseline is sufficient to
// treat that snapshot as initialized.
func DetectCPUFeatures() CPUFeatures {
	return snapshotCPUFeatures(
		runtime.GOARCH,
		cpu.Initialized,
		cpu.X86.HasAVX2,
		cpu.ARM64.HasASIMD,
		cpu.X86.HasAES || cpu.ARM64.HasAES,
	)
}

func snapshotCPUFeatures(architecture string, initialized, avx2, asimd, aes bool) CPUFeatures {
	f := CPUFeatures{Architecture: architecture, Initialized: initialized}

	switch architecture {
	case "amd64":
		if !f.Initialized {
			return f
		}
		f.AVX2 = avx2
		f.AES = aes
	case "arm64":
		// ASIMD is part of the ARM64 baseline. x/sys/cpu can expose it from
		// runtime auxv or its /proc/cpuinfo fallback without setting Initialized.
		f.Initialized = f.Initialized || asimd
		if !f.Initialized {
			return f
		}
		f.ASIMD = asimd
		f.AES = aes
	}

	return f
}

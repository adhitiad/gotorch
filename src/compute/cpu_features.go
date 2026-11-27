package compute

import (
	"runtime"

	"golang.org/x/sys/cpu"
)

// CPUFeatures is an immutable snapshot of OS-usable instruction-set capabilities.
// A false Initialized value means detection could not be completed; callers must
// conservatively select the scalar backend.
type CPUFeatures struct {
	Architecture string
	Initialized  bool
	AVX2         bool
	ASIMD        bool
	AES          bool
}

// DetectCPUFeatures snapshots the current process CPU capabilities. x/sys/cpu
// accounts for OS support required to safely use extended x86 register state.
func DetectCPUFeatures() CPUFeatures {
	f := CPUFeatures{Architecture: runtime.GOARCH, Initialized: cpu.Initialized}
	switch runtime.GOARCH {
	case "amd64":
		f.AVX2 = cpu.X86.HasAVX2
		f.AES = cpu.X86.HasAES
	case "arm64":
		f.ASIMD = cpu.ARM64.HasASIMD
		f.AES = cpu.ARM64.HasAES
	}
	return f
}

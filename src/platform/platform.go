// Package platform exposes GoTorch's supported operating-system and architecture policy.
package platform

import goruntime "runtime"

// SupportLevel describes GoTorch's validation commitment for a target.
type SupportLevel uint8

const (
	Unsupported SupportLevel = iota
	Compatibility
	Primary
)

func (s SupportLevel) String() string {
	switch s {
	case Primary:
		return "primary"
	case Compatibility:
		return "compatibility"
	default:
		return "unsupported"
	}
}

// Target identifies an operating-system and CPU-architecture pair.
type Target struct {
	OS   string
	Arch string
}

// Current returns the target on which the current process was built.
func Current() Target { return Target{OS: goruntime.GOOS, Arch: goruntime.GOARCH} }

// SupportLevel returns the project's declared support commitment for t.
func (t Target) SupportLevel() SupportLevel {
	switch {
	case t.OS == "windows" && t.Arch == "amd64":
		return Primary
	case t.OS == "linux" && (t.Arch == "amd64" || t.Arch == "arm64"):
		return Compatibility
	default:
		return Unsupported
	}
}

// Supported reports whether t is a primary or compatibility target.
func (t Target) Supported() bool { return t.SupportLevel() != Unsupported }

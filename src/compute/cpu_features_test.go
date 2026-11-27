package compute

import (
	"runtime"
	"testing"
)

func TestDetectCPUFeaturesIsConservativeAndIdentifiesArchitecture(t *testing.T) {
	got := DetectCPUFeatures()
	if got.Architecture != runtime.GOARCH {
		t.Fatalf("architecture=%q want %q", got.Architecture, runtime.GOARCH)
	}
	if !got.Initialized && (got.AVX2 || got.ASIMD || got.AES) {
		t.Fatal("uninitialized detection exposed usable features")
	}
	if runtime.GOARCH != "amd64" && got.AVX2 {
		t.Fatal("AVX2 reported outside amd64")
	}
	if runtime.GOARCH != "arm64" && got.ASIMD {
		t.Fatal("ASIMD reported outside arm64")
	}
}

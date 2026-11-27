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

func TestSnapshotCPUFeaturesHandlesARM64FallbackInitialization(t *testing.T) {
	t.Run("ARM64 baseline proves usable snapshot", func(t *testing.T) {
		got := snapshotCPUFeatures("arm64", false, false, true, true)
		if !got.Initialized || !got.ASIMD || !got.AES {
			t.Fatalf("snapshot=%+v, want initialized ARM64 ASIMD/AES snapshot", got)
		}
	})

	t.Run("ARM64 without feature evidence remains conservative", func(t *testing.T) {
		got := snapshotCPUFeatures("arm64", false, false, false, true)
		if got.Initialized || got.ASIMD || got.AES {
			t.Fatalf("snapshot=%+v, want uninitialized scalar-only snapshot", got)
		}
	})

	t.Run("AMD64 does not infer initialization from feature bits", func(t *testing.T) {
		got := snapshotCPUFeatures("amd64", false, true, false, true)
		if got.Initialized || got.AVX2 || got.AES {
			t.Fatalf("snapshot=%+v, want uninitialized scalar-only snapshot", got)
		}
	})
}

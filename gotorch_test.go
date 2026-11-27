package gotorch

import "testing"

func TestVersionIsDevelopmentVersion(t *testing.T) {
	if Version == "" {
		t.Fatal("Version must not be empty")
	}
}

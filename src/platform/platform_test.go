package platform

import "testing"

func TestSupportPolicy(t *testing.T) {
	cases := []struct {
		target Target
		want   SupportLevel
	}{
		{Target{OS: "windows", Arch: "amd64"}, Primary},
		{Target{OS: "linux", Arch: "amd64"}, Compatibility},
		{Target{OS: "linux", Arch: "arm64"}, Compatibility},
		{Target{OS: "darwin", Arch: "arm64"}, Unsupported},
	}
	for _, tc := range cases {
		if got := tc.target.SupportLevel(); got != tc.want {
			t.Fatalf("%+v: got %s, want %s", tc.target, got, tc.want)
		}
	}
}

func TestCurrentTarget(t *testing.T) {
	target := Current()
	if target.OS == "" || target.Arch == "" {
		t.Fatalf("incomplete target: %+v", target)
	}
}

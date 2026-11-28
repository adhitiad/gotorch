package memory

import (
	"context"
	"errors"
	"testing"

	"github.com/adhitiad/gotorch/src/errs"
)

func TestWorkspaceLifecycle(t *testing.T) {
	workspace := NewWorkspace()
	first, err := workspace.Add("activations", 64)
	if err != nil {
		t.Fatalf("Add first: %v", err)
	}
	if _, err := workspace.Add("activations", 64); !errors.Is(err, ErrWorkspaceDuplicate) {
		t.Fatalf("duplicate Add: got %v", err)
	}
	if _, err := workspace.Add(" ", 64); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("empty name: got %v", err)
	}
	second, err := workspace.Add("scratch", 32)
	if err != nil {
		t.Fatalf("Add second: %v", err)
	}
	found, err := workspace.Arena("scratch")
	if err != nil || found != second {
		t.Fatalf("Arena: %v, %v", found, err)
	}
	if _, err := workspace.Arena("missing"); !errors.Is(err, ErrWorkspaceMissing) {
		t.Fatalf("missing Arena: got %v", err)
	}

	handle, err := first.Allocate(context.Background(), 16)
	if err != nil {
		t.Fatalf("Allocate: %v", err)
	}
	if err := workspace.Reset(); !errors.Is(err, ErrArenaLive) {
		t.Fatalf("Reset with live handle: got %v", err)
	}
	if err := handle.Release(); err != nil {
		t.Fatalf("Release: %v", err)
	}
	if err := workspace.Reset(); err != nil {
		t.Fatalf("Reset: %v", err)
	}
	stats := workspace.Stats()
	if stats.Arenas != 2 || stats.Capacity != 96 || stats.Used != 0 || stats.ActiveBytes != 0 {
		t.Fatalf("stats = %+v", stats)
	}
	if err := workspace.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if !workspace.Stats().Closed {
		t.Fatal("workspace should be closed")
	}
	if _, err := workspace.Arena("scratch"); !errors.Is(err, ErrWorkspaceClosed) {
		t.Fatalf("Arena after close: got %v", err)
	}
	if _, err := workspace.Add("late", 8); !errors.Is(err, ErrWorkspaceClosed) {
		t.Fatalf("Add after close: got %v", err)
	}
}

func TestWorkspaceCloseWithLiveHandle(t *testing.T) {
	workspace := NewWorkspace()
	arena, err := workspace.Add("live", 16)
	if err != nil {
		t.Fatalf("Add: %v", err)
	}
	handle, err := arena.Allocate(context.Background(), 8)
	if err != nil {
		t.Fatalf("Allocate: %v", err)
	}
	if err := workspace.Close(); !errors.Is(err, ErrArenaLive) {
		t.Fatalf("Close with live handle: got %v", err)
	}
	if workspace.Stats().Closed {
		t.Fatal("workspace must remain open when Close fails")
	}
	if err := handle.Release(); err != nil {
		t.Fatalf("Release: %v", err)
	}
	if err := workspace.Close(); err != nil {
		t.Fatalf("retry Close: %v", err)
	}
}

func TestWorkspaceInvalidReceivers(t *testing.T) {
	var workspace *Workspace
	if _, err := workspace.Add("x", 8); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("nil Add: got %v", err)
	}
	if _, err := workspace.Arena("x"); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("nil Arena: got %v", err)
	}
	if err := workspace.Reset(); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("nil Reset: got %v", err)
	}
	if err := workspace.Close(); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("nil Close: got %v", err)
	}
	if stats := workspace.Stats(); stats.Arenas != 0 {
		t.Fatalf("nil Stats = %+v", stats)
	}
}

package memory

import (
	"errors"
	"fmt"
	"strings"
	"sync"

	"github.com/adhitiad/gotorch/src/errs"
)

// Workspace mengelola sejumlah arena bernama dengan reset dan penutupan yang
// deterministik. Reset menolak operasi bila arena mana pun masih memiliki handle hidup.
type Workspace struct {
	mu     sync.Mutex
	closed bool
	arenas map[string]*Arena
	order  []string
}

// NewWorkspace membuat workspace kosong.
func NewWorkspace() *Workspace {
	return &Workspace{arenas: make(map[string]*Arena)}
}

// Add membuat arena baru dengan nama unik di workspace.
func (w *Workspace) Add(name string, capacity int64) (*Arena, error) {
	if w == nil || strings.TrimSpace(name) == "" {
		return nil, errs.ErrInvalidArgument
	}
	arena, err := NewArena(capacity)
	if err != nil {
		return nil, err
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return nil, ErrWorkspaceClosed
	}
	if _, exists := w.arenas[name]; exists {
		return nil, fmt.Errorf("%w: %q", ErrWorkspaceDuplicate, name)
	}
	w.arenas[name] = arena
	w.order = append(w.order, name)
	return arena, nil
}

// Arena mengambil arena berdasarkan nama.
func (w *Workspace) Arena(name string) (*Arena, error) {
	if w == nil || strings.TrimSpace(name) == "" {
		return nil, errs.ErrInvalidArgument
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return nil, ErrWorkspaceClosed
	}
	arena, ok := w.arenas[name]
	if !ok {
		return nil, fmt.Errorf("%w: %q", ErrWorkspaceMissing, name)
	}
	return arena, nil
}

// Reset mengosongkan seluruh arena bila tidak ada alokasi hidup di workspace.
func (w *Workspace) Reset() error {
	if w == nil {
		return errs.ErrInvalidArgument
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return ErrWorkspaceClosed
	}
	for _, name := range w.order {
		if w.arenas[name].Stats().Active > 0 {
			return fmt.Errorf("%w: arena %q", ErrArenaLive, name)
		}
	}
	var result []error
	for _, name := range w.order {
		if err := w.arenas[name].Reset(); err != nil {
			result = append(result, fmt.Errorf("reset arena %q: %w", name, err))
		}
	}
	return errors.Join(result...)
}

// Close menutup arena secara urutan terbalik dan mengembalikan error gabungan.
func (w *Workspace) Close() error {
	if w == nil {
		return errs.ErrInvalidArgument
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return nil
	}
	var result []error
	for index := len(w.order) - 1; index >= 0; index-- {
		name := w.order[index]
		if err := w.arenas[name].Close(); err != nil {
			result = append(result, fmt.Errorf("close arena %q: %w", name, err))
		}
	}
	if len(result) == 0 {
		w.closed = true
	}
	return errors.Join(result...)
}

// Stats mengembalikan agregat kapasitas, pemakaian, dan alokasi hidup workspace.
func (w *Workspace) Stats() WorkspaceStats {
	if w == nil {
		return WorkspaceStats{}
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	stats := WorkspaceStats{Arenas: len(w.order), Closed: w.closed}
	for _, name := range w.order {
		arena := w.arenas[name].Stats()
		stats.Capacity += arena.Capacity
		stats.Used += arena.Used
		stats.ActiveBytes += arena.ActiveBytes
	}
	return stats
}

// WorkspaceStats adalah snapshot agregat workspace.
type WorkspaceStats struct {
	Arenas      int
	Capacity    int64
	Used        int64
	ActiveBytes int64
	Closed      bool
}

// Package memory menyediakan pengelolaan memori dengan kepemilikan eksplisit,
// penggunaan ulang buffer, arena, workspace, dan statistik yang deterministik.
package memory

import "errors"

// ErrMemoryLimit dikembalikan saat anggaran memori tidak memungkinkan alokasi.
var ErrMemoryLimit = errors.New("gotorch: memory limit exceeded")

// ErrPoolClosed dikembalikan saat pool tidak menerima alokasi baru.
var ErrPoolClosed = errors.New("gotorch: memory pool is closed")

// ErrArenaClosed dikembalikan saat arena sudah ditutup.
var ErrArenaClosed = errors.New("gotorch: arena is closed")

// ErrArenaFull dikembalikan saat arena tidak punya kapasitas tersisa.
var ErrArenaFull = errors.New("gotorch: arena capacity exceeded")

// ErrArenaLive dikembalikan saat reset/close dilakukan dengan alokasi masih hidup.
var ErrArenaLive = errors.New("gotorch: arena has live allocations")

// ErrWorkspaceClosed dikembalikan saat workspace sudah ditutup.
var ErrWorkspaceClosed = errors.New("gotorch: workspace is closed")

// ErrWorkspaceDuplicate dikembalikan saat nama arena sudah dipakai.
var ErrWorkspaceDuplicate = errors.New("gotorch: duplicate workspace arena name")

// ErrWorkspaceMissing dikembalikan saat arena tidak ditemukan di workspace.
var ErrWorkspaceMissing = errors.New("gotorch: workspace arena not found")

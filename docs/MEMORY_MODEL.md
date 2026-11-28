# Memory and Ownership Model

## Memory classes

- Parameter storage: persistent for model/session lifetime.
- User input storage: owned by caller or explicitly transferred.
- Output storage: ownership and validity documented by API.
- Activation storage: temporary for one execution/training step.
- Workspace: reusable scratch with explicit completion/lifetime.
- Distributed staging: buffers used for serialization/communication.

## Rules

- Views retain backing storage.
- A buffer cannot be reused while any view or asynchronous operation references it.
- Pooling is an optimization, not an ownership model.
- `sync.Pool` is opportunistic and may be cleared by GC; do not use it for deterministic resource accounting.
- Arena reset is legal only after all allocations from the arena are dead.
- Track allocated, active, cached, and peak bytes separately.
- Define behavior under allocation failure and memory pressure.

## P4 implementation

- `memory.Pool` implements `storage.Allocator` with an explicit cache keyed by exact byte size. It never uses `sync.Pool`, so active, allocated, cached and peak bytes are deterministic.
- `Pool.Allocate` returns zero-initialized storage. `Retain` creates an independent handle, and the backing buffer returns to the cache only after the final handle is released. A reused buffer is zeroed before it becomes visible again.
- `PoolConfig.MaxCachedBytes` caps cached bytes, and `MaxTotalBytes` caps all bytes counted by the pool including cached buffers. A zero limit means no bound for that dimension; the default cache is 64 MiB. Exceeding a limit returns `ErrMemoryLimit` instead of growing without bound.
- `memory.Arena` is a fixed-capacity bump allocator with 8-byte alignment. `Reset` and `Close` return `ErrArenaLive` while any handle from the arena is alive; they never invalidate a live handle.
- `memory.Workspace` owns named arenas. `Reset` validates every arena before clearing any of them, and `Close` stops arenas in reverse registration order. A failed `Close` leaves the workspace open so the caller can release handles and retry.
- `PoolStats`, `ArenaStats` and `WorkspaceStats` report allocated, active, cached and peak byte counts. Cache reuse and rejection counters make pressure and backpressure observable.
- Allocation failure, cancellation and closed lifecycles are ordinary errors (`ErrMemoryLimit`, `ErrPoolClosed`, `ErrArenaClosed`, `ErrArenaFull`, `ErrArenaLive`, `ErrWorkspaceClosed`); no invalid user input panics.

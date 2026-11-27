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

# Tensor and Memory Rules

- Specify shape, strides, dtype, layout, device/placement, and storage ownership.
- Distinguish owning tensors from views; define aliasing and mutation semantics.
- Check `numel` and byte-size arithmetic for integer overflow.
- Keep persistent parameters separate from temporary execution buffers.
- A buffer may be reused only after all consumers have completed.
- Never expose an arena or pool buffer beyond its lease/lifetime.
- Define zero-sized tensor behavior and scalar/rank-0 behavior explicitly.
- Test non-contiguous views, negative/invalid dimensions, and overlapping aliases.

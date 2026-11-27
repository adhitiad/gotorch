# Core Contracts

This document records contracts that must be settled before implementation expands.

## Tensor contract

A tensor must define dtype, shape, strides, layout, storage, ownership, mutability, and placement. Views share storage and must retain its lifetime. Shape and byte-size arithmetic must detect overflow. Rank-0 and zero-sized tensor semantics must be explicit.

## Kernel contract

A kernel declares supported operator, dtype, layout, shape constraints, aliasing rules, and determinism. The Pure Go implementation is the correctness reference. Optimized kernels are selected only after capability and compatibility checks.

## Execution contract

Execution accepts cancellation, uses bounded resources, and has defined output ownership. Concurrent calls must not mutate shared model parameters or scratch buffers unsafely.

## Memory contract

Persistent model parameters, user-owned inputs/outputs, and temporary execution buffers have distinct lifetimes. Reuse is allowed only after all consumers complete.

## Distributed contract

Distributed values identify placement and logical partitioning. Collective operations define ranks, ordering, shape compatibility, timeout, cancellation, and failure behavior. Local execution must not incur distributed network overhead.

## Public API contract

Public APIs must document error behavior, concurrency safety, mutation, ownership, dtype conversion, and compatibility guarantees.

# Distributed Systems Rules

- Define rank, world size, worker identity, membership, and run/session identity.
- Every RPC/collective needs deadlines, cancellation, and bounded payload sizes.
- Specify ordering and semantics for AllReduce, AllGather, ReduceScatter, and Broadcast.
- Avoid unbounded retries; use retry budgets and idempotency where applicable.
- Distinguish worker failure from slow worker and network partition.
- Checkpoint metadata must include model, optimizer, scheduler, RNG, and progress state as applicable.
- State consistency and recovery semantics must be documented before implementation.
- Never assume exactly-once delivery from ordinary network transports.
- Authenticate cluster peers and protect transport; do not expose unauthenticated worker control endpoints.

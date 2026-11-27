# Skill: Implement a Distributed Feature

1. Define worker/rank identity, world size, topology, and membership lifecycle.
2. Specify message schema, payload limits, authentication, deadlines, and cancellation.
3. Define ordering, consistency, retry, idempotency, and failure semantics.
4. Implement a local/in-process reference path when feasible.
5. Add multi-process integration tests before multi-node complexity.
6. Test worker crash, network timeout, duplicate message, partial collective, and shutdown.
7. Measure communication overhead separately from compute time.
8. Document operational requirements and recovery guarantees.

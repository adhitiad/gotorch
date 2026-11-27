# ADR-0003: Distributed by Design, Local Fast Path

- Status: Accepted
- Date: 2026-11-27

## Decision

Distributed identity, placement, serialization, and execution contracts are considered from the foundation phase. Ordinary local tensor operations remain in-process and must not incur network or coordinator overhead.

## Consequences

- Tensor and execution metadata need an extensible placement model.
- Distributed collectives are implemented after local tensor/autograd/training semantics stabilize.
- Local and distributed execution share semantics but may use different planners and transports.

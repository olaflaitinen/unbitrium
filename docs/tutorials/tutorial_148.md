# Tutorial 148: FL System Design Patterns

This tutorial covers common design patterns for FL systems.

## Patterns

1. **Actor Model**: Clients as actors.
2. **Event Sourcing**: Log all updates.
3. **CQRS**: Separate read/write paths.
4. **Circuit Breaker**: Handle client failures.

## Architecture

```mermaid
graph TD
    O[Orchestrator]
    Q[Message Queue]
    W1[Worker 1]
    W2[Worker 2]
    S[State Store]

    O --> Q
    Q --> W1
    Q --> W2
    W1 --> S
    W2 --> S
```

## Exercises

1. Choosing patterns for cross-device vs cross-silo.
2. Fault tolerance considerations.

# Tutorial 141: Profiling FL Experiments

This tutorial covers performance profiling.

## Metrics to Profile

- Client compute time.
- Communication latency.
- Server aggregation time.
- Memory usage.

## Tools

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()
# ... run simulation ...
profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumtime')
stats.print_stats(20)
```

## Exercises

1. Identifying bottlenecks.
2. Optimizing for throughput vs latency.

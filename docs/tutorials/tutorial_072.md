# Tutorial 072: Profiling Memory Usage

## Overview
How much RAM does my simulation need?

## Calculation
$$ \text{Mem} \approx N_{clients} \times \text{Size}_{dataset} + N_{threads} \times \text{Size}_{model} $$
(If using efficient in-memory dataset, otherwise disk pointers).

## Tool
Use `memory_profiler`.

## Command
```bash
mprof run python -m unbitrium.bench.run ...
mprof plot
```

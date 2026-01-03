# Tutorial 044: Artifact Management

## Overview
Where do the models go?

## Directory Structure
```
results/
  exp_name/
    run_0/
        manifest.json
        events.jsonl
        final_model.pt
        checkpoints/
            round_50.pt
```

## Policy
You can configure `checkpoint_frequency` (e.g., every 10 rounds) to save disk space.

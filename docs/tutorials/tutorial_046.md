# Tutorial 046: Provenance Tracking

## Overview
How do I know exactly what code produced this result 2 years later?

## `manifest.json`
Every run captures:
- Git Commit Hash (`git rev-parse HEAD`)
- Git Diff (if uncommitted changes exist)
- Full environment (pip freeze)
- CLI arguments

## Tutorial
Inspect a generated `manifest.json`.

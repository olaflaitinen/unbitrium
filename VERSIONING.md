# Versioning Policy

This document describes the versioning strategy and release process for Unbitrium.

---

## Table of Contents

1. [Semantic Versioning](#semantic-versioning)
2. [Version Format](#version-format)
3. [Version Increments](#version-increments)
4. [Pre-release Versions](#pre-release-versions)
5. [Deprecation Policy](#deprecation-policy)
6. [Compatibility](#compatibility)
7. [Release Process](#release-process)

---

## Semantic Versioning

Unbitrium follows [Semantic Versioning 2.0.0](https://semver.org/) (SemVer).

Given a version number **MAJOR.MINOR.PATCH**, increment the:

| Component | When to Increment |
|-----------|-------------------|
| **MAJOR** | Incompatible API changes |
| **MINOR** | Backward-compatible new functionality |
| **PATCH** | Backward-compatible bug fixes |

---

## Version Format

### Standard Format

```
MAJOR.MINOR.PATCH
```

Examples:
- `1.0.0` - First stable release
- `1.1.0` - New features added
- `1.1.1` - Bug fix release

### Pre-release Format

```
MAJOR.MINOR.PATCH-PRERELEASE
```

Examples:
- `2.0.0-alpha.1` - First alpha release
- `2.0.0-beta.3` - Third beta release
- `2.0.0-rc.1` - First release candidate

### Build Metadata

```
MAJOR.MINOR.PATCH+BUILD
```

Examples:
- `1.0.0+20260103` - Build date
- `1.0.0+git.abc1234` - Git commit

---

## Version Increments

### MAJOR Version (X.0.0)

Increment when making incompatible API changes:

- Removing public API functions or classes
- Changing function signatures in breaking ways
- Changing return types
- Removing or renaming parameters
- Changing default behavior significantly

```python
# Before (1.x)
def aggregate(updates: list) -> dict:
    pass

# After (2.0) - Breaking change
def aggregate(updates: list, weights: list) -> Model:
    pass
```

### MINOR Version (1.X.0)

Increment when adding backward-compatible functionality:

- New public functions or classes
- New optional parameters with defaults
- New features that don't break existing code
- Deprecating (but not removing) API

```python
# Before (1.0)
def aggregate(updates: list) -> dict:
    pass

# After (1.1) - Backward compatible
def aggregate(updates: list, momentum: float = 0.0) -> dict:
    pass
```

### PATCH Version (1.0.X)

Increment for backward-compatible bug fixes:

- Bug fixes that don't change API
- Performance improvements
- Documentation updates
- Internal refactoring

---

## Pre-release Versions

### Alpha (X.Y.Z-alpha.N)

- Early development stage
- API may change significantly
- Not feature complete
- May have bugs

### Beta (X.Y.Z-beta.N)

- Feature complete
- API mostly stable
- Testing in progress
- Some bugs expected

### Release Candidate (X.Y.Z-rc.N)

- Production ready candidate
- API frozen
- Final testing phase
- Minimal changes expected

### Precedence

```
1.0.0-alpha.1 < 1.0.0-alpha.2 < 1.0.0-beta.1 < 1.0.0-rc.1 < 1.0.0
```

---

## Deprecation Policy

### Timeline

| Phase | Duration | Action |
|-------|----------|--------|
| Deprecation | 1 minor version | Add deprecation warning |
| Removal | Next major version | Remove deprecated code |

### Deprecation Warnings

```python
import warnings

def old_function():
    """Deprecated: Use new_function instead."""
    warnings.warn(
        "old_function is deprecated, use new_function instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return new_function()
```

### Documentation

Deprecated features are marked in:

- Docstrings
- API documentation
- CHANGELOG.md

---

## Compatibility

### Python Version Support

| Unbitrium | Python |
|-----------|--------|
| 1.x | >= 3.10 |
| 2.x | >= 3.11 |

### Dependency Compatibility

Core dependencies follow compatible release constraints:

```toml
[project]
dependencies = [
    "torch>=2.0",
    "numpy>=2.0",
    "scipy>=1.12",
]
```

### PyTorch Compatibility

| Unbitrium | PyTorch |
|-----------|---------|
| 1.0.x | >= 2.0 |
| 1.1.x | >= 2.1 |
| 2.0.x | >= 2.4 |

---

## Release Process

### Release Checklist

- [ ] All tests passing
- [ ] Coverage requirements met
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped in:
  - [ ] `src/unbitrium/__init__.py`
  - [ ] `pyproject.toml`
  - [ ] `CITATION.cff`
- [ ] Git tag created
- [ ] Package published to PyPI
- [ ] GitHub release created
- [ ] Announcement posted

### Version Locations

Update version in these files:

| File | Location |
|------|----------|
| `src/unbitrium/__init__.py` | `__version__ = "X.Y.Z"` |
| `pyproject.toml` | `version = "X.Y.Z"` |
| `CITATION.cff` | `version: X.Y.Z` |

### Git Tags

```bash
# Create annotated tag
git tag -a v1.0.0 -m "Release version 1.0.0"

# Push tag
git push origin v1.0.0
```

### PyPI Release

```bash
# Build package
python -m build

# Upload to PyPI
twine upload dist/*
```

---

## Version History

See [CHANGELOG.md](CHANGELOG.md) for complete version history.

### Current Version

- **1.0.0** (January 2026)
  - First stable production release
  - Complete API documentation
  - 200+ tutorials

---

*Last updated: January 2026*

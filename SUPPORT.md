# Support

This document describes how to get help with Unbitrium.

---

## Table of Contents

1. [Documentation](#documentation)
2. [Getting Help](#getting-help)
3. [Issue Tracker](#issue-tracker)
4. [Community Resources](#community-resources)
5. [Professional Support](#professional-support)
6. [Frequently Asked Questions](#frequently-asked-questions)

---

## Documentation

Before seeking help, please consult the available documentation:

| Resource | Description | Link |
|----------|-------------|------|
| User Guide | Installation and quick start | [Documentation](https://olaflaitinen.github.io/unbitrium/) |
| API Reference | Complete API documentation | [API Docs](https://olaflaitinen.github.io/unbitrium/api/) |
| Tutorials | 200+ comprehensive tutorials | [Tutorials](docs/tutorials/) |
| Examples | Working code examples | [Examples](examples/) |
| FAQ | Common questions | Below |

---

## Getting Help

### Step 1: Search Existing Resources

Before asking a question:

1. Search the [documentation](https://olaflaitinen.github.io/unbitrium/)
2. Review the [tutorials](docs/tutorials/)
3. Search [existing issues](https://github.com/olaflaitinen/unbitrium/issues)
4. Check the FAQ below

### Step 2: Prepare Your Question

When asking for help, include:

| Information | Description |
|-------------|-------------|
| Unbitrium version | `pip show unbitrium` |
| Python version | `python --version` |
| PyTorch version | `python -c "import torch; print(torch.__version__)"` |
| Operating system | Windows, macOS, Linux |
| Code example | Minimal reproducible example |
| Error message | Complete error traceback |
| Expected behavior | What you expected to happen |

### Step 3: Choose the Right Channel

| Channel | Use Case |
|---------|----------|
| GitHub Issues | Bug reports, feature requests |
| GitHub Discussions | Questions, ideas, show-and-tell |
| Email | Private inquiries, collaborations |

---

## Issue Tracker

### Bug Reports

If you've found a bug:

1. Go to [GitHub Issues](https://github.com/olaflaitinen/unbitrium/issues)
2. Click "New Issue"
3. Select "Bug Report" template
4. Fill in all required fields
5. Submit the issue

**Required information:**
- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment details
- Minimal code example

### Feature Requests

If you have an idea for a new feature:

1. Go to [GitHub Issues](https://github.com/olaflaitinen/unbitrium/issues)
2. Click "New Issue"
3. Select "Feature Request" template
4. Describe your use case
5. Submit the request

---

## Community Resources

### GitHub Repository

| Resource | Link |
|----------|------|
| Main Repository | [github.com/olaflaitinen/unbitrium](https://github.com/olaflaitinen/unbitrium) |
| Issues | [Issues](https://github.com/olaflaitinen/unbitrium/issues) |
| Discussions | [Discussions](https://github.com/olaflaitinen/unbitrium/discussions) |
| Releases | [Releases](https://github.com/olaflaitinen/unbitrium/releases) |

### Research Resources

| Resource | Link |
|----------|------|
| Bibliography | [docs/references/bibliography.md](docs/references/bibliography.md) |
| Research Notes | [docs/research/notes.md](docs/research/notes.md) |
| Validation | [docs/validation/](docs/validation/) |

---

## Professional Support

### Academic Collaborations

For research collaborations or academic inquiries, contact:

| Contact | Details |
|---------|---------|
| Name | Olaf Yunus Laitinen Imanov |
| Email | <oyli@dtu.dk> |
| Institution | Technical University of Denmark (DTU) |
| Department | DTU Compute |

### Commercial Support

Unbitrium is an academic research project and does not offer commercial support packages at this time.

---

## Frequently Asked Questions

### Installation

**Q: How do I install Unbitrium?**

```bash
pip install unbitrium
```

For development installation:
```bash
git clone https://github.com/olaflaitinen/unbitrium.git
cd unbitrium
pip install -e ".[dev]"
```

**Q: Which Python versions are supported?**

Python 3.10 and later are supported. Python 3.12+ is recommended.

**Q: Is GPU support required?**

No. Unbitrium works with CPU-only installations, but GPU support (via PyTorch CUDA) is recommended for large-scale experiments.

### Usage

**Q: How do I create a non-IID partition?**

```python
from unbitrium.partitioning import DirichletPartitioner

partitioner = DirichletPartitioner(
    num_clients=100,
    alpha=0.5,  # Lower = more heterogeneous
    seed=42,
)
client_indices = partitioner.partition(labels)
```

**Q: How do I use FedAvg aggregation?**

```python
from unbitrium.aggregators import FedAvg

aggregator = FedAvg()
new_model, metrics = aggregator.aggregate(client_updates, global_model)
```

**Q: How do I compute heterogeneity metrics?**

```python
from unbitrium.metrics import compute_label_entropy, compute_emd

entropy = compute_label_entropy(labels, client_indices)
emd = compute_emd(labels, client_indices)
```

### Troubleshooting

**Q: I get an ImportError when importing unbitrium.**

Ensure you have installed the package:
```bash
pip install unbitrium
```

If installing from source:
```bash
pip install -e "."
```

**Q: My simulation is slow.**

Consider:
1. Reducing the number of clients or rounds
2. Using GPU acceleration
3. Reducing local epochs
4. Using smaller batch sizes

**Q: I get CUDA out of memory errors.**

Try:
1. Reducing batch size
2. Reducing model size
3. Using `torch.cuda.empty_cache()`
4. Running on CPU for debugging

---

## Response Times

| Channel | Expected Response |
|---------|-------------------|
| GitHub Issues | Within 7 days |
| Security Issues | Within 48 hours |
| Email | Within 14 days |

*Note: Response times are best-effort and may vary based on workload.*

---

## Contact

For questions not covered above:

| Contact | Email |
|---------|-------|
| Olaf Yunus Laitinen Imanov | <oyli@dtu.dk> |

---

*Last updated: January 2026*

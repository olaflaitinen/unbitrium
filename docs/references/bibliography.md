# Bibliography

This document lists the key references for federated learning research that informed the design of Unbitrium.

---

## Foundational Papers

### FedAvg

**McMahan, H. B., Moore, E., Ramage, D., Hampson, S., and Arcas, B. A. y.** (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. *Proceedings of AISTATS*.

The foundational paper introducing Federated Averaging (FedAvg), demonstrating that averaging locally-trained models can produce a global model competitive with centralized training.

---

### FedProx

**Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., and Smith, V.** (2020). Federated Optimization in Heterogeneous Networks. *MLSys*.

Introduces proximal regularization to handle statistical heterogeneity by penalizing deviation from the global model.

---

### FedDyn

**Acar, D. A. E., Zhao, Y., Navarro, R. M., Mattina, M., Whatmough, P. N., and Sabar, V.** (2021). Federated Learning Based on Dynamic Regularization. *ICLR*.

Proposes dynamic regularization using dual variables to achieve improved convergence under heterogeneity.

---

### SCAFFOLD

**Karimireddy, S. P., Kale, S., Mohri, M., Reddi, S. J., Stich, S. U., and Suresh, A. T.** (2020). SCAFFOLD: Stochastic Controlled Averaging for Federated Learning. *ICML*.

Introduces control variates for variance reduction, achieving communication-efficient convergence.

---

## Byzantine Robustness

### Krum

**Blanchard, P., Mhamdi, E. M. E., Guerraoui, R., and Stainer, J.** (2017). Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent. *NeurIPS*.

Introduces robust aggregation methods including Krum and Multi-Krum.

---

### Trimmed Mean and Median

**Yin, D., Chen, Y., Kannan, R., and Bartlett, P.** (2018). Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates. *ICML*.

Theoretical analysis of coordinate-wise trimmed mean and median aggregators.

---

## Benchmarks

### LEAF

**Caldas, S., Duddu, S. M. K., Wu, P., Li, T., Konecny, J., McMahan, H. B., Smith, V., and Talwalkar, A.** (2018). LEAF: A Benchmark for Federated Settings. *arXiv:1812.01097*.

Standard benchmark suite for federated learning research.

---

## Adaptive Optimization

### FedOpt

**Reddi, S., Charles, Z., Zaheer, M., Garrett, Z., Rush, K., Konecny, J., Kumar, S., and McMahan, H. B.** (2021). Adaptive Federated Optimization. *ICLR*.

Extends server-side momentum and adaptive learning rates to federated settings.

---

## Non-IID Data

### Measuring Heterogeneity

**Hsu, T.-M. H., Qi, H., and Brown, M.** (2019). Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification. *arXiv:1909.06335*.

Systematic study of Dirichlet partitioning and its effects on federated learning.

---

## Citation Format

All references are available in BibTeX format in `bibliography.bib` and CSL-JSON format in `bibliography.json`.

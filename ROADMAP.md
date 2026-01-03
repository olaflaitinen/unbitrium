# Roadmap

This document outlines the detailed development roadmap for Unbitrium from version 1.0.0 through 2.0.0.
All milestones are tentative and subject to research priorities.

---

## [1.1.0] - Privacy and Security Hardening

### 1.1.0.1 - RDP Foundation
- [ ] Create `RenyiAccountant` base class with moment generation.
- [ ] Add unit tests for basic RDP accounting.

### 1.1.0.2 - RDP Composition
- [ ] Implement sequential composition for RDP.
- [ ] Add heterogeneous composition support.

### 1.1.0.3 - RDP to DP Conversion
- [ ] Implement optimal RDP to (epsilon, delta)-DP conversion.
- [ ] Add numerical stability improvements.

### 1.1.0.4 - Vectorized RDP
- [ ] Implement `VectorizedGaussianMechanism` for batch accounting.
- [ ] Benchmark performance improvement (target: 10x).

### 1.1.0.5 - Autograd Integration
- [ ] Add `autograd` hooks for automatic privacy tracking.
- [ ] Integrate with PyTorch backward pass.

### 1.1.1.0 - Subsampled Gaussian
- [ ] Implement `SubsampledGaussianMechanism` with Poisson sampling.
- [ ] Add privacy amplification by subsampling.

### 1.1.1.1 - Subsampling Analysis
- [ ] Implement numerical privacy amplification bounds.
- [ ] Add analytical bounds for comparison.

### 1.1.1.2 - Subsampling Validation
- [ ] Create validation tests against reference implementations.
- [ ] Document subsampling privacy guarantees.

### 1.1.2.0 - Laplace Mechanism
- [ ] Implement `LaplaceMechanism` for pure epsilon-DP.
- [ ] Add sensitivity calibration utilities.

### 1.1.2.1 - Exponential Mechanism
- [ ] Implement `ExponentialMechanism` for discrete outputs.
- [ ] Add utility function interface.

### 1.1.2.2 - Report Noisy Max
- [ ] Implement `ReportNoisyMax` mechanism.
- [ ] Add sparse vector technique.

### 1.1.3.0 - Opacus Integration Setup
- [ ] Add `opacus` as optional dependency.
- [ ] Create compatibility layer.

### 1.1.3.1 - Model Validation
- [ ] Integrate Opacus validators for BatchNorm detection.
- [ ] Add automatic layer replacement suggestions.

### 1.1.3.2 - Privacy Engine Hook
- [ ] Create `PrivacyEngine` wrapper for easy model wrapping.
- [ ] Add gradient accumulation support.

### 1.1.3.3 - Per-Sample Gradients
- [ ] Optimize per-sample gradient computation.
- [ ] Add memory-efficient variants.

### 1.1.4.0 - Timing Attack Mitigation
- [ ] Implement constant-time aggregation.
- [ ] Add timing noise injection.

### 1.1.4.1 - Memory Padding
- [ ] Implement memory padding for side-channel resistance.
- [ ] Add configurable padding policies.

### 1.1.4.2 - Secure Random
- [ ] Use cryptographically secure random number generation.
- [ ] Validate entropy sources.

### 1.1.5.0 - Privacy Audit Tools
- [ ] Create privacy budget visualization.
- [ ] Add real-time privacy consumption tracking.

### 1.1.5.1 - Privacy Reports
- [ ] Generate machine-readable privacy reports.
- [ ] Add human-readable summaries.

### 1.1.5.2 - Compliance Checking
- [ ] Add GDPR compliance checklist generator.
- [ ] Add HIPAA compliance checklist generator.

---

## [1.2.0] - Vertical Federated Learning

### 1.2.0.1 - Vertical Dataset Interface
- [ ] Define `VerticalDataset` abstract base class.
- [ ] Specify input/output contract.

### 1.2.0.2 - Feature Alignment
- [ ] Implement entity alignment interface.
- [ ] Add ID hashing utilities.

### 1.2.0.3 - Vertical DataLoader
- [ ] Implement `AlignedDataLoader` for synchronized batching.
- [ ] Add caching for alignment results.

### 1.2.0.4 - Vertical Partitioning
- [ ] Add feature-based partitioning utilities.
- [ ] Support overlapping features.

### 1.2.1.0 - PSI Foundation
- [ ] Define PSI protocol interface.
- [ ] Add security model documentation.

### 1.2.1.1 - RSA-PSI Simulation
- [ ] Implement RSA-blind signature based PSI (simulated).
- [ ] Add computational cost modeling.

### 1.2.1.2 - Circuit-PSI
- [ ] Implement garbled circuit PSI simulation.
- [ ] Add communication cost modeling.

### 1.2.1.3 - PSI Cardinality
- [ ] Implement PSI-cardinality for set size estimation.
- [ ] Add threshold-based filtering.

### 1.2.2.0 - SplitNN Foundation
- [ ] Define `SplitNN` protocol interface.
- [ ] Specify message formats.

### 1.2.2.1 - SplitNN Forward Pass
- [ ] Implement forward pass message passing.
- [ ] Add activation serialization.

### 1.2.2.2 - SplitNN Backward Pass
- [ ] Implement gradient passing protocol.
- [ ] Add gradient serialization.

### 1.2.2.3 - SplitNN Optimization
- [ ] Implement pipelining for SplitNN.
- [ ] Add U-shaped splitting.

### 1.2.3.0 - Vertical FedSGD
- [ ] Implement basic vertical federated SGD.
- [ ] Add convergence tests.

### 1.2.3.1 - Vertical FedAvg
- [ ] Extend FedAvg for vertical setting.
- [ ] Handle partial feature updates.

### 1.2.3.2 - Vertical SecureBoost
- [ ] Implement SecureBoost for XGBoost.
- [ ] Add histogram-based splitting.

### 1.2.3.3 - Vertical Neural Networks
- [ ] Implement end-to-end vertical NN training.
- [ ] Add label protection mechanisms.

### 1.2.4.0 - Label Protection
- [ ] Implement label hiding techniques.
- [ ] Add differential privacy for labels.

### 1.2.4.1 - Gradient Leakage Prevention
- [ ] Add gradient perturbation for VFL.
- [ ] Implement secure gradients.

### 1.2.5.0 - VFL Tutorials
- [ ] Create Tutorial 201: VFL Introduction.
- [ ] Create Tutorial 202: VFL for Credit Scoring.

---

## [1.3.0] - Decentralized and Gossip Learning

### 1.3.0.1 - Graph Topology Base
- [ ] Define `GraphTopology` abstract base class.
- [ ] Specify neighbor interface.

### 1.3.0.2 - Ring Topology
- [ ] Implement `RingTopology` generator.
- [ ] Add bidirectional support.

### 1.3.0.3 - Torus Topology
- [ ] Implement `TorusTopology` generator.
- [ ] Add 2D and 3D variants.

### 1.3.0.4 - Grid Topology
- [ ] Implement `GridTopology` generator.
- [ ] Add wraparound options.

### 1.3.0.5 - Small World
- [ ] Implement `SmallWorldTopology` (Watts-Strogatz).
- [ ] Add rewiring probability parameter.

### 1.3.0.6 - Scale Free
- [ ] Implement `ScaleFreeTopology` (Barabasi-Albert).
- [ ] Add preferential attachment.

### 1.3.0.7 - Random Graph
- [ ] Implement `ErdosRenyiTopology`.
- [ ] Add edge probability parameter.

### 1.3.0.8 - Custom Topology
- [ ] Add support for custom adjacency matrices.
- [ ] Validate connectivity.

### 1.3.1.0 - Mixing Matrix
- [ ] Implement mixing matrix construction.
- [ ] Add Metropolis-Hastings weighting.

### 1.3.1.1 - Spectral Gap Analysis
- [ ] Implement spectral gap computation.
- [ ] Add convergence rate estimation.

### 1.3.1.2 - Dynamic Topology
- [ ] Support time-varying topologies.
- [ ] Add topology change events.

### 1.3.2.0 - Push Protocol
- [ ] Implement basic push gossip.
- [ ] Add message queueing.

### 1.3.2.1 - Pull Protocol
- [ ] Implement pull gossip.
- [ ] Add request-response pattern.

### 1.3.2.2 - Push-Sum
- [ ] Implement Push-Sum algorithm.
- [ ] Add weight tracking.

### 1.3.2.3 - Push-Pull
- [ ] Implement Push-Pull gossip.
- [ ] Add symmetric communication.

### 1.3.2.4 - Random Walk Sampling
- [ ] Implement random walk for neighbor selection.
- [ ] Add Metropolis-Hastings sampling.

### 1.3.3.0 - D-PSGD Foundation
- [ ] Implement Decentralized Parallel SGD base.
- [ ] Add synchronization barriers.

### 1.3.3.1 - D-PSGD Async
- [ ] Implement asynchronous D-PSGD.
- [ ] Add delay tolerance.

### 1.3.3.2 - DSGT
- [ ] Implement Decentralized Stochastic Gradient Tracking.
- [ ] Add variance tracking.

### 1.3.3.3 - EXTRA
- [ ] Implement EXTRA algorithm.
- [ ] Add acceleration.

### 1.3.3.4 - D2
- [ ] Implement D2 (decentralized with momentum).
- [ ] Add momentum tracking.

### 1.3.4.0 - Decentralized Privacy
- [ ] Add DP to decentralized protocols.
- [ ] Implement neighbor-level privacy.

### 1.3.4.1 - Decentralized Robustness
- [ ] Add Byzantine tolerance to D-FL.
- [ ] Implement local filtering.

### 1.3.5.0 - D-FL Tutorials
- [ ] Create Tutorial 203: Gossip Learning.
- [ ] Create Tutorial 204: D-PSGD on Ring.

---

## [1.4.0] - Federated LLMs and Generative AI

### 1.4.0.1 - HuggingFace Setup
- [ ] Add `transformers` as optional dependency.
- [ ] Create compatibility layer.

### 1.4.0.2 - Model Loading
- [ ] Implement `HuggingFaceAdapter` for model loading.
- [ ] Add tokenizer management.

### 1.4.0.3 - Model Serialization
- [ ] Optimize state dict serialization for large models.
- [ ] Add delta compression.

### 1.4.0.4 - Quantization Support
- [ ] Add `bitsandbytes` integration for quantization.
- [ ] Support 8-bit and 4-bit models.

### 1.4.1.0 - LoRA Foundation
- [ ] Implement LoRA layer injection.
- [ ] Add rank configuration.

### 1.4.1.1 - LoRA Aggregation
- [ ] Implement LoRA weight aggregation strategies.
- [ ] Add weighted averaging for adapters.

### 1.4.1.2 - LoRA + DP
- [ ] Combine LoRA with differential privacy.
- [ ] Optimize privacy budget for adapters.

### 1.4.1.3 - QLoRA
- [ ] Implement QLoRA (quantized LoRA).
- [ ] Add 4-bit training support.

### 1.4.2.0 - Prompt Tuning
- [ ] Implement soft prompt tuning.
- [ ] Add prompt aggregation.

### 1.4.2.1 - Prefix Tuning
- [ ] Implement prefix tuning.
- [ ] Add layer-wise prefixes.

### 1.4.2.2 - P-Tuning
- [ ] Implement P-Tuning v1 and v2.
- [ ] Add prompt encoder.

### 1.4.3.0 - FedGAN Foundation
- [ ] Implement federated GAN framework.
- [ ] Add discriminator aggregation.

### 1.4.3.1 - FedGAN Generator
- [ ] Implement generator aggregation.
- [ ] Handle mode collapse mitigation.

### 1.4.3.2 - MD-GAN
- [ ] Implement Multi-Discriminator GAN.
- [ ] Add client-specific discriminators.

### 1.4.4.0 - FedDiffusion
- [ ] Implement federated diffusion model training.
- [ ] Add score matching aggregation.

### 1.4.4.1 - Latent Diffusion
- [ ] Support latent diffusion models.
- [ ] Add VAE handling.

### 1.4.5.0 - LLM Tutorials
- [ ] Create Tutorial 205: Federated GPT-2 Fine-tuning.
- [ ] Create Tutorial 206: FedLoRA for BERT.

---

## [1.5.0] - Cross-Silo Orchestration

### 1.5.0.1 - RPC Interface
- [ ] Define `RPCEngine` abstract interface.
- [ ] Specify message protocols.

### 1.5.0.2 - gRPC Protos
- [ ] Create protobuf definitions for aggregator service.
- [ ] Define model update messages.

### 1.5.0.3 - gRPC Server
- [ ] Implement gRPC server for central aggregator.
- [ ] Add authentication.

### 1.5.0.4 - gRPC Client
- [ ] Implement gRPC client for silos.
- [ ] Add retry logic.

### 1.5.0.5 - TLS Support
- [ ] Add TLS encryption for gRPC.
- [ ] Support client certificates.

### 1.5.1.0 - Docker Foundation
- [ ] Create base Dockerfile for Unbitrium.
- [ ] Optimize image size.

### 1.5.1.1 - Docker Compose
- [ ] Create `docker-compose.yml` for multi-container simulation.
- [ ] Add service discovery.

### 1.5.1.2 - Docker GPU
- [ ] Add GPU support to Docker images.
- [ ] Configure NVIDIA runtime.

### 1.5.2.0 - Kubernetes Basics
- [ ] Create Kubernetes deployment manifests.
- [ ] Add service definitions.

### 1.5.2.1 - Helm Charts
- [ ] Create Helm chart for Unbitrium.
- [ ] Add configurable values.

### 1.5.2.2 - Auto-scaling
- [ ] Add HPA (Horizontal Pod Autoscaler) support.
- [ ] Configure scaling policies.

### 1.5.3.0 - Monitoring Setup
- [ ] Add Prometheus metrics endpoint.
- [ ] Configure key metrics.

### 1.5.3.1 - Grafana Dashboards
- [ ] Create Grafana dashboard templates.
- [ ] Add real-time training visualization.

### 1.5.3.2 - Alerting
- [ ] Configure alerting rules.
- [ ] Add failure notifications.

### 1.5.4.0 - Orchestration Tutorials
- [ ] Create Tutorial 207: Docker Deployment.
- [ ] Create Tutorial 208: Kubernetes FL.

---

## [1.6.0] - Fairness and Robustness 2.0

### 1.6.0.1 - q-FedAvg Core
- [ ] Implement q-FedAvg algorithm.
- [ ] Add q parameter tuning.

### 1.6.0.2 - q-FedAvg Analysis
- [ ] Add fairness-accuracy trade-off analysis.
- [ ] Create visualization tools.

### 1.6.1.0 - AgnosticFL
- [ ] Implement Agnostic Federated Learning.
- [ ] Add minimax optimization.

### 1.6.1.1 - AFL Variants
- [ ] Add AFL with different domains.
- [ ] Support grouped fairness.

### 1.6.2.0 - Demographic Parity
- [ ] Implement `DemographicParity` metric.
- [ ] Add group-based computation.

### 1.6.2.1 - Equalized Odds
- [ ] Implement `EqualizedOdds` metric.
- [ ] Add TPR/FPR balance.

### 1.6.2.2 - Calibration
- [ ] Implement `CalibrationMetric`.
- [ ] Add reliability diagrams.

### 1.6.2.3 - Individual Fairness
- [ ] Implement `IndividualFairness` metric.
- [ ] Add Lipschitz-based measures.

### 1.6.3.0 - Backdoor Detection
- [ ] Implement spectral signature detection.
- [ ] Add clustering-based detection.

### 1.6.3.1 - Backdoor Removal
- [ ] Implement fine-tuning based removal.
- [ ] Add pruning-based removal.

### 1.6.3.2 - FoolsGold
- [ ] Implement FoolsGold aggregation.
- [ ] Add historical tracking.

### 1.6.3.3 - FLAME
- [ ] Implement FLAME defense.
- [ ] Add clustering and clipping.

### 1.6.4.0 - Fairness Tutorials
- [ ] Create Tutorial 209: Fair FL.
- [ ] Create Tutorial 210: Robust FL.

---

## [1.7.0] - Mobile and Edge Optimization

### 1.7.0.1 - TFLite Basics
- [ ] Add TFLite conversion utility.
- [ ] Support common model architectures.

### 1.7.0.2 - TFLite Optimization
- [ ] Add post-training quantization.
- [ ] Support float16 and int8.

### 1.7.0.3 - ONNX Export
- [ ] Add ONNX export utility.
- [ ] Validate exported models.

### 1.7.1.0 - QAT Foundation
- [ ] Implement Quantization-Aware Training hooks.
- [ ] Add fake quantization.

### 1.7.1.1 - QAT Integration
- [ ] Integrate QAT with FL training loop.
- [ ] Handle quantized aggregation.

### 1.7.1.2 - Mixed Precision
- [ ] Add mixed precision training support.
- [ ] Use AMP (Automatic Mixed Precision).

### 1.7.2.0 - Android Example Setup
- [ ] Create Android starter project.
- [ ] Add TFLite Interpreter.

### 1.7.2.1 - Android Training
- [ ] Implement on-device training for Android.
- [ ] Add background training service.

### 1.7.2.2 - Android Communication
- [ ] Add REST client for server communication.
- [ ] Implement secure upload.

### 1.7.3.0 - iOS Example Setup
- [ ] Create iOS starter project.
- [ ] Add Core ML integration.

### 1.7.3.1 - iOS Training
- [ ] Implement on-device training for iOS.
- [ ] Add background task support.

### 1.7.4.0 - Edge Tutorials
- [ ] Create Tutorial 211: Android FL.
- [ ] Create Tutorial 212: iOS FL.

---

## [1.8.0] - Federated Analytics

### 1.8.0.1 - SecureSum Foundation
- [ ] Implement SecureSum protocol interface.
- [ ] Add security documentation.

### 1.8.0.2 - SecureSum MPC
- [ ] Implement MPC-based SecureSum (simulated).
- [ ] Add secret sharing.

### 1.8.0.3 - SecureSum HE
- [ ] Implement HE-based SecureSum (simulated).
- [ ] Add Paillier encryption.

### 1.8.1.0 - SecureMedian
- [ ] Implement SecureMedian protocol.
- [ ] Add comparison-based approach.

### 1.8.1.1 - SecureMax
- [ ] Implement SecureMax protocol.
- [ ] Add argmax variant.

### 1.8.2.0 - FederatedSQL Parser
- [ ] Implement SQL parser for federated queries.
- [ ] Support SELECT, COUNT, SUM.

### 1.8.2.1 - FederatedSQL Aggregates
- [ ] Implement aggregate functions (AVG, MIN, MAX).
- [ ] Add GROUP BY support.

### 1.8.2.2 - FederatedSQL Privacy
- [ ] Add DP to federated SQL queries.
- [ ] Implement sensitivity analysis.

### 1.8.3.0 - Heavy Hitters
- [ ] Implement Count-Min Sketch.
- [ ] Add frequency estimation.

### 1.8.3.1 - HyperLogLog
- [ ] Implement HyperLogLog for cardinality estimation.
- [ ] Add union operation.

### 1.8.3.2 - Bloom Filters
- [ ] Implement federated Bloom filters.
- [ ] Add set membership queries.

### 1.8.4.0 - Analytics Tutorials
- [ ] Create Tutorial 213: Federated Analytics.
- [ ] Create Tutorial 214: Private Histograms.

---

## [1.9.0] - Extreme Scalability

### 1.9.0.1 - TopK Compression
- [ ] Implement TopK gradient compression.
- [ ] Add error feedback.

### 1.9.0.2 - RandomK Compression
- [ ] Implement RandomK compression.
- [ ] Add variance reduction.

### 1.9.0.3 - QSGD
- [ ] Implement Quantized SGD.
- [ ] Add multi-level quantization.

### 1.9.0.4 - SignSGD
- [ ] Implement SignSGD compression.
- [ ] Add majority vote.

### 1.9.0.5 - TernGrad
- [ ] Implement ternary gradient compression.
- [ ] Add layer-wise scaling.

### 1.9.1.0 - Server Parallelism
- [ ] Implement multi-threaded aggregation.
- [ ] Add thread pool management.

### 1.9.1.1 - Distributed Aggregation
- [ ] Implement distributed aggregation across machines.
- [ ] Add reduce-scatter.

### 1.9.1.2 - Pipeline Parallelism
- [ ] Implement pipelined aggregation.
- [ ] Overlap communication and computation.

### 1.9.2.0 - Memory Optimization
- [ ] Optimize state_dict memory usage.
- [ ] Add streaming aggregation.

### 1.9.2.1 - Client Sampling
- [ ] Implement efficient client sampling.
- [ ] Add stratified sampling.

### 1.9.2.2 - Lazy Loading
- [ ] Implement lazy model loading.
- [ ] Reduce memory footprint for 1M+ clients.

### 1.9.3.0 - Scalability Benchmarks
- [ ] Create benchmark for 10k clients.
- [ ] Create benchmark for 100k clients.
- [ ] Create benchmark for 1M clients.

### 1.9.4.0 - Scalability Tutorials
- [ ] Create Tutorial 215: Scaling to 100k Clients.
- [ ] Create Tutorial 216: Million-Client Simulation.

---

## [2.0.0] - The Unbitrium Platform

### 2.0.0.1 - Web UI Foundation
- [ ] Select frontend framework (React/Vue).
- [ ] Create project structure.

### 2.0.0.2 - Dashboard Layout
- [ ] Implement dashboard layout.
- [ ] Add navigation.

### 2.0.0.3 - Real-time Monitoring
- [ ] Add WebSocket for real-time updates.
- [ ] Display training progress.

### 2.0.0.4 - Metrics Visualization
- [ ] Add charts for accuracy/loss.
- [ ] Add per-client visualizations.

### 2.0.0.5 - Experiment Builder
- [ ] Implement drag-and-drop experiment builder.
- [ ] Add component library.

### 2.0.0.6 - Configuration Editor
- [ ] Add YAML configuration editor.
- [ ] Add validation.

### 2.0.1.0 - User Authentication
- [ ] Implement user authentication.
- [ ] Add role-based access control.

### 2.0.1.1 - Experiment Management
- [ ] Implement experiment CRUD.
- [ ] Add experiment comparison.

### 2.0.1.2 - Result Export
- [ ] Add result export to CSV/JSON.
- [ ] Generate LaTeX tables.

### 2.0.2.0 - Cloud AWS
- [ ] Add AWS EC2 spot instance support.
- [ ] Implement auto-scaling.

### 2.0.2.1 - Cloud GCP
- [ ] Add GCP Compute Engine support.
- [ ] Implement preemptible VMs.

### 2.0.2.2 - Cloud Azure
- [ ] Add Azure VM support.
- [ ] Implement spot instances.

### 2.0.2.3 - Serverless
- [ ] Implement serverless FL orchestration.
- [ ] Use AWS Lambda/GCP Functions.

### 2.0.3.0 - Platform Tutorials
- [ ] Create Tutorial 217: Web Dashboard.
- [ ] Create Tutorial 218: Cloud Deployment.
- [ ] Create Tutorial 219: Serverless FL.
- [ ] Create Tutorial 220: Advanced Platform Features.

---

## Beyond 2.0.0

### Future Considerations
- Quantum-resistant cryptography for FL.
- Neuromorphic computing integration.
- Federated reinforcement learning at scale.
- Multi-modal FL (text, image, audio).
- Federated continual learning with catastrophic forgetting prevention.
- Cross-modal FL (transfer between modalities).
- Federated foundation models.
- Real-time FL for streaming data.
- Geographically distributed FL with latency awareness.
- Energy-aware scheduling for green FL.

---

*This roadmap is a living document and will be updated as the project evolves.*
*Contributions and suggestions are welcome via GitHub Issues.*

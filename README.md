# U.M.E.R Engine — Universal Synthetic Data Foundry

![Domain](https://img.shields.io/badge/Domain-Computer%20Vision%20%26%20ML-ff9800?style=flat-square)
![Architecture](https://img.shields.io/badge/Architecture-Synthetic%20Reality%20Compiler-blue?style=flat-square)
![Status](https://img.shields.io/badge/Status-Sim--to--Real%20Ready-success?style=flat-square)

> **Part of the U.M.E.R (Uniform Memory Encoded Representation) GPU Compute Core**

---

## Overview: The Synthetic Reality Compiler

The **U.M.E.R Synthetic Data Foundry** addresses the most expensive bottleneck in modern AI: the computer vision cold-start problem. Training robust, domain-adaptable Deep Learning models requires millions of perfectly annotated, edge-case heavy images. Real-world data collection for this is fundamentally unscalable, prone to manual annotation errors, and often physically unsafe.

This repository demonstrates that the U.M.E.R. architecture is not a game engine. It is a **Synthetic Reality Compiler for AI**—a general-purpose procedural world generator with omniscient, pixel-perfect annotation. By leveraging the engine's absolute, deterministic $O(1)$ spatial math, it transforms arbitrary physical or conceptual assets into perfectly labeled, large-scale dataset substrates.

## 1. Universal Input Abstraction Layer

The engine supports dynamic ingestion across five progressive levels of asset fidelity, allowing for maximum flexibility based on client or research availability:

* **Level 1 — Procedural Generation:** Fully algorithmic scene and agent generation (e.g., executing team logic and spatial constraints for sports analytics).
* **Level 2 — Single-Image 3D Reconstruction:** Integration pathways for image-to-3D generative inference (e.g., TripoSR) to extract geometry from flat images.
* **Level 3 — Video-Based Photogrammetry:** Multi-view dynamic reconstruction.
* **Level 4 — Physical Object Scanning:** Direct digitization pathways for NeRF and LiDAR-scanned realities.
* **Level 5 — CAD / Bare-Metal 3D Ingestion:** Direct simulation of `.obj` / `.fbx` models (such as the showcased lithium battery and player mechanics).

## 2. Deterministic Simulation & Logic Core

At the heart of the pipeline is the **U.M.E.R. GPU-Accelerated Spatial Compute Core**. Traditional engines use probabilistic physics, leading to stochastic drift. U.M.E.R. uses absolute spatial hashing to guarantee identical topological collapse.

* **Dynamics:** Resolves both rigid and deformable object interactions in bare-metal PyCUDA (`synthetic-dataset.ipynb`).
* **Agentic Behavior:** Governs complex contextual positioning and team differentiation natively within the memory blocks.
* **Omniscient State Tracking:** Because the engine physically commands the spatial location of every voxel, generating bounding boxes and segmentation masks is reduced to a simple memory offset read, achieving 100% annotation accuracy at zero performance cost.

## 3. Domain Randomization (Closing the Sim-to-Real Gap)

To prevent AI model overfitting on "perfect" simulated pixels, the Python Director module algorithmically orchestrates intense environmental permutations:

* **Sensor Noise Modeling:** Film grain, motion blur, focal variations, and algorithmic depth-of-field.
* **Environmental Permutations:** Randomized sun trajectories, varying color temperatures, weather states (fog, rain), and HDR background mapping.
* **Texture & Material Overrides:** Dynamic scattering of surface reflectivity to force neural networks to learn shape and context over simple color profiles.

## 4. Output: The Labeled Substrate

The engine compiles these localized physics, inputs, and permutations into master `umer_synthetic_data.json` registries. When ingested by the rendering layer, the system automatically produces:
* Large-scale synthetic image datasets.
* Pixel-accurate bounding boxes and contextual metadata.
* Controlled, automated long-tail scenario generation (e.g., massive occlusions, rare collision events, and extreme spatial densities).

---

## Execution Pipeline

The backend orchestrator is housed in `synthetic-dataset.ipynb`.

### Prerequisites
* CUDA-capable GPU (For $O(1)$ hardware spatial compilation)
* Python 3.12+ with PyCUDA
* Unity / Corresponding headless renderer configured for ingestion

### Generating a Reality Substrate
1. Execute the target generation logic inside the Python orchestrator.
2. The PyCUDA backend will compute the absolute $O(1)$ centroids and inject them into the randomized domain matrices.
3. Output files (`umer_synthetic_data_clean.json`) are automatically sanitized and exported.
4. Pass the generated timeline to the rendering frontend to physically write the dataset to disk.

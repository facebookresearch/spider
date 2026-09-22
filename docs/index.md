---
layout: home

hero:
  name: "SPIDER"
  text: "Scalable Physics-Informed DExterous Retargeting"
  tagline: A general framework for physics-based retargeting from human to diverse robot embodiments
  image:
    src: /figs/teaser.png
    alt: SPIDER
  actions:
    - theme: brand
      text: Get Started
      link: /guide/quick-start
    - theme: alt
      text: Released Dataset
      link: /usage/lifted-datasets
    - theme: alt
      text: View on GitHub
      link: https://github.com/facebookresearch/spider

features:
  - icon: 🔬
    title: Physics-Based
    details: First general physics-based retargeting pipeline for both dexterous hand and humanoid robot manipulation

  - icon: ⚡
    title: Fast Simulation
    details: GPU-accelerated batched simulation with MuJoCo Warp achieving 10-20x speedup over sequential execution

  - icon: 📊
    title: Rich Datasets, Robots and Simulators
    details: Works with 6+ datasets out of the box including GigaHand, Hot3D, OakInk, and more. Supports 9+ robot embodiments including dexterous hands (Allegro, Inspire, Xhand) and humanoid robots (G1, H1, T1). Supports multiple simulators including MuJoCo Warp, Genesis, and HDMI.

  - icon: 🔄
    title: Sim2Real Ready
    details: Optimized trajectories can be directly deployed to real-world robots with minimal adjustments
---

## News

- **2026-09-22:** [SPIDER retarget_full is released](https://huggingface.co/datasets/retarget/retarget_full): **7,876 successful trajectories** from 2,885 source episodes, covering DexYCB, HOT3D v2, HRDexDB, and OakInk with Allegro, Xhand, Inspire, and Sharpa. [Download and inspect with Viser](./usage/lifted-datasets.md).

The downloader and Viser inspector are included in SPIDER; see the
[loading and inspection guide](./usage/lifted-datasets.md) to get started.

## Quick Example

```bash
# Clone example datasets
git clone https://huggingface.co/datasets/retarget/retarget_example example_datasets

# Install with uv
uv sync --python 3.12
pip install --ignore-requires-python --no-deps -e .

# Run retargeting
uv run examples/run_mjwp.py
```

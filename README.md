# Situating and Comparing Artificial Neural Networks in a Unified Brain-like Space

## Overview
This repository provides code for computing graph metrics from spatial attention patterns across different models and brain functional networks.

## Repository Structure

### 1. Model Processing (`model-process/`)
Contains code for computing graph metrics from spatial attention patterns of various model families.

#### Key Files:
### Vision Transformers (ViT, ViT-Variants)
  - `vit-process.py`
  - `dinov3-process.py`

### Large Language Models (LLMs)
  - `gpt2-process.py`
  - `bert-process.py`

### Large Multimodal Models (LMMs)
  - `clip-languag-process.py`, `clip-vision-process.py`
  - `blip-language-process.py`, `blip-vision-process.py`

### RoPE-based Models （LLM-RoPEs、LMM-RoPEs）
- **LLMs with Rotary Position Embeddings**
  - `qwen-process.py`
- **LMMs with Rotary Position Embeddings**
  - `qwen-language-process.py`, `qwen-vision-process.py`
  - `deepseek-language-process.py`

#### Usage Requirements:
Before running any model processing scripts, modify these key parameters:
1. **Model List**: Specify target list of model names in `models`
2. **Cache Directory**: Set `your_cache_dir` for model downloads
3. **Output Directory**: Set `your_base_dir` for results storage
4. **Access Tokens**: For gated models, add your HuggingFace access token in the download section (request from official model pages)

#### Extension Notes:
For other and new RoPE-based LLM/LMM models, adapt existing examples by:
- Modifying the model loader
- Adjusting configuration extraction based on model architecture
- Updating Q/K weight extraction methods according to model nesting structure

### 2. Brain Network Processing (`brain-process.ipynb`)
Computes graph metrics for brain functional networks.

#### Prerequisites:
- Prepare multiple pre-extracted brain network connectivity matrices (`.mat` files)
- Ensure main diagonal elements are set to zero in all matrices

### 3. Pre-computed Results
- `model-result.zip`: Graph metrics for a representative subset of 60 models covering the main model families reported in the paper (this is a partial release; the full study covers 151 models). Extract the archive to a `model-result/` directory.
- `brainnetwork-result.json`: Graph metrics for 7 group-level brain functional networks
- `pca-reference.json`: Frozen two-component PCA reference (head-embedding mean, PC1/PC2 components, explained variance ratio) used to project attention heads into the brain-like space. The five graph-metric and seven-ICN orders are fixed by this file and must not be reordered.

### 4. Brain-Like Space Analysis (`brain-like_space.ipynb`)
Computes per-head brain-reference similarity vectors, then builds the brain-like space and derives model-level scores — `S_sum` (sum of PC1 projections), `S_mean` (mean PC1 projection), `S_plus` (sum of positive PC1 projections only) — under one of two PCA modes, toggled by `PCA_MODE` in the notebook:

- **frozen** (default): project onto the published PCA reference fitted on the full 151-model collection (`pca-reference.json`); all heads share the released coordinate system.
- **fit**: fit a two-component PCA from the heads of the locally available 60-model subset. For local reproduction only; the space and scores then reflect the local fit rather than the published reference.

#### Quick Start:
Use the provided `model-result.zip` (representative subset), `brainnetwork-result.json`, and `pca-reference.json` to:
- Visualize the brain-like space
- Generate brain-like score result files (`brain_like_scores.csv`)

Switch `PCA_MODE` between `"frozen"` and `"fit"` to choose the PCA construction above.

## Requirements

### Core Dependencies
- torch==2.8.0
- transformers==4.56.1
- accelerate==1.10.1
- safetensors==0.6.2
- timm==1.0.14
- networkx==3.4.2
- python-louvain==0.16
- numpy==2.2.6
- scipy==1.15.3

**Note:** `transformers` and `accelerate` versions may need adjustment for specific models. Some models require different versions for compatibility.

## Citation
If you use this code in your research, please cite our accompanying paper：Situating and Comparing Artificial Neural Networks in a Unified Brain-like Space

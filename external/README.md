
## Overview
This directory contains external repositories and raw datasets required for the VAD training pipeline.

Includes:
- Model repositories (NAS-VAD, Self-Attentive VAD)
- Raw public datasets (MS-SNSD, VOiCES, VoiceBank+DEMAND)

These are used as inputs to preprocessing scripts, which generate structured datasets under 'datasets/`.

---

## Directory Structure

```bash
external/
├── NAS_VAD/                   # NAS-VAD official repository
├── voice-activity-detection/  # Self-attentive VAD repository
├── MS-SNSD/                   # MS-SNSD dataset
├── VOiCES/                    # VOiCES dataset
└── Voicebank28/               # VoiceBank+DEMAND dataset

Note: Datasets are not included due to GitHub storage limits. Please refer to the sources listed below
```

---

## Model Repositories
### NAS-VAD
```bash
https://github.com/daniel03c1/NAS_VAD
```
- Used as the main training framework in this project

### Self-attentive VAD
```bash
https://github.com/voithru/voice-activity-detection
```
- Original implementation of the attention-based VAD model
- In this project, the model is re-implemented within the NAS-VAD framework for consistency

---

## Datasets

These datasets must be placed under `external/` before running preprocessing scripts.

### MS-SNSD
```bash
- Source: https://github.com/microsoft/MS-SNSD
- After setup: clean files → clean/, mixed noisy files → noisy/
```
### VOiCES
```bash
- Source: https://iqtlabs.github.io/voices/
- Only selected subsets are used — YAML configs in datasets/VOiCES/_cfg/ define splits
```
### VoiceBank + DEMAND (Voicebank28)
```bash
- Source: https://datashare.ed.ac.uk/handle/10283/2791
```
---

## Pipeline
```bash
external/ (raw data + source repos) -> scripts/*.py (preprocessing) -> datasets/ (structured training format)
```

---

## Important
- Do **NOT** modify files inside `external/`
- These act as raw data sources and reference implementations only
- All processed outputs go into `datasets/`

---
## Setup
1. Clone or download required repositories into `external/`
2. Refer to main README for preprocessing scripts from the root directory

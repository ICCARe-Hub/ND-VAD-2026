# ND-VAD-2026 (Neurodegenerative Voice Activity Detection)
# Cross-Domain Voice Activity Detection on Healthy and Neurodegenerative Speech Across Narrative and Diadochokinetic Tasks


This repository accompanies our research paper:

> **Cross-Domain Voice Activity Detection on Healthy and Neurodegenerative Speech Across Narrative and Diadochokinetic Tasks**  
> Humzah Zahid Malik, Arman Hassanpour, Benjamin James Haller, Angela C. Roberts  
> Western University, London, Ontario, Canada  
> [Paper (PDF)](malik.pdf)

This is a PyTorch-based framework for training and evaluating VAD models on noisy speech (MS-SNSD, VOiCES, Voicebank+DEMAND),
fine-tuning them on clinical recordings from the Ontario Neurodegenerative Disease Research Initiative (ONDRI), and evaluating them on held-out neurodegenerative (ONDRI) and healthy-control (BEAM) speech across narrative and diadochokinetic (DDK) tasks.

Additional details that were omitted from the paper due to space (participant allocation, hyperparameter search space, final hyperparameters, full validation results, and additional test metrics) are provided below under [Supplementary Material](#supplementary-material).

Note: ONDRI and BEAM recordings and annotations are private clinical data and are not distributed with this repository.

---

## Features
- Modular PyTorch training pipeline for VAD models
- Supports datasets: MS-SNSD, VOiCES, Voicebank+DEMAND, and others (Please refer to external/README and datasets/README for additional setup information for public noisy datasets)
- Pseudo-label generation using Silero VAD
- Spectrogram-based feature extraction
- Evaluation metrics: AUC, Accuracy, Precision, Recall, F1-score
---

## Project Structure
- Please refer to datasets/README and external/README for additional details

---

## Setup

### 1. Clone the repo
```bash
$ git clone https://github.com/ICCARe-Hub/ND-VAD-2026
$ cd HM-Thesis-2026
```
---

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
---

### 3. Prepare datasets
Place raw datasets under the following paths before running scripts:
```
external/MS-SNSD/
external/VOiCES/
external/Voicebank28/
```
- (The same procedure will apply for any other dataset. Place each dataset directly under external/)
---

### 4. Prepare training pipeline(s)
- NAS_VAD/ and voice-activity-detection/ (self-attentive VAD) repositories provided under external/
- Please refer to their respective README files for additional training and testing the models.

---

### 5. Run preprocessing
#### MS-SNSD
```
python scripts/ms_snsd_speaker_split.py
python scripts/ms_snsd_mix.py
```

#### VOiCES training/validation custom splits using voices_train.yaml (Provided under VOiCES/_cfg)
```bash
python scripts/voices_prepare.py
```

#### VOiCES test split using voices_test.yaml (Provided under VOiCES/_cfg)
```bash
python scripts/voices_prepare.py
```

#### Voicebank28 training/validation splits
```bash
python scripts/voicebank28_split.py
```

#### All datasets - label + spectrogram generation
```bash
python scripts/run_silero_vad.py
python scripts/silero_json_to_npy.py
python scripts/make_specs.py
```

Note: Please refer to datasets/README for each dataset's directory setup before running scripts.

---

### 6. Prepare ONDRI data (requires approved data access)
ONDRI/BEAM recordings and annotations are not distributed with this repository.

#### Fine-tuning pool (Silero pseudo-labels)
```bash
python scripts/run_silero_vad.py
python scripts/silero_json_to_npy.py
python scripts/make_specs.py
```
- Output: datasets/ondri_silero_train/ and datasets/ondri_silero_valid/ (participant-disjoint, 80/20, stratified by cohort)

#### Ground-truth test sets (manually corrected TextGrids)
```bash
python scripts/textgrid_to_npy.py --audio_dir datasets/ondri_narrative_ddk --textgrid_dir datasets/ondri_narrative_ddk --out_dir datasets/ondri_narrative_ddk_npy --speech_mode nonempty --target_tier_index 2 --require_sr 16000
python scripts/make_specs.py
```
- Narrative TextGrids: Whisper (large-v3) → Montreal Forced Aligner (v3.3.9) → manual correction in Praat
- Narrative labeling conditions: WNW (words + non-words only, primary) and AV (all vocalizations, supplementary)

---

## Training
Example:
```bash
python trainer.py --model SL_model --mode train --dataset Voicebank28 --save_path ./SLTrain
```
- Please refer to external/NAS_VAD for additional details about training

#### ONDRI fine-tuning (Example)
```bash
python trainer.py --model NewSearch --mode train --dataset ONDRI-SILERO --init_checkpoint fullTrain/026_NewSearch_TRAIN.pth --save_path ./ondri_silero_nasvad_f2
```
- Best epoch is selected by validation F2

---

## Evaluation
Example:
```bash
python trainer.py --model NewSearch --mode test --dataset VOiCES --save_path ./fullTrain
```

- Evaluates on the held-out test sets (VOiCES/test/TEST/, Voicebank28/test/TEST/)
- Please refer to external/NAS_VAD for additional details about testing


#### ONDRI evaluation
```bash
python trainer.py --model SL_model --mode test --dataset ONDRI-NARRATIVE --init_checkpoint <checkpoint.pth> --test_path <test_dir>
python trainer.py --model NewSearch --mode test --dataset ONDRI-DDK --init_checkpoint <checkpoint.pth> --test_path <test_dir>
```
- Metrics: AUC, Accuracy, Precision, Recall, F1, F2, Miss Rate, False Alarm Rate (all except AUC were computed at a threshold of 0.5)
- Global metrics are pooled at the frame level across all recordings
---

## Pipeline Overview

```
1. Raw audio → dataset splits (train/val/test)
        
2. Silero VAD → pseudo-labels (labels_json/)
        
3. JSON timestamps → frame-level binary labels (labels_npy/)
        
4. Spectrogram extraction (spec_npy/)
        
5. Model training on TRAIN/ and validation on VALID/(merged across datasets)

6. Evaluation on dataset-specific TEST/ sets

7. Silero pseudo-labels on ONDRI fine-tuning pool

8. Fine-tuning from public-data checkpoints (+ Optuna search)

9. Evaluation on manually corrected ONDRI/BEAM TextGrids (narrative + DDK speech tasks)
```

---

## Supplementary Material

### Participant Allocation
| Cohort | Eligible | Representative (held out) | Tested | Fine-tuning pool |
|---|---:|---:|---:|---:|
| ADMCI | 139 | 12 | 10 | 127 |
| ALS | 36 | 12 | 0 | 24 |
| FTD | 53 | 12 | 0 | 41 |
| PD | 138 | 12 | 10 | 126 |
| VCI | 148 | 12 | 0 | 136 |
| **ONDRI Total** | **514** | **60** | **20** | **454** (364 train / 90 valid) |
| BEAM | — | — | 10 | 0 |

### Hyperparameter Search Space
Optuna (TPE sampler), 50 trials per model, objective: validation F2
| Hyperparameter | Values |
|---|---|
| loss_type | {bce, weighted_bce, focal} |
| batch_size | {32, 64, 128, 256} |
| learning_rate | {1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3} |
| weight_decay | {0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2} |
| patience | {5, 10, 15, 20} |
| sa_dropout (Self-Attentive VAD) | {0, 0.1, 0.2, 0.3, 0.4} |
| drop_path (NAS-VAD) | {0, 0.05, 0.1, 0.2, 0.3} |
| epochs | {30, 60, 75, 100} |
| grad_clip | {1, 2, 5, 10} |

### Final Hyperparameters
| Hyperparameter | NAS-VAD | Self-Attentive VAD |
|---|---|---|
| loss_type | bce | focal |
| batch_size | 128 | 32 |
| learning_rate | 1e-3 | 3e-3 |
| weight_decay | 0 | 1e-5 |
| patience | 10 | 15 |
| sa_dropout | — | 0.4 |
| drop_path | 0 | — |
| epochs | 30 | 75 |
| grad_clip | 5 | 10 |
| best epoch | 20 | 38 |

### Validation Checkpoint Comparison
| Checkpoint | AUC | Accuracy | Precision | Recall | F1 | F2 | Val Loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| NAS-VAD baseline (final) | 0.9983 | 0.9847 | 0.9860 | 0.9876 | 0.9868 | 0.9873 | 0.0500 |
| NAS-VAD trial 49 | 0.9953 | 0.9721 | 0.9600 | 0.9946 | 0.9770 | 0.9874 | 0.0618 |
| Self-Attentive VAD baseline | 0.9944 | 0.9727 | 0.9785 | 0.9741 | 0.9763 | 0.9749 | 0.0878 |
| Self-Attentive VAD trial 2 (final) | 0.9919 | 0.9662 | 0.9551 | 0.9861 | 0.9704 | 0.9797 | 0.0283 |
| Self-Attentive VAD trial 19 | 0.9882 | 0.9523 | 0.9257 | 0.9954 | 0.9593 | 0.9806 | 0.1598 |

### Additional Test Metrics
| Model | Task | Labels | Miss Rate | False Alarm Rate | TP | TN | FP | FN |
|---|---|---|---:|---:|---:|---:|---:|---:|
| NAS-VAD | Narrative | WNW | 0.0171 | 0.1688 | 140,879 | 69,984 | 14,216 | 2,454 |
| NAS-VAD | Narrative | AV | 0.1221 | 0.1488 | 145,980 | 52,138 | 9,115 | 20,300 |
| Self-Attentive VAD | Narrative | WNW | 0.0250 | 0.2822 | 139,748 | 60,436 | 23,764 | 3,585 |
| Self-Attentive VAD | Narrative | AV | 0.1099 | 0.2531 | 148,010 | 45,751 | 15,502 | 18,270 |

---

## References

### Models
#### NAS-VAD
```
  Rho, D., et al. (2022). *NAS-VAD: Neural Architecture Search for Voice Activity Detection*.  
  In Proceedings of Interspeech 2022.  
  https://doi.org/10.21437/Interspeech.2022-975
```

#### Self-Attentive VAD
```
  Jo, H., et al. (2021). *Self-Attentive Voice Activity Detection*.  
  In IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2021).  
  https://doi.org/10.1109/ICASSP39728.2021.9413961  
```
---

### Implementations
#### NAS-VAD Repository (Official implementation used in this project)  
```
https://github.com/daniel03c1/NAS_VAD
```
#### Self-Attentive VAD Repository (Original implementation)
```
  https://github.com/voithru/voice-activity-detection
```

### Labeling Model
#### Silero VAD
```
  https://github.com/snakers4/silero-vad  
```

### Datasets
**VOiCES Dataset** 
```
  Richey, C., et al. (2018). *VOiCES: Voices Obscured in Complex Environmental Settings*.  
  In Proceedings of Interspeech 2018.  
  https://www.isca-archive.org/interspeech_2018/richey18_interspeech.html  
```

**VoiceBank + DEMAND Dataset**  
```
  Valentini-Botinhao, C., et al. (2016). *Speech Enhancement for a Noise-Robust TTS System*.  
  In Proceedings of Interspeech 2016.  
  https://www.isca-archive.org/interspeech_2016/valentini_botinhao16_interspeech.html  
```

**MS-SNSD Dataset**  
```
  Reddy, C. K. A., et al. (2019). *A Scalable Noisy Speech Dataset and Online Subjective Test Framework*.  
  In Proceedings of Interspeech 2019.  
  https://www.isca-archive.org/interspeech_2019/reddy19_interspeech.html  
```

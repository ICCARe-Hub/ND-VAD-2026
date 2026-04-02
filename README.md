````
# Cross-Domain Evaluation of Neural Network-Based Voice Activity Detection on Healthy and Neurodegenerative Speech

This is a PyTorch-based framework for training and evaluating VAD models on noisy speech (MS-SNSD, VOiCES, Voicebank+DEMAND),
with a focus on generalization to a private neurodegenerative speech dataset (ONDRI DDK).

---

## Features
- Modular PyTorch training pipeline for VAD models
- Supports datasets: MS-SNSD, VOiCES, Voicebank+DEMAND, and others (Please refer to external/README and datasets/README for additional setup information for public noisy datasets)
- Pseudo-label generation using Silero VAD
- Spectrogram-based feature extraction
- Evaluation metrics: AUC, Accuracy, Precision, Recall, F1-score

## Project Structure

Please refer to datasets/README and external/README for additional details

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/ICCARe-Hub/HM-Thesis-2026
cd HM-Thesis-2026
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare datasets
Place raw datasets under the following paths before running scripts:
```
external/MS-SNSD/
external/VOiCES/
external/Voicebank28/
(The same procedure will apply for any other dataset. Place each dataset directly under external/)
```

### 4. Prepare training pipeline(s)
NAS_VAD/ and voice-activity-detection/ (self-attentive VAD) repositories provided under external/
Please refer to their respective README files for additional training and testing the models.

### 4. Run preprocessing
```bash
# MS-SNSD
python scripts/ms_snsd_speaker_split.py
python scripts/ms_snsd_mix.py

# VOiCES training/validation custom splits using voices_train.yaml (Provided under VOiCES/_cfg)
python scripts/voices_prepare.py

# VOiCES test split using voices_test.yaml (Provided under VOiCES/_cfg)
python scripts/voices_prepare.py

# Voicebank28 training/validation splits
python scripts/voicebank28_split.py

# All datasets — label + spectrogram generation
python scripts/run_silero_vad.py
python scripts/silero_json_to_npy.py
python scripts/make_specs.py

Note: Please refer to datasets/README for each dataset's directory setup before running scripts.

```

---

## Training
Example:
```bash
python trainer.py --model SL_model --mode train --dataset Voicebank28 --save_path ./SLTrain
```
Please refer to external/NAS_VAD for additional details about training

---

## Evaluation
Example:
```bash
!python trainer.py --model NewSearch --mode test --dataset VOiCES --save_path ./fullTrain
```

Evaluates on the held-out test sets (VOiCES/test/TEST/, Voicebank28/test/TEST/)


Please refer to external/NAS_VAD for additional details about testing
---

## Pipeline Overview

```
1. Raw audio → dataset splits (train/val/test)
        ↓
2. Silero VAD → pseudo-labels (labels_json/)
        ↓
3. JSON timestamps → frame-level binary labels (labels_npy/)
        ↓
4. Spectrogram extraction (spec_npy/)
        ↓
5. Model training on TRAIN/ and validation on VALID/(merged across datasets)
        ↓
6. Evaluation on dataset-specific TEST/ sets
```

---

## Key Insights

To be filled

---

## Future Work

To be filled

---

## Tech Stack

- Python · PyTorch · torchaudio · NumPy · scikit-learn · librosa

---

## References

### Models
- **NAS-VAD**  
  Rho, D., et al. (2022). *NAS-VAD: Neural Architecture Search for Voice Activity Detection*.  
  In Proceedings of Interspeech 2022.  
  https://doi.org/10.21437/Interspeech.2022-975  

- **Self-Attentive VAD**  
  Jo, H., et al. (2021). *Self-Attentive Voice Activity Detection*.  
  In IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2021).  
  https://doi.org/10.1109/ICASSP39728.2021.9413961  

### Implementations
- NAS-VAD Repository (Official implementation used in this project):  
  https://github.com/daniel03c1/NAS_VAD  

- Self-Attentive VAD Repository (Original implementation):  
  https://github.com/voithru/voice-activity-detection  

### Labeling Model
- **Silero VAD**  
  Silero Team. *Silero Voice Activity Detector*.  
  https://github.com/snakers4/silero-vad  

### Datasets
- **VOiCES Dataset**  
  Richey, C., et al. (2018). *VOiCES: Voices Obscured in Complex Environmental Settings*.  
  In Proceedings of Interspeech 2018.  
  https://www.isca-archive.org/interspeech_2018/richey18_interspeech.html  

- **VoiceBank + DEMAND Dataset**  
  Valentini-Botinhao, C., et al. (2016). *Speech Enhancement for a Noise-Robust TTS System*.  
  In Proceedings of Interspeech 2016.  
  https://www.isca-archive.org/interspeech_2016/valentini_botinhao16_interspeech.html  

- **MS-SNSD Dataset**  
  Reddy, C. K. A., et al. (2019). *A Scalable Noisy Speech Dataset and Online Subjective Test Framework*.  
  In Proceedings of Interspeech 2019.  
  https://www.isca-archive.org/interspeech_2019/reddy19_interspeech.html  

### Pathological Dataset

- **ONDRI DDK Dataset**  
  Ontario Neurodegenerative Disease Research Initiative (ONDRI).  
  (Private dataset used for evaluation; not publicly available)
````

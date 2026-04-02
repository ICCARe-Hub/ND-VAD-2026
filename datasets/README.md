# datasets directory setup used for training and testing

datasets/
- MS-SNSD/
      train/
        clean/ (contains clean files from MS-SNSD repo's 'clean_train' obtained from scripts/ms_snsd_speaker_split.py)
        noisy/ (contains noisy files after running ms_snsd_mix.py)
        labels_json/ (contains Silero VAD sample indices obtained after running scripts/run_silero_vad.py)
        labels_npy/ (contains converted .json sample indices -> .npy frame-level labels obtained after scripts/running silero_json_to_npy.py)
        spec_npy/ (contains computed spectrogram transformations obtained after running scripts/make_specs.py)
      val/ (same subdirectories as MS-SNSD/train)

- VOiCES/
      train/
        noisy/ (contains noisy files from VOiCES dataset after running scripts/voices_prepare.py)
        labels_json/
        labels_npy/
        spec_npy/
      val/ (same setup as VOiCES/train)
      test/ (same setup as train/ and val/ except TEST/. Also noisy/ contains .wav files from the official VOiCES_devkit test set)
        - TEST/ (contains _spec.npy files and .npy files for the official VOiCES_devkit .wav files)
      _cfg/ (contains yaml files used by voices_prepare.py to subset VOiCES noisy files into train/, val/ and test/ subfolders)

- Voicebank28/
      train/
        noisy/ (contains noisy wav files after running scripts/voicebank28_split.py)
        labels_json/
        labels_npy/
        spec_npy/
      val/ (same setup. noisy/ includes validation files after running running scripts/voicebank28_split.py)
      test/ (same setup, other than the following)
        - TEST/
        - contains all official Voicebank28 test set wav files
      log_testset.txt
      log_trainset_28spk.txt

- ONDRI_DDK_Test3
      - Private dataset. Contains _spec.npy and .npy label corresponding to every .wav file from the original dataset.

- TRAIN/
      - Contains training _spec.npy and .npy files obtained from train/spec_npy/ and train/labels_npy/ subfolders from the three public datasets mentioned above.
      
      
- VALID/ (same setup as TRAIN/)
   

# Spoken Digit Recognition

EE708 course project for robust ten-class spoken-digit recognition from variable-duration audio recordings.

## What the notebook does

- Loads 37,800 labelled recordings and 16,200 unlabelled test recordings.
- Converts audio to mono, resamples to 16 kHz, and pads or truncates it to one second.
- Builds three-channel log-Mel, delta, and delta-delta spectrograms.
- Applies waveform augmentation, SpecAugment, mixup, label smoothing, and gradient clipping.
- Trains EfficientNet-B2 from scratch across three stratified folds.
- Averages three fold-specific checkpoints over five temporal shifts for inference.

The experiment obtained **99.39% mean validation accuracy** across the three folds.

## Repository layout

```text
notebooks/spoken_digit_recognition.ipynb  complete Kaggle training and inference workflow
requirements.txt                          Python dependencies
```

The competition audio is not committed because the raw archive is approximately 2 GB. The notebook expects the Kaggle dataset under the paths defined in its `CFG` dictionary.

## Run

Open the notebook in a Kaggle GPU session, attach the EE708 spoken-digit dataset, verify the paths in `CFG`, and run all cells. Model checkpoints and the submission CSV are generated in the notebook working directory.

## Core ideas

The project treats audio classification as image classification over spectrograms. The base log-Mel channel captures spectral energy, while delta and delta-delta channels represent first- and second-order temporal change. Stratified folds estimate generalisation, and checkpoint/shift averaging reduces sensitivity to a single split or temporal alignment.
"# Spoken-Digit-Recognition" 

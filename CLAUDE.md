# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Kepler exoplanet detection project that implements a CNN model using TensorFlow 2.x/Keras to classify Kepler light curves as planet candidates or non-candidates. It's a rewrite of the Google Research AstroNet project with modifications for the Keras framework.

## Core Architecture

### CNN Model (`astro_net/Kepler_cnn.py`)
- Dual-input architecture with global view (2001 points) and local view (201 points) time series
- 1D convolutional layers with max pooling for feature extraction
- Merged features processed through dense layers with dropout
- Binary classification (PC vs NON_PC) with softmax output

### Data Pipeline
- **Raw Data**: Kepler light curves stored in FITS files
- **Processing**: Phase folding, bucketization, global/local view extraction (`data/preprocess.py`)
- **Training Data**: Pickled record files with [global_view, local_view] pairs (`data/training_data_io.py`)
- **Data Organization**: Train/Validation/Test splits with class folders (0_PC, 1_NON_PC)

### Key Components
- **Training**: Thread-safe data generators, GPU/CPU configuration (`astro_net/train_Kepler_CNN.py`)
- **Prediction**: Single TCE prediction with visualization (`astro_net/predict_Kepler.py`)
- **Data Generation**: CSV-to-record conversion with multi-threading (`data/generate_training_data.py`)
- **Download**: Kepler data fetching from MAST archive (`download/`)

## Development Commands

### Training
```bash
python astro_net/train_Kepler_CNN.py
```

### Prediction
```bash
python astro_net/predict_Kepler.py
```

### Data Processing
```bash
# Step 1: Query file list
python download/Download-Kepler-data-Step1-query-file-list.py

# Step 2: Download files
python download/Download-Kepler-data-Step2-get-files.py

# Generate training data (slow process)
python data/generate_training_data.py

# Distribute to Train/Validation/Test folders
python data/distribute_training_data.py
```

## Configuration

All paths and parameters are configured in `configure/environment.py`:
- Data folders: KEPLER_DATA_FOLDER, TRAINING_FOLDER
- Model paths: KEPLER_TRAINED_MODEL_FOLDER
- Classification: NB_CLASSES (2 for binary), ALLOWED_LABELS
- CSV input: KEPLER_CSV_FILE path for q1_q17_dr24_tce.csv

## Training Configuration

Key training parameters in `astro_net/train_Kepler_CNN.py`:
- GPU/CPU selection via `GPU = True/False`
- Batch size: 32, Epochs: 200
- Sample sizes: TRAINING_SAMPLE_SIZE, VALIDATION_SAMPLE_SIZE
- Early stopping patience: 30
- Multi-threading: 24 workers for data generation

## Data Flow

1. **Raw Input**: Kepler FITS files → CSV labels
2. **Processing**: Light curve preprocessing → Global/Local views
3. **Training**: Pickled records → Keras generators → CNN model
4. **Inference**: TCE parameters → Light curves → Model prediction → Visualization

## Model Files

- Trained weights: `astro_net/trained_model/kepler-model-two-classes.h5`
- Checkpoints: `{KEPLER_TRAINED_MODEL_FOLDER}/checkpoints/`
- Logs: `{KEPLER_TRAINED_MODEL_FOLDER}/logs/`

## Testing

Test evaluation is built into `train_Kepler_CNN.py` after training completion, measuring accuracy on the Test set.
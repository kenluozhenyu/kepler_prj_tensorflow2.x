# Kepler Exoplanet Detection with TensorFlow 2.x

A modern implementation of the AstroNet project for detecting exoplanets from Kepler light curve data using TensorFlow 2.x and Keras. This project implements a dual-input CNN architecture that processes both global and local views of light curves to classify potential exoplanet candidates.

## 🌟 Project Overview

This project is a modern rewrite of the Google Research AstroNet project, updated to use TensorFlow 2.x best practices with improved performance, reliability, and maintainability. The system automatically processes Kepler light curve data and predicts whether observed signals represent potential exoplanet candidates.

### Key Features

- **Modern TensorFlow 2.x Architecture**: Utilizes the latest TensorFlow features including tf.data API, mixed precision training, and GPU memory management
- **Dual-Input CNN Model**: Processes both global (2001 points) and local (201 points) views of light curves for comprehensive analysis
- **Automated Data Pipeline**: Complete workflow from raw Kepler data to model predictions with minimal manual intervention
- **Advanced Training Features**: Validation monitoring, early stopping, automatic weight management, and comprehensive evaluation
- **High-Performance Inference**: Batch processing support and optimized prediction pipeline
- **Rich Visualizations**: Automatic generation of global/local view plots with prediction results
- **Comprehensive Logging**: Detailed progress tracking and error reporting

## 🏗️ Code Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Source    │    │   Data Pipeline   │    │   ML Model       │
│                 │    │                  │    │                 │
│ • Kepler FITS   │───▶│ • Download &      │───▶│ • Dual-Input CNN  │
│ • CSV Labels    │    │   Preprocess      │    │ • Binary Class.  │
│ • Raw Light      │    │ • Distribution    │    │ • TF2.x Training │
│   Curves         │    │ • Validation      │    │ • Weights Mgmt    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Configuration  │    │  Training &       │    │  Prediction &    │
│                 │    │  Evaluation       │    │  Visualization   │
│ • Environment   │    │                  │    │                 │
│ • Paths         │    │ • Validation      │    │ • Inference      │
│ • Hyperparams    │    │ • Early Stop      │    │ • Results Plot   │
│ • GPU/CPU        │    │ • Test Eval       │    │ • Confidence     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Core Components

1. **Data Processing Pipeline**
   - `download/`: Kepler data acquisition modules
   - `data/`: Data preprocessing, generation, and distribution
   - `preprocess.py`: Light curve preprocessing and view generation

2. **Machine Learning Model**
   - `astro_net/Kepler_cnn.py`: Modern CNN implementation with TensorFlow 2.x
   - `astro_net/train_Kepler_CNN.py`: Advanced training pipeline with monitoring
   - `astro_net/predict_Kepler.py`: Production-ready inference system

3. **Configuration Management**
   - `configure/environment.py`: Centralized configuration and paths

## 📁 Code Structure

```
kepler_prj_tensorflow2.x/
├── astro_net/                          # Machine learning modules
│   ├── Kepler_cnn.py                  # Modern CNN model (TF 2.x)
│   ├── train_Kepler_CNN.py            # Training pipeline with validation
│   ├── predict_Kepler.py              # Inference and visualization
│   └── trained_model/                 # Trained model weights
├── data/                               # Data processing modules
│   ├── generate_training_data.py      # Training data generation
│   ├── distribute_training_data.py    # Train/val/test distribution
│   ├── preprocess.py                  # Light curve preprocessing
│   └── training_data_io.py            # Data I/O operations
├── download/                           # Data acquisition modules
│   ├── Download_one_Kepler_ID.py      # Single target download
│   ├── Download-Kepler-data-Step1-*.py # Multi-step download process
│   └── Download-Kepler-data-Step2-*.py
├── configure/                          # Configuration
│   └── environment.py                  # Environment variables and paths
├── backup_code/                        # Original code backups
├── light_curve_util/                   # Light curve utilities
├── third_party/                        # Third-party dependencies
└── README.md                          # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- TensorFlow 2.x
- Required packages: `numpy`, `pandas`, `matplotlib`, `scikit-learn`

### Installation

1. Clone the repository
2. Install dependencies: `pip install tensorflow numpy pandas matplotlib scikit-learn`
3. Configure environment settings in `configure/environment.py`

### Basic Usage

#### 1. Prediction (Quickest Start)

```python
# Use the existing trained model for prediction
python astro_net/predict_Kepler.py
```

The script will:
- Load the trained model automatically
- Download required light curve data if needed
- Generate predictions with confidence scores
- Create visualization plots showing global/local views
- Save results to the configured output folder

#### 2. Training Your Own Model

**Step 1: Prepare Data**
```bash
# Get the Kepler TCE CSV file from the original AstroNet project
# Place it at the path specified in environment.KEPLER_CSV_FILE

# Download Kepler data (2+ days)
python download/Download-Kepler-data-Step1-query-file-list.py
python download/Download-Kepler-data-Step2-get-files.py
```

**Step 2: Generate Training Data**
```bash
# Convert raw data to CNN-compatible format (several days)
python data/generate_training_data.py
```

**Step 3: Distribute Data**
```bash
# Split data into train/validation/test sets
python data/distribute_training_data.py
```

**Step 4: Train Model**
```bash
# Start training with automatic validation monitoring
python astro_net/train_Kepler_CNN.py
```

## 📊 Model Architecture

### Dual-Input CNN Design

The model processes two complementary views of light curve data:

1. **Global View (2001 points)**: Complete orbital phase information
2. **Local View (201 points)**: Detailed transit information

### Network Structure

```
Global View Branch:        Local View Branch:
Input(2001,1)              Input(201,1)
    ↓                           ↓
Conv1D(16)×2               Conv1D(16)×2
    ↓                           ↓
MaxPool1D                  MaxPool1D
    ↓                           ↓
Conv1D(32)×2               Conv1D(32)×2
    ↓                           ↓
MaxPool1D                  MaxPool1D
    ↓                           ↓
Conv1D(64)×2               (Additional processing)
    ↓
MaxPool1D
    ↓
Conv1D(128)×2
    ↓
MaxPool1D
    ↓
Conv1D(256)×2
    ↓
MaxPool1D
    ↓
Flatten                    Flatten
    └─────────────┬─────────┘
                  ↓
              Concatenate
                  ↓
              Dense(512)×4 + Dropout(0.5)
                  ↓
              Output Layer (Binary Classification)
```

## ⚙️ Configuration

### Environment Setup (`configure/environment.py`)

```python
# Data paths
KEPLER_CSV_FILE = "path/to/q1_q17_dr24_tce.csv"
KEPLER_DATA_FOLDER = "path/to/kepler/data"
TRAINING_FOLDER = "path/to/training/data"
KEPLER_TRAINED_MODEL_FOLDER = "path/to/saved/models"

# Classification settings
NB_CLASSES = 2  # Binary classification: PC vs NON_PC
ALLOWED_LABELS = ["PC", "AFP", "NTP"]

# Output folders
DATA_FOR_PREDICTION_FOLDER = "path/to/prediction/data"
PREDICT_OUTPUT_FOLDER = "path/to/output/images"
```

### Training Configuration

```python
# In astro_net/train_Kepler_CNN.py
config = TrainingConfig(
    batch_size=32,
    max_epochs=200,
    early_stopping_patience=30,
    learning_rate=1e-5,
    use_gpu=True
)
```

## 📈 Training Features

### Advanced Training Pipeline

- **Validation Monitoring**: Real-time validation loss tracking
- **Early Stopping**: Automatic stopping when validation loss doesn't improve for 30 epochs
- **Weight Management**: Automatic saving of best weights during training
- **Learning Rate Scheduling**: Adaptive learning rate adjustment
- **Mixed Precision**: Automatic mixed precision training for faster training (when available)
- **TensorBoard Integration**: Comprehensive training visualization

### Training Process

1. **Data Loading**: Efficient tf.data pipeline with parallel processing
2. **Model Training**: Modern Keras training loop with callbacks
3. **Validation Monitoring**: Custom ValidationMonitor callback tracks progress
4. **Best Weight Selection**: Automatically saves weights when validation loss improves
5. **Early Stopping**: Stops training when no improvement for 30 epochs
6. **Final Evaluation**: Comprehensive test set evaluation with detailed metrics
7. **Model Saving**: Final model weights saved for future use

## 🔮 Prediction Features

### Inference Pipeline

- **Automatic Data Acquisition**: Downloads missing light curve data automatically
- **Smart Preprocessing**: Handles data loading and preprocessing transparently
- **Batch Processing**: Support for multiple TCE predictions
- **Confidence Scoring**: Detailed probability estimates for predictions
- **Rich Visualizations**: Automatic generation of global/local view plots
- **Performance Monitoring**: Processing time and resource usage tracking

### Prediction Output

For each prediction, the system provides:
- **Classification Result**: PC (Planet Candidate) or NON_PC
- **Confidence Score**: 0-100% confidence estimate
- **Visualization**: High-quality plots showing global and local views
- **Detailed Metrics**: Processing time and model performance information

## 📊 Performance and Optimization

### Training Optimizations

- **GPU Memory Management**: Dynamic memory growth to prevent OOM errors
- **Mixed Precision Training**: 2-3x speed improvement on compatible hardware
- **Parallel Data Loading**: Multi-threaded data preprocessing
- **Efficient Batching**: Optimized batch processing for better GPU utilization

### Inference Optimizations

- **Model Caching**: Loaded model stays in memory for batch predictions
- **GPU Acceleration**: Automatic GPU utilization when available
- **Memory Efficiency**: Optimized memory usage for large datasets
- **Error Handling**: Robust error recovery and graceful degradation

## 🛠️ Advanced Usage

### Custom Training

```python
from astro_net.train_Kepler_CNN import KeplerTrainer, TrainingConfig

# Custom configuration
config = TrainingConfig(
    batch_size=64,
    max_epochs=300,
    early_stopping_patience=50,
    learning_rate=5e-6,
    use_gpu=True
)

# Create and run trainer
trainer = KeplerTrainer(config)
stats = trainer.run_complete_training()
```

### Batch Prediction

```python
from astro_net.predict_Kepler import KeplerPredictor, TCEData

# Create predictor
predictor = KeplerPredictor()
predictor.load_model()

# Prepare TCE data
tce_list = [
    TCEData(kepid=11442793, tce_period=14.44, tce_time0bk=2.2, tce_duration=0.112),
    TCEData(kepid=757450, tce_period=8.88, tce_time0bk=134.45, tce_duration=0.087)
]

# Run batch prediction
results = predictor.predict_batch(tce_list)
```

### Custom Data Processing

```python
from data.generate_training_data import TrainingDataGenerator

# Configure generator
generator = TrainingDataGenerator(
    num_workers=16,
    enable_multiprocessing=True,
    log_level=logging.DEBUG
)

# Generate training data
stats = generator.generate_training_data()
```

## 📝 Troubleshooting

### Common Issues

1. **GPU Memory Issues**: The system automatically manages GPU memory with dynamic growth
2. **Missing Data**: Automatic download system handles missing light curve files
3. **Training Time**: Use multi-threading and GPU acceleration for faster training
4. **Model Loading**: Ensure trained model weights are in the correct location

### Performance Tips

- Use GPU for training and inference when available
- Increase `num_workers` for faster data processing
- Use mixed precision training on compatible hardware
- Batch predictions for better throughput

## 📚 Technical Details

### Model Specifications

- **Input Shapes**: Global view (2001, 1), Local view (201, 1)
- **Architecture**: Dual-input 1D CNN with 5 convolutional blocks
- **Parameters**: ~2M trainable parameters
- **Output**: Binary classification with softmax activation
- **Loss Function**: Binary crossentropy
- **Optimizer**: Adam with learning rate scheduling

### Data Processing Pipeline

1. **Light Curve Loading**: Reads Kepler FITS files
2. **Preprocessing**: Normalization and cleaning
3. **Phase Folding**: Aligns data based on orbital period
4. **View Generation**: Creates global and local views
5. **Batching**: Efficient batching for model training/inference

## 🤝 Contributing

This project is a modern rewrite of the original AstroNet. Contributions are welcome for:

- Performance optimizations
- Additional visualization features
- New model architectures
- Enhanced error handling
- Documentation improvements

## 📄 License

This project maintains the same license as the original AstroNet project from Google Research.

## 🔗 References

- [Original AstroNet Project](https://github.com/google-research/exoplanet-ml/tree/master/exoplanet-ml/astronet)
- [Kepler Mission](https://archive.stsci.edu/missions-and-data/kepler)
- [TensorFlow 2.x Documentation](https://www.tensorflow.org/)

---

**Note**: This project represents a significant modernization of the original AstroNet codebase, incorporating TensorFlow 2.x best practices, improved error handling, and enhanced performance while maintaining the scientific accuracy and methodology of the original research.
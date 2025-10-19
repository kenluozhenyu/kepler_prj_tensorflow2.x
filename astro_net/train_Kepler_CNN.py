"""
Kepler CNN Training (TensorFlow 2.x Implementation)

This module implements a comprehensive training pipeline for the Kepler CNN model
using TensorFlow 2.x architecture. It includes validation monitoring, early stopping,
weight management, and comprehensive evaluation.

Features:
- Modern TensorFlow 2.x training with tf.data and Keras APIs
- Custom validation loss monitoring with patience-based early stopping
- Automatic weight management and checkpointing
- Comprehensive test set evaluation with detailed metrics
- Progress tracking and logging
- GPU/CPU configuration with memory management
"""

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, models, optimizers, utils
from tensorflow.keras.callbacks import Callback

from astro_net import Kepler_cnn
from configure import environment
from data import training_data_io


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    batch_size: int = 32
    max_epochs: int = 200
    early_stopping_patience: int = 30
    training_samples_per_epoch: int = 12600
    validation_samples: int = 1600
    learning_rate: float = 1e-5
    decay: float = 1e-6
    use_gpu: bool = True
    num_workers: int = 4
    buffer_size: int = 1000
    prefetch_size: int = tf.data.AUTOTUNE


@dataclass
class TrainingStats:
    """Statistics for training progress."""
    total_epochs: int = 0
    best_epoch: int = 0
    best_val_loss: float = float('inf')
    training_time: float = 0.0
    test_accuracy: float = 0.0
    test_loss: float = 0.0
    class_accuracies: Dict[str, float] = None
    learning_curves: Dict[str, List[float]] = None

    def __post_init__(self):
        """Initialize dictionaries."""
        if self.class_accuracies is None:
            self.class_accuracies = {}
        if self.learning_curves is None:
            self.learning_curves = {
                'train_loss': [], 'val_loss': [],
                'train_acc': [], 'val_acc': []
            }


class ValidationMonitor(Callback):
    """
    Custom callback for monitoring validation loss and managing weights.

    This callback implements:
    - Validation loss monitoring with patience-based early stopping
    - Automatic weight saving when validation loss improves
    - Final weight restoration after training completion
    """

    def __init__(self, patience: int = 30, restore_best_weights: bool = True, verbose: int = 1):
        """
        Initialize validation monitor.

        Args:
            patience: Number of epochs to wait for improvement
            restore_best_weights: Whether to restore best weights at end
            verbose: Verbosity level
        """
        super().__init__()
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = 0
        self.best_val_loss = float('inf')

    def on_train_begin(self, logs=None):
        """Initialize training state."""
        self.wait = 0
        self.stopped_epoch = 0
        self.best_val_loss = float('inf')
        self.best_weights = None

        if self.verbose > 0:
            self.model.logger.info("Validation monitoring started")

    def on_epoch_end(self, epoch, logs=None):
        """
        Monitor validation loss at end of each epoch.

        Args:
            epoch: Current epoch number
            logs: Training logs
        """
        logs = logs or {}
        current_val_loss = logs.get('val_loss')

        if current_val_loss is None:
            if self.verbose > 0:
                self.model.logger.warning("Validation loss not found in logs")
            return

        # Check if validation loss improved
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.wait = 0
            self.best_weights = self.model.get_weights().copy()

            if self.verbose > 0:
                self.model.logger.info(
                    f"Epoch {epoch + 1}: Validation loss improved to {current_val_loss:.6f}. "
                    f"Saving weights."
                )

            # Save best weights to file
            self._save_best_weights(epoch + 1, current_val_loss)

        else:
            self.wait += 1
            if self.verbose > 0:
                self.model.logger.info(
                    f"Epoch {epoch + 1}: Validation loss did not improve from {self.best_val_loss:.6f}. "
                    f"Patience: {self.wait}/{self.patience}"
                )

            # Check for early stopping
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True

                if self.verbose > 0:
                    self.model.logger.info(
                        f"Early stopping triggered at epoch {epoch + 1}. "
                        f"Best validation loss: {self.best_val_loss:.6f}"
                    )

    def on_train_end(self, logs=None):
        """Restore best weights if requested."""
        if self.restore_best_weights and self.best_weights is not None:
            try:
                self.model.set_weights(self.best_weights)
                if self.verbose > 0:
                    best_epoch = max(0, self.stopped_epoch - self.patience + 1)
                    self.model.logger.info(
                        f"Restored best weights from epoch {best_epoch} "
                        f"with validation loss {self.best_val_loss:.6f}"
                    )
            except Exception as e:
                if hasattr(self.model, 'logger'):
                    self.model.logger.error(f"Failed to restore best weights: {e}")
                else:
                    print(f"Failed to restore best weights: {e}")

    def _save_best_weights(self, epoch: int, val_loss: float) -> None:
        """Save best weights to file."""
        try:
            weights_path = os.path.join(
                environment.KEPLER_TRAINED_MODEL_FOLDER,
                'best_weights.h5'
            )
            os.makedirs(os.path.dirname(weights_path), exist_ok=True)
            self.model.save_weights(weights_path)
        except Exception as e:
            self.model.logger.error(f"Failed to save best weights: {e}")


class KeplerDataGenerator:
    """
    Modern data generator using tf.data for efficient training.

    This class replaces the original thread-based generator with TensorFlow's
    tf.data API for better performance and resource management.
    """

    def __init__(self, config: TrainingConfig):
        """Initialize data generator."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def create_dataset(self, data_folder: str, is_training: bool = True) -> tf.data.Dataset:
        """
        Create tf.data dataset from folder.

        Args:
            data_folder: Path to data folder
            is_training: Whether this is training data

        Returns:
            tf.data.Dataset: Ready-to-use dataset
        """
        # Get class folders
        class_folders = [
            os.path.join(data_folder, class_name)
            for class_name in sorted(os.listdir(data_folder))
            if os.path.isdir(os.path.join(data_folder, class_name))
        ]

        if not class_folders:
            raise ValueError(f"No class folders found in {data_folder}")

        self.logger.info(f"Found {len(class_folders)} classes in {data_folder}")

        # Create dataset of file paths
        file_paths = []
        labels = []

        for class_idx, class_folder in enumerate(class_folders):
            class_files = [
                os.path.join(class_folder, f)
                for f in os.listdir(class_folder)
                if f.endswith('.record')
            ]

            file_paths.extend(class_files)
            labels.extend([class_idx] * len(class_files))

        self.logger.info(f"Found {len(file_paths)} files in {data_folder}")

        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

        if is_training:
            dataset = dataset.shuffle(buffer_size=len(file_paths))

        # Map file paths to data
        dataset = dataset.map(
            self._load_and_preprocess,
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Batch and prefetch
        dataset = dataset.batch(self.config.batch_size)
        dataset = dataset.prefetch(self.config.prefetch_size)

        return dataset

    def _load_and_preprocess(self, file_path: tf.Tensor, label: tf.Tensor) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
        """
        Load and preprocess a single sample.

        Args:
            file_path: Path to record file (Tensor)
            label: Class label (Tensor)

        Returns:
            Tuple of (features_dict, label_tensor)
        """
        def _parse_function(file_path, label):
            # Read the pickle file
            import pickle
            try:
                with open(file_path.numpy().decode('utf-8'), 'rb') as f:
                    global_view, local_view = pickle.load(f)

                # Validate data shapes
                if len(global_view) != 2001 or len(local_view) != 201:
                    raise ValueError(f"Invalid data shapes: global={len(global_view)}, local={len(local_view)}")

                # Convert to tensors and reshape
                global_view = tf.reshape(tf.convert_to_tensor(global_view, dtype=tf.float32), (2001, 1))
                local_view = tf.reshape(tf.convert_to_tensor(local_view, dtype=tf.float32), (201, 1))

                # Create features dictionary
                features = {
                    'global_input': global_view,
                    'local_input': local_view
                }

                # Convert label to one-hot
                label_one_hot = tf.one_hot(label, depth=environment.NB_CLASSES)

                return features, label_one_hot

            except Exception as e:
                # Return dummy data if file loading fails
                self.logger.error(f"Error loading file {file_path.numpy().decode('utf-8')}: {e}")
                dummy_global = tf.zeros((2001, 1), dtype=tf.float32)
                dummy_local = tf.zeros((201, 1), dtype=tf.float32)
                dummy_features = {
                    'global_input': dummy_global,
                    'local_input': dummy_local
                }
                dummy_label = tf.zeros(environment.NB_CLASSES, dtype=tf.float32)
                return dummy_features, dummy_label

        # Use tf.py_function for file I/O
        features, label = tf.py_function(
            _parse_function,
            [file_path, label],
            ([tf.float32, tf.float32], tf.float32)
        )

        # Set shapes for better performance
        features[0].set_shape((2001, 1))
        features[1].set_shape((201, 1))
        label.set_shape((environment.NB_CLASSES,))

        return {'global_input': features[0], 'local_input': features[1]}, label


class KeplerTrainer:
    """
    Modern Kepler CNN trainer with TensorFlow 2.x.

    This class provides a complete training pipeline including model compilation,
    data preparation, training with monitoring, and final evaluation.
    """

    def __init__(self, config: TrainingConfig, log_level: int = logging.INFO):
        """
        Initialize trainer.

        Args:
            config: Training configuration
            log_level: Logging level
        """
        self.config = config
        self.stats = TrainingStats()
        self._setup_logging(log_level)
        self._setup_tensorflow()
        self._build_model()

    def _setup_logging(self, log_level: int) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('kepler_training.log')
            ]
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    def _setup_tensorflow(self) -> None:
        """Setup TensorFlow configuration."""
        # Configure GPU memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus and self.config.use_gpu:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                self.logger.info(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
            except RuntimeError as e:
                self.logger.error(f"GPU configuration failed: {e}")
                self.config.use_gpu = False

        # Set mixed precision if available
        try:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            self.logger.info("Mixed precision training enabled")
        except:
            self.logger.info("Mixed precision not available, using float32")

    def _build_model(self) -> None:
        """Build and compile the model."""
        self.logger.info("Building Kepler CNN model")

        # Build model using the modern implementation
        cnn = Kepler_cnn.KeplerCNN()
        self.model = cnn.build_compiled_model(
            learning_rate=self.config.learning_rate,
            decay=self.config.decay
        )

        # Add logger reference to model for callbacks
        self.model.logger = self.logger

        self.logger.info(f"Model built with {self.model.count_params():,} parameters")

    def _prepare_data(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Prepare training, validation, and test datasets."""
        self.logger.info("Preparing datasets")

        generator = KeplerDataGenerator(self.config)

        # Create datasets
        train_folder = os.path.join(environment.TRAINING_FOLDER, 'Train')
        validation_folder = os.path.join(environment.TRAINING_FOLDER, 'Validation')
        test_folder = os.path.join(environment.TRAINING_FOLDER, 'Test')

        train_dataset = generator.create_dataset(train_folder, is_training=True)
        validation_dataset = generator.create_dataset(validation_folder, is_training=False)
        test_dataset = generator.create_dataset(test_folder, is_training=False)

        self.logger.info("Datasets prepared successfully")

        return train_dataset, validation_dataset, test_dataset

    def _create_callbacks(self) -> List[Callback]:
        """Create training callbacks."""
        callbacks_list = []

        # Validation monitor with early stopping
        val_monitor = ValidationMonitor(
            patience=self.config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks_list.append(val_monitor)

        # CSV logger
        csv_logger = callbacks.CSVLogger(
            filename=os.path.join(environment.KEPLER_TRAINED_MODEL_FOLDER, 'training_log.csv'),
            append=True
        )
        callbacks_list.append(csv_logger)

        # TensorBoard
        tensorboard = callbacks.TensorBoard(
            log_dir=os.path.join(environment.KEPLER_TRAINED_MODEL_FOLDER, 'tensorboard'),
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
        callbacks_list.append(tensorboard)

        # Learning rate scheduler
        lr_scheduler = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        )
        callbacks_list.append(lr_scheduler)

        return callbacks_list

    def train(self, train_dataset: tf.data.Dataset, validation_dataset: tf.data.Dataset) -> Dict:
        """
        Train the model.

        Args:
            train_dataset: Training data
            validation_dataset: Validation data

        Returns:
            Dict: Training history
        """
        self.logger.info("Starting training")

        # Calculate steps per epoch
        train_steps = self.config.training_samples_per_epoch // self.config.batch_size
        val_steps = self.config.validation_samples // self.config.batch_size

        self.logger.info(f"Training steps per epoch: {train_steps}")
        self.logger.info(f"Validation steps per epoch: {val_steps}")

        # Create callbacks
        callback_list = self._create_callbacks()

        # Start training
        start_time = time.time()

        history = self.model.fit(
            train_dataset,
            epochs=self.config.max_epochs,
            validation_data=validation_dataset,
            steps_per_epoch=train_steps,
            validation_steps=val_steps,
            callbacks=callback_list,
            verbose=1
        )

        training_time = time.time() - start_time
        self.stats.training_time = training_time

        self.logger.info(f"Training completed in {training_time:.2f} seconds")

        # Update stats
        self.stats.total_epochs = len(history.history['loss'])
        self.stats.best_val_loss = min(history.history['val_loss'])
        self.stats.learning_curves = history.history

        return history

    def evaluate(self, test_dataset: tf.data.Dataset) -> Dict:
        """
        Evaluate the model on test set.

        Args:
            test_dataset: Test data

        Returns:
            Dict: Evaluation results
        """
        self.logger.info("Starting evaluation on test set")

        # Load best weights
        best_weights_path = os.path.join(
            environment.KEPLER_TRAINED_MODEL_FOLDER,
            'best_weights.h5'
        )

        if os.path.exists(best_weights_path):
            self.model.load_weights(best_weights_path)
            self.logger.info("Loaded best weights for evaluation")
        else:
            self.logger.warning("Best weights not found, using current weights")

        # Evaluate on test set
        results = self.model.evaluate(test_dataset, verbose=1)

        # Get predictions for detailed analysis
        self.logger.info("Computing detailed predictions")
        predictions = []
        true_labels = []

        for batch_x, batch_y in test_dataset:
            pred = self.model.predict(batch_x, verbose=0)
            predictions.extend(pred)
            true_labels.extend(batch_y.numpy())

        predictions = np.array(predictions)
        true_labels = np.array(true_labels)

        # Calculate detailed metrics
        self._calculate_detailed_metrics(predictions, true_labels)

        # Update stats
        self.stats.test_loss = results[0]
        self.stats.test_accuracy = results[1] if len(results) > 1 else 0.0

        self.logger.info(f"Test evaluation completed:")
        self.logger.info(f"  Test loss: {self.stats.test_loss:.6f}")
        self.logger.info(f"  Test accuracy: {self.stats.test_accuracy:.4f}")

        return {
            'loss': self.stats.test_loss,
            'accuracy': self.stats.test_accuracy,
            'class_accuracies': self.stats.class_accuracies
        }

    def _calculate_detailed_metrics(self, predictions: np.ndarray, true_labels: np.ndarray) -> None:
        """Calculate detailed metrics including class-wise accuracies."""
        if len(predictions) == 0 or len(true_labels) == 0:
            self.logger.warning("Empty predictions or true labels arrays")
            return

        try:
            # Convert predictions to class indices
            if environment.NB_CLASSES == 2:
                # Binary classification: use threshold of 0.5
                pred_classes = (predictions[:, 0] > 0.5).astype(int)
                true_classes = true_labels[:, 0].astype(int)
                class_names = ['NON_PC', 'PC']  # 0: NON_PC, 1: PC
            else:
                # Multi-class classification
                pred_classes = np.argmax(predictions, axis=1)
                true_classes = np.argmax(true_labels, axis=1)
                class_names = [f'Class_{i}' for i in range(environment.NB_CLASSES)]

            # Overall accuracy
            overall_accuracy = np.mean(pred_classes == true_classes)

            # Class-wise accuracies
            for i, class_name in enumerate(class_names):
                if i < len(class_names):
                    class_mask = true_classes == i
                    if np.sum(class_mask) > 0:
                        class_accuracy = np.mean(pred_classes[class_mask] == true_classes[class_mask])
                        self.stats.class_accuracies[class_name] = class_accuracy
                        self.logger.info(f"  {class_name} accuracy: {class_accuracy:.4f} "
                                       f"({np.sum(class_mask)} samples)")
                    else:
                        self.logger.warning(f"  No samples found for class {class_name}")

        except Exception as e:
            self.logger.error(f"Error calculating detailed metrics: {e}")

    def save_final_model(self) -> None:
        """Save the final trained model."""
        model_path = os.path.join(
            environment.KEPLER_TRAINED_MODEL_FOLDER,
            'kepler-model-two-classes.h5'
        )

        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self.model.save_weights(model_path)
            self.logger.info(f"Final model saved to {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to save final model: {e}")

    def run_complete_training(self) -> TrainingStats:
        """
        Run the complete training pipeline.

        Returns:
            TrainingStats: Complete training statistics
        """
        try:
            # Prepare data
            train_dataset, validation_dataset, test_dataset = self._prepare_data()

            # Train model
            self.train(train_dataset, validation_dataset)

            # Evaluate on test set
            self.evaluate(test_dataset)

            # Save final model
            self.save_final_model()

            # Log final statistics
            self._log_final_statistics()

            return self.stats

        except Exception as e:
            self.logger.error(f"Training pipeline failed: {e}")
            raise

    def _log_final_statistics(self) -> None:
        """Log final training statistics."""
        self.logger.info("=" * 60)
        self.logger.info("TRAINING COMPLETED - FINAL STATISTICS")
        self.logger.info("=" * 60)
        self.logger.info(f"Total epochs trained: {self.stats.total_epochs}")
        self.logger.info(f"Best validation loss: {self.stats.best_val_loss:.6f}")
        self.logger.info(f"Total training time: {self.stats.training_time:.2f} seconds")
        self.logger.info(f"Test accuracy: {self.stats.test_accuracy:.4f}")
        self.logger.info(f"Test loss: {self.stats.test_loss:.6f}")

        if self.stats.class_accuracies:
            self.logger.info("Class-wise accuracies:")
            for class_name, accuracy in self.stats.class_accuracies.items():
                self.logger.info(f"  {class_name}: {accuracy:.4f}")

        self.logger.info("=" * 60)


def main():
    """
    Main function to run Kepler CNN training.

    This function sets up the training configuration and runs the complete
    training pipeline with proper error handling and logging.
    """
    # Training configuration
    config = TrainingConfig(
        batch_size=32,
        max_epochs=200,
        early_stopping_patience=30,
        training_samples_per_epoch=12600,
        validation_samples=1600,
        learning_rate=1e-5,
        decay=1e-6,
        use_gpu=True,
        num_workers=4
    )

    try:
        # Create trainer
        trainer = KeplerTrainer(config, log_level=logging.INFO)

        # Run complete training
        stats = trainer.run_complete_training()

        # Return success based on test accuracy
        return 0 if stats.test_accuracy > 0.7 else 1

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return 1
    except Exception as e:
        print(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
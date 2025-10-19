"""
Kepler Exoplanet Prediction (TensorFlow 2.x Implementation)

This module implements a comprehensive prediction pipeline for Kepler light curve data
using TensorFlow 2.x. It loads trained models and predicts whether a given light curve
represents a potential exoplanet candidate.

Features:
- Modern TensorFlow 2.x model loading and inference
- Automatic data acquisition and preprocessing
- Detailed prediction analysis with confidence scores
- Visualization of global and local views
- Comprehensive error handling and logging
- Batch prediction support
"""

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib

from astro_net import Kepler_cnn
from data import preprocess
from configure import environment
from download import Download_one_Kepler_ID


# Set matplotlib backend for headless environments
matplotlib.use('Agg')


@dataclass
class TCEData:
    """Threshold Crossing Event data structure."""
    kepid: int
    tce_period: float
    tce_time0bk: float
    tce_duration: float  # in days

    def __post_init__(self):
        """Validate TCE data."""
        if self.kepid <= 0:
            raise ValueError("Kepler ID must be positive")
        if self.tce_period <= 0:
            raise ValueError("Period must be positive")
        if self.tce_duration <= 0:
            raise ValueError("Duration must be positive")


@dataclass
class PredictionResult:
    """Prediction result with detailed information."""
    tce_data: TCEData
    prediction_score: float
    predicted_class: str
    confidence: float
    processing_time: float
    global_view: np.ndarray
    local_view: np.ndarray
    output_image_path: str
    model_version: str = "unknown"


class KeplerPredictor:
    """
    Modern Kepler exoplanet predictor using TensorFlow 2.x.

    This class provides a complete prediction pipeline including model loading,
    data preprocessing, prediction, and visualization.
    """

    def __init__(self,
                 model_path: Optional[str] = None,
                 log_level: int = logging.INFO,
                 enable_gpu: bool = True):
        """
        Initialize the predictor.

        Args:
            model_path: Path to trained model weights
            log_level: Logging level
            enable_gpu: Whether to use GPU for inference
        """
        self.model_path = model_path or os.path.join(
            environment.KEPLER_TRAINED_MODEL_FOLDER,
            'kepler-model-two-classes.h5'
        )
        self.enable_gpu = enable_gpu
        self.model = None
        self.logger = self._setup_logging(log_level)
        self._setup_tensorflow()

    def _setup_logging(self, log_level: int) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('kepler_prediction.log')
            ]
        )
        return logging.getLogger(self.__class__.__name__)

    def _setup_tensorflow(self) -> None:
        """Setup TensorFlow configuration for inference."""
        # Configure GPU if available
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus and self.enable_gpu:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                self.logger.info(f"GPU enabled for inference: {len(gpus)} GPU(s)")
            except RuntimeError as e:
                self.logger.error(f"GPU configuration failed: {e}")
                self.enable_gpu = False
        else:
            self.logger.info("Using CPU for inference")

    def load_model(self) -> None:
        """Load the trained model."""
        try:
            self.logger.info(f"Loading model from {self.model_path}")

            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            # Build model using the modern implementation
            cnn = Kepler_cnn.KeplerCNN()
            self.model = cnn.build_compiled_model()

            # Load weights
            self.model.load_weights(self.model_path)
            self.logger.info("Model loaded successfully")

            # Log model information
            total_params = self.model.count_params()
            self.logger.info(f"Model parameters: {total_params:,}")

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def _get_light_curve_data_path(self, kepid: int) -> Tuple[str, bool]:
        """
        Get the path to light curve data.

        Args:
            kepid: Kepler ID

        Returns:
            Tuple of (data_folder, from_unclassified_flag)
        """
        kepid_formatted = f"{kepid:09d}"

        # First check KEPLER_DATA_FOLDER
        data_folder = os.path.join(
            environment.KEPLER_DATA_FOLDER,
            kepid_formatted[0:4],
            kepid_formatted
        )

        if os.path.exists(data_folder):
            return data_folder, False

        # Check DATA_FOR_PREDICTION_FOLDER
        data_folder = os.path.join(
            environment.DATA_FOR_PREDICTION_FOLDER,
            kepid_formatted[0:4],
            kepid_formatted
        )

        if os.path.exists(data_folder):
            self.logger.info(f"Using data from DATA_FOR_PREDICTION_FOLDER for KIC {kepid}")
            return data_folder, True

        # Need to download data
        self.logger.info(f"Downloading data for KIC {kepid}")
        Download_one_Kepler_ID.download_one_kepler_id_files(kepid)
        return data_folder, True

    def _preprocess_light_curve(self, tce_data: TCEData) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess light curve data for prediction.

        Args:
            tce_data: TCE data structure

        Returns:
            Tuple of (global_view, local_view)
        """
        # Get data path
        data_folder, from_unclassified = self._get_light_curve_data_path(tce_data.kepid)

        # Choose base folder
        base_folder = (environment.DATA_FOR_PREDICTION_FOLDER
                      if from_unclassified else environment.KEPLER_DATA_FOLDER)

        # Read and process light curve
        self.logger.info(f"Processing light curve for KIC {tce_data.kepid}")
        time, flux = preprocess.read_and_process_light_curve(
            tce_data.kepid, base_folder, 0.75
        )

        # Phase fold and sort
        time, flux = preprocess.phase_fold_and_sort_light_curve(
            time, flux, tce_data.tce_period, tce_data.tce_time0bk
        )

        # Generate views
        global_view = preprocess.global_view(time, flux, tce_data.tce_period)
        local_view = preprocess.local_view(time, flux, tce_data.tce_period, tce_data.tce_duration)

        return global_view, local_view

    def _create_visualization(self,
                           tce_data: TCEData,
                           global_view: np.ndarray,
                           local_view: np.ndarray,
                           prediction_result: PredictionResult) -> str:
        """
        Create visualization of global and local views.

        Args:
            tce_data: TCE data
            global_view: Global view array
            local_view: Local view array
            prediction_result: Prediction result

        Returns:
            Path to saved image
        """
        try:
            # Create output directory
            os.makedirs(environment.PREDICT_OUTPUT_FOLDER, exist_ok=True)

            # Generate filename with safe characters
            filename = (f"{tce_data.kepid}_period={tce_data.tce_period:.6f}_"
                       f"time0bk={tce_data.tce_time0bk:.6f}_duration={tce_data.tce_duration*24:.3f}.png")
            output_path = os.path.join(environment.PREDICT_OUTPUT_FOLDER, filename)

            # Remove existing file
            if os.path.exists(output_path):
                os.remove(output_path)

            # Validate input data
            if global_view.size == 0 or local_view.size == 0:
                raise ValueError("Empty global or local view data")

            # Create figure with error handling
            plt.style.use('default')  # Ensure consistent style
            fig, axes = plt.subplots(1, 2, figsize=(20, 8), squeeze=False)

            # Set main title
            fig.suptitle(f'Kepler ID: {tce_data.kepid} - Prediction: {prediction_result.predicted_class} '
                        f'({prediction_result.confidence:.1%})', fontsize=14, fontweight='bold')

            # Plot global view
            axes[0, 0].plot(global_view, '.', markersize=2, alpha=0.7, color='blue')
            axes[0, 0].set_title("Global View", fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel("Bucketized Time (days)", fontsize=10)
            axes[0, 0].set_ylabel("Normalized Flux", fontsize=10)
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_xlim(left=0, right=len(global_view)-1)

            # Plot local view
            axes[0, 1].plot(local_view, '.', markersize=2, alpha=0.7, color='red')
            axes[0, 1].set_title("Local View", fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel("Bucketized Time (days)", fontsize=10)
            axes[0, 1].set_ylabel("Normalized Flux", fontsize=10)
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_xlim(left=0, right=len(local_view)-1)

            # Add prediction information box
            info_text = (f"TCE Parameters:\n"
                        f"Period: {tce_data.tce_period:.3f} days\n"
                        f"Duration: {tce_data.tce_duration*24:.2f} hours\n"
                        f"Time0bk: {tce_data.tce_time0bk:.3f}\n\n"
                        f"Prediction Results:\n"
                        f"Class: {prediction_result.predicted_class}\n"
                        f"Confidence: {prediction_result.confidence:.1%}\n"
                        f"Score: {prediction_result.prediction_score:.4f}")

            fig.text(0.02, 0.98, info_text, transform=fig.transFigure,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))

            # Add processing time info
            time_text = f"Processing time: {prediction_result.processing_time:.3f}s"
            fig.text(0.98, 0.02, time_text, transform=fig.transFigure,
                    fontsize=9, horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

            plt.tight_layout(rect=[0, 0.03, 1, 0.96])  # Adjust for text boxes

            # Save figure with error handling
            plt.savefig(output_path, bbox_inches="tight", dpi=150, facecolor='white')
            plt.close(fig)  # Explicitly close figure

            self.logger.info(f"Visualization saved to {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"Failed to create visualization: {e}")
            # Return empty path if visualization fails
            return ""

    def _interpret_prediction(self, prediction: np.ndarray) -> Tuple[str, float, str]:
        """
        Interpret model prediction.

        Args:
            prediction: Model prediction output

        Returns:
            Tuple of (predicted_class, confidence, description)
        """
        if environment.NB_CLASSES == 2:
            # Binary classification
            pc_probability = float(prediction[0][0])
            non_pc_probability = float(prediction[0][1])

            if pc_probability > 0.5:
                predicted_class = "PC (planet candidate)"
                confidence = pc_probability
                description = f"Planet candidate with {confidence:.1%} confidence"
            else:
                predicted_class = "NON_PC (not a planet candidate)"
                confidence = non_pc_probability
                description = f"Not a planet candidate with {confidence:.1%} confidence"

            return predicted_class, confidence, description

        else:
            # Multi-class classification
            class_idx = np.argmax(prediction[0])
            confidence = float(np.max(prediction[0]))

            # Map class index to label
            if class_idx < len(environment.ALLOWED_LABELS):
                label = environment.ALLOWED_LABELS[class_idx]
            else:
                label = f"Class_{class_idx}"

            predicted_class = f"{label} (class {class_idx})"
            description = f"{label} with {confidence:.1%} confidence"

            return predicted_class, confidence, description

    def predict_single(self, tce_data: TCEData) -> PredictionResult:
        """
        Predict for a single TCE.

        Args:
            tce_data: TCE data structure

        Returns:
            PredictionResult: Detailed prediction result
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        start_time = time.time()

        try:
            self.logger.info(f"Starting prediction for KIC {tce_data.kepid}")

            # Preprocess light curve
            global_view, local_view = self._preprocess_light_curve(tce_data)

            # Validate data shapes
            if global_view.shape != (2001,) or local_view.shape != (201,):
                raise ValueError(f"Invalid data shapes: global={global_view.shape}, local={local_view.shape}")

            # Reshape for model input
            global_view = np.reshape(global_view, (2001, 1))
            local_view = np.reshape(local_view, (201, 1))

            # Prepare input for model
            model_input = {
                'global_input': np.expand_dims(global_view, axis=0),
                'local_input': np.expand_dims(local_view, axis=0)
            }

            # Make prediction with error handling
            self.logger.info("Running model inference")
            try:
                prediction = self.model.predict(model_input, verbose=0)
            except Exception as e:
                self.logger.error(f"Model inference failed: {e}")
                raise RuntimeError(f"Model inference failed: {e}")

            # Validate prediction output
            if prediction is None or len(prediction) == 0:
                raise ValueError("Model returned empty prediction")

            # Interpret prediction
            predicted_class, confidence, description = self._interpret_prediction(prediction)
            prediction_score = float(np.max(prediction))

            # Create visualization
            result = PredictionResult(
                tce_data=tce_data,
                prediction_score=prediction_score,
                predicted_class=predicted_class,
                confidence=confidence,
                processing_time=time.time() - start_time,
                global_view=global_view,
                local_view=local_view,
                output_image_path=""
            )

            output_path = self._create_visualization(tce_data, global_view, local_view, result)
            result.output_image_path = output_path

            # Log result
            self._log_prediction_result(result)

            return result

        except Exception as e:
            self.logger.error(f"Prediction failed for KIC {tce_data.kepid}: {e}")
            # Re-raise with more context
            raise RuntimeError(f"Prediction failed for KIC {tce_data.kepid}: {e}") from e

    def _log_prediction_result(self, result: PredictionResult) -> None:
        """Log prediction result."""
        self.logger.info("=" * 60)
        self.logger.info("PREDICTION RESULT")
        self.logger.info("=" * 60)
        self.logger.info(f"Kepler ID: {result.tce_data.kepid}")
        self.logger.info(f"Period: {result.tce_data.tce_period:.6f} days")
        self.logger.info(f"Time0bk: {result.tce_data.tce_time0bk:.6f}")
        self.logger.info(f"Duration: {result.tce_data.tce_duration*24:.3f} hours")
        self.logger.info(f"Predicted class: {result.predicted_class}")
        self.logger.info(f"Confidence: {result.confidence:.2%}")
        self.logger.info(f"Processing time: {result.processing_time:.3f} seconds")
        self.logger.info(f"Output image: {result.output_image_path}")
        self.logger.info("=" * 60)

    def predict_batch(self, tce_list: List[TCEData]) -> List[PredictionResult]:
        """
        Predict for multiple TCEs.

        Args:
            tce_list: List of TCE data structures

        Returns:
            List of prediction results
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        results = []
        total_time = 0

        self.logger.info(f"Starting batch prediction for {len(tce_list)} TCEs")

        for i, tce_data in enumerate(tce_list):
            try:
                self.logger.info(f"Processing TCE {i+1}/{len(tce_list)}")
                result = self.predict_single(tce_data)
                results.append(result)
                total_time += result.processing_time

            except Exception as e:
                self.logger.error(f"Failed to process TCE {i+1}: {e}")
                # Continue with next TCE
                continue

        # Log batch statistics
        avg_time = total_time / len(results) if results else 0
        self.logger.info(f"Batch prediction completed:")
        self.logger.info(f"  Successful: {len(results)}/{len(tce_list)}")
        self.logger.info(f"  Average time per TCE: {avg_time:.3f} seconds")

        return results


def create_sample_tce_data() -> List[TCEData]:
    """
    Create sample TCE data for testing.

    Returns:
        List of sample TCE data
    """
    return [
        TCEData(
            kepid=11442793,
            tce_period=14.44912,
            tce_time0bk=2.2,
            tce_duration=0.11267  # 2.70408 hours / 24
        ),
        # Additional sample TCEs can be added here
        # TCEData(
        #     kepid=757450,
        #     tce_period=8.88492,
        #     tce_time0bk=134.452,
        #     tce_duration=0.08658  # 2.078 hours / 24
        # ),
    ]


def main():
    """
    Main function to run Kepler prediction.

    This function demonstrates the usage of the KeplerPredictor class
    with sample data and proper error handling.
    """
    # Configuration
    config = {
        'model_path': None,  # Use default path
        'log_level': logging.INFO,
        'enable_gpu': True
    }

    try:
        # Create predictor
        predictor = KeplerPredictor(**config)

        # Load model
        predictor.load_model()

        # Create sample TCE data
        tce_list = create_sample_tce_data()

        # Run predictions
        results = predictor.predict_batch(tce_list)

        # Print summary
        print(f"\nPrediction Summary:")
        print(f"Processed {len(results)} TCEs successfully")

        for result in results:
            print(f"\nKIC {result.tce_data.kepid}:")
            print(f"  {result.predicted_class}")
            print(f"  Confidence: {result.confidence:.2%}")
            print(f"  Image: {os.path.basename(result.output_image_path)}")

        return 0 if results else 1

    except KeyboardInterrupt:
        print("\nPrediction interrupted by user")
        return 1
    except Exception as e:
        print(f"Prediction failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
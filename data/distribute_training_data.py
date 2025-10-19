"""
Kepler Training Data Distribution (sklearn train_test_split Implementation)

This module distributes Kepler TCE (Threshold Crossing Event) data into training,
validation, and test sets using sklearn's train_test_split method for stratified
sampling. It ensures proper class distribution across all splits and provides
comprehensive error handling and logging.

Features:
- Stratified sampling using sklearn.model_selection.train_test_split
- Configurable split ratios with validation
- Comprehensive error handling and logging
- File existence verification and validation
- Progress tracking and statistics
- Class balance analysis and reporting
"""

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from configure import environment


@dataclass
class DistributionStats:
    """Statistics for data distribution operations."""
    total_records: int
    train_count: int
    validation_count: int
    test_count: int
    successful_copies: int = 0
    failed_copies: int = 0
    skipped_copies: int = 0
    class_distribution: Dict[str, Dict[str, int]] = None

    def __post_init__(self):
        """Initialize class distribution dictionary."""
        if self.class_distribution is None:
            self.class_distribution = {}

    @property
    def train_ratio(self) -> float:
        """Calculate actual train ratio."""
        return self.train_count / self.total_records if self.total_records > 0 else 0.0

    @property
    def validation_ratio(self) -> float:
        """Calculate actual validation ratio."""
        return self.validation_count / self.total_records if self.total_records > 0 else 0.0

    @property
    def test_ratio(self) -> float:
        """Calculate actual test ratio."""
        return self.test_count / self.total_records if self.total_records > 0 else 0.0

    @property
    def success_rate(self) -> float:
        """Calculate file copy success rate."""
        total_attempts = self.successful_copies + self.failed_copies + self.skipped_copies
        return self.successful_copies / total_attempts if total_attempts > 0 else 0.0


class TrainingDataDistributor:
    """
    Modern training data distributor using sklearn's train_test_split.

    This class handles the distribution of Kepler TCE data into training,
    validation, and test sets with proper stratification and validation.
    """

    def __init__(self,
                 train_ratio: float = 0.8,
                 validation_ratio: float = 0.1,
                 test_ratio: float = 0.1,
                 random_state: int = 42,
                 log_level: int = logging.INFO,
                 log_file: str = 'data_distribution.log'):
        """
        Initialize the training data distributor.

        Args:
            train_ratio: Proportion of data for training (0.0-1.0)
            validation_ratio: Proportion of data for validation (0.0-1.0)
            test_ratio: Proportion of data for testing (0.0-1.0)
            random_state: Random seed for reproducibility
            log_level: Logging level
            log_file: Log file path

        Raises:
            ValueError: If ratios don't sum to 1.0 or are invalid
        """
        # Validate ratios
        self._validate_split_ratios(train_ratio, validation_ratio, test_ratio)

        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.log_file = log_file
        self.stats = DistributionStats(
            total_records=0,
            train_count=0,
            validation_count=0,
            test_count=0
        )

        # Setup logging
        self._setup_logging(log_level)

        # Validate environment
        self._validate_environment()

    def _validate_split_ratios(self, train: float, validation: float, test: float) -> None:
        """Validate that split ratios are valid and sum to 1.0."""
        total = train + validation + test
        tolerance = 1e-6

        if not (0.0 <= train <= 1.0 and 0.0 <= validation <= 1.0 and 0.0 <= test <= 1.0):
            raise ValueError("All split ratios must be between 0.0 and 1.0")

        if abs(total - 1.0) > tolerance:
            raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f}")

    def _setup_logging(self, log_level: int) -> None:
        """Setup logging configuration."""
        # Clear existing handlers to avoid duplicates
        logger = logging.getLogger(__name__)
        logger.handlers.clear()

        # Set logging level
        logger.setLevel(log_level)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        try:
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except (PermissionError, FileNotFoundError) as e:
            logger.warning(f"Could not create log file {self.log_file}: {e}")

        self.logger = logger

    def _validate_environment(self) -> None:
        """Validate environment configuration."""
        required_paths = [
            environment.KEPLER_CSV_FILE,
            environment.KEPLER_DATA_FOLDER,
            environment.TRAINING_FOLDER
        ]

        for path in required_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required path does not exist: {path}")

        self.logger.info("Environment validation passed")

    def load_and_filter_tce_data(self) -> pd.DataFrame:
        """
        Load and filter TCE data from CSV file.

        Returns:
            pd.DataFrame: Filtered TCE data

        Raises:
            FileNotFoundError: If CSV file doesn't exist
            pd.errors.EmptyDataError: If CSV file is empty
        """
        try:
            self.logger.info(f"Loading TCE data from {environment.KEPLER_CSV_FILE}")

            # Read CSV file
            tce_table = pd.read_csv(
                environment.KEPLER_CSV_FILE,
                index_col="loc_rowid",
                comment="#"
            )

            # Convert duration from hours to days
            tce_table["tce_duration"] /= 24

            initial_count = len(tce_table)
            self.logger.info(f"Loaded {initial_count} TCE records from CSV")

            # Filter for allowed labels
            allowed_mask = tce_table[environment.LABEL_COLUMN].apply(
                lambda label: label in environment.ALLOWED_LABELS
            )
            tce_table = tce_table[allowed_mask]

            filtered_count = len(tce_table)
            removed_count = initial_count - filtered_count

            self.logger.info(
                f"Filtered to {filtered_count} TCEs with labels in {environment.ALLOWED_LABELS}"
            )
            self.logger.info(f"Removed {removed_count} records (including 'UNK' labels)")

            return tce_table

        except FileNotFoundError:
            self.logger.error(f"CSV file not found: {environment.KEPLER_CSV_FILE}")
            raise
        except pd.errors.EmptyDataError:
            self.logger.error(f"CSV file is empty: {environment.KEPLER_CSV_FILE}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading TCE data: {e}")
            raise

    def create_stratified_splits(self, tce_table: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create stratified train/validation/test splits.

        Args:
            tce_table: Filtered TCE data

        Returns:
            Tuple of (train_df, validation_df, test_df)

        Raises:
            ValueError: If data is insufficient for stratified splitting
        """
        self.logger.info("Creating stratified data splits")

        # Check minimum records per class for stratification
        class_counts = tce_table[environment.LABEL_COLUMN].value_counts()
        min_class_count = class_counts.min()
        total_records = len(tce_table)

        # Validate minimum samples required for stratification
        min_required_samples = max(2, int(10 / self.test_ratio))  # At least 10 samples in test set
        if min_class_count < 2:
            raise ValueError(
                f"Insufficient data for stratified splitting. "
                f"Minimum class count: {min_class_count}, required: >= 2"
            )

        if total_records < 10:
            raise ValueError(
                f"Insufficient total data for splitting. "
                f"Total records: {total_records}, required: >= 10"
            )

        # Create binary labels for stratification (PC vs NON_PC)
        binary_labels = tce_table[environment.LABEL_COLUMN].apply(
            lambda label: 'PC' if label == 'PC' else 'NON_PC'
        )

        try:
            # First split: separate test set
            remaining_data, test_data = train_test_split(
                tce_table,
                test_size=self.test_ratio,
                stratify=binary_labels,
                random_state=self.random_state
            )

            # Calculate validation ratio from remaining data
            validation_ratio_adjusted = self.validation_ratio / (1 - self.test_ratio)

            # Create binary labels for remaining data
            remaining_labels = remaining_data[environment.LABEL_COLUMN].apply(
                lambda label: 'PC' if label == 'PC' else 'NON_PC'
            )

            # Second split: separate train and validation
            train_data, validation_data = train_test_split(
                remaining_data,
                test_size=validation_ratio_adjusted,
                stratify=remaining_labels,
                random_state=self.random_state
            )

        except ValueError as e:
            if "cannot allocate array of size" in str(e):
                raise ValueError(
                    f"Memory error during data splitting. "
                    f"Try reducing the dataset size or using a simpler splitting method."
                ) from e
            elif "Minimum number of samples required" in str(e):
                # Fall back to non-stratified splitting if stratification fails
                self.logger.warning(
                    "Stratified splitting failed due to insufficient samples. "
                    "Falling back to random splitting."
                )
                remaining_data, test_data = train_test_split(
                    tce_table, test_size=self.test_ratio, random_state=self.random_state
                )
                validation_ratio_adjusted = self.validation_ratio / (1 - self.test_ratio)
                train_data, validation_data = train_test_split(
                    remaining_data, test_size=validation_ratio_adjusted, random_state=self.random_state
                )
            else:
                raise

        # Validate splits
        for split_name, split_data in [("Train", train_data), ("Validation", validation_data), ("Test", test_data)]:
            if len(split_data) == 0:
                raise ValueError(f"Empty {split_name} split created")

        # Log split statistics
        self.logger.info(f"Data splits created:")
        self.logger.info(f"  Train: {len(train_data)} records ({len(train_data)/len(tce_table):.1%})")
        self.logger.info(f"  Validation: {len(validation_data)} records ({len(validation_data)/len(tce_table):.1%})")
        self.logger.info(f"  Test: {len(test_data)} records ({len(test_data)/len(tce_table):.1%})")

        # Log class distributions
        self._log_class_distribution(train_data, "Train")
        self._log_class_distribution(validation_data, "Validation")
        self._log_class_distribution(test_data, "Test")

        return train_data, validation_data, test_data

    def _log_class_distribution(self, data: pd.DataFrame, split_name: str) -> None:
        """Log class distribution for a data split."""
        if len(data) == 0:
            self.logger.warning(f"No data in {split_name} split")
            return

        class_counts = data[environment.LABEL_COLUMN].value_counts()
        total = len(data)

        self.logger.info(f"{split_name} class distribution:")
        for label in environment.ALLOWED_LABELS:  # Ensure consistent order
            count = class_counts.get(label, 0)
            percentage = (count / total) * 100
            self.logger.info(f"  {label}: {count} ({percentage:.1f}%)")

        # Store in stats
        self.stats.class_distribution[split_name] = class_counts.to_dict()

    def _get_source_filename(self, tce_data: pd.Series) -> str:
        """Generate source filename for TCE data."""
        kep_id = f"{int(tce_data.kepid):09d}"
        kepid_dir = os.path.join(
            environment.KEPLER_DATA_FOLDER,
            kep_id[0:4],
            kep_id
        )
        return os.path.join(
            kepid_dir,
            f"{int(tce_data.kepid):09d}_plnt_num-{tce_data.tce_plnt_num:02d}_tce.record"
        )

    def _get_label_folder(self, tce_data: pd.Series) -> str:
        """Get label folder name based on TCE classification."""
        if tce_data[environment.LABEL_COLUMN] == 'PC':
            return '0_PC'
        else:
            return '1_NON_PC'

    def _create_output_directory(self, data_type: str, label_folder: str) -> str:
        """Create output directory structure."""
        dest_folder = os.path.join(
            environment.TRAINING_FOLDER,
            data_type,
            label_folder
        )
        os.makedirs(dest_folder, exist_ok=True)
        return dest_folder

    def distribute_data_split(self,
                            tce_data: pd.DataFrame,
                            data_type: str) -> None:
        """
        Distribute a data split to appropriate folders.

        Args:
            tce_data: DataFrame containing TCE records for this split
            data_type: Type of data ('Train', 'Validation', 'Test')
        """
        self.logger.info(f"Distributing {data_type} data ({len(tce_data)} records)")

        # Create base directory for this data type
        data_type_folder = os.path.join(environment.TRAINING_FOLDER, data_type)

        # Remove existing directory if it exists
        if os.path.exists(data_type_folder):
            self.logger.info(f"Removing existing directory: {data_type_folder}")
            shutil.rmtree(data_type_folder)

        # Create fresh directory
        os.makedirs(data_type_folder)
        self.logger.info(f"Created directory: {data_type_folder}")

        # Process each TCE record
        for i, (_, tce_record) in enumerate(tce_data.iterrows()):
            try:
                # Get source filename
                source_file = self._get_source_filename(tce_record)

                # Verify source file exists
                if not os.path.isfile(source_file):
                    self.logger.warning(f"Source file not found: {source_file}")
                    self.stats.skipped_copies += 1
                    continue

                # Get label folder and create destination
                label_folder = self._get_label_folder(tce_record)
                dest_folder = self._create_output_directory(data_type, label_folder)

                # Copy file
                shutil.copy2(source_file, dest_folder)
                self.stats.successful_copies += 1

                # Log progress every 100 files or at completion
                if (i + 1) % 100 == 0 or (i + 1) == len(tce_data):
                    self.logger.info(
                        f"Processed {i + 1}/{len(tce_data)} files for {data_type}"
                    )

            except Exception as e:
                self.logger.error(f"Error processing TCE {i}: {e}")
                self.stats.failed_copies += 1

        self.logger.info(f"Completed {data_type} distribution")

    def distribute_training_data(self) -> DistributionStats:
        """
        Main method to distribute training data into train/validation/test sets.

        Returns:
            DistributionStats: Statistics about the distribution

        Raises:
            Exception: If distribution fails critically
        """
        try:
            # Load and validate TCE data
            tce_table = self.load_and_filter_tce_data()

            if len(tce_table) == 0:
                self.logger.warning("No TCE data to distribute")
                return self.stats

            # Update statistics
            self.stats.total_records = len(tce_table)

            # Create stratified splits
            train_data, validation_data, test_data = self.create_stratified_splits(tce_table)

            # Update split counts
            self.stats.train_count = len(train_data)
            self.stats.validation_count = len(validation_data)
            self.stats.test_count = len(test_data)

            # Distribute data
            self.distribute_data_split(train_data, 'Train')
            self.distribute_data_split(validation_data, 'Validation')
            self.distribute_data_split(test_data, 'Test')

            # Log final statistics
            self._log_final_statistics()

            return self.stats

        except Exception as e:
            self.logger.error(f"Critical error in data distribution: {e}")
            raise

    def _log_final_statistics(self) -> None:
        """Log final distribution statistics."""
        self.logger.info("=" * 60)
        self.logger.info("TRAINING DATA DISTRIBUTION COMPLETED")
        self.logger.info("=" * 60)
        self.logger.info(f"Total records: {self.stats.total_records}")
        self.logger.info(f"Train set: {self.stats.train_count} ({self.stats.train_ratio:.1%})")
        self.logger.info(f"Validation set: {self.stats.validation_count} ({self.stats.validation_ratio:.1%})")
        self.logger.info(f"Test set: {self.stats.test_count} ({self.stats.test_ratio:.1%})")
        self.logger.info(f"Successful copies: {self.stats.successful_copies}")
        self.logger.info(f"Failed copies: {self.stats.failed_copies}")
        self.logger.info(f"Skipped copies: {self.stats.skipped_copies}")
        self.logger.info(f"Success rate: {self.stats.success_rate:.1%}")
        self.logger.info("=" * 60)

        # Log class balance summary
        if self.stats.class_distribution:
            self.logger.info("Class Balance Summary:")
            for split_name, class_counts in self.stats.class_distribution.items():
                total = sum(class_counts.values())
                self.logger.info(f"  {split_name}:")
                for label, count in class_counts.items():
                    percentage = (count / total) * 100
                    self.logger.info(f"    {label}: {count} ({percentage:.1f}%)")


def main():
    """
    Main function to run training data distribution.

    This function maintains backward compatibility with the original script
    while providing enhanced functionality through sklearn's train_test_split.
    """
    # Configuration
    config = {
        'train_ratio': 0.8,
        'validation_ratio': 0.1,
        'test_ratio': 0.1,
        'random_state': 42,
        'log_level': logging.INFO,
        'log_file': 'data_distribution.log'
    }

    try:
        # Create distributor instance
        distributor = TrainingDataDistributor(**config)

        # Distribute training data
        stats = distributor.distribute_training_data()

        # Return success
        return 0 if stats.failed_copies == 0 else 1

    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
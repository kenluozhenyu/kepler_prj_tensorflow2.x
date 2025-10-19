"""
Kepler Training Data Generation (Modern Python Multi-threading Implementation)

This module processes Kepler TCE (Threshold Crossing Event) data from CSV files
and generates training data in the form of pickled [global_view, local_view] pairs.
It uses modern Python multi-threading frameworks for efficient parallel processing.

Features:
- Modern concurrent.futures.ThreadPoolExecutor for thread management
- Comprehensive error handling and logging
- Progress tracking and statistics
- Configurable thread count and batch processing
- Graceful shutdown and resource cleanup
"""

import logging
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np

from data import training_data_io
from configure import environment


@dataclass
class ProcessingStats:
    """Statistics for data processing progress tracking."""
    total_tces: int
    processed_tces: int = 0
    successful_tces: int = 0
    failed_tces: int = 0
    skipped_tces: int = 0
    start_time: float = 0.0

    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage."""
        return (self.processed_tces / self.total_tces * 100) if self.total_tces > 0 else 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return (self.successful_tces / self.processed_tces * 100) if self.processed_tces > 0 else 0.0

    @property
    def elapsed_time(self) -> float:
        """Calculate elapsed time in seconds."""
        return time.time() - self.start_time

    @property
    def estimated_remaining_time(self) -> float:
        """Estimate remaining processing time in seconds."""
        if self.processed_tces == 0:
            return 0.0
        avg_time_per_tce = self.elapsed_time / self.processed_tces
        remaining_tces = self.total_tces - self.processed_tces
        return avg_time_per_tce * remaining_tces


class TrainingDataGenerator:
    """
    Modern training data generator with improved multi-threading capabilities.

    This class handles the conversion of Kepler TCE data from CSV format
    to pickled [global_view, local_view] pairs suitable for CNN training.
    """

    def __init__(self,
                 num_workers: int = 8,
                 enable_multiprocessing: bool = True,
                 log_level: int = logging.INFO,
                 log_file: str = 'training_data_generation.log'):
        """
        Initialize the training data generator.

        Args:
            num_workers: Number of worker threads
            enable_multiprocessing: Whether to use multi-threading
            log_level: Logging level
            log_file: Log file path
        """
        self.num_workers = max(1, min(num_workers, os.cpu_count() or 1))  # Validate thread count
        self.enable_multiprocessing = enable_multiprocessing
        self.log_file = log_file
        self.stats = ProcessingStats(total_tces=0)
        self._shutdown_requested = False

        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()

        # Setup logging
        self._setup_logging(log_level)

        # Validate environment configuration
        self._validate_environment()

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            print(f"\nReceived signal {signum}. Initiating graceful shutdown...")
            self._shutdown_requested = True

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

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
        console_handler = logging.StreamHandler(sys.stdout)
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
            environment.KEPLER_DATA_FOLDER
        ]

        for path in required_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required path does not exist: {path}")

        self.logger.info("Environment validation passed")

    def load_tce_data(self) -> pd.DataFrame:
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

    def process_single_tce(self, tce_data: pd.Series, worker_id: int) -> Tuple[bool, str]:
        """
        Process a single TCE record.

        Args:
            tce_data: Single TCE record
            worker_id: ID of the worker processing this TCE

        Returns:
            Tuple[bool, str]: (success, message)
        """
        # Check for shutdown request
        if self._shutdown_requested:
            return False, "Shutdown requested"

        try:
            # Validate TCE data
            if not hasattr(tce_data, 'kepid') or not hasattr(tce_data, 'tce_plnt_num'):
                return False, "Invalid TCE data structure"

            kepid = int(tce_data.kepid)
            plnt_num = int(tce_data.tce_plnt_num)

            self.logger.debug(
                f"Worker {worker_id}: Processing Kepler ID {kepid:09d}, "
                f"planet number {plnt_num:2d}"
            )

            # Check if file already exists to skip processing
            output_file = self._get_output_filename(kepid, plnt_num)
            if os.path.exists(output_file):
                self.logger.debug(f"Skipping existing file: {output_file}")
                return True, "File already exists"

            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            # Process TCE data
            training_data_io.tce_global_view_local_view_to_file(tce_data)

            self.logger.debug(
                f"Worker {worker_id}: Successfully processed Kepler ID {kepid:09d}"
            )
            return True, "Success"

        except ValueError as e:
            error_msg = f"Invalid data format for TCE (KepID: {tce_data.get('kepid', 'unknown')}): {e}"
            self.logger.error(error_msg)
            return False, str(e)
        except OSError as e:
            error_msg = f"File system error for TCE (KepID: {tce_data.get('kepid', 'unknown')}): {e}"
            self.logger.error(error_msg)
            return False, str(e)
        except Exception as e:
            error_msg = f"Unexpected error processing TCE (KepID: {tce_data.get('kepid', 'unknown')}): {e}"
            self.logger.error(error_msg)
            return False, str(e)

    def _get_output_filename(self, kepid: int, plnt_num: int) -> str:
        """Generate output filename for TCE data."""
        kep_id_formatted = f"{kepid:09d}"
        kepid_dir = os.path.join(
            environment.KEPLER_DATA_FOLDER,
            kep_id_formatted[0:4],
            kep_id_formatted
        )
        return os.path.join(
            kepid_dir,
            f"{kepid:09d}_plnt_num-{plnt_num:02d}_tce.record"
        )

    def process_tces_parallel(self, tce_table: pd.DataFrame) -> None:
        """
        Process TCE data using multiple threads.

        Args:
            tce_table: DataFrame containing TCE records
        """
        self.stats.total_tces = len(tce_table)
        self.stats.start_time = time.time()

        self.logger.info(
            f"Starting parallel processing with {self.num_workers} workers"
        )
        self.logger.info(f"Total TCEs to process: {self.stats.total_tces}")

        # Convert DataFrame to list for easier indexing
        tce_list = [tce_table.iloc[i] for i in range(len(tce_table))]

        # Use ThreadPoolExecutor with proper context management
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit tasks in batches to avoid memory issues
            batch_size = 1000
            all_futures = []

            for batch_start in range(0, len(tce_list), batch_size):
                if self._shutdown_requested:
                    self.logger.info("Shutdown requested, stopping submission of new tasks")
                    break

                batch_end = min(batch_start + batch_size, len(tce_list))
                batch_futures = {
                    executor.submit(
                        self.process_single_tce,
                        tce_data,
                        worker_id=i % self.num_workers + 1
                    ): (i, tce_data)
                    for i, tce_data in enumerate(tce_list[batch_start:batch_end], batch_start)
                }
                all_futures.extend(batch_futures.items())

                self.logger.debug(f"Submitted batch {batch_start//batch_size + 1}, "
                                f"tasks {batch_start}-{batch_end-1}")

            # Process completed tasks
            for future, (tce_index, tce_data) in all_futures:
                if self._shutdown_requested:
                    self.logger.info("Shutdown requested, stopping processing")
                    break

                try:
                    success, message = future.result(timeout=300)  # 5 minute timeout per task
                    self.stats.processed_tces += 1

                    if success:
                        self.stats.successful_tces += 1
                        if message != "File already exists":
                            self.logger.info(
                                f"Processed {self.stats.processed_tces}/{self.stats.total_tces}: "
                                f"Kepler ID {int(tce_data.kepid):09d}, "
                                f"planet {tce_data.tce_plnt_num:2d}"
                            )
                        else:
                            self.stats.skipped_tces += 1
                    else:
                        self.stats.failed_tces += 1
                        if message != "Shutdown requested":
                            self.logger.error(
                                f"Failed {self.stats.processed_tces}/{self.stats.total_tces}: "
                                f"Kepler ID {int(tce_data.kepid):09d}, "
                                f"planet {tce_data.tce_plnt_num:2d} - {message}"
                            )

                    # Log progress every 100 TCEs or at completion
                    if self.stats.processed_tces % 100 == 0 or \
                       self.stats.processed_tces == self.stats.total_tces:
                        self._log_progress()

                except TimeoutError:
                    self.stats.processed_tces += 1
                    self.stats.failed_tces += 1
                    self.logger.error(f"Timeout processing TCE {tce_index}")
                except Exception as e:
                    self.stats.processed_tces += 1
                    self.stats.failed_tces += 1
                    self.logger.error(f"Exception processing TCE {tce_index}: {e}")

    def process_tces_sequential(self, tce_table: pd.DataFrame) -> None:
        """
        Process TCE data sequentially (single-threaded).

        Args:
            tce_table: DataFrame containing TCE records
        """
        self.stats.total_tces = len(tce_table)
        self.stats.start_time = time.time()

        self.logger.info(f"Starting sequential processing")
        self.logger.info(f"Total TCEs to process: {self.stats.total_tces}")

        for i in range(len(tce_table)):
            tce_data = tce_table.iloc[i]

            self.stats.processed_tces += 1

            success, message = self.process_single_tce(tce_data, worker_id=0)

            if success:
                self.stats.successful_tces += 1
                if message != "File already exists":
                    self.logger.info(
                        f"Processed {self.stats.processed_tces}/{self.stats.total_tces}: "
                        f"Kepler ID {int(tce_data.kepid):09d}, "
                        f"planet {tce_data.tce_plnt_num:2d}"
                    )
                else:
                    self.stats.skipped_tces += 1
            else:
                self.stats.failed_tces += 1
                self.logger.error(
                    f"Failed {self.stats.processed_tces}/{self.stats.total_tces}: "
                    f"Kepler ID {int(tce_data.kepid):09d}, "
                    f"planet {tce_data.tce_plnt_num:2d} - {message}"
                )

            # Log progress every 50 TCEs or at completion
            if self.stats.processed_tces % 50 == 0 or \
               self.stats.processed_tces == self.stats.total_tces:
                self._log_progress()

    def _log_progress(self) -> None:
        """Log current processing progress."""
        progress = self.stats.progress_percentage
        success_rate = self.stats.success_rate
        elapsed = self.stats.elapsed_time
        remaining = self.stats.estimated_remaining_time

        self.logger.info(
            f"Progress: {progress:.1f}% ({self.stats.processed_tces}/{self.stats.total_tces}) | "
            f"Success: {success_rate:.1f}% | "
            f"Skipped: {self.stats.skipped_tces} | "
            f"Failed: {self.stats.failed_tces} | "
            f"Elapsed: {elapsed:.1f}s | "
            f"ETA: {remaining:.1f}s"
        )

    def generate_training_data(self) -> ProcessingStats:
        """
        Main method to generate training data from TCE CSV file.

        Returns:
            ProcessingStats: Statistics about the processing

        Raises:
            Exception: If processing fails critically
        """
        try:
            # Load and validate TCE data
            tce_table = self.load_tce_data()

            if len(tce_table) == 0:
                self.logger.warning("No TCE data to process")
                return self.stats

            # Process data
            if self.enable_multiprocessing:
                self.process_tces_parallel(tce_table)
            else:
                self.process_tces_sequential(tce_table)

            # Log final statistics
            self._log_final_stats()

            return self.stats

        except Exception as e:
            self.logger.error(f"Critical error in training data generation: {e}")
            raise

    def _log_final_stats(self) -> None:
        """Log final processing statistics."""
        total_time = self.stats.elapsed_time
        success_rate = self.stats.success_rate

        self.logger.info("=" * 60)
        self.logger.info("TRAINING DATA GENERATION COMPLETED")
        self.logger.info("=" * 60)
        self.logger.info(f"Total TCEs: {self.stats.total_tces}")
        self.logger.info(f"Successfully processed: {self.stats.successful_tces}")
        self.logger.info(f"Skipped (already exist): {self.stats.skipped_tces}")
        self.logger.info(f"Failed: {self.stats.failed_tces}")
        self.logger.info(f"Success rate: {success_rate:.2f}%")
        self.logger.info(f"Total processing time: {total_time:.2f} seconds")

        if self.stats.processed_tces > 0:
            avg_time = total_time / self.stats.processed_tces
            self.logger.info(f"Average time per TCE: {avg_time:.2f} seconds")

        self.logger.info("=" * 60)


def main():
    """
    Main function to run training data generation.

    This function maintains backward compatibility with the original script
    while providing enhanced functionality through the new implementation.
    """
    # Configuration
    num_workers = 8  # Adjust based on your system
    enable_multiprocessing = True  # Set to False for single-threaded processing
    log_level = logging.INFO  # Set to logging.DEBUG for verbose output

    try:
        # Create generator instance
        generator = TrainingDataGenerator(
            num_workers=num_workers,
            enable_multiprocessing=enable_multiprocessing,
            log_level=log_level
        )

        # Generate training data
        stats = generator.generate_training_data()

        # Return success
        return 0 if stats.failed_tces == 0 else 1

    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
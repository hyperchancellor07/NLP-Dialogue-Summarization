from pathlib import Path
from datasets import load_dataset, DatasetDict
import json
import numpy as np
from datetime import datetime
from src.utils.common import read_yaml, create_directories, PROJECT_ROOT
from src.exception.custom_exception import CustomException
from src.logger.logger import logging
import sys


class DataIngestion:
    def __init__(self, config_path: Path):
        try:
            self.config = read_yaml(config_path)

            self.raw_data_dir = (
                PROJECT_ROOT
                / self.config.data_ingestion.raw_data_dir
            )

            self.ingested_data_dir = (
                PROJECT_ROOT
                / self.config.data_ingestion.processed_data_dir
            )

            create_directories(
                [self.ingested_data_dir],
                verbose=True
            )

        except Exception as e:
            raise CustomException(e, sys)


    def load_raw_data(self) -> DatasetDict:
        """
        Load raw SAMSum data from JSONL artifacts.
        """
        try:
            logging.info("Loading raw SAMSum dataset from artifacts")

            data_files = {
                "train": str(self.raw_data_dir / "samsum_train.json"),
                "validation": str(self.raw_data_dir / "samsum_validation.json"),
                "test": str(self.raw_data_dir / "samsum_test.json"),
            }

            dataset = load_dataset(
                "json",
                data_files=data_files
            )

            logging.info("Raw dataset successfully loaded")
            return dataset

        except Exception as e:
            raise CustomException(e, sys)


    def validate_schema(self, dataset: DatasetDict):
        """
        Validate expected schema of the dataset.
        """
        try:
            logging.info("Validating dataset schema")

            expected_columns = {"id", "dialogue", "summary"}

            for split in dataset.keys():
                actual_columns = set(dataset[split].column_names)
                if not expected_columns.issubset(actual_columns):
                    raise ValueError(
                        f"Schema mismatch in {split}: "
                        f"Expected {expected_columns}, "
                        f"Found {actual_columns}"
                    )

            logging.info("Dataset schema validation passed")

        except Exception as e:
            raise CustomException(e, sys)


    def ingest(self) -> DatasetDict:
        """
        Main ingestion method (Part 1).
        """
        try:
            dataset = self.load_raw_data()
            self.validate_schema(dataset)

            logging.info("Data Ingestion Part 1 completed successfully")
            return dataset

        except Exception as e:
            raise CustomException(e, sys)
    def compute_basic_stats(self, dataset: DatasetDict) -> dict:
        """
        Compute basic statistics for each split.
        """
        try:
            logging.info("Computing basic dataset statistics")

            stats = {}

            for split in dataset.keys():
                dialogues = dataset[split]["dialogue"]
                summaries = dataset[split]["summary"]

                dialogue_lengths = [len(d.split()) for d in dialogues]
                summary_lengths = [len(s.split()) for s in summaries]

                stats[split] = {
                    "num_samples": len(dialogues),
                    "dialogue_length": {
                        "mean": float(np.mean(dialogue_lengths)),
                        "median": float(np.median(dialogue_lengths)),
                        "p95": float(np.percentile(dialogue_lengths, 95)),
                    },
                    "summary_length": {
                        "mean": float(np.mean(summary_lengths)),
                        "median": float(np.median(summary_lengths)),
                        "p95": float(np.percentile(summary_lengths, 95)),
                    },
                }

            logging.info("Basic statistics computed successfully")
            return stats

        except Exception as e:
            raise CustomException(e, sys)
    def detect_pretraining_drift(self, stats: dict) -> dict:
        """
        Detect potential drift across dataset splits using simple heuristics.
        """
        try:
            logging.info("Detecting pre-training data drift")

            drift_report = {}

            train_stats = stats.get("train")

            for split in ["validation", "test"]:
                split_stats = stats.get(split)

                drift_report[split] = {
                    "dialogue_length_mean_shift": (
                        split_stats["dialogue_length"]["mean"]
                        - train_stats["dialogue_length"]["mean"]
                    ),
                    "summary_length_mean_shift": (
                        split_stats["summary_length"]["mean"]
                        - train_stats["summary_length"]["mean"]
                    ),
                }

            logging.info("Pre-training drift analysis completed")
            return drift_report

        except Exception as e:
            raise CustomException(e, sys)
    def save_ingestion_report(
        self,
        stats: dict,
        drift_report: dict
        ):
        """
        Save ingestion metadata (stats + drift).
        """
        try:
            report = {
                "timestamp": datetime.utcnow().isoformat(),
                "stats": stats,
                "pretraining_drift": drift_report,
            }

            report_path = (
                self.ingested_data_dir / "ingestion_report.json"
            )

            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=4)

            logging.info(
                f"Ingestion report saved at {report_path}"
            )

        except Exception as e:
            raise CustomException(e, sys)
    def run(self) -> DatasetDict:
        """
        Full data ingestion pipeline (Part 1 + Part 2).
        """
        try:
            dataset = self.ingest()

            stats = self.compute_basic_stats(dataset)
            drift_report = self.detect_pretraining_drift(stats)

            self.save_ingestion_report(stats, drift_report)

            logging.info("Full data ingestion pipeline completed")
            return dataset

        except Exception as e:
            raise CustomException(e, sys)


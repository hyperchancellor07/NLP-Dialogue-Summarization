from pathlib import Path
import sys
import json
from datetime import datetime
from datasets import DatasetDict
from src.utils.common import read_yaml, PROJECT_ROOT, create_directories
from src.exception.custom_exception import CustomException
from src.logger.logger import logging

class DataValidation:
    def __init__(self, config_path: Path):
        try:
            self.config = read_yaml(config_path)

            self.validation_dir = (
                PROJECT_ROOT / "artifacts/data/validation"
            )

            create_directories([self.validation_dir], verbose=True)

            logging.info("DataValidation initialized")

        except Exception as e:
            raise CustomException(e, sys)

    def validate_structure(self, dataset: DatasetDict) -> dict:
        """
        Validate schema and null values.
        """
        try:
            logging.info("Validating dataset structure")

            report = {}

            required_columns = {"id", "dialogue", "summary"}

            for split in dataset.keys():
                split_report = {
                    "missing_columns": [],
                    "null_dialogues": 0,
                    "null_summaries": 0,
                    "empty_dialogues": 0,
                    "empty_summaries": 0,
                }

                columns = set(dataset[split].column_names)

                missing = required_columns - columns
                if missing:
                    split_report["missing_columns"] = list(missing)

                for item in dataset[split]:
                    if item["dialogue"] is None:
                        split_report["null_dialogues"] += 1
                    elif item["dialogue"].strip() == "":
                        split_report["empty_dialogues"] += 1

                    if item["summary"] is None:
                        split_report["null_summaries"] += 1
                    elif item["summary"].strip() == "":
                        split_report["empty_summaries"] += 1

                report[split] = split_report

            logging.info("Structural validation completed")
            return report

        except Exception as e:
            raise CustomException(e, sys)

    def validate_lengths(self, dataset: DatasetDict) -> dict:
        """
        Validate minimum text length constraints.
        """
        try:
            logging.info("Validating text lengths")

            report = {}

            min_dialogue_words = 3
            min_summary_words = 3

            for split in dataset.keys():
                short_dialogues = 0
                short_summaries = 0

                for item in dataset[split]:
                    if len(item["dialogue"].split()) < min_dialogue_words:
                        short_dialogues += 1

                    if len(item["summary"].split()) < min_summary_words:
                        short_summaries += 1

                report[split] = {
                    "short_dialogues": short_dialogues,
                    "short_summaries": short_summaries,
                }

            logging.info("Length validation completed")
            return report

        except Exception as e:
            raise CustomException(e, sys)

    def save_validation_report(self, structure_report: dict, length_report: dict):
        """
        Persist validation results.
        """
        try:
            final_report = {
                "timestamp": datetime.utcnow().isoformat(),
                "structure_validation": structure_report,
                "length_validation": length_report,
            }

            report_path = self.validation_dir / "validation_report.json"

            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(final_report, f, indent=4)

            logging.info(f"Validation report saved at {report_path}")

        except Exception as e:
            raise CustomException(e, sys)

    def run(self, dataset: DatasetDict):
        """
        Execute full data validation pipeline.
        """
        try:
            structure_report = self.validate_structure(dataset)
            length_report = self.validate_lengths(dataset)

            self.save_validation_report(structure_report, length_report)

            logging.info("Data validation completed successfully")

        except Exception as e:
            raise CustomException(e, sys)

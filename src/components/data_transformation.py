from pathlib import Path
import sys
from datasets import DatasetDict
from transformers import AutoTokenizer

from src.utils.common import read_yaml, create_directories, PROJECT_ROOT
from src.exception.custom_exception import CustomException
from src.logger.logger import logging


class DataTransformation:
    def __init__(self, config_path: Path):
        try:
            self.config = read_yaml(config_path)

            self.tokenizer_name = (
                self.config.data_transformation.tokenizer_name
            )

            self.max_input_length = (
                self.config.data_transformation.max_input_length
            )

            self.max_target_length = (
                self.config.data_transformation.max_target_length
            )

            self.tokenized_data_dir = (
                PROJECT_ROOT
                / self.config.data_transformation.tokenized_data_dir
            )

            create_directories([self.tokenized_data_dir], verbose=True)

            # ✅ Tokenizer is a class attribute
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name
            )

            logging.info(
                f"Tokenizer loaded successfully: {self.tokenizer_name}"
            )

        except Exception as e:
            raise CustomException(e, sys)

    def tokenize_function(self, examples):
        """
        Tokenize dialogue-summary pairs for Pegasus.
        """
        try:
            model_inputs = self.tokenizer(
                examples["dialogue"],
                max_length=self.max_input_length,
                truncation=True,
                padding="max_length"
            )
            labels = self.tokenizer(
                text_target=examples["summary"],
                max_length=self.max_target_length,
                truncation=True,
                padding="max_length"
            )
            labels_ids = []
            for label in labels["input_ids"]:
                labels_ids.append([
                    token_id if token_id != self.tokenizer.pad_token_id else -100
                    for token_id in label
                ])

            model_inputs["labels"] = labels_ids
            return model_inputs

        except Exception as e:
            raise CustomException(e, sys)

    def transform(self, dataset: DatasetDict) -> DatasetDict:
        """
        Apply tokenization to entire dataset.
        """
        try:
            logging.info("Starting data transformation (tokenization)")

            tokenized_dataset = dataset.map(
                self.tokenize_function,
                batched=True,
                remove_columns=dataset["train"].column_names
            )

            logging.info("Tokenization completed successfully")
            return tokenized_dataset

        except Exception as e:
            raise CustomException(e, sys)

    def save_tokenized_data(self, tokenized_dataset: DatasetDict):
        """
        Save tokenized dataset to disk.
        """
        try:
            logging.info("Saving tokenized dataset")

            for split in tokenized_dataset.keys():
                split_path = self.tokenized_data_dir / split
                tokenized_dataset[split].save_to_disk(split_path)

            logging.info("Tokenized dataset saved successfully")

        except Exception as e:
            raise CustomException(e, sys)

    def run(self, dataset: DatasetDict) -> DatasetDict:
        """
        Full data transformation pipeline.
        """
        try:
            tokenized_dataset = self.transform(dataset)
            self.save_tokenized_data(tokenized_dataset)

            logging.info("Data transformation pipeline completed")
            return tokenized_dataset

        except Exception as e:
            raise CustomException(e, sys)

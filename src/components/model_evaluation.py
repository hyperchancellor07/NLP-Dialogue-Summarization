from pathlib import Path
import sys
import json
from datetime import datetime

import evaluate
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.utils.common import read_yaml, PROJECT_ROOT
from src.exception.custom_exception import CustomException
from src.logger.logger import logging


class ModelEvaluation:
    def __init__(self, config_path: Path):
        try:
            self.config = read_yaml(config_path)

            self.model_dir = (
                PROJECT_ROOT / self.config.model_trainer.output_dir
            )

            self.tokenized_data_dir = (
                PROJECT_ROOT / self.config.data_transformation.tokenized_data_dir
            )

            self.evaluation_dir = (
                PROJECT_ROOT / self.config.model_evaluation.evaluation_dir
            )

            self.max_summary_length = (
                self.config.prediction.max_summary_length
            )

            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            self.rouge = evaluate.load("rouge")

            logging.info("ModelEvaluation initialized")

        except Exception as e:
            raise CustomException(e, sys)

    def load_model_and_tokenizer(self):
        """
        Load trained model and tokenizer.
        """
        try:
            logging.info("Loading trained model and tokenizer")

            tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_dir)
            model.to(self.device)
            model.eval()

            logging.info("Model and tokenizer loaded")
            return model, tokenizer

        except Exception as e:
            raise CustomException(e, sys)

    def load_evaluation_data(self, split: str = "test"):
        """
        Load tokenized evaluation data.
        """
        try:
            logging.info(f"Loading {split} dataset for evaluation")

            dataset = load_from_disk(
                self.tokenized_data_dir / split
            )

            return dataset

        except Exception as e:
            raise CustomException(e, sys)

    def generate_predictions(self, model, tokenizer, dataset):
        """
        Generate summaries for evaluation.
        """
        try:
            predictions = []
            references = []

            for sample in dataset:
                input_ids = torch.tensor(sample["input_ids"]).unsqueeze(0).to(self.device)
                attention_mask = torch.tensor(sample["attention_mask"]).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    generated_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=self.max_summary_length,
                        num_beams=4,
                        early_stopping=True,
                    )
                pred = tokenizer.decode(
                    generated_ids[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                ref = tokenizer.decode(
                    sample["labels"],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                predictions.append(pred)
                references.append(ref)

            return predictions, references

        except Exception as e:
            raise CustomException(e, sys)

    def compute_metrics(self, predictions, references):
        """
        Compute ROUGE metrics.
        """
        try:
            logging.info("Computing ROUGE metrics")

            scores = self.rouge.compute(
                predictions=predictions,
                references=references,
            )

            return scores

        except Exception as e:
            raise CustomException(e, sys)

    def save_evaluation_report(self, scores: dict):
        """
        Save evaluation metrics to disk.
        """
        try:
            report = {
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": scores,
            }

            report_path = self.evaluation_dir / "evaluation_report.json"
            self.evaluation_dir.mkdir(parents=True, exist_ok=True)

            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=4)

            logging.info(f"Evaluation report saved at {report_path}")

        except Exception as e:
            raise CustomException(e, sys)

    def run(self, split: str = "test"):
        """
        Full evaluation pipeline.
        """
        try:
            model, tokenizer = self.load_model_and_tokenizer()
            dataset = self.load_evaluation_data(split)

            predictions, references = self.generate_predictions(
                model, tokenizer, dataset
            )

            scores = self.compute_metrics(predictions, references)
            self.save_evaluation_report(scores)

            logging.info("Model evaluation completed successfully")
        except Exception as e:
            raise CustomException(e, sys)
            
            
            
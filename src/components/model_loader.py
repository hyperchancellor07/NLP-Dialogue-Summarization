import sys
from dataclasses import dataclass
from typing import Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.utils.common import read_yaml
from src.exception.custom_exception import CustomException
from src.logger.logger import logging


@dataclass
class ModelArtifacts:
    model: AutoModelForSeq2SeqLM
    tokenizer: AutoTokenizer
    device: str


class ModelLoader:
    """
    Loads a fine-tuned Hugging Face model for evaluation and inference.
    No training logic is allowed in this class.
    """

    def __init__(self, config_path: str):
        try:
            self.config = read_yaml(config_path)

            self.model_id = self.config.model.model_id
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            logging.info(f"ModelLoader initialized with model_id={self.model_id}")

        except Exception as e:
            raise CustomException(e, sys)

    def load(self) -> ModelArtifacts:
        """
        Load tokenizer and model from Hugging Face Hub.
        """
        try:
            logging.info("Loading tokenizer from Hugging Face Hub")
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)

            logging.info("Loading model from Hugging Face Hub")
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id)

            model.to(self.device)
            model.eval()

            logging.info(f"Model loaded successfully on device={self.device}")

            return ModelArtifacts(
                model=model,
                tokenizer=tokenizer,
                device=self.device,
            )

        except Exception as e:
            raise CustomException(e, sys)
            
            
            
            
            
            
            
            
            
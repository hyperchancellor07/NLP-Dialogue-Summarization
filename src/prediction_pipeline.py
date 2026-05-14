
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)


MODEL_NAME = "your-username/pegasus-samsum-full-finetuned"


class DialogueSummarizer:

    def __init__(self):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )


        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )


        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME,
            use_safetensors=False,
        )


        self.model.to(self.device)


    def summarize(self, dialogue_text):
        dialogue_text = "summarize: " + dialogue_text


        inputs = self.tokenizer(
            dialogue_text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512,
        )


        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }


        summary_ids = self.model.generate(
            **inputs,

            max_length=64,
            min_length=8,

            num_beams=8,

            repetition_penalty=2.5,
            no_repeat_ngram_size=3,

            length_penalty=1.0,

            early_stopping=True,
        )


        generated_summary = self.tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True,
        )


        return generated_summary

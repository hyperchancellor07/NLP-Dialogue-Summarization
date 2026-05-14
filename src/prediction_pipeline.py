```python id="4tmx8q"
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

from rank_bm25 import BM25Okapi


MODEL_NAME = (
    "hyperchancellor07/pegasus-samsum-dialogue-summarizer"
)


class DialogueSummarizer:

    def __init__(self):

        self.device = torch.device(

            "cuda"

            if torch.cuda.is_available()

            else "cpu"
        )


        # ====================================================
        # TOKENIZER
        # ====================================================

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )


        # ====================================================
        # MODEL
        # ====================================================

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME,
        )


        self.model.to(self.device)


    # ========================================================
    # SUMMARY GENERATION
    # ========================================================

    def summarize(
        self,
        dialogue_text,
    ):

        dialogue_text = (
            "summarize: " + dialogue_text
        )


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

            max_length=120,

            min_length=12,

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


    # ========================================================
    # SPARSE SIMILARITY
    # ========================================================

    def sparse_similarity(
        self,
        dialogue,
        summary,
    ):

        corpus = [
            dialogue.split()
        ]


        bm25 = BM25Okapi(
            corpus
        )


        tokenized_summary = (
            summary.split()
        )


        score = bm25.get_scores(
            tokenized_summary
        )[0]


        normalized_score = min(
            score / 20,
            1.0,
        )


        return round(
            float(normalized_score),
            4,
        )


    # ========================================================
    # HDRS SCORE
    # ========================================================

    def compute_hdrs(
        self,
        dialogue,
        summary,
    ):

        sparse_score = (
            self.sparse_similarity(
                dialogue,
                summary,
            )
        )


        hdrs = sparse_score


        return {

            "Sparse Similarity":
                sparse_score,

            "HDRS":
                round(
                    hdrs,
                    4,
                ),
        }
```

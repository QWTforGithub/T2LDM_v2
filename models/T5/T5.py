from transformers import logging
logging.set_verbosity_error()

import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5EncoderModel

# export HF_ENDPOINT=https://hf-mirror.com

ROOT_PATH = "/ihoment/youjie10/qwt/download"

# huggingface-cli download t5-small --local-dir /ihoment/youjie10/qwt/download/T5/T5-small
MODEL_PATH_SMALL = f"{ROOT_PATH}/T5/T5-small" # 512

# huggingface-cli download t5-base --local-dir /ihoment/youjie10/qwt/download/T5/T5-base
MODEL_PATH_BASE = f"{ROOT_PATH}/T5/T5-base" # 768

# huggingface-cli download t5-large --local-dir /ihoment/youjie10/qwt/download/T5/T5-large
MODEL_PATH_LARGE = f"{ROOT_PATH}/T5/T5-large" # 1024


class t5(nn.Module):

    def __init__(
            self,
            model_type="base",
            fp16=True,
            max_length=128
    ):
        super().__init__()

        self.fp16 = fp16
        self.max_length = max_length

        if model_type == "small":
            model_path = MODEL_PATH_SMALL
        elif model_type == "base":
            model_path = MODEL_PATH_BASE
        elif model_type == "large":
            model_path = MODEL_PATH_LARGE
        else:
            raise ValueError("Unsupported type")

        self.tokenizer = T5Tokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            legacy=True
        )

        self.encoder = T5EncoderModel.from_pretrained(
            model_path,
            local_files_only=True,
            ignore_mismatched_sizes=True
        )

        # 冻结
        for p in self.encoder.parameters():
            p.requires_grad = False

        # 精度处理（不放 cuda，这一步交给外部）
        if self.fp16:
            self.encoder = self.encoder.half()

    def tokenize(self, texts, **kwargs):
        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )

        device = next(self.parameters()).device
        tokens = {k: v.to(device) for k, v in tokens.items()}

        return tokens

    def encode_text(self, tokens, **kwargs):

        outputs = self.encoder(**tokens)

        return outputs.last_hidden_state

    def forward(self, texts):

        device = next(self.parameters()).device

        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )

        tokens = {k: v.to(device) for k, v in tokens.items()}

        with torch.no_grad():
            outputs = self.encoder(**tokens)

        return outputs.last_hidden_state


if __name__ == '__main__':
    # pip install sentencepiece

    text_encoder = t5(
        model_type="large",
        fp16=True,
    ).cuda().eval()

    texts = [
        "Rainy. Two cars.",
        "A car is to the right of a truck.",
    ]

    tokens = text_encoder.tokenize(texts)
    outputs = text_encoder.encode_text(tokens)

    print(outputs.shape)
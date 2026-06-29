import torch
from transformers import T5Tokenizer, T5EncoderModel



if __name__ == '__main__':


    texts = [
        "Rainy. Two cars.",
        "A car is to the right of a truck.",
    ]

    model_path = "/ihoment/youjie10/qwt/download/T5/T5-base"

    tokenizer = T5Tokenizer.from_pretrained(
        model_path,
        local_files_only=True
    )

    model = T5EncoderModel.from_pretrained(
        model_path,
        local_files_only=True
    )

    # 放到 GPU
    model = model.cuda().half()

    # 冻结（建议）
    for p in model.parameters():
        p.requires_grad = False

    tokens = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    # tokens 也放 GPU
    tokens = {k: v.cuda() for k, v in tokens.items()}

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            outputs = model(**tokens)

    print(outputs.last_hidden_state.shape)
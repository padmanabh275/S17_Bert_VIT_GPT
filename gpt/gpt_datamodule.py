import torch
from transformers import AutoTokenizer  


def encode(text_seq: str, tokenizer: any) -> torch.Tensor:
    """
    Function to encode input text using a pre-trained tokenizer and vectorized lookups
    """
    # tokenize the input text
    tokens = tokenizer.tokenize(text_seq)
    # convert the tokens to their corresponding ids
    token_indices = tokenizer.convert_tokens_to_ids(tokens)
    token_indices = torch.tensor(token_indices, dtype=torch.long)
    return token_indices


def decode(enc_sec: torch.Tensor, tokenizer: any) -> str:
    """
    Function to decode a sequence of token indices back to a string
    """
    # convert the indices to a list
    enc_sec = enc_sec.tolist()
    # decode the indices to a string
    text = tokenizer.decode(enc_sec)
    return text

def prepare_data(path_of_dataset):
    # raw data
    path_do_data = "data/english.txt"
    data_raw = open(path_do_data, encoding="utf-8").read()
    # we use pretrained BERT tokenizer for performance improvements
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # data_raw = data_raw[4000000:] # short dataset

    # train/val split
    data = encode(text_seq=data_raw, tokenizer=tokenizer)
    n = int(0.9 * len(data))  # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data
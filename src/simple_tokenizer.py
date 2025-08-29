import re
from collections import defaultdict

class SimpleTokenizer:
    def __init__(self, lower=True):
        self.lower = lower
        self.vocab = {}
        self.inv_vocab = {}
        self.unk_token = "<UNK>"
        self.pad_token = "<PAD>"
        self.start_token = "<START>"
        self.end_token = "<END>"
        self.special_tokens = [self.start_token, self.end_token, self.pad_token, self.unk_token]

    def tokenize(self, text):
        if self.lower:
            text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        return tokens

    def build_vocab(self, texts):
        # texts: list of strings
        token_set = set()
        for text in texts:
            token_set.update(self.tokenize(text))
        # Add special tokens first
        all_tokens = self.special_tokens + sorted(token_set)
        self.vocab = {token: idx for idx, token in enumerate(all_tokens)}
        self.inv_vocab = {idx: token for token, idx in self.vocab.items()}

    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens]

    def decode(self, ids):
        return [self.inv_vocab.get(idx, self.unk_token) for idx in ids]

    def detokenize(self, tokens):
        return " ".join(tokens)

    def save_vocab(self, filepath):
        """Save the vocabulary mapping to a file."""
        with open(filepath, "w") as f:
            for token, idx in self.vocab.items():
                f.write(f"{token}\t{idx}\n")

if __name__ == "__main__":
    import pandas as pd
    # Load the medical Q&A dataset
    df = pd.read_csv("../data/raw/mle_screening_dataset.csv")
    # Combine all questions and answers
    texts = list(df['question'].dropna()) + list(df['answer'].dropna())
    # Build vocabulary
    tokenizer = SimpleTokenizer(lower=True)
    tokenizer.build_vocab(texts)
    # Save vocabulary to file
    with open("../data/processed/vocab.txt", "w") as f:
        for token, idx in tokenizer.vocab.items():
            f.write(f"{token}\t{idx}\n")
    tokenizer = SimpleTokenizer()
    texts = [
        "Hello, world!",
        "This is a test.",
        "Tokenization is important."
    ]
    tokenizer.build_vocab(texts)
    print("Vocabulary:", tokenizer.vocab)

    encoded = tokenizer.encode("Hello, world!")
    print("Encoded:", encoded)

    decoded = tokenizer.decode(encoded)
    print("Decoded:", decoded)

    detokenized = tokenizer.detokenize(decoded)
    print("Detokenized:", detokenized)
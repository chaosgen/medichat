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
    
    def update_vocab(self, texts, filepath=None):
        """
        Update the vocabulary with new tokens from the given texts.
        Only adds new tokens, keeps existing token-ID pairs unchanged.
        If filepath is provided, saves the updated vocab to file.
        """
        # Find new tokens
        new_tokens = set()
        for text in texts:
            new_tokens.update(self.tokenize(text))
        # Remove tokens already in vocab
        new_tokens = new_tokens - set(self.vocab.keys())
        # Add new tokens with new IDs
        next_id = max(self.vocab.values(), default=-1) + 1
        for token in sorted(new_tokens):
            self.vocab[token] = next_id
            self.inv_vocab[next_id] = token
            next_id += 1
        # Optionally save
        if filepath:
            self.save_vocab(filepath)

    @classmethod
    def load_vocab(cls, filepath, lower=True):
        """Load the vocabulary mapping from a file and return a SimpleTokenizer instance."""
        vocab = {}
        with open(filepath, "r") as f:
            for line in f:
                token, idx = line.rstrip("\n").split("\t")
                vocab[token] = int(idx)
        tokenizer = cls(lower=lower)
        tokenizer.vocab = vocab
        tokenizer.inv_vocab = {idx: token for token, idx in vocab.items()}
        return tokenizer

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Simple Tokenizer Vocabulary Builder")
    parser.add_argument('--db_path', type=str, default="data/raw/mle_screening_dataset.csv", help="Path to the raw CSV dataset")
    parser.add_argument('--output_vocab', type=str, default="data/processed/vocab.txt", help="Path to save vocabulary file")
    args = parser.parse_args()

    import pandas as pd

    # Load your dataset
    df = pd.read_csv("data/raw/mle_screening_dataset.csv")

    # Combine questions and answers into one list of texts
    texts = df['question'].tolist() + df['answer'].tolist()

    # Build tokenizer
    tokenizer = SimpleTokenizer(lower=True)
    tokenizer.build_vocab(texts)

    # Save vocabulary
    tokenizer.save_vocab("data/processed/vocab.txt")
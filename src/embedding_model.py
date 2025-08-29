import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from simple_tokenizer import SimpleTokenizer
import pandas as pd

class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Optionally add more layers here

    def forward(self, input_ids):
        return self.embedding(input_ids)

class EmbeddingTrainer:
    def __init__(self, tokenizer, embedding_dim=128, model_path="embedding_model.pt"):
        self.tokenizer = tokenizer
        self.embedding_dim = embedding_dim
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EmbeddingModel(len(tokenizer.vocab), embedding_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()  # Dummy loss for demonstration
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))
            print(f"Loaded model weights from {self.model_path}")

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)
        print(f"Saved model weights to {self.model_path}")

    def train(self, texts, epochs=5, save_interval=1):
        # Dummy target: train embeddings to be close to one-hot (for demonstration)
        for epoch in range(epochs):
            total_loss = 0.0
            for text in texts:
                input_ids = torch.tensor(self.tokenizer.encode(text), dtype=torch.long).to(self.device)
                if len(input_ids) == 0:
                    continue
                embeds = self.model(input_ids)
                # Dummy target: zeros
                target = torch.zeros_like(embeds)
                loss = self.criterion(embeds, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
            if (epoch + 1) % save_interval == 0:
                self.save_model()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Embedding Model from Scratch")
    parser.add_argument('--train_path', type=str, default="data/processed/train.csv", help="Path to training data CSV")
    parser.add_argument('--tokenizer_vocab', type=str, default="data/processed/vocab.txt", help="Path to tokenizer vocab file")
    parser.add_argument('--model_path', type=str, default="embedding_model.pt", help="Path to save/load model weights")
    parser.add_argument('--embedding_dim', type=int, default=128, help="Embedding dimension size")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs")
    parser.add_argument('--save_interval', type=int, default=1, help="Save model every N epochs")
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = SimpleTokenizer.load_vocab(filepath=args.tokenizer_vocab)

    # Load training data
    df = pd.read_csv(args.train_path)
    texts = df['question'].dropna().tolist() + df['answer'].dropna().tolist()

    # Train embedding model
    trainer = EmbeddingTrainer(tokenizer, embedding_dim=args.embedding_dim, model_path=args.model_path)
    trainer.train(texts, epochs=args.epochs, save_interval=args.save_interval)

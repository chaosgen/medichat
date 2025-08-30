import torch
import torch.nn as nn
import argparse
import os
from torch.utils.data import Dataset, DataLoader
from simple_tokenizer import SimpleTokenizer

class MedicalTransformerConfig:
    def __init__(
        self,
        vocab_size: int,
        max_seq_length: int = 512,
        hidden_size: int = 256,
        num_attention_heads: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dropout: float = 0.1
    ):
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dropout = dropout

class MedicalTransformer(nn.Module):
    def __init__(self, config: MedicalTransformerConfig):
        super().__init__()
        self.config = config
        
        # Embedding layers
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_encoder = nn.Embedding(config.max_seq_length, config.hidden_size)
        
        # Transformer layers
        self.transformer = nn.Transformer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dropout=config.dropout
        )
        
        # Output layer
        self.output_layer = nn.Linear(config.hidden_size, config.vocab_size)
        
    def forward(self, src, tgt):
        # Create position indices
        pos = torch.arange(0, src.size(1), device=src.device).unsqueeze(0)
        
        # Combine token embeddings with position encodings
        src = self.embedding(src) + self.pos_encoder(pos)
        tgt = self.embedding(tgt) + self.pos_encoder(pos[:, :tgt.size(1)])
        
        # Transform input sequences
        output = self.transformer(
            src.transpose(0, 1),
            tgt.transpose(0, 1)
        ).transpose(0, 1)
        
        return self.output_layer(output)

class MedicalQADataset(Dataset):
    def __init__(self, questions, answers, tokenizer, max_length=512):
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answers[idx]
        
        # Tokenize and truncate to max_length - 2 to account for special tokens
        q_tokens = self.tokenizer.encode(question)[:self.max_length-2]
        a_tokens = self.tokenizer.encode(answer)[:self.max_length-2]
        
        # Add start/end tokens
        q_tokens = [self.tokenizer.vocab[self.tokenizer.start_token]] + q_tokens + [self.tokenizer.vocab[self.tokenizer.end_token]]
        a_tokens = [self.tokenizer.vocab[self.tokenizer.start_token]] + a_tokens + [self.tokenizer.vocab[self.tokenizer.end_token]]
        
        # Pad sequences
        q_tokens = q_tokens + [self.tokenizer.vocab[self.tokenizer.pad_token]] * (self.max_length - len(q_tokens))
        a_tokens = a_tokens + [self.tokenizer.vocab[self.tokenizer.pad_token]] * (self.max_length - len(a_tokens))
        
        return {
            'question': torch.tensor(q_tokens),
            'answer': torch.tensor(a_tokens)
        }

class TransformerTrainer:
    def __init__(
        self,
        model: MedicalTransformer,
        tokenizer: SimpleTokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "checkpoints"
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.optimizer = torch.optim.Adam(model.parameters())
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is pad_token_id
        
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, epoch, loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss
        }
        path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']

    def train(
        self,
        train_dataloader,
        eval_dataloader,
        epochs=10,
        save_interval=1,
        checkpoint_path=None
    ):
        start_epoch = 0
        if checkpoint_path and os.path.exists(checkpoint_path):
            start_epoch = self.load_checkpoint(checkpoint_path)
            print(f"Resuming from epoch {start_epoch}")

        for epoch in range(start_epoch, epochs):
            self.model.train()
            total_loss = 0
            
            for batch in train_dataloader:
                q_tokens = batch['question'].to(self.device)
                a_tokens = batch['answer'].to(self.device)
                
                self.optimizer.zero_grad()
                # Pad a_tokens to max length for decoder input
                max_a_len = a_tokens.size(1)
                output = self.model(q_tokens, a_tokens[:, :-1].contiguous())
                
                loss = self.criterion(
                    output.reshape(-1, output.size(-1)),
                    a_tokens[:, 1:].reshape(-1)
                )
                
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
            # Validation
            eval_loss = self.evaluate(eval_dataloader)
            print(f"Validation Loss: {eval_loss:.4f}")

            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(epoch + 1, avg_loss)

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                q_tokens = batch['question'].to(self.device)
                a_tokens = batch['answer'].to(self.device)
                
                output = self.model(q_tokens, a_tokens[:, :-1])
                loss = self.criterion(
                    output.reshape(-1, output.size(-1)),
                    a_tokens[:, 1:].reshape(-1)
                )
                total_loss += loss.item()

        return total_loss / len(dataloader)

    def generate(self, question: str, max_length: int = 100):
        self.model.eval()
        q_tokens = torch.tensor(
            self.tokenizer.encode(question),
            dtype=torch.long
        ).unsqueeze(0).to(self.device)
        
        # Start with start token
        output_tokens = [self.tokenizer.vocab[self.tokenizer.start_token]]
        
        with torch.no_grad():
            for _ in range(max_length):
                tgt_tensor = torch.tensor(
                    output_tokens,
                    dtype=torch.long
                ).unsqueeze(0).to(self.device)
                
                output = self.model(q_tokens, tgt_tensor)
                next_token = output[0, -1].argmax().item()
                
                output_tokens.append(next_token)
                
                if next_token == self.tokenizer.vocab[self.tokenizer.end_token]:
                    break
        
        return self.tokenizer.decode(output_tokens)

if __name__ == "__main__":
    import pandas as pd

    parser = argparse.ArgumentParser(description="Train Medical Transformer Model")
    parser.add_argument('--train_path', type=str, required=True, help="Path to training data")
    parser.add_argument('--eval_path', type=str, required=True, help="Path to validation data")
    parser.add_argument('--test_path', type=str, required=True, help="Path to test data")
    parser.add_argument('--vocab_path', type=str, required=True, help="Path to vocabulary file")
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument('--hidden_size', type=int, default=256, help="Hidden size of transformer")
    parser.add_argument('--num_layers', type=int, default=3, help="Number of transformer layers")
    parser.add_argument('--batch_size', type=int, default=32, help="Training batch size")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = SimpleTokenizer.load_vocab(args.vocab_path)

    # Create model config and model
    config = MedicalTransformerConfig(
        vocab_size=len(tokenizer.vocab),
        hidden_size=args.hidden_size,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=args.num_layers
    )
    model = MedicalTransformer(config)

    # Create datasets and dataloaders
    train_data = pd.read_csv(args.train_path)
    eval_data = pd.read_csv(args.eval_path)
    
    train_dataset = MedicalQADataset(
        train_data['question'].tolist(),
        train_data['answer'].tolist(),
        tokenizer
    )
    eval_dataset = MedicalQADataset(
        eval_data['question'].tolist(),
        eval_data['answer'].tolist(),
        tokenizer
    )

    def collate_fn(batch):
        # Find max lengths
        max_q_len = max(len(item['question']) for item in batch)
        max_a_len = max(len(item['answer']) for item in batch)
        
        # Pad sequences
        questions = [
            torch.nn.functional.pad(item['question'], (0, max_q_len - len(item['question'])))
            for item in batch
        ]
        answers = [
            torch.nn.functional.pad(item['answer'], (0, max_a_len - len(item['answer'])))
            for item in batch
        ]
        
        return {
            'question': torch.stack(questions),
            'answer': torch.stack(answers)
        }

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=collate_fn
    )
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=args.batch_size,
        collate_fn=collate_fn
    )

    # Create trainer and train
    trainer = TransformerTrainer(model, tokenizer, checkpoint_dir=args.checkpoint_dir)
    trainer.train(train_loader, eval_loader, epochs=args.epochs)

"""
python src/transformer_model.py \
    --train_path data/processed/train.csv \
    --eval_path data/processed/eval.csv \
    --test_path data/processed/test.csv \
    --vocab_path data/processed/vocab.txt \
    --checkpoint_dir models/checkpoints \
    --hidden_size 256 \
    --num_layers 3 \
    --batch_size 32 \
    --epochs 10
"""
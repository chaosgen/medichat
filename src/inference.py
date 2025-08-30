import argparse
import torch
from transformer_model import MedicalTransformer, MedicalTransformerConfig
from simple_tokenizer import SimpleTokenizer

class MedicalQAInferenceService:
    def __init__(self, model_path, vocab_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nInitializing Medical QA Service...")
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = SimpleTokenizer.load_vocab(vocab_path)
        
        # Initialize model
        print("Initializing model...")
        config = MedicalTransformerConfig(
            vocab_size=len(self.tokenizer.vocab),
            hidden_size=512,  # Match the checkpoint architecture
            num_attention_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6
        )
        self.model = MedicalTransformer(config).to(self.device)
        
        # Load trained weights
        print(f"Loading model weights from {model_path}...")
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
        
        print(f"Model loaded on {self.device}")
        print(f"Vocabulary size: {len(self.tokenizer.vocab)}")

    def generate_response(self, question: str, max_length: int = 100):
        print(f"\nProcessing question: {question}")
        
        # Tokenize and encode the question
        q_tokens = self.tokenizer.encode(question)
        print(f"Question tokens: {self.tokenizer.decode(q_tokens)}")
        
        # Add special tokens and pad
        q_tokens = [self.tokenizer.vocab[self.tokenizer.start_token]] + q_tokens + [self.tokenizer.vocab[self.tokenizer.end_token]]
        q_tokens = q_tokens + [self.tokenizer.vocab[self.tokenizer.pad_token]] * (max_length - len(q_tokens))
        q_tokens = torch.tensor(q_tokens, dtype=torch.long).unsqueeze(0).to(self.device)
        
        # Initialize target with start token
        output_tokens = [self.tokenizer.vocab[self.tokenizer.start_token]]
        
        # Generate tokens one by one
        with torch.no_grad():
            for _ in range(max_length - 1):
                tgt_tensor = torch.tensor(output_tokens, dtype=torch.long).unsqueeze(0).to(self.device)
                output = self.model(q_tokens, tgt_tensor)
                next_token = output[0, -1].argmax().item()
                output_tokens.append(next_token)
                
                print(f"Generated token: {self.tokenizer.decode([next_token])}")
                
                if next_token == self.tokenizer.vocab[self.tokenizer.end_token]:
                    break
        
        # Decode generated tokens
        decoded_tokens = self.tokenizer.decode(output_tokens)
        print(f"\nFull generated sequence: {decoded_tokens}")
        
        # Remove special tokens and join
        response = ' '.join(token for token in decoded_tokens 
                          if token not in [self.tokenizer.start_token, 
                                         self.tokenizer.end_token,
                                         self.tokenizer.pad_token,
                                         self.tokenizer.unk_token])
        
        print(f"Final response: {response}")
        return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical QA Inference Service")
    parser.add_argument('--model_path', type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument('--vocab_path', type=str, required=True, help="Path to vocabulary file")
    parser.add_argument('--interactive', action='store_true', help="Run in interactive mode")
    parser.add_argument('--question', type=str, help="Single question to answer (non-interactive mode)")
    args = parser.parse_args()

    try:
        # Initialize service
        service = MedicalQAInferenceService(args.model_path, args.vocab_path)

        if args.interactive:
            print("\nMedical QA System. Type 'quit' to exit.")
            while True:
                question = input("\nQuestion: ").strip()
                if question.lower() in ['quit', 'exit']:
                    break
                if not question:
                    continue
                
                try:
                    response = service.generate_response(question)
                    print(f"\nAnswer: {response}")
                except Exception as e:
                    print(f"Error generating response: {str(e)}")
        else:
            if args.question:
                response = service.generate_response(args.question)
                print(f"\nQuestion: {args.question}")
                print(f"Answer: {response}")
            else:
                print("Please provide a question using --question or use --interactive mode")
    except Exception as e:
        print(f"Error: {str(e)}")

"""
python src/inference.py \
    --model_path models/checkpoints/checkpoint_epoch_10.pt \
    --vocab_path data/processed/vocab.txt \
    --question "What are the symptoms of diabetes?"
"""
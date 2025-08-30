import pandas as pd
from sklearn.model_selection import train_test_split
import os


from config_loader import config

class MedicalDataProcessor:
    def __init__(self, data_path=None, output_path=None):
        self.data_path = data_path or config.get_path('data', 'raw_data_path')
        self.output_path = output_path or config.get_path('data', 'processed_data_dir')

    def load_data(self):
        # Load CSV file into a pandas dataframe
        self.data = pd.read_csv(self.data_path)
        return self.data

    def preprocess_data(self, eval_size=None, test_size=None):
        # Use config values if not provided
        eval_size = eval_size if eval_size is not None else config.data['eval_size']
        test_size = test_size if test_size is not None else config.data['test_size']
        # Drop rows with missing values
        df = self.data.dropna(subset=['question', 'answer']).copy()

        # Remove leading/trailing whitespace using .loc
        df.loc[:, 'question'] = df['question'].str.strip()
        df.loc[:, 'answer'] = df['answer'].str.strip()

        # Remove duplicates
        df = df.drop_duplicates(subset=['question', 'answer'])

        # Split train/eval/test data
        train_df, temp_df = train_test_split(df, test_size=eval_size + test_size, random_state=42)
        eval_df, test_df = train_test_split(temp_df, test_size=test_size / (eval_size + test_size), random_state=42)

        self.processed_data = {
            'train': train_df,
            'eval': eval_df,
            'test': test_df
        }
        

    def save_data(self, output_path=None):
        # Save train/eval/test data to csv files
        if output_path is None:
            output_path = self.output_path
        
        df = pd.DataFrame(self.processed_data['train'])
        df.to_csv(os.path.join(output_path, 'train.csv'), index=False)

        df = pd.DataFrame(self.processed_data['eval'])
        df.to_csv(os.path.join(output_path, 'eval.csv'), index=False)

        df = pd.DataFrame(self.processed_data['test'])
        df.to_csv(os.path.join(output_path, 'test.csv'), index=False)

        return output_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Medical Q&A Data Processor")
    parser.add_argument('--db_path', type=str, default="data/raw/mle_screening_dataset.csv", help="Path to the raw CSV dataset")
    parser.add_argument('--output_dir', type=str, default="data/processed", help="Directory to save processed files")
    args = parser.parse_args()

    processor = MedicalDataProcessor(args.db_path, args.output_dir)
    processor.load_data()
    processor.preprocess_data()
    processor.save_data()
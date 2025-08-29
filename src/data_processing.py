import pandas as pd
from sklearn.model_selection import train_test_split
import os


class MedicalDataProcessor:
    def __init__(self, data_path, output_path):
        self.data_path = data_path
        self.output_path = output_path or os.path.dirname(self.data_path)

    def load_data(self):
        # Load CSV file into a pandas dataframe
        self.data = pd.read_csv(self.data_path)
        return self.data

    def preprocess_data(self, eval_size=0.1, test_size=0.1):
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
    processor = MedicalDataProcessor("data/raw/mle_screening_dataset.csv", "data/processed")
    processor.load_data()
    processor.preprocess_data()
    processor.save_data()
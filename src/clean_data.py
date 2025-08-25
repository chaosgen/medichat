
import pandas as pd
import argparse
import os
from sklearn.model_selection import train_test_split

def clean_and_split_data(in_path, out_dir, test_size=0.1, val_size=0.1):
	# Loading
	df = pd.read_csv(in_path)

	# preprocessing
	df['question'] = df['question'].str.lower().str.strip()
	df['answer'] = df['answer'].str.lower().str.strip()

	df = df.drop_duplicates(subset=['question', 'answer'])
	df = df.dropna(subset=['question', 'answer'])

	df = df.sort_values('answer', key=lambda x: x.str.len(), ascending=False)

	# Split data
	test_size = float(test_size)
	val_size = float(val_size)
	train_size = 1 - test_size - val_size
	if train_size <= 0:
		raise ValueError("train_size must be positive. Reduce test_size or val_size.")

	train_df, temp_df = train_test_split(df, test_size=(test_size + val_size), random_state=42)
	test_relative_size = test_size / (test_size + val_size)
	val_df, test_df = train_test_split(temp_df, test_size=test_relative_size, random_state=42)

	# Dump data
	os.makedirs(out_dir, exist_ok=True)
	cleaned_path = os.path.join(out_dir, "mle_screening_dataset_cleaned.csv")
	train_path = os.path.join(out_dir, "train.csv")
	val_path = os.path.join(out_dir, "val.csv")
	test_path = os.path.join(out_dir, "test.csv")

	df.to_csv(cleaned_path, index=False)
	train_df.to_csv(train_path, index=False)
	val_df.to_csv(val_path, index=False)
	test_df.to_csv(test_path, index=False)

	print(f"Data cleaned and saved to {cleaned_path}")
	print(f"Train/Val/Test splits saved to {train_path}, {val_path}, {test_path}")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Clean and split dataset.")
	parser.add_argument("--in_path", type=str, required=True, help="Path to input CSV file.")
	parser.add_argument("--out_dir", type=str, required=True, help="Directory to save processed files.")
	parser.add_argument("--test_size", type=float, default=0.1, help="Fraction of data for test set.")
	parser.add_argument("--val_size", type=float, default=0.1, help="Fraction of data for validation set.")
	args = parser.parse_args()

	clean_and_split_data(args.in_path, args.out_dir, args.test_size, args.val_size)

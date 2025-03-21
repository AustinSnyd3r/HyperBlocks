import pandas as pd
from sklearn.model_selection import StratifiedKFold
import argparse

def main():
    parser = argparse.ArgumentParser(description='Split CSV data into stratified k-fold train/test sets.')
    parser.add_argument('file', type=str, help='Input CSV file name')
    parser.add_argument('k', type=int, help='Number of folds (k)')
    args = parser.parse_args()

    # Load the dataset from the provided CSV file.
    data = pd.read_csv(args.file)

    # Assume the target column is the last column.
    label_col = data.columns[-1]
    print("Using '%s' as the target column." % label_col)

    # Create a StratifiedKFold object to maintain class proportions.
    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=42)

    fold = 1
    for train_index, test_index in skf.split(data, data[label_col]):
        train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]

        # Write each train and test split to separate CSV files.
        train_file = 'train%s.csv' % fold
        test_file = 'test%s.csv' % fold
        train_data.to_csv(train_file, index=False)
        test_data.to_csv(test_file, index=False)

        print("Fold %s: Train samples: %s, Test samples: %s" % (fold, len(train_data), len(test_data)))
        fold += 1

    print("Stratified k-fold files created successfully.")

if __name__ == '__main__':
    main()


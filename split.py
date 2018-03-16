from argparse import ArgumentParser
import pandas as pd
import numpy as np

if __name__ == '__main__':
    parser = ArgumentParser(description='Split a CSV describing paintings into train and test sets.')
    parser.add_argument('file', help='CSV file with "style" and "filename" columns.')
    parser.add_argument('train_set', help='CSV file containing the split training rows.')
    parser.add_argument('test_set', help='CSV file containing the split testing rows.')
    parser.add_argument('split', type=float, help='Fraction of rows to consider for training.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for sampling')

    args = parser.parse_args()

    # Split data into training, validation,  and test sets
    train_split = args.split    # as a fraction of total instances
    test_split = 1 - train_split

    alldata = pd.read_csv(args.file, header=0)
    alldata = alldata.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    num_instances = len(alldata)
    num_train = int(np.ceil(train_split * num_instances))

    test = alldata.iloc[num_train:]
    train = alldata.iloc[:num_train]

    test.to_csv(args.test_set, index=False)
    train.to_csv(args.train_set, index=False)
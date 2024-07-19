from train import transform_data, prepare_datasets, train
import argparse

parser = argparse.ArgumentParser(description='Train parameters')
parser.add_argument('--train_data', type=str, help='train data directory', nargs='?',
                    const='./data/train.csv', default='./data/train.csv')

if __name__ == "__main__":
    args = parser.parse_args()
    vargs = vars(args)
    train_data_path = vargs.get('train_data')
    transform_data(train_data_path)
    train_ds, val_ds = prepare_datasets()
    train(train_ds, val_ds)

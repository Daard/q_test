from predict import predict
import argparse

parser = argparse.ArgumentParser(description='Predict parameters')
parser.add_argument('--data_path', type=str, help='file with data', nargs='?',
                    const='./data/hidden_test.csv', default='./data/hidden_test.csv')
parser.add_argument('--model_path', type=str, help='directory with the model', nargs='?',
                    const='./cached_model', default='./cached_model')
parser.add_argument('--tfx_root', type=str, help='root directory for TFX artefacts', nargs='?',
                    const='cached_tfx', default='cached_tfx')

if __name__ == "__main__":
    args = parser.parse_args()
    vargs = vars(args)
    data_path = vargs.get('data_path')
    model_path = vargs.get('model_path')
    tfx_root = vargs.get('tfx_root')
    predict(data_path, model_path, tfx_root)

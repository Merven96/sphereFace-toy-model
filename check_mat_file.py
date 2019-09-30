import scipy.io as io
import os
import argparse
import numpy as np

def parameter():
    parser = argparse.ArgumentParser(description="Transformer")
    parser.add_argument('--train_data_mat', type=str, default="./mat_file/high_dimension_model/high_feature_train_input.mat")
    parser.add_argument('--label_data_mat', type=str, default="./mat_file/high_dimension_model/high_feature_train_label.mat")
    parser.add_argument('--test_input_mat', type=str, default="./mat_file/high_dimension_model/high_feature_test_input.mat")
    parser.add_argument('--test_label_mat', type=str, default="./mat_file/high_dimension_model/high_feature_test_label.mat")

    parser.add_argument('--use_gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epoch_num', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--mid_num', type=int, default=4)
    parser.add_argument('--mid_dimen', type=int, default=4)

    return parser


if __name__ == "__main__":
    args = parameter()
    args = args.parse_args()

    print("check train dataset:")
    train_data = io.loadmat(args.train_data_mat)
    train_input = train_data['name']
    label_data = io.loadmat(args.label_data_mat)
    train_label = label_data['name']
    print(np.shape(train_input), np.shape(train_label))

    print("check test dataset:")
    test_input = io.loadmat(args.test_input_mat)
    test_input = test_input['name']
    test_label = io.loadmat(args.test_label_mat)
    test_label = test_label['name']
    print(np.shape(test_input), np.shape(test_label))

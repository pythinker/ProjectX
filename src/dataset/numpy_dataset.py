import pickle

import numpy as np


np.random.seed(0)


def create_xor_dataset(N, gauss_std, data_path):

    M = N // 4

    X_0_0 = np.array([0, 0]) + gauss_std * np.random.randn(M, 2)
    X_0_1 = np.array([0, 1]) + gauss_std * np.random.randn(M, 2)
    X_1_0 = np.array([1, 0]) + gauss_std * np.random.randn(M, 2)
    X_1_1 = np.array([1, 1]) + gauss_std * np.random.randn(M, 2)

    y_0_0 = np.zeros(M)
    y_0_1 = np.ones(M)
    y_1_0 = np.ones(M)
    y_1_1 = np.zeros(M)

    X = np.concatenate((X_0_0, X_0_1, X_1_0, X_1_1))
    y = np.concatenate((y_0_0, y_0_1, y_1_0, y_1_1))

    data = {"X": X, "y": y}

    with open (data_path, "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    create_xor_dataset(10000, 0.1, "/home/mos/workshop/ProjectX/data/valid_data.pickle")
    create_xor_dataset(90000, 0.1, "/home/mos/workshop/ProjectX/data/train_data.pickle")

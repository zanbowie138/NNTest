import idx2numpy
import numpy as np

import matplotlib.pyplot as plt

def main():
    # 10000 x 28 x 28
    arr = idx2numpy.convert_from_file('data/t10k-images.idx3-ubyte')
    # 10000
    labels = idx2numpy.convert_from_file('data/t10k-labels.idx1-ubyte')

    # 784 inputs
    # Two hidden layers
    # 10 outputs 0-9

    h_1 = np.zeros(10)
    h_2 = np.zeros(10)
    out = np.zeros(10)

    # 3 sets of weights & biases
    # 1st: 784 * 10 weights, 10 biases
    # 2nd: 10 * 10, 10 biases
    # 3rd: 10 * 10, 10 biases

    w_1 = np.random.rand(784, 10)
    w_2 = np.random.rand(10, 10)
    w_3 = np.random.rand(10, 10)

    b_1 = np.random.rand(10)
    b_2 = np.random.rand(10)
    b_3 = np.random.rand(10)

    # calculate cost
    iters = 5
    for i in range(iters):
        # calculate activations
        for n in h_1:
            h_1[n] = np.dot(arr[i].flatten(), w_1) + b_1
        for n in h_2:
            h_2[n] = np.dot(h_1[n], w_2) + b_2
        for n in out:
            out[n] = np.dot(h_2[n], w_3) + b_3

        # calculate cost
        cost = 0
        for n in out:
            cost += (out[n] - labels[i]) ** 2




if __name__ == '__main__':
    main()

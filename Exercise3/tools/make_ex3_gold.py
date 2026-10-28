# -*- coding: utf-8 -*-
"""
Generate golden expected outputs for Exercise3 and save to Data/ex3_gold.pkl.
Stores only pure data (numpy arrays, scalars) to be Python-version agnostic.
Parts:
 1) lrCostFunction(theta, X, y, lambda_) -> check J and grad
 2) oneVsAll(X, y, num_labels, lambda_)   -> check predictions on Xm
 3) predictOneVsAll(all_theta, X)         -> check predictions
 4) predict(Theta1, Theta2, X)            -> check predictions (forward NN)
"""
from __future__ import annotations
import os
import pickle
import numpy as np


def sigmoid(z):
    z = np.array(z)
    return 1.0 / (1.0 + np.exp(-z))


def lr_cost_grad(theta, X, y, lambda_):
    m = y.size
    z = X @ theta
    h = sigmoid(z)
    temp = theta.copy()
    temp[0] = 0.0
    J = (1.0/m) * np.sum(-y*np.log(h) - (1-y)*np.log(1-h)) + (lambda_/(2*m))*np.sum(temp**2)
    grad = (1.0/m) * ((h - y) @ X) + (lambda_/m) * temp
    return float(J), grad


def build_golden():
    # Part 1 test case (matches notebook sample)
    theta_t = np.array([-2, -1, 1, 2], dtype=float)
    X_t = np.concatenate([np.ones((5, 1)), np.arange(1, 16).reshape(5, 3, order='F')/10.0], axis=1)
    y_t = np.array([1, 0, 1, 0, 1])
    lambda_t = 3.0
    J1, grad1 = lr_cost_grad(theta_t, X_t, y_t, lambda_t)

    # Part 2 dataset and expected predictions (use labels as golden p)
    Xm = np.array([[-1, -1],
                   [-1, -2],
                   [-2, -1],
                   [-2, -2],
                   [1, 1],
                   [1, 2],
                   [2, 1],
                   [2, 2],
                   [-1, 1],
                   [-1, 2],
                   [-2, 1],
                   [-2, 2],
                   [1, -1],
                   [1, -2],
                   [-2, -1],
                   [-2, -2]])
    # Ensure consistent labels (Quadrant mapping); duplicates consistent
    ym = np.array([0, 0, 0, 0,   # QIII
                   1, 1, 1, 1,   # QI
                   2, 2, 2, 2,   # QII
                   3, 3,         # QIV
                   0, 0])        # QIII duplicates
    p2 = ym.copy()

    # Part 3 small all_theta test
    all_theta3 = np.array([[0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0],
                           [0.5, -1.0, 0.5],
                           [-0.5, 0.5, -1.0]])
    X3 = np.array([[2.0, 0.0],
                   [0.0, 2.0],
                   [1.0, -0.5],
                   [-1.0, 1.0]])
    p3 = np.argmax(sigmoid(np.c_[np.ones(X3.shape[0]), X3] @ all_theta3.T), axis=1)

    # Part 4 small NN forward test (non-trivial predictions)
    Theta1 = np.array([[0.0, 5.0, 0.0],
                       [0.0, 0.0, 5.0],
                       [0.0, 0.0, 0.0]])
    Theta2 = np.array([[0.2, 0.0, 0.0, 0.0],
                       [-0.4, 1.0, 0.0, 0.0]])
    X4 = np.array([[1.0, 2.0],
                   [-1.0, 0.5],
                   [0.6, 0.0]])
    a2 = sigmoid(np.c_[np.ones(X4.shape[0]), X4] @ Theta1.T)
    p4 = np.argmax(sigmoid(np.c_[np.ones(a2.shape[0]), a2] @ Theta2.T), axis=1)

    gold = {
        1: {
            'args': (theta_t, X_t, y_t, lambda_t),
            'expected': {'J': J1, 'grad': grad1}
        },
        2: {
            'args': (Xm, ym, 4, 0.1),
            'expected': p2
        },
        3: {
            'args': (all_theta3, X3),
            'expected': p3
        },
        4: {
            'args': (Theta1, Theta2, X4),
            'expected': p4
        }
    }
    return gold


def main():
    root = os.path.dirname(os.path.dirname(__file__))  # Exercise3
    data_dir = os.path.join(root, 'Data')
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, 'ex3_gold.pkl')

    gold = build_golden()
    with open(out_path, 'wb') as f:
        pickle.dump(gold, f, protocol=5)
    print(f'Wrote golden data to: {out_path}')


if __name__ == '__main__':
    main()

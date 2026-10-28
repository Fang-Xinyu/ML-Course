import sys
import os
import pickle
import numpy as np
from matplotlib import pyplot

sys.path.append('..')


def mapFeature(X1, X2, degree=6):
    """
    Maps the two input features to quadratic features used in the regularization exercise.

    Returns a new feature array with more features, comprising of
    X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..

    Parameters
    ----------
    X1 : array_like
        A vector of shape (m, 1), containing one feature for all examples.

    X2 : array_like
        A vector of shape (m, 1), containing a second feature for all examples.
        Inputs X1, X2 must be the same size.

    degree: int, optional
        The polynomial degree.

    Returns
    -------
    : array_like
        A matrix of of m rows, and columns depend on the degree of polynomial.
    """
    if X1.ndim > 0:
        out = [np.ones(X1.shape[0])]
    else:
        out = [np.ones(1)]

    for i in range(1, degree + 1):
        for j in range(i + 1):
            out.append((X1 ** (i - j)) * (X2 ** j))

    if X1.ndim > 0:
        return np.stack(out, axis=1)
    else:
        return np.array(out)


def plotDecisionBoundary(plotData, theta, X, y):
    """
    Plots the data points X and y into a new figure with the decision boundary defined by theta.
    Plots the data points with * for the positive examples and o for  the negative examples.

    Parameters
    ----------
    plotData : func
        A function reference for plotting the X, y data.

    theta : array_like
        Parameters for logistic regression. A vector of shape (n+1, ).

    X : array_like
        The input dataset. X is assumed to be  a either:
            1) Mx3 matrix, where the first column is an all ones column for the intercept.
            2) MxN, N>3 matrix, where the first column is all ones.

    y : array_like
        Vector of data labels of shape (m, ).
    """
    # This utility is for visualization in the notebook; grading does not depend on it.
    if X.shape[1] <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plotData(X[:, 1:3], y)
        if X.shape[1] == 3:
            plot_x = np.array([np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2])
            plot_y = (-1. / theta[2]) * (theta[1] * plot_x + theta[0])
            pyplot.plot(plot_x, plot_y)
        return
    
    # Here is the grid range
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    z = np.zeros((u.size, v.size))
    for i in range(u.size):
        for j in range(v.size):
            mapped = mapFeature(np.array([u[i]]), np.array([v[j]]))
            z[i, j] = mapped @ theta
    z = z.T
    plotData(X[:, 1:3], y)
    pyplot.contour(u, v, z, levels=[0], linewidths=2)

# Exercise2 Grader: local scoring, no SubmissionBase dependency
class Grader:

    def __init__(self):
        # Scoring weights per spec: 10, 20, 20, 10, 20, 20
        self.part_scores = [10, 20, 20, 10, 20, 20]
        self.max_score = 100
        self.part_names = [
            'Sigmoid Function',
            'Logistic Regression Cost',
            'Logistic Regression Gradient',
            'Predict',
            'Regularized Logistic Regression Cost',
            'Regularized Logistic Regression Gradient'
        ]
        self.functions = {}
        # Load golden expected outputs from pickle if present; otherwise use built-in defaults
        self.gold = self._load_golden_data()

    def __setitem__(self, key, value):
        self.functions[key] = value

    def __iter__(self):
        for part_id in range(1, 7):
            try:
                func = self.functions[part_id]
                yield part_id, func
            except KeyError:
                yield part_id, 0

    def grade(self):
        print('\n本地评分结果\n')
        print('%43s | %-8s | %-s' % ('Part Name', 'Score', 'Result'))
        print('%43s | %-8s | %-s' % ('---------', '--------', '--------'))
        total_score = 0
        for part_id in range(1, 7):
            score = self.part_scores[part_id - 1]
            try:
                func = self.functions.get(part_id)
                if func is None:
                    print('%43s | %-8d | 未通过' % (self.part_names[part_id - 1], 0))
                    continue
                g = self.gold.get(part_id)
                if g is None:
                    print('%43s | %-8d | 未通过' % (self.part_names[part_id - 1], 0))
                    continue
                args = g['args']
                try:
                    # 调用学生函数
                    if part_id == 1:
                        stu_out = func(*args)
                        passed = self._allclose(stu_out, g['expected'], atol=1e-4)
                    elif part_id == 2:
                        J_stu, _ = func(*args)
                        passed = np.isclose(J_stu, g['expected'], atol=1e-3, rtol=1e-6)
                    elif part_id == 3:
                        _, grad_stu = func(*args)
                        passed = self._allclose(grad_stu, g['expected'], atol=1e-2)
                    elif part_id == 4:
                        p_stu = func(*args)
                        passed = self._array_equal(p_stu, g['expected'])
                    elif part_id == 5:
                        J_stu, _ = func(*args)
                        passed = np.isclose(J_stu, g['expected'], atol=1e-3, rtol=1e-6)
                    elif part_id == 6:
                        _, grad_stu = func(*args)
                        passed = self._allclose(grad_stu, g['expected'], atol=1e-3)
                    else:
                        passed = False
                except Exception as e:
                    passed = False
                part_score = score if passed else 0
                total_score += part_score
                print('%43s | %-8d | %-s' % (self.part_names[part_id - 1], part_score, '通过' if passed else '未通过'))
            except Exception:
                print('%43s | %-8d | 未通过' % (self.part_names[part_id - 1], 0))
        print('\n总分: %d / %d' % (total_score, self.max_score))

    # Helpers
    @staticmethod
    def _allclose(a, b, atol=1e-6, rtol=1e-6):
        a = np.array(a)
        b = np.array(b)
        return a.shape == b.shape and np.allclose(a, b, atol=atol, rtol=rtol)

    @staticmethod
    def _array_equal(a, b):
        a = np.array(a)
        b = np.array(b)
        return a.shape == b.shape and np.array_equal(a, b)

    def _load_golden_data(self):
        # Try to load from pickle file if present
        pkl_path = os.path.join(os.path.dirname(__file__), 'Data', 'ex2_gold.pkl')
        if os.path.exists(pkl_path):
            try:
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)
                    return data
            except Exception:
                pass
        # Fallback: compute deterministic golden outputs for fixed tiny testcases
        return self._default_golden()

    def _default_golden(self):
        def _sigmoid(z):
            z = np.array(z)
            return 1.0 / (1.0 + np.exp(-z))

        # Tiny testcases
        z = np.array([0.0, 2.0, -2.0])
        X_small = np.array([[1.0, 45.0, 85.0],
                            [1.0, 50.0, 43.0],
                            [1.0, 60.0, 60.0]])
        y_small = np.array([1.0, 0.0, 1.0])
        theta_test = np.array([-24.0, 0.2, 0.2])

        # Regularized tiny case
        Xr = np.array([[1.0, 0.1, 0.6],
                       [1.0, 0.2, 0.7],
                       [1.0, 0.3, 0.8]])
        yr = np.array([1.0, 0.0, 1.0])
        thetar = np.ones(3)
        lambda_ = 1.0

        # Cost (unregularized)
        h = _sigmoid(X_small @ theta_test)
        J = (1.0 / y_small.size) * np.sum(-y_small * np.log(h) - (1 - y_small) * np.log(1 - h))
        # Grad (unregularized)
        grad = (1.0 / y_small.size) * ((h - y_small) @ X_small)

        # Predict
        p = np.round(_sigmoid(X_small @ theta_test))

        # Regularized cost
        hR = _sigmoid(Xr @ thetar)
        temp = thetar.copy(); temp[0] = 0.0
        Jr = (1.0 / yr.size) * np.sum(-yr * np.log(hR) - (1 - yr) * np.log(1 - hR)) + (lambda_ / (2.0 * yr.size)) * np.sum(temp ** 2)
        # Regularized grad
        gr = (1.0 / yr.size) * ((hR - yr) @ Xr) + (lambda_ / yr.size) * temp

        return {
            1: {
                'args': (z,),
                'expected': _sigmoid(z)
            },
            2: {
                'args': (theta_test, X_small, y_small),
                'expected': float(J)
            },
            3: {
                'args': (theta_test, X_small, y_small),
                'expected': grad
            },
            4: {
                'args': (theta_test, X_small),
                'expected': p.astype(int)
            },
            5: {
                'args': (thetar, Xr, yr, lambda_),
                'expected': float(Jr)
            },
            6: {
                'args': (thetar, Xr, yr, lambda_),
                'expected': gr
            }
        }

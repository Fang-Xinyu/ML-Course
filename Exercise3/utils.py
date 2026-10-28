import sys
import os
import pickle
import numpy as np
from matplotlib import pyplot

sys.path.append('..')


def displayData(X, example_width=None, figsize=(10, 10)):
    """
    Displays 2D data stored in X in a nice grid.
    """
    # Compute rows, cols
    if X.ndim == 2:
        m, n = X.shape
    elif X.ndim == 1:
        n = X.size
        m = 1
        X = X[None]  # Promote to a 2 dimensional array
    else:
        raise IndexError('Input X should be 1 or 2 dimensional.')

    example_width = example_width or int(np.round(np.sqrt(n)))
    example_height = n / example_width

    # Compute number of items to display
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    fig, ax_array = pyplot.subplots(display_rows, display_cols, figsize=figsize)
    fig.subplots_adjust(wspace=0.025, hspace=0.025)

    ax_array = [ax_array] if m == 1 else ax_array.ravel()

    for i, ax in enumerate(ax_array):
        ax.imshow(X[i].reshape(example_width, example_width, order='F'),
                  cmap='Greys', extent=[0, 1, 0, 1])
        ax.axis('off')


def sigmoid(z):
    """
    Computes the sigmoid of z.
    """
    return 1.0 / (1.0 + np.exp(-z))


class Grader:
    def __init__(self):
        self.part_names = [
            'Regularized Logistic Regression',
            'One-vs-All Classifier Training',
            'One-vs-All Classifier Prediction',
            'Neural Network Prediction Function'
        ]
        # Scores per spec: 30, 20, 20, 30
        self.part_scores = [30, 20, 20, 30]
        self.max_score = 100
        self.functions = {}
        self.gold = self._load_golden_data()

    def __setitem__(self, key, value):
        self.functions[key] = value

    def grade(self):
        print('\n本地评分结果\n')
        print('%43s | %-8s | %-s' % ('Part Name', 'Score', 'Result'))
        print('%43s | %-8s | %-s' % ('---------', '--------', '--------'))
        total_score = 0
        for part_id in range(1, 5):
            score = self.part_scores[part_id - 1]
            try:
                func = self.functions.get(part_id)
                g = self.gold.get(part_id)
                if func is None or g is None:
                    print('%43s | %-8d | 未通过' % (self.part_names[part_id - 1], 0))
                    continue
                passed = False
                if part_id == 1:
                    # lrCostFunction: expect J and grad correct
                    args = self._deepcopy_args(g['args'])  # (theta, X, y, lambda_)
                    J_stu, grad_stu = func(*args)
                    J_ok = np.isclose(J_stu, g['expected']['J'], atol=1e-6, rtol=1e-6)
                    grad_ok = self._allclose(grad_stu, g['expected']['grad'], atol=1e-5, rtol=1e-5)
                    passed = bool(J_ok and grad_ok)
                elif part_id == 2:
                    # oneVsAll: verify predictions on provided small set
                    args = self._deepcopy_args(g['args'])  # (X, y, num_labels, lambda_)
                    all_theta = func(*args)
                    X, y, num_labels, _ = args
                    p = self._predict_one_vs_all(all_theta, X)
                    passed = self._array_equal(p, g['expected'])
                elif part_id == 3:
                    # predictOneVsAll: compare predictions
                    args = self._deepcopy_args(g['args'])  # (all_theta, X)
                    p = func(*args)
                    passed = self._array_equal(p, g['expected'])
                elif part_id == 4:
                    # Neural net predict: compare predictions
                    args = self._deepcopy_args(g['args'])  # (Theta1, Theta2, X)
                    p = func(*args)
                    passed = self._array_equal(p, g['expected'])
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

    @staticmethod
    def _deepcopy_args(args):
        """Safely deep-copy a tuple of args containing numpy arrays/scalars to avoid in-place mutation by student code."""
        def _copy(x):
            if isinstance(x, np.ndarray):
                return x.copy()
            if isinstance(x, (list, tuple)):
                t = type(x)
                return t(_copy(v) for v in x)
            # Scalars (int/float/None/str)
            return x
        if isinstance(args, tuple):
            return tuple(_copy(a) for a in args)
        return _copy(args)

    @staticmethod
    def _predict_one_vs_all(all_theta, X):
        m = X.shape[0]
        Xb = np.concatenate([np.ones((m, 1)), X], axis=1)
        return np.argmax(sigmoid(Xb @ all_theta.T), axis=1)

    def _load_golden_data(self):
        pkl_path = os.path.join(os.path.dirname(__file__), 'Data', 'ex3_gold.pkl')
        if os.path.exists(pkl_path):
            try:
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)
                    return data
            except Exception:
                pass
        return self._default_golden()

    def _default_golden(self):
        # Part 1 test case (from notebook Section 1.3 vectorized check)
        theta_t = np.array([-2, -1, 1, 2], dtype=float)
        X_t = np.concatenate([np.ones((5, 1)), np.arange(1, 16).reshape(5, 3, order='F')/10.0], axis=1)
        y_t = np.array([1, 0, 1, 0, 1])
        lambda_t = 3.0

        def _lr_cost_grad(theta, X, y, lambda_):
            m = y.size
            z = X @ theta
            h = sigmoid(z)
            temp = theta.copy(); temp[0] = 0.0
            J = (1.0/m) * np.sum(-y*np.log(h) - (1-y)*np.log(1-h)) + (lambda_/(2*m))*np.sum(temp**2)
            grad = (1.0/m) * ((h - y) @ X) + (lambda_/m) * temp
            return float(J), grad

        J1, grad1 = _lr_cost_grad(theta_t, X_t, y_t, lambda_t)

        # Part 2 small dataset for OVA training validation
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
        # Ensure consistent labels: Quadrant III should be class 0 (not 3)
        ym = np.array([0, 0, 0, 0,   # QIII points correctly labeled 0
                       1, 1, 1, 1,   # QI
                       2, 2, 2, 2,   # QII
                       3, 3,         # QIV
                       0, 0])        # duplicates consistent with QIII
        # Build a simple reference all_theta by training is brittle; instead we only keep expected predictions
        # We can derive expected predictions using a hand-crafted linear classifier
        # For simplicity, expect classes by quadrant-like mapping with a fixed all_theta_ref
        # However to decouple from training, we store only expected predictions from an idealized mapping
        # We'll construct them by a simple rule here for golden data stability
        # Golden predictions for Xm as per ym
        p2 = ym.copy()

        # Part 3: provide a small all_theta and X to test predictOneVsAll
        all_theta3 = np.array([[0.0, 1.0, 0.0],   # class 0 weights
                               [0.0, 0.0, 1.0],   # class 1
                               [0.5, -1.0, 0.5],  # class 2
                               [-0.5, 0.5, -1.0]])  # class 3
        X3 = np.array([[2.0, 0.0],   # likely class 0
                       [0.0, 2.0],   # likely class 1
                       [1.0, -0.5],  # likely class 2 (depends)
                       [-1.0, 1.0]]) # likely class 3
        p3 = np.argmax(sigmoid(np.c_[np.ones(X3.shape[0]), X3] @ all_theta3.T), axis=1)

        # Part 4: small NN forward test (ensure non-trivial predictions)
        # Design Theta1/Theta2 so that predictions vary across X4
        Theta1 = np.array([[0.0, 5.0, 0.0],
                           [0.0, 0.0, 5.0],
                           [0.0, 0.0, 0.0]])  # (3 x 3) hidden=3, input=2(+bias)
        Theta2 = np.array([[0.2, 0.0, 0.0, 0.0],
                           [-0.4, 1.0, 0.0, 0.0]])  # (2 x 4) labels=2, hidden=3(+bias)
        X4 = np.array([[1.0, 2.0],
                       [-1.0, 0.5],
                       [0.6, 0.0]])
        a2 = sigmoid(np.c_[np.ones(X4.shape[0]), X4] @ Theta1.T)
        p4 = np.argmax(sigmoid(np.c_[np.ones(a2.shape[0]), a2] @ Theta2.T), axis=1)

        return {
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

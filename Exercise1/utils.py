import numpy as np

class Grader:
    part_scores = [20, 20, 20, 20, 20, 20, 20]
    max_score = 140

    def grade(self):
        print('\n本地评分结果\n')
        print('%43s | %-8s | %-s' % ('Part Name', 'Score', 'Result'))
        print('%43s | %-8s | %-s' % ('---------', '--------', '--------'))
        total_score = 0
        for part_id in range(1, 8):
            score = self.part_scores[part_id - 1]
            try:
                func = self.functions.get(part_id)
                std_func = self.std_funcs.get(part_id)
                if func is None or std_func is None:
                    print('%43s | %-8d | 未通过' % (self.part_names[part_id - 1], 0))
                    continue
                # 使用固定的测试输入，分别调用标准答案和学生函数进行对比
                passed = False
                if part_id == 1:
                    std_out = std_func()
                    stu_out = func()
                    passed = isinstance(stu_out, np.ndarray) and std_out.shape == stu_out.shape and np.allclose(stu_out, std_out, atol=1e-8, rtol=1e-8)
                elif part_id == 2:
                    X1 = self.X1; Y1 = self.Y1; theta = np.array([0.5, -0.5])
                    std_J = std_func(X1, Y1, theta)
                    stu_J = func(X1, Y1, theta)
                    passed = np.isclose(stu_J, std_J, atol=1e-8, rtol=1e-7)
                elif part_id == 3:
                    X1 = self.X1; Y1 = self.Y1; theta0 = np.array([0.5, -0.5])
                    std_theta, std_hist = std_func(X1, Y1, theta0, 0.01, 10)
                    stu_theta, stu_hist = func(X1, Y1, theta0, 0.01, 10)
                    passed = isinstance(stu_theta, np.ndarray) and std_theta.shape == stu_theta.shape and np.allclose(stu_theta, std_theta, atol=1e-6, rtol=1e-6)
                elif part_id == 4:
                    X3 = self.X2[:, 1:4]
                    std_Xn, std_mu, std_sigma = std_func(X3)
                    stu_Xn, stu_mu, stu_sigma = func(X3)
                    passed = (std_Xn.shape == stu_Xn.shape and std_mu.shape == stu_mu.shape and std_sigma.shape == stu_sigma.shape and
                              np.allclose(stu_Xn, std_Xn, atol=1e-6, rtol=1e-6) and
                              np.allclose(stu_mu, std_mu, atol=1e-8, rtol=1e-8) and
                              np.allclose(stu_sigma, std_sigma, atol=1e-8, rtol=1e-8))
                elif part_id == 5:
                    X2 = self.X2; Y2 = self.Y2; theta = np.array([0.1, 0.2, 0.3, 0.4])
                    std_J = std_func(X2, Y2, theta)
                    stu_J = func(X2, Y2, theta)
                    passed = np.isclose(stu_J, std_J, atol=1e-8, rtol=1e-7)
                elif part_id == 6:
                    X2 = self.X2; Y2 = self.Y2; theta0 = np.array([-0.1, -0.2, -0.3, -0.4])
                    std_theta, _ = std_func(X2, Y2, theta0, 0.01, 10)
                    stu_theta, _ = func(X2, Y2, theta0, 0.01, 10)
                    passed = isinstance(stu_theta, np.ndarray) and std_theta.shape == stu_theta.shape and np.allclose(stu_theta, std_theta, atol=1e-6, rtol=1e-6)
                elif part_id == 7:
                    X2 = self.X2; Y2 = self.Y2
                    std_theta = std_func(X2, Y2)
                    stu_theta = func(X2, Y2)
                    passed = isinstance(stu_theta, np.ndarray) and std_theta.shape == stu_theta.shape and np.allclose(stu_theta, std_theta, atol=1e-6, rtol=1e-6)
                else:
                    passed = False
                part_score = score if passed else 0
                total_score += part_score
                print('%43s | %-8d | %-s' % (self.part_names[part_id - 1], part_score, '通过' if passed else '未通过'))
            except Exception:
                print('%43s | %-8d | 未通过' % (self.part_names[part_id - 1], 0))
        print('\n总分: %d / %d' % (total_score, self.max_score))
    X1 = np.column_stack((np.ones(20), np.exp(1) + np.exp(2) * np.linspace(0.1, 2, 20)))
    Y1 = X1[:, 1] + np.sin(X1[:, 0]) + np.cos(X1[:, 1])
    X2 = np.column_stack((X1, X1[:, 1]**0.5, X1[:, 1]**0.25))
    Y2 = np.power(Y1, 0.5) + Y1

    def __init__(self):
        self.part_names = [
            'Warm up exercise',
            'Computing Cost (for one variable)',
            'Gradient Descent (for one variable)',
            'Feature Normalization',
            'Computing Cost (for multiple variables)',
            'Gradient Descent (for multiple variables)',
            'Normal Equations'
        ]
        self.functions = {}
        # 载入标准答案函数
        try:
            import answers as _answers
        except Exception:
            _answers = None
        self.std_funcs = {}
        if _answers is not None:
            # 优先从模块属性获取，其次从 _answers.answers 映射获取
            std_map = getattr(_answers, 'answers', {}) if hasattr(_answers, 'answers') else {}
            def pick(name, pid):
                fn = getattr(_answers, name, None)
                if fn is None:
                    fn = std_map.get(pid)
                return fn
            self.std_funcs = {
                1: pick('warmUpExercise', 1),
                2: pick('computeCost', 2),
                3: pick('gradientDescent', 3),
                4: pick('featureNormalize', 4),
                5: pick('computeCostMulti', 5),
                6: pick('gradientDescentMulti', 6),
                7: pick('normalEqn', 7),
            }
        else:
            self.std_funcs = {i: None for i in range(1, 8)}

    def __setitem__(self, key, value):
        self.functions[key] = value

    def __iter__(self):
        for part_id in range(1, 8):
            try:
                func = self.functions[part_id]
                if part_id == 1:
                    res = func()
                elif part_id == 2:
                    res = func(self.X1, self.Y1, np.array([0.5, -0.5]))
                elif part_id == 3:
                    res = func(self.X1, self.Y1, np.array([0.5, -0.5]), 0.01, 10)
                elif part_id == 4:
                    res = func(self.X2[:, 1:4])
                elif part_id == 5:
                    res = func(self.X2, self.Y2, np.array([0.1, 0.2, 0.3, 0.4]))
                elif part_id == 6:
                    res = func(self.X2, self.Y2, np.array([-0.1, -0.2, -0.3, -0.4]), 0.01, 10)
                elif part_id == 7:
                    res = func(self.X2, self.Y2)
                else:
                    raise KeyError
                yield part_id, res
            except KeyError:
                yield part_id, 0

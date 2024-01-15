import numpy as np
import math
import pickle
from sklearn.feature_selection import SelectPercentile, f_classif
from integral_image import integralimage
from haar import get_all_haar_features

class WeakClassifier:
    def __init__(self, positive_regions, negative_regions, threshold, polarity):
        self.positive_regions = positive_regions
        self.negative_regions = negative_regions
        self.threshold = threshold
        self.polarity = polarity

    def classify(self, x):
        feature = lambda ii: sum([pos.compute_feature(ii) for pos in self.positive_regions]) - \
                             sum([neg.compute_feature(ii) for neg in self.negative_regions])

        if self.polarity * feature(x) < self.polarity * self.threshold:
            return 1
        return 0

    def __str__(self):
        return "Weak Clf (threshold= %d, polarity= %d, %s, %s)" % \
               (self.threshold, self.polarity, str(self.positive_regions), str(self.negative_regions))

class ViolaJones:

    def __init__(self, T = 10):
        self.T = T
        self.alphas = []
        self.clfs = []

    def train(self, training, pos_num, neg_num):
        weights = np.zeros(len(training))
        training_data = []

        print('computing integral images')
        for x in range(len(training)):
            # Tuple(data, label) -->  Tuple(integral_image, label)
            training_data.append((integralimage(training[x][0]), training[x][1]))
            if training[x][1] == 1:
                weights[x] = 1.0 / (2 * pos_num)
            else:
                weights[x] = 1.0 / (2 * neg_num)

        print("building features")
        features = self.build_features(training_data[0][0].shape)

        # features1 = self.build_features2(training_data[0][0].shape)
        print("applying features to training examples")
        X, y = self.apply_features(features, training_data)

        print('selecting best features')
        selector = SelectPercentile(f_classif, percentile=10)  #
        selector.fit(X.T, y)
        selector = selector.get_support(indices=True)
        X = X[selector]
        features = [features[i] for i in selector]
        print('Selected {} potential features'.format(len(X)))

        for t in range(self.T):
            weights = weights / np.linalg.norm(weights)
            weak_classifiers = self.train_weak(X, y, features, weights)
            clf, error, accuracy = self.select_best(weak_classifiers, weights, training_data)
            beta = error / (1.0 - error)
            for i in range(len(accuracy)):
                weights[i] = weights[i] * (beta ** (1 - accuracy[i]))
            alpha = math.log(1.0/beta)
            self.alphas.append(alpha)
            self.clfs.append(clf)
            print("chose classifier: %s with accuracy: %f and alpha: %f" % (str(clf), len(accuracy) - sum(accuracy), alpha))

    def train_weak(self, X, y, features, weights):
        total_pos, total_neg = 0, 0
        for w, label in zip(weights, y):
            if label == 1:
                total_pos += w
            else:
                total_neg += w
        classifiers = []
        total_features = X.shape[0]

        for index, feature in enumerate(X):
            if len(classifiers) % 1000 == 0 and len(classifiers) != 0:
                print("Trained %d classifiers out of %d" % (len(classifiers), total_features))

            applied_feature = sorted(zip(weights, feature, y), key=lambda x: x[1])
            pos_seen, neg_seen = 0, 0
            pos_weights, neg_weights = 0, 0
            min_error, best_feature, best_threshold, best_polarity = float('inf'), None, None, None
            for w, f, label in applied_feature:
                error = min(neg_weights + total_pos - pos_weights, pos_weights + total_neg - neg_weights)
                if error < min_error:
                    min_error = error
                    best_feature = features[index]
                    best_threshold = f
                    best_polarity = 1 if pos_seen > neg_seen else -1

                if label == 1:
                    pos_seen += 1
                    pos_weights += w
                else:
                    neg_seen += 1
                    neg_weights += w
            clf = WeakClassifier(best_feature[0], best_feature[1], best_threshold, best_polarity)
            classifiers.append(clf)
        return classifiers

    # 基于haar特征的每个模式，构建特征
    def build_features(self, image_shape):
        h, w = image_shape
        return get_all_haar_features(h, w)

    # 基于分类准确率，更新最优模型
    def select_best(self, classifiers, weights, training_data):
        best_clf, best_error, best_accuracy = None, float('inf'), None
        for clf in classifiers:
            error, accuracy = 0, []
            for data, w in zip(training_data, weights):
                correctness = abs(clf.classify(data[0]) - data[1])  # 测试结果-标签
                accuracy.append(correctness)
                error += w * correctness
            error = error / len(training_data)
            if error < best_error:  # 当前分类器 误差小于 记录最优，更新
                best_clf, best_error, best_accuracy = clf, error, accuracy
        return best_clf, best_error, best_accuracy


    def apply_features(self, features, training_data):
        y = np.array(list(map(lambda data: data[1], training_data)))  # 直接取存放的标签
        # 计算特征
        X = np.zeros((len(features), len(training_data)))
        i = 0
        for positive_regions, negative_regions in features:
            #print(i)
            feature = lambda ii: sum([pos.compute_feature(ii) for pos in positive_regions]) - \
                                 sum([neg.compute_feature(ii) for neg in negative_regions])
            X[i] = list(map(lambda data: feature(data[0]), training_data))
            i += 1
        return X, y

    def classify(self, image):
        total = 0
        ii = integralimage(image)
        for alpha, clf in zip(self.alphas, self.clfs):
            total += alpha * clf.classify(ii)  # 带权累加分类结果= 累加每个弱分类器结果 * alpha
        #
        return 1 if total >= 0.5 * sum(self.alphas) else 0
    # 存模型
    def save(self, filename):
        with open(filename+'.pkl', 'wb') as f:
            pickle.dump(self, f)

    # 从filename.pkl载入模型参数
    @staticmethod
    def load(filename):
        with open(filename+'.pkl', 'rb') as f:
            return pickle.load(f)




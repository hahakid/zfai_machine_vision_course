from viola_jones import ViolaJones
import pickle

class CascadeClassifier():
    def __init__(self, layers):
        self.layers = layers
        self.clfs = []  # list of classifiers

    def train(self, training):
        pos, neg = [], []
        for ex in training:  # iter train set data
            if ex[1] == 1:  # positive label
                pos.append(ex)
            else:
                neg.append(ex)

        for feature_num in self.layers:
            if len(neg) == 0:  # early stop condition: current classifiers can detect all sample without adding more clf
                print("stopping early. FPR=0")
                break
            clf = ViolaJones(T=feature_num)  # new a classifier
            clf.train(pos+neg, len(pos), len(neg))  # train on pos+neg
            self.clfs.append(clf)  # add to clfs list
            false_positives = []  #
            for ex in neg:
                if self.classify(ex[0]) == 1:
                    false_positives.append(ex)
            neg = false_positives

    def classify(self, image):
        for clf in self.clfs:
            if clf.classify(image) == 0:
                return 0
        return 1

    def save(self, filename):
        with open(filename+'.pkl', 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename+'.pkl', 'rb') as f:
            return pickle.load(f)







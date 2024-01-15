import numpy as np
import pickle
from viola_jones import ViolaJones
from cascade import CascadeClassifier
import time
import cv2
import matplotlib.pyplot as plt
from sklearn import metrics

# @layers @data @save name
def train_cascade(layers, train_data, filename='cascade'):
    with open(train_data, 'rb') as f:
        training = pickle.load(f)

    clf = CascadeClassifier(layers)
    clf.train(training)
    evaluation(clf, training)
    clf.save(filename)

# @data @ model paramaters
def test_cascade(test_data, filename='cascade'):
    with open(test_data, 'rb') as f:
        test = pickle.load(f)
    clf = CascadeClassifier.load(filename)
    evaluation(clf, test)

def evaluation(clf, data):
    correct = 0
    all_negatives, all_positive = 0, 0
    true_negatives, false_negatives = 0, 0
    true_positives, false_positives = 0, 0
    classification_time = 0
    y_arr = []
    y_pred = []

    # iter all data
    for x, y in data:
        if y == 1:
            all_positive += 1
        else:
            all_negatives += 1

        start = time.time()
        prediction = clf.classify(x)  # result from classifier
        y_pred.append(prediction)  # add predicted
        y_arr.append(y)  # add label
        classification_time += time.time() - start
        if prediction == 1 and y == 0:
            false_positives += 1
        if prediction == 0 and y == 1:
            false_negatives += 1

        correct += 1 if prediction == y else 0

    confusion_matrix = metrics.confusion_matrix(y_arr, y_pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels=["non-face", "face"])
    cm_display.plot()
    plt.savefig("confusion matrix.png")
    print("False Positive Rate: %d/%d (%f)" % (false_positives, all_negatives, false_positives/all_negatives))
    print("False Negative Rate: %d/%d (%f)" % (false_negatives, all_positive, false_negatives/all_positive))
    print("Accuracy: %d/%d (%f)" % (correct, len(data), correct/len(data)))
    print("Average Classification Time: %f FPS") % (classification_time/len(data))







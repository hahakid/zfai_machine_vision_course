# https://vovkos.github.io/doxyrest-showcase/opencv/sphinx_rtd_theme/page_tutorial_py_face_detection.html#doxid-d7-d8b-tutorial-py-face-detection
# Haar Feature-based Cascade Classifiers
# https://vovkos.github.io/doxyrest-showcase/opencv/sphinx_rtd_theme/page_tutorial_traincascade.html#doxid-dc-d88-tutorial-traincascade
'''
step1: kernel size
step2: feature calculation for each kernel, sum of pixels under white and black rectangle
step3: Adaboost training.
'''
import cv2
import numpy as np
import os
import pickle
from cascade import CascadeClassifier
from evol import test_cascade, train_cascade


# return pixel at (i,j) of a integral image

def detect(clf, img):
    img = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    w_size = 24
    step = 4
    w, h, _ = img.shape
    for i in range(0, w - w_size, step):  # sliding window with (24, 24) + step(2, 2)
        for j in range(0, h - w_size, step):
            print(i, j)
            window = gray[i:i + w_size, j:j + w_size]  # cur_window pixels
            if clf.classify(window) == 1:
                cv2.rectangle(img, (j, i), (j + w_size, i + w_size), (0, 255, 0), 2)  # if True add a Green bounding box

    cv2.imwrite("result.jpg", img)
    cv2.imshow("", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    mode = "valid"

    if mode == 'train':
        # train data is needed
        train_data = './training.pkl'
        # train_cascade([1, 2, 5, 10, 50], train_data, "cascade")
        train_cascade([1, 5, 10], train_data, "cascade")

    if mode == 'valid':  # valid data has label, can use evaluation
        # eval on test set
        test_data = 'test.pkl'
        model_path = 'cascade'
        test_cascade(test_data, model_path)

    if mode == 'test':  # test data has no label, can only show result
        # test on single img or video
        model_path = 'cascade'
        img = 'solvay.jpg'
        clf = CascadeClassifier.load(model_path)  # load auto add '.pkl'
        detect(clf, img)


if __name__ == '__main__':
    main()

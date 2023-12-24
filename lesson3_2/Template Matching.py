import cv2
import numpy as np
from matplotlib import pyplot as plt

def nothing(x):
    pass


img = cv2.imread("../data/trafficlight2.jpg", 0)
img2 = img.copy()

template = cv2.imread("../data/trafficlight2_template1.jpg", 0)
template = cv2.blur(template, (7, 7))
h, w = template.shape
# https://vovkos.github.io/doxyrest-showcase/opencv/sphinx_rtd_theme/enum_cv_TemplateMatchModes.html?highlight=tm_ccorr#details-df-dfb-group-imgproc-object-1ga3a7850640f1fe1f58fe91a2d7583695d
methods = ['cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED']
normed_methods = ['cv2.TM_CCOEFF_NORMED'] #, 'cv2.TM_CCORR_NORMED', 'cv2.TM_CCOEFF_NORMED']
for meth in normed_methods:
    '''
    
    img = img.copy()
    method = eval(meth)  # index
    #print(meth)
    # @ input @ template @ method
    # return a score matrix of sliding window of (img_width - template_width + 1, img_high - template_high + 1)
    res = cv2.matchTemplate(img, template, method)
    #print(res)
    # @ precentage
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    #print(min_val, max_val, min_loc, max_loc)


    # adjust coord
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    image plane:
    top_left----- *
    |             |
    |             |
    *----------down_right
    
    # be careful of the order of h and w
    bottom_right = (top_left[0] + h, top_left[1] + w)

    #cv2.rectangle(img, top_left, bottom_right, 255, 2)
    #cv2.imshow("", img)
    #cv2.waitKey()
    
    plt.subplot(121)
    plt.imshow(res, cmap='gray')
    plt.title('matching result')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(122)
    plt.imshow(img, cmap='gray')
    plt.title('detected point')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    '''
    if meth in normed_methods: # normalized can be used to select thresh

        res = cv2.matchTemplate(img, template, eval(meth))
        cv2.namedWindow("test")
        cv2.createTrackbar("thresh", 'test', 1, 10, nothing)
        cv2.setTrackbarMin("thresh", 'test', 1)

        while True:
            img3 = img2.copy()
            thresh = cv2.getTrackbarPos("thresh", 'test') / 10.0
            # print(thresh)
            loc = np.where(res >= thresh)
            # print(len(loc))
            #cv2.imshow('test', img3)
            #k = cv2.waitKey()

            for pt in zip(*loc[::-1]):
                print(pt)
                # @point1 @point2 @ color @ thickness
                cv2.rectangle(img3, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)

            cv2.imshow('test', img3)
            k = cv2.waitKey(100)
            if k == 27:
                break
        #'''
        cv2.destroyAllWindows()





















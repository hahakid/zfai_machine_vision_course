import numpy as np
from integral_image import integralimage as ii

class RectangleRegion:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def compute_feature(self, ii):
        # D' = D + A - B - C
        return ii[self.y + self.height][self.x + self.width] + ii[self.y][self.x]\
               - ii[self.y + self.height][self.x] - ii[self.y][self.x + self.width]

    def __str__(self):
        return "(x= %d, y= %d, width= %d, height= %d)" % (self.x, self.y, self.width, self.height)

    def __repr__(self):
        return "RectangleRegion(%d, %d, %d, %d)" % (self.x, self.y, self.width, self.height)

def get_all_haar_features(height, width):
    features = []
    for w in range(1, width+1):
        for h in range(1, height+1):
            for i in range(width - w):  #row
                for j in range(height - h):  # column
                    '''
                    (i,j)------(i+w,j)
                    |                |
                    (i,j+h)---(i+w,j+h)
                    
                    location:
                    
                    immmediate ------ right ------------- right_2
                    |                   |                   |
                    bottom ---------- bottom_right--------  |
                    |                   |                   |
                    bottom2----------   | ------------------|
                    1: immmediate-right
                    2: immmediate-right-right_2
                    3: immmediate-bottom
                    4: immmediate-bottom-bottom_2
                    5: immmediate-right-bottom-bottom_right
                    '''
                    immediate = RectangleRegion(i, j, w, h)
                    right = RectangleRegion(i+w, j, w, h)
                    right_2 = RectangleRegion(i+2*w, j, w, h)
                    bottom = RectangleRegion(i, j+h, w, h)
                    bottom_2 = RectangleRegion(i, j+2*h, w, h)
                    bottom_right = RectangleRegion(i+w, j+h, w, h)
                    if i + 2 * w < width: # right-left
                        features.append(([right], [immediate]))
                    if j + 2 * h < height: # bottom-up
                        features.append(([immediate], [bottom]))
                    if i + 3 * w < width: #
                        features.append(([right], [right_2, immediate]))
                    if j + 3 * h < height:
                        features.append(([bottom], [bottom_2, immediate]))
                    if i + 2 * w < width and j + 2 * h < height:
                        features.append(([right, bottom], [immediate, bottom_right]))
                    j += 1
                i += 1
    return np.array(features, dtype=object)


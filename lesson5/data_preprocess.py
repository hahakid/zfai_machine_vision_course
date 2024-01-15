import os
import pickle
import numpy as np
from PIL import Image
'''

http://cbcl.mit.edu/software-datasets/FaceData2.html

Each image is 19x19 and greyscale. There are Training set: 2,429 faces, 4,548 non-faces Test set: 472 faces, 23,573 non-faces

training.pkl
An array of tuples. 
The first element of each tuple is a numpy array representing the image. 
The second element is its classification (1 for face, 0 for non-face)
2429 face images, 4548 non-face images, total = 6977

test.pkl 
An array of tuples. The first element of each tuple is a numpy array representing the image. The second element is its classification (1 for face, 0 for non-face)
472 faces, 23573 non-face images, total = 24045

'''
label = ['face', 'non-face']
path1 = r'./data/train'
path2 = r'./data/test'

def save_pkl(path, save_name):
    label_flag = 1
    arr = []
    for l in label:
        full_path = os.path.join(path, l)
        if l == 'face':
            label_flag = 1
        else:
            label_flag = 0
        images = os.listdir(full_path)
        for im in images:
            #print(im)
            data = Image.open(os.path.join(full_path, im))
            data = np.asarray(data)
            tup = (data, label_flag)
            arr.append(tup)
    #print(arr)
    with open(save_name, 'wb') as f:
        pickle.dump(arr, f)

save_pkl(path1, './training.pkl')
save_pkl(path2, './test.pkl')
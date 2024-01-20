from mmdet.apis import DetInferencer
import mmcv
# models = DetInferencer.list_models('mmdet')  # this will show all available models in this repo
# inferencer = DetInferencer(model='rtmdet_tiny_8xb32-300e_coco')  # download model from url in config

# use weights from local file
#inferencer = DetInferencer(model='rtmdet_tiny_8xb32-300e_coco', weights='./weights/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth')  # refer names in configs/

# assign a specific device: cuda-0 when the machine contains multiple cards (e.g. a GPU server)
inferencer = DetInferencer(model='rtmdet_tiny_8xb32-300e_coco'
                           , weights='./weights/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
                           , device='cuda:0')

img_path = '../demo/demo.jpg'
# source data 1: url
inferencer('../demo/demo.jpg', show=True)

# source data 2: numpy array
array = mmcv.imread("../demo/demo.jpg")  # can use opencv or camera api
#inferencer(array)  # array can be [array1, array2, ...]

# with save function
# generate both visualized results and a json
#inferencer(img_path, out_dir='./save', no_save_pred=False)  # only accept url for image, cannot be an array


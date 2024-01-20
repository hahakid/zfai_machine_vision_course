import cv2
import mmcv
import torch
import time
from mmdet.apis import inference_detector, init_detector, DetInferencer
from mmdet.registry import VISUALIZERS

def main():
    config = '../configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py'
    checkpoint = './weights/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
    score_thr = 0.3
    # build the model from a config file and a checkpoint file
    device = torch.device('cuda:0')
    model = init_detector(config, checkpoint, device=device)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_detector
    visualizer.dataset_meta = model.dataset_meta

    camera = cv2.VideoCapture(0)  # camera
    ret, img = camera.read()
    h, w, _ = img.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 10
    writer = cv2.VideoWriter('./save/test.mp4', fourcc, fps, (w, h))
    print('Press "Esc", "q" or "Q" to exit.')
    count = 0
    bt = time.time()
    while ret:
        #camera.grab()
        count += 1
        ret, img = camera.read()

        result = inference_detector(model, img)
        ct = time.time() - bt
        img = mmcv.imconvert(img, 'bgr', 'rgb')
        #'''
        visualizer.add_datasample(
            name='result',
            image=img,
            data_sample=result,
            draw_gt=False,
            pred_score_thr=score_thr,
            show=False)
        #'''
        img = visualizer.get_image()
        img = mmcv.imconvert(img, 'bgr', 'rgb')
        fps = str(round(count * 1.0 / ct, 2))
        img = cv2.putText(img, str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        writer.write(img)
        cv2.imshow('', img)
        ch = cv2.waitKey(1)
        #camera.retrieve()
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break
    writer.release()
    camera.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()
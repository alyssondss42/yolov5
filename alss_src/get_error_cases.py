import os
import json
import torch
import cv2 as cv
from alss_src.utils import count_signatures, read_json_gt
from alss_src.mark_img import mark_detection


def load_model():
    mdl = torch.hub.load('C:/Users/Alysson/PycharmProjects/yolov5', 'custom',
                         path='../signature_detector/model_v3/best.pt', source='local')
    return mdl


def get_error(img_path, json_path):
    model = load_model()

    for (root, dirnames, files) in os.walk(json_path):
        for filename in files:
            # current_img = cv.imread(os.path.join(path_img, filename))
            # gt_filename = filename.split('.')[0] + '.json'

            sig_qtd, sig_list, img_name = read_json_gt(json_path, filename)

            new_img_name = img_name.split('.')[0] + '.jpg'
            current_img = cv.imread(os.path.join(img_path, new_img_name))

            detection = model(current_img, size=1280)
            pred_count = count_signatures(result_pred=detection.pandas().xyxy[0], thr=0.5)

            if pred_count != sig_qtd:
                mark_detection(new_img_name, current_img, sig_list=[], detection_pts=detection.pandas().xyxy[0])
                print('entrei')

            print('File {} processed.'.format(filename))


if __name__ == '__main__':
    img_path = r'C:\Users\Alysson\Downloads\tcc_dataset_v2\YOLO\yolo\images\test'
    gt_path = r'C:\Users\Alysson\Downloads\tcc_dataset_v2\partition\test_partition\json_gt'

    get_error(img_path=img_path, json_path=gt_path)

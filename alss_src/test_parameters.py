import os
import json
import torch
import cv2 as cv
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from infer_evaluate import count_signatures_with_iou
from utils import read_json_gt, root_mean_squared_error, count_signatures


def load_model():
    mdl = torch.hub.load('C:/Users/Alysson/PycharmProjects/yolov5', 'custom',
                         path='../signature_detector/model_v3_trlr/best.pt', source='local')
    return mdl


def test_param(path_img, json_path):
    model = load_model()

    gt_list = []
    list_of_sigs = []
    pred_list = []
    detection_list = []
    thr_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # only detect
    for (root, dirnames, files) in os.walk(json_path):
        for filename in files:
            # current_img = cv.imread(os.path.join(path_img, filename))
            # gt_filename = filename.split('.')[0] + '.json'

            sig_qtd, sig_list, img_name = read_json_gt(json_path, filename)
            gt_list.append(sig_qtd)

            new_img_name = img_name.split('.')[0]+'.jpg'
            current_img = cv.imread(os.path.join(path_img, new_img_name))

            detection = model(current_img, size=1280)
            detection_list.append(detection)
            # detection.save()
            # pred_list.append(count_signatures(result_pred=detection.pandas().xyxy[0], thr=0.5))

            print('File {} processed.'.format(filename))

    for thrx in thr_list:
        pred_list = []

        for element_det in detection_list:
            pred_list.append(count_signatures(result_pred=element_det.pandas().xyxy[0], thr=thrx))
            # pred_list.append(count_signatures_with_iou(result_pred=element_det.pandas().xyxy[0],
            #                                            list_sig=l_sig, thr=thrx))

        rmse = root_mean_squared_error(predictions=np.array(pred_list), targets=np.array(gt_list))
        mae = mean_absolute_error(y_true=gt_list, y_pred=pred_list)

        json_name = 'model_metrics_' + str(thrx) + '.json'
        with open(os.path.join('./detection_output', json_name), 'w', encoding='utf-8') as result_file:
            result_file.write(json.dumps({
                'result': {
                    'mae': mae,
                    'rmse': rmse
                }
            }))
            result_file.close()

        print('--------------------------------')


if __name__ == '__main__':
    img_path = r'C:\Users\Alysson\Downloads\TCC_dataset_xerox\LoteCertidao\yolo\images\train'
    gt_path = r'C:\Users\Alysson\Downloads\TCC_dataset_xerox\LoteCertidao\json_alss'

    test_param(path_img=img_path, json_path=gt_path)


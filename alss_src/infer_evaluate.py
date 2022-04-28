import os
import json
import torch
import cv2 as cv
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def compute_iou(bbox_gt, bbox_pred):
    xA = max(bbox_gt[0], bbox_pred[0])
    yA = max(bbox_gt[1], bbox_pred[1])
    xB = min(bbox_gt[2], bbox_pred[2])
    yB = min(bbox_gt[3], bbox_pred[3])

    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    if inter_area == 0:
        return 0
    else:
        bbox_gt_area = abs((bbox_gt[2] - bbox_gt[0]) * (bbox_gt[3] - bbox_gt[1]))
        bbox_pred_area = abs((bbox_pred[2] - bbox_pred[0]) * (bbox_pred[3] - bbox_pred[1]))

        iou = inter_area/float(bbox_gt_area + bbox_pred_area - inter_area)

        return iou


# Conta assinaturas com a avaliação do IoU
def count_signatures_with_iou(result_pred, list_sig, thr=0.5):
    object_count = 0
    for idx, element in result_pred.iterrows():
        det_x_min = element.xmin
        det_y_min = element.ymin
        det_x_max = element.xmax
        det_y_max = element.ymax

        conf = element.confidence

        best_iou = 0
        for sig in list_sig:
            x_min = sig['bounding_box']['x']
            y_min = sig['bounding_box']['y']
            x_max = x_min + sig['bounding_box']['width']
            y_max = y_min + sig['bounding_box']['height']

            current_iou = compute_iou(bbox_gt=[x_min, y_min, x_max, y_max],
                                      bbox_pred=[det_x_min, det_y_min, det_x_max, det_y_max])

            if current_iou > best_iou:
                best_iou = current_iou

        if conf > thr and best_iou > 0.5:
            object_count += 1
            best_iou = 0

    return object_count


# Conta as assinaturas
def count_signatures(result_pred, thr=0.5):
    object_count = 0
    elements_conf = result_pred.confidence
    for element in elements_conf:
        if element > thr:
            object_count += 1

    return object_count


def load_model():
    mdl = torch.hub.load('C:/Users/Alysson/PycharmProjects/yolov5', 'custom',
                         path='signature_detector/model_v2_yoloM/best.pt', source='local')
    return mdl


def read_json_gt(json_pth, filename):
    with open(os.path.join(json_pth, filename), 'r', encoding='utf-8') as gt_file:
        data = json.load(gt_file)
        gt_file.close()

    sig_count = data['count_sig']
    sig_list = data['signatures']
    return sig_count, sig_list


def infer_sign(path_img, json_path):
    model = load_model()

    gt_list = []
    pred_list = []

    for (root, dirnames, files) in os.walk(path_img):
        for filename in files:
            current_img = cv.imread(os.path.join(path_img, filename))
            gt_filename = filename.split('.')[0]+'.json'

            sig_qtd, sig_list = read_json_gt(json_path, gt_filename)
            gt_list.append(sig_qtd)

            detection = model(current_img, size=1280)
            detection.save()
            pred_list.append(count_signatures_with_iou(result_pred=detection.pandas().xyxy[0], list_sig=sig_list))
            # pred_list.append(count_signatures(result_pred=detection.pandas().xyxy[0], thr=0.1))

            print('File {} processed.'.format(filename))

    root_mean_squared_error = rmse(predictions=np.array(pred_list), targets=np.array(gt_list))
    mae = mean_absolute_error(y_true=gt_list, y_pred=pred_list)

    json_name = 'model_metrics'+'.json'
    with open(os.path.join('./detection_output', json_name), 'w', encoding='utf-8') as result_file:
        result_file.write(json.dumps({
            'result': {
                'mae': mae,
                'rmse': root_mean_squared_error
            }
        }))
        result_file.close()


if __name__ == '__main__':
    img_path = r'C:\Users\Alysson\Downloads\TCC_dataset_xerox\LoteCertidao\yolo\images\train'
    gt_path = r'C:\Users\Alysson\Downloads\TCC_dataset_xerox\LoteCertidao\json_alss'

    infer_sign(path_img=img_path, json_path=gt_path)


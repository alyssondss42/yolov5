import os
import json
import numpy as np


def read_json_gt(json_pth, filename):
    with open(os.path.join(json_pth, filename), 'r', encoding='utf-8') as gt_file:
        data = json.load(gt_file)
        gt_file.close()

    sig_count = data['count_sig']
    sig_list = data['signatures']
    img_name = data['img']
    return sig_count, sig_list, img_name


def root_mean_squared_error(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


# Conta as assinaturas
def count_signatures(result_pred, thr=0.5):
    object_count = 0
    elements_conf = result_pred.confidence
    for element in elements_conf:
        if element > thr:
            object_count += 1

    return object_count

import os
import cv2 as cv
import json
import torch


def mark_detection(img_name, img, sig_list, detection_pts):
    for idx, element in detection_pts.iterrows():
        det_x_min = element.xmin
        det_y_min = element.ymin
        det_x_max = element.xmax
        det_y_max = element.ymax
        conf = element.confidence

        if conf >= 0.5:

            start_point = (int(det_x_min), int(det_y_min))
            end_point = (int(det_x_max), int(det_y_max))

            img = cv.rectangle(img, start_point, end_point, color=(255, 0, 0), thickness=2)
            cv.putText(img, str(round(conf, 3)), (int(det_x_min), int(det_y_min) - 10), cv.FONT_HERSHEY_SIMPLEX,
                       0.4, color=(255, 0, 0), thickness=2)

    # for sig_gt in sig_list:
    #     x_min = sig_gt['bounding_box']['x']
    #     y_min = sig_gt['bounding_box']['y']
    #     x_max = x_min + sig_gt['bounding_box']['width']
    #     y_max = y_min + sig_gt['bounding_box']['height']
    #
    #     start_point = (int(x_min), int(y_min))
    #     end_point = (int(x_max), int(y_max))
    #
    #     img = cv.rectangle(img, start_point, end_point, color=(0, 255, 0), thickness=2)
    #     cv.putText(img, 'GT', (int(x_min), int(y_min) - 10), cv.FONT_HERSHEY_SIMPLEX,
    #                0.4, color=(0, 255, 0), thickness=2)

    cv.imwrite('./out_img/' + img_name, img)


def mark_gt():
    pass
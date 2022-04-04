"""Human facial landmark detector based on Convulutional Neural Network."""
import os
import cv2
import torch
from faceboxes_pytorch.faceboxes_face_detector import FaceBoxesFaceDetector


class Detector:
    """face detecting model for face recognition"""

    def __init__(self, detector_params):
        """Initialization"""
        self.model_name = str(detector_params['model_name'])
        self.model_path = ''
        if 'model_path' in detector_params.keys():
            self.model_path = str(detector_params['model_path'])

        self.resize_max_y = 2400
        if 'resize_max_y' in detector_params.keys():
            self.resize_max_y = int(detector_params['resize_max_y'])

        self.pad_ratio = 0.0
        if 'pad_ratio' in detector_params.keys():
            self.pad_ratio = float(detector_params['pad_ratio'])

        self.conf_thresh = 0.0
        if 'conf_thresh' in detector_params.keys():
            self.conf_thresh = float(detector_params['conf_thresh'])

        self.bb_inter_ratio_thresh = 0.8
        if 'bb_inter_ratio_thresh' in detector_params.keys():
            self.bb_inter_ratio_thresh = float(detector_params['bb_inter_ratio_thresh'])

        self.detector_params = detector_params
        self.detector = FaceBoxesFaceDetector(use_gpu=(torch.cuda.device_count() > 0))

    def detect(self, image_path):
        face_info = {'success': False, 'conf': 0.0,
                     'img_wid': 0, 'img_hei': 0, 'face_wid': 0, 'face_hei': 0,
                     'bb_l': 0, 'bb_t': 0, 'bb_b': 0, 'bb_r': 0}
        if os.path.exists(image_path):
            image = cv2.imread(image_path)

            return self.detect_image(image)
        else:
            print(f'detect error!! {image_path} not found!')
            # return face_info, face_image
            return face_info

    def detect_image(self, image):
        """
        顔が写っている画像パスを受け取って、検出された顔画像を返す
        """
        success = False

        h, w = image.shape[:2]
        face_info = {'img_hei': h, 'img_wid': w}

        if self.model_name == 'raw':
            success = True
            conf = 1.0
            bb = [0, 0, face_info['img_hei'] - 1, face_info['img_wid'] - 1]
            # face_image = image

        else:
            # preprocess
            # 高速化のためのリサイズ
            resize_scale = 1
            while image.shape[0] > self.resize_max_y:
                image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
                resize_scale *= 2

            # 顔検出成功率高めるためのパディング
            y_bordersize, x_bordersize = [0, 0]
            if self.pad_ratio > 0.0:
                image, y_bordersize, x_bordersize = add_border(image, border_ratio=self.pad_ratio)

            # main detect
            if self.model_name == 'faceboxes':
                # success, bb, conf, face_image = self._detect(image)
                success, bb, conf = self._detect(image, self.conf_thresh)

            # postprocess
            if success:
                # パディングした場合その影響の排除
                if y_bordersize > 0 or x_bordersize > 0:
                    # この補正によってBBが画像の範囲外になることもあるが、ここではあえてそのままにしておく
                    bb = [bb[0] - x_bordersize, bb[1] - y_bordersize, bb[2] - x_bordersize, bb[3] - y_bordersize]

                # リサイズした場合その影響の排除
                if resize_scale > 1:
                    bb = [int(e * resize_scale) for e in bb]

                # bb枠内の元画像の面積比が一定閾値より小さければbb検出失敗とする
                if calc_bb_inter_ratio(h, w, bb) < self.bb_inter_ratio_thresh:
                    success = False

        face_info['success'] = success

        if success:
            face_info['conf'] = float(conf)
            face_info['bb_l'] = bb[0]
            face_info['bb_t'] = bb[1]
            face_info['bb_r'] = bb[2]
            face_info['bb_b'] = bb[3]
            face_info['face_wid'] = face_info['bb_r'] - face_info['bb_l']
            face_info['face_hei'] = face_info['bb_b'] - face_info['bb_t']

        # return face_info, face_image
        return face_info

    def batch_detect_image(self, images, *, faceboxes_target_height=640, faceboxes_target_width=360):

        if self.model_name == 'raw':
            results = []
            add_border_images = images
            for i in images:
                h, w = i.shape[:2]
                scores = [1.0]
                bbs = [0, 0, h - 1, w - 1]
                results.append((scores, bbs))

        else:
            add_border_images = [add_border(image)[0] for image in images]
            results = self.detector.get_batch_faceboxes_with_resize(add_border_images, resize_target_height=faceboxes_target_height, resize_target_width=faceboxes_target_width)

        return add_border_images, results

    def _detect(self, image, conf_thresh):
        confs, rects_float = self.detector.get_faceboxes(image, threshold=conf_thresh)
        if len(rects_float) == 0:
            success = False
            bb = [0, 0, 0, 0]
            conf = 0.0
        else:
            success = True
            rects = []
            for i in range(len(rects_float)):
                rects.append([int(r) for r in rects_float[i]])
            bb = rects[0]
            conf = confs[0]

        return success, bb, conf


def add_border(img, border_ratio=0.2):
    y, x = img.shape[:2]
    y_bordersize = int(y * border_ratio)
    x_bordersize = int(x * border_ratio)
    bordered = cv2.copyMakeBorder(
        img,
        top=y_bordersize,
        bottom=y_bordersize,
        left=x_bordersize,
        right=x_bordersize,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )

    return bordered, y_bordersize, x_bordersize

def calc_bb_inter_ratio(image_hei, image_wid, bb):
    if bb[0] >= image_wid or bb[1] >= image_hei or bb[2] < 0 or bb[3] < 0:
        return 0.0
    bb_area = (bb[2] - bb[0]) * (bb[3] - bb[1])
    inter_hei = min(bb[3], image_hei) - max(bb[1], 0)
    inter_wid = min(bb[2], image_wid) - max(bb[0], 0)
    bb_inter_ratio = (inter_hei * inter_wid) / bb_area

    return bb_inter_ratio

from .detector import Detector
from .cropper import Cropper
import cv2


class Preprocessor:
    """
    detectしてcropした画像を返す
    """

    def __init__(self, detector_params, cropper_params):
        """
        """
        self.detector = Detector(detector_params)
        self.cropper = Cropper(cropper_params)

    def preprocess(self, img):
        face_info = self.detector.detect_image(img)
        if face_info['success']:
            bb = [face_info['bb_l'], face_info['bb_t'], face_info['bb_r'], face_info['bb_b']]
            return face_info['conf'], self.cropper.crop(img, bb)
        else:
            return 0, img

    def batch_preprocess(self, imgs, *, faceboxes_target_height=640, faceboxes_target_width=360, resize_size=112):
        bordered_images, scores_boxes_pairs = self.detector.batch_detect_image(imgs, faceboxes_target_height=faceboxes_target_height, faceboxes_target_width=faceboxes_target_width)

        confidences = []
        preprocessed_images = []
        for image, sbs in zip(bordered_images, scores_boxes_pairs):
            # 顔が検出できなかった場合
            if len(sbs[0]) == 0:
                scores = [0]
                face = image

            else:
                scores, bbs = sbs
                face = self.cropper.bordered_image_crop(image, bbs[0])

            image = cv2.resize(face, (resize_size, resize_size))
            confidences.append(scores[0])
            preprocessed_images.append(image)

        return confidences, preprocessed_images

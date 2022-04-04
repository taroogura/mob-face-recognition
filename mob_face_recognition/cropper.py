from .detector import add_border


class Cropper:
    """face cropping model for face recognition"""

    def __init__(self, cropper_params):
        """Initialization"""
        self.resize_scale = 1.0
        if 'resize_scale' in cropper_params.keys():
            self.resize_scale = float(cropper_params['resize_scale'])

        self.face_y_offset = 0.0
        if 'face_y_offset' in cropper_params.keys():
            self.face_y_offset = float(cropper_params['face_y_offset'])

        self.padding_type = 'none'
        if 'padding_type' in cropper_params.keys():
            self.padding_type = str(cropper_params['padding_type'])

        self.cropper_params = cropper_params

    def crop(self, img, bb):
        bordered, y_bordersize, x_bordersize = add_border(img, border_ratio=1.0)
        h, w = bordered.shape[:2]
        bb = [bb[0] + x_bordersize, bb[1] + y_bordersize, bb[2] + x_bordersize, bb[3] + y_bordersize]
        if self.padding_type == 'none':
            bb[0] = max(0, bb[0])
            bb[1] = max(0, bb[1])
            bb[2] = min(w - 1, bb[2])
            bb[3] = min(h - 1, bb[3])
        elif self.padding_type == 'raw':
            bb = self.__resize_bb(bb, w, h, self.resize_scale, self.face_y_offset)

        return bordered[bb[1]:bb[3], bb[0]:bb[2]]

    def bordered_image_crop(self, img, bb):
        h, w = img.shape[:2]
        if self.padding_type == 'none':
            bb[0] = max(0, bb[0])
            bb[1] = max(0, bb[1])
            bb[2] = min(w - 1, bb[2])
            bb[3] = min(h - 1, bb[3])
        elif self.padding_type == 'raw':
            bb = self.__resize_bb(bb, w, h, self.resize_scale, self.face_y_offset)

        return img[bb[1]:bb[3], bb[0]:bb[2]]

    def __resize_bb(self, rawbox, wid, hei, scale, offset):
        """
        connoisseurからのコピー
        """
        old_size = (rawbox[2] - rawbox[0] + rawbox[3] - rawbox[1]) / 2.0
        center_x = (rawbox[0] + rawbox[2]) / 2.0
        center_y = (rawbox[1] + rawbox[3]) / 2.0 + old_size * offset
        size = int(old_size * scale)
        newbox = []
        newbox.append(max(0, int(center_x - size / 2)))
        newbox.append(max(0, int(center_y - size / 2)))
        newbox.append(min(wid - 1, int(newbox[0] + size)))
        newbox.append(min(hei - 1, int(newbox[1] + size)))
        return newbox

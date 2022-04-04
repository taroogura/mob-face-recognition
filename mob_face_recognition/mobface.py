import os
import cv2
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mob_facex.backbone.backbone_def import BackboneFactory
from .preprocessor import Preprocessor

builtin_models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/mobface')
print('built-in models_dir: ', builtin_models_dir)

def compare(f1, f2):
    return np.sum(f1*f2)

def read_params(parampath):
    if not os.path.exists(parampath):
        print("{} doesn't exist!!".format(parampath))
        return None

    print("parampath: ", parampath)
    with open(parampath, 'r') as stream:
        try:
            params_dict = yaml.safe_load(stream)
            if params_dict is None:
                params_dict = {}
            return params_dict
        except yaml.YAMLError as exc:
            print(exc)

    return None


def get_model_params(model_id, models_dir):
    """
    detect、cropなどのパラメータファイルを探して読み込む
    """
    base_model_id = model_id[:-12]
    base_model_dir = os.path.join(models_dir, base_model_id)

    print('base_model_dir')
    print(base_model_dir)

    model_path = os.path.join(base_model_dir, f"{model_id}.pt")
    model_params = {'model_path': model_path}

    base_model_params = read_params(os.path.join(base_model_dir, f'{base_model_id}.yml'))
    cropper_id = base_model_params['cropper_id']

    cropper_parampath = os.path.join(base_model_dir, f'{cropper_id}.yml')
    cropper_params = read_params(cropper_parampath)
    if cropper_params is None:
        print(f'{cropper_parampath} not found!!')
        return None

    detector_id = cropper_params['detector_id']
    detector_parampath = os.path.join(base_model_dir, f'{detector_id}.yml')
    detector_params = read_params(detector_parampath)
    if detector_params is None:
        print(f'{detector_parampath} not found!!')
        return None

    train_id = base_model_params['train_id']
    train_params = read_params(os.path.join(base_model_dir, f'{train_id}.yml'))

    model_params['backbone_type'] = train_params['backbone_type']
    if 'update_backbone_param' in train_params:
        model_params['update_backbone_param'] = train_params['update_backbone_param']

    model_params['model_id'] = model_id
    model_params['detector_params'] = detector_params
    model_params['cropper_params'] = cropper_params

    return model_params


def load_model(model_path):
    """
    model_path: [models_dir]/LQD_XXXX_YYYY/LQD_XXXX_YYYY_epochZZZZZZ.pt
    指定するmodel_pathの入っているディレクトリには、以下ファイル群が存在することが前提となっている
    - LQD_XXXX_YYYY/
        - LQD_XXXX_YYYY_epochZZZZZZ.pt (model_pathで指定するモデルファイル)
        - LQD_XXXX_YYYY.yml
        - 上記ymlファイル内で指定されている各パラメータのIDごとのymlファイル
            - [detector_id].yml
            - [cropper_id].yml
            - [train_id].yml
    """
    model_id = os.path.splitext(os.path.basename(model_path))[0]
    models_dir = os.path.dirname(os.path.dirname(model_path))
    return Mobface(model_id, models_dir)


class Mobface:
    def __init__(self, model_id='PUBLIC_MOB_epoch000029', models_dir=builtin_models_dir):
        """
        model_id: [base_model_id]_epochXXXXXX
        """
        self.model_id = model_id
        self.use_gpu = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.model_params = get_model_params(model_id, models_dir)
        if self.model_params is None:
            print(f'model_id: {model_id} init failed!!')
            raise Exception
        print('model_id')
        print(self.model_id)
        print('model_params')
        print(self.model_params)

        self.resize_size = 112   # 現状固定

        update_backbone_param = {}
        if 'update_backbone_param' in self.model_params.keys():
            update_backbone_param = self.model_params['update_backbone_param']
        backbone_factory = BackboneFactory(self.model_params['backbone_type'], update_backbone_param)
        self.feat_dim = backbone_factory.backbone_param['feat_dim']

        print('calc on local ')
        self.pp = Preprocessor(self.model_params['detector_params'],
                               self.model_params['cropper_params'])
        self.model = backbone_factory.get_backbone()
        self.model = nn.DataParallel(self.model)
        with open(self.model_params['model_path'], 'rb') as f:
            self.model.load_state_dict(torch.load(f, map_location=self.device))
        if self.use_gpu:
            self.model.cuda()
        self.model.eval()

    def preprocess(self, image):
        mean = 127.5
        std = 128.0

        confidence, face = self.pp.preprocess(image)
        if face is None or confidence == 0:
            # 顔検出が失敗していた場合input_sizeの黒画像を返すように
            return 0, torch.from_numpy(np.zeros((3, self.resize_size, self.resize_size), dtype=np.float32))

        img = cv2.resize(face, (self.resize_size, self.resize_size))
        img = (img.transpose((2, 0, 1)) - mean) / std
        img = torch.from_numpy(img.astype(np.float32))

        return confidence, img

    def extract(self, image):
        """
        imageにcv2.imread されたimageを渡せばそれ使って特徴量抽出する
        """
        feature = np.zeros(self.feat_dim)
        confidence, img = self.preprocess(image)
        if img is None or confidence == 0:
            return 0, np.zeros((self.feat_dim))
        images = torch.unsqueeze(img, 0)
        with torch.no_grad():
            if self.device != 'cpu':
                images = images.to(self.device)
            features = self.model(images)
            features = F.normalize(features)
            if self.device != 'cpu':
                features = features.cpu().numpy()
            else:
                features = features.cpu().numpy()
        feature = features[0]

        return confidence, feature

    def batch_preprocess(self, images):
        mean = 127.5
        std = 128.0

        confidences, faces = self.pp.batch_preprocess(imgs=images, resize_size=self.resize_size)
        rim_height, rim_width, rim_ch = faces[0].shape
        faces = np.array(faces).reshape(len(faces), rim_height, rim_width, rim_ch)

        imgs = torch.from_numpy(faces)
        imgs = imgs.to(self.device)
        imgs = imgs.to(torch.float32)

        imgs = torch.sub(imgs, torch.tensor((mean, mean, mean)).to(self.device))
        imgs = torch.div(imgs, torch.tensor((std, std, std)).to(self.device))
        imgs = imgs.permute(0, 3, 1, 2)

        return confidences, imgs

    def batch_extract(self, images):
        if self.device != torch.device('cuda'):
            return []

        # confidences, imgs = self.batch_preprocess(images)
        confidences = []
        tmp_imgs = []
        for image in images:
            confidence, img = self.preprocess(image)
            confidences.append(confidence)
            tmp_imgs.append(img)
        imgs = torch.stack(tmp_imgs)

        with torch.no_grad():
            features = self.model(imgs)
            features = F.normalize(features)
            features = features.cpu().numpy()

        return confidences, features


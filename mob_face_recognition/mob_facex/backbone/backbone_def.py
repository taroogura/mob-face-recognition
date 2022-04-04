"""
@author: Jun Wang 
@date: 20201019 
@contact: jun21wangustc@gmail.com    
"""

import os, sys
import yaml

backbone_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if backbone_dir not in sys.path:
    sys.path.append(backbone_dir)

from backbone.MobileFaceNets import MobileFaceNet


class BackboneFactory:
    """Factory to produce backbone according the backbone_conf.yaml.
    
    Attributes:
        backbone_type(str): which backbone will produce.
        backbone_param(dict):  parsed params and it's value. 
    """
    def __init__(self, backbone_type, update_backbone_param=None):
        self.backbone_type = backbone_type
        backbone_conf_file = os.path.dirname(os.path.realpath(__file__)) + '_conf.yaml'
        print('backbone_conf_file')
        print(backbone_conf_file)
        with open(backbone_conf_file) as f:
            backbone_conf = yaml.safe_load(f)
            self.backbone_param = backbone_conf[backbone_type]
        if update_backbone_param is not None:
            self.backbone_param.update(update_backbone_param)
        print('backbone param:')
        print(self.backbone_param)

    def get_backbone(self):
        if self.backbone_type == 'MobileFaceNet':
            feat_dim = int(self.backbone_param['feat_dim']) # dimension of the output features, e.g. 512.
            out_h = int(self.backbone_param['out_h']) # height of the feature map before the final features.
            out_w = int(self.backbone_param['out_w']) # width of the feature map before the final features.
            backbone = MobileFaceNet(feat_dim, out_h, out_w)
        else:
            pass
        return backbone

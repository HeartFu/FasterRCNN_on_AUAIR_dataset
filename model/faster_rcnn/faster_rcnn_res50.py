from model.faster_rcnn.faster_rcnn import _fasterRCNN

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import resnet50
from model.relation.relation_module import RelationModule
from model.relation.relation_tools import extract_position_matrix, extract_position_embedding

class FasterRCNNResNet50(_fasterRCNN):
    def __init__(self, classes, num_layers=50, relation_module=False, pretrained=False, class_agnostic=False):
        self.dout_base_model = 1024
        self.pretrained = pretrained,
        self.class_agnostic = class_agnostic
        self.relation_module = relation_module
        _fasterRCNN.__init__(self, classes, class_agnostic, relation_module=relation_module)
        self.fc_dim = 16
        self.emb_ft_dim = 64
        self.relation_output_dim = 2048

    def _init_modules(self):
        resnet = resnet50(pretrained=self.pretrained)
        # print(resnet)
        # Build resnet.
        self.RCNN_base = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
                                       resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3)

        self.RCNN_top = nn.Sequential(resnet.layer4)
        if self.relation_module:
            self.RCNN_relation = RelationModule(self.fc_dim, self.emb_ft_dim, self.relation_output_dim)
        # self.RCNN_fc7 = nn.Linear(2048, 2048)
        self.RCNN_cls_score = nn.Linear(2048, self.n_classes)
        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(2048, 4)
        else:
            self.RCNN_bbox_pred = nn.Linear(2048, 4 * self.n_classes)

        # weights = torch.load('model/resnet50-19c8e357.pth')
        # # Fix blocks
        # for _n, p in self.RCNN_base[0].parameters():
        #     if _n in weights.keys():
        #         p.requires_grad = False
        #         p.copy_(weights[_n])
        for p in self.RCNN_base[0].parameters(): p.requires_grad = False
        for p in self.RCNN_base[1].parameters(): p.requires_grad = False

        # assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
        # if cfg.RESNET.FIXED_BLOCKS >= 3:
        for p in self.RCNN_base[6].parameters():
            p.requires_grad = False
        # if cfg.RESNET.FIXED_BLOCKS >= 2:
        for p in self.RCNN_base[5].parameters(): p.requires_grad = False
        # if cfg.RESNET.FIXED_BLOCKS >= 1:
        for p in self.RCNN_base[4].parameters(): p.requires_grad = False

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad = False

        self.RCNN_base.apply(set_bn_fix)
        self.RCNN_top.apply(set_bn_fix)

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)
        if mode:
            # Set fixed blocks to be in eval mode
            self.RCNN_base.eval()
            self.RCNN_base[5].train()
            self.RCNN_base[6].train()

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.RCNN_base.apply(set_bn_eval)
            self.RCNN_top.apply(set_bn_eval)

    def _head_to_tail(self, pool5):
        fc7 = self.RCNN_top(pool5).mean(3).mean(2)
        # fc7 = self.RCNN_fc7(fc6)
        # output = F.relu(fc7)
        return fc7

    def _head_to_tail_relation(self, pool5, rois):
        sliced_rois = rois[:,1:5]
        if self.training:
            nongt_dim = 2000
        else:
            nongt_dim = 300
        position_matrix = extract_position_matrix(sliced_rois, nongt_dim)
        position_embedding = extract_position_embedding(position_matrix, feat_dim=64)
        fc6 = self.RCNN_top(pool5).mean(3).mean(2)
        fc7 = self.RCNN_fc7(fc6)
        # import pdb
        # pdb.set_trace()
        attention_1 = self.RCNN_relation(fc7, position_embedding, nongt_dim=nongt_dim, fc_dim=self.fc_dim,
                                         feat_dim=self.relation_output_dim, index=1, group=16,
                                         dim=(self.relation_output_dim, self.relation_output_dim, self.relation_output_dim))
        fc_all = fc7 + attention_1
        fc_all_2_relu = F.relu(fc_all)

        return fc_all_2_relu


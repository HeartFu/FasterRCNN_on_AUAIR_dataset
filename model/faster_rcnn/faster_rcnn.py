import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# import torchvision.models as models
from torchvision.ops import RoIPool, RoIAlign, nms
# from model.roi_layers import ROIAlign, ROIPool
import numpy as np
from model.rpn.rpn import _RPN
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from model.utils.utils import _smooth_l1_loss
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes

class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic, pooling_size=7, pooling_mode="align", thresh=0.05, relation_module=False):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        self.thresh = thresh
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.bbox_normalize_targets_precomputed = True,
        self.bbox_normalize_means = [0.0, 0.0, 0.0, 0.0]
        self.bbox_normalize_stds = [0.1, 0.1, 0.2, 0.2]
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes,
                                                         bbox_normalize_targets_precomputed=self.bbox_normalize_targets_precomputed,
                                                         bbox_normalize_means=self.bbox_normalize_means,
                                                         bbox_normalize_stds=self.bbox_normalize_stds)
        self.pooling_mode = pooling_mode

        self.RCNN_roi_pool = RoIPool((pooling_size, pooling_size), 1.0 / 16.0)
        self.RCNN_roi_align = RoIAlign((pooling_size, pooling_size), 1.0 / 16.0, 0)

        self.relation_module = relation_module # whether add relation network

    def forward(self, im_data, gt_boxes, im_info=None, num_boxes=None):
        batch_size = im_data.size(0)
        # get the base im_info, shape is [N, H, W, S]
        # N: Batch size
        # H: image height
        # W: image weight
        # S: scale
        if im_info is None:
            im_info_list = []
            for i in range(0, batch_size):
                im_info_list.append([im_data.size(2), im_data.size(3), 1])  # Default the image doesn't have scale
            im_info = torch.from_numpy(np.array(im_info_list))
            if torch.cuda.is_available():
                im_info = im_info.cuda()
        # get the base number of boxes, shape is [N, 1]
        # N: Batch size
        if num_boxes is None:
            num_boxes_list = []
            for i in range(0, batch_size):
                number_boxex_one = 0
                for boxes in gt_boxes[i]:
                    if boxes.size(4) != 0:
                        # the boxes is real bounding box
                        number_boxex_one += 1
                num_boxes_list.append(number_boxex_one)
            num_boxes = np.asarray(num_boxes_list)
        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)
        # self.visualise_feature(base_feat)
        # import pdb
        # pdb.set_trace()
        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)
        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        # do roi pooling based on predicted rois

        if self.pooling_mode == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        else:
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))
        # feed pooled features to top model
        if self.relation_module:
            # add relation network
            pooled_feat = self._head_to_tail_relation(pooled_feat, rois.view(-1, 5))
        else:
            pooled_feat = self._head_to_tail(pooled_feat)
        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1,
                                            rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:

            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        if self.training:
            return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label

        # eval mode
        prediction = self.prediction(rois, cls_prob, bbox_pred, im_info)
        return prediction

    def prediction(self, rois, cls_prob, bbox_pred, im_info):
        score_all = cls_prob.data
        boxes = rois.data[:, :, 1:5] # shape: [N, C, Box.shape], N: Batch size, C: Number of Boxes
        # test_a = np.tile(boxes.cpu().numpy(), (1, scores.shape[1]))
        # apply bounding-box regression deltas
        box_deltas = bbox_pred.data
        if self.bbox_normalize_targets_precomputed:
            # because we have already done the bbox normalization in train process
            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(self.bbox_normalize_stds).cuda() \
                         + torch.FloatTensor(self.bbox_normalize_means).cuda()
            box_deltas = box_deltas.view(1, -1, 4 * self.n_classes)

        pred_boxes_all = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes_all = clip_boxes(pred_boxes_all, im_info.data, 1)
        prediction = []
        # if it has already clipped, it should recover original
        pred_boxes_all /= im_info.data[0][2].item()
        for i in range(0, pred_boxes_all.shape[0]):
            scores = score_all[i]
            pred_boxes = pred_boxes_all[i]
            # all_boxes = [[[] for _ in range(self.n_classes)]
            #              for _ in range(self.classes)]
            boxes_list = []
            labels_list = []
            scores_list = []
            for j in range(1, self.n_classes):
                inds = torch.nonzero(scores[:, j] > self.thresh).view(-1)
                if inds.numel() > 0:
                    cls_scores = scores[:, j][inds]
                    _, order = torch.sort(cls_scores, 0, True)
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
                    keep = nms(cls_boxes[order, :], cls_scores[order], 0.3)
                    # ############# test ####################
                    # cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                    # cls_dets = cls_dets[order]
                    # cls_dets = cls_dets[keep.view(-1).long()]
                    # ############## end test ####################
                    cls_boxes = cls_boxes[order]
                    cls_boxes = cls_boxes[keep.view(-1).long()]
                    cls_scores = cls_scores[order]
                    cls_scores = cls_scores[keep.view(-1).long()]
                    boxes_list += cls_boxes.cpu().numpy().tolist()
                    scores_list += cls_scores.cpu().numpy().tolist()
                    labels_list += [j for box in cls_boxes]
                    # all_boxes[j][i] = cls_dets.cpu().numpy()

            prediction.append(
                {'boxes': np.asarray(boxes_list), 'labels': np.asarray(labels_list), 'scores': np.asarray(scores_list)})
            # import pdb
            # pdb.set_trace()
        # for i in range(0, len(boxes)):
        #     box = boxes[i]
        #     score = scores[i]
        #     boxes_batch = []
        #     scores_batch = []
        #     labels_batch = []
        #     for j in range(0, len(box)):
        #         # delete boxes which the class is 0 because it is background
        #         box_one = box[j]
        #         score_one = score[j]
        #         max_score, index = torch.max(score_one, 0)
        #         if index == 0:
        #             continue
        #         if max_score >= 0.0:
        #             boxes_batch.append(box_one.cpu().numpy().tolist())
        #             scores_batch.append(max_score.cpu().numpy().tolist())
        #             labels_batch.append(index.cpu().numpy().tolist())
        #     # boxes_list.append(boxes_batch)
        #     # scores_list.append(scores_batch)
        #     # labels_list.append(labels_batch)
        #     prediction.append({'boxes': np.asarray(boxes_batch), 'labels': np.asarray(labels_batch), 'scores': np.asarray(scores_batch)})

        return prediction
        # prediction

    def visualise_feature(self, base_feat):
        img_list = []
        for i in range(0, base_feat.size(1)):
            feature = base_feat[:,i,:,:]
            feature=feature.view(feature.shape[1],feature.shape[2])
            feature = feature.data.cpu().numpy()

            feature = 1.0/(1+np.exp(-1 * feature))
            feature = np.round(feature * 255)

            cv2.imwrite('feature_map/feature{}.jpg'.format(i), feature)
        #     img_list.append(feature)
        # imgs = np.hstack(img_list)
        # cv2.imwrite('feature.jpg', imgs)

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, False)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, False)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, False)
        # normal_init(self.RCNN_fc7, 0, 0.01, False)
        if self.relation_module:
            normal_init(self.RCNN_relation.relation_fc, 0, 0.01, False)
            normal_init(self.RCNN_relation.relation_fc2, 0, 0.01, False)
            normal_init(self.RCNN_relation.relation_fc3, 0, 0.01, False)
            normal_init(self.RCNN_relation.relation_conv1, 0, 0.01, False)
        normal_init(self.RCNN_cls_score, 0, 0.01, False)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, False)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

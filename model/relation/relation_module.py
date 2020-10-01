import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np

class RelationModule(nn.Module):

    def __init__(self, fc_dim=16, emb_ft_dim=64, output_dim=2048):
        super(RelationModule, self).__init__()
        self.relation_fc = nn.Linear(emb_ft_dim, fc_dim)
        self.relation_fc2 = nn.Linear(output_dim, output_dim)
        self.relation_fc3 = nn.Linear(output_dim, output_dim)
        self.relation_conv1 = nn.Conv2d(fc_dim * output_dim, output_dim, (1, 1), groups=fc_dim)

    def forward(self, roi_feat, position_embedding,
                nongt_dim, fc_dim, feat_dim,
                dim=(1024,1024,1024), group=16,
                index=1):
        dim_group = (dim[0] / group, dim[1] / group, dim[2] / group)
        # nongt_roi_feat = torch.chunk(roi_feat, nongt_dim, dim=0)
        nongt_roi_feat = roi_feat[0:nongt_dim, :]
        # position_embedding = torch.stack(position_embedding)
        position_embedding_reshape = position_embedding.view(position_embedding.size(0) * position_embedding.size(1), position_embedding.size(2))
        # position_embedding_reshape = torch.reshape(position_embedding, shape=(-3, -2))

        position_feat_1 = self.relation_fc(position_embedding_reshape.float())
        position_feat_1_relu = F.relu(position_feat_1)

        aff_weight = torch.reshape(position_feat_1_relu, shape=(-1, position_embedding.size(1), fc_dim))
        aff_weight = aff_weight.permute(0,2,1)

        # multi head
        assert dim[0] == dim[1], 'Matrix multiply requires same dimensions!'
        q_data = self.relation_fc2(roi_feat)
        q_data_batch = torch.reshape(q_data, shape=(-1, group, int(dim_group[0])))
        q_data_batch = q_data_batch.permute(1,0,2)

        # k_data = wK of formula 4, nongt_roi_feat = fA of formula 4
        k_data = self.relation_fc3(nongt_roi_feat)
        k_data_batch = torch.reshape(k_data, shape=(-1, group, int(dim_group[0])))
        k_data_batch = k_data_batch.permute(1,0,2)

        v_data = nongt_roi_feat

        # aff_scale == wA of formula 4
        k_data_batch_t = k_data_batch.permute(0,2,1)
        aff = torch.bmm(q_data_batch, k_data_batch_t)
        aff_scale = (1.0 / math.sqrt(float(dim_group[1]))) * aff
        aff_scale = aff_scale.permute(1,0,2)

        assert fc_dim == group
        min_value = torch.from_numpy(np.asarray([1e-3])).float().cuda()
        # bak_aff_weight = torch.full(aff_weight.shape, 1e-3)
        weighted_aff = torch.log(torch.max(aff_weight, min_value)) + aff_scale
        aff_softmax = F.softmax(weighted_aff, dim=2)

        aff_softmax_reshape = aff_softmax.view(aff_softmax.size(0) * aff_softmax.size(1), aff_softmax.size(2))

        output_t = torch.mm(aff_softmax_reshape, v_data)
        output_t = torch.reshape(output_t, shape=(-1, fc_dim * feat_dim, 1, 1))

        linear_out = self.relation_conv1(output_t)
        output = torch.squeeze(linear_out)
        # output = torch.reshape(linear_out, shape=(0,0))

        return output









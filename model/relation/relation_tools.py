import torch
import numpy as np

# formula 1
def extract_position_matrix(bboxes, nongt_dim=2000):
    # Extract position matrix, Fg
    # bboxes shape: [batch_size, num_bbox, 4]
        # shape: [num_bbox, 1]
    xmin, ymin, xmax, ymax = bboxes[:,0:1], bboxes[:,1:2], bboxes[:,2:3], bboxes[:,3:4]
    bbox_width = xmax - xmin + 1
    bbox_height = ymax - ymin + 1
    center_x = 0.5 * (xmin + xmax)
    center_y = 0.5 * (ymin + ymax)

    min_value = torch.from_numpy(np.asarray([1e-3])).float().cuda()

    delta_x = torch.abs(torch.sub(center_x, center_x.t()))
    delta_x = torch.div(delta_x, bbox_width)
    delta_x = torch.max(delta_x, min_value)
    delta_x = torch.log(delta_x)

    delta_y = torch.abs(torch.sub(center_y,center_y.t()))
    delta_y = torch.div(delta_y, bbox_height)
    delta_y = torch.log(torch.max(delta_y, min_value))

    delta_width = torch.div(bbox_width, bbox_width.t())
    delta_width = torch.log(delta_width)

    delta_height = torch.div(bbox_height, bbox_height.t())
    delta_height = torch.log(delta_height)

    concat_list = [delta_x, delta_y, delta_width, delta_height]

    # TODO: need to check
    for idx, sym in enumerate(concat_list):
        sym = sym[:, 0:nongt_dim]
        concat_list[idx] = sym.unsqueeze(-1)

    # It is ok, shape: 256,256,4
    position_matrix = torch.cat(concat_list, dim=2)

    return position_matrix

# fomula 5
def extract_position_embedding(position_mat, feat_dim, wave_length=1000):
    feat_range = torch.range(0, feat_dim / 8 - 1)
    # TODO: should be more efficial
    dim_mat = torch.from_numpy(np.asarray([[[[1.,2.3713737,5.623413,13.335215,31.622776,74.98942,177.82794,421.6965]]]])).cuda()
    dim_mat = torch.reshape(dim_mat, shape=(1, 1, 1, -1))

    position_mat = (position_mat * 100.0).unsqueeze(-1)
    div_mat = torch.div(position_mat, dim_mat)

    sin_mat = torch.sin(div_mat)
    cos_mat = torch.cos(div_mat)

    embedding = torch.cat([sin_mat, cos_mat], dim=3)
    embedding = torch.reshape(embedding, shape=(embedding.size(0), embedding.size(1), feat_dim))
    # embedding_list.append(embedding)

    return embedding


import torch
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

from dataset.auairData import AuairData
from model.faster_rcnn.faster_rcnn_res50 import FasterRCNNResNet50
# from model.faster_rcnn.faster_rcnn_vgg16 import FasterRCNNVGG16
from utils.Logger import Logger
from utils.tools import split_data, collate_fn

device = torch.device('cpu')
devices = []
if torch.cuda.is_available():
   device = torch.device('cuda:0')

print(device)

class Train():
    def __init__(self, **kwargs):
        # self.img_dir = kwargs['img_dir']
        # self.label_dir = kwargs['label_dir']
        self.original_dir = kwargs['original_dir']
        self.train_dir = kwargs['train_dir']
        self.test_dir = kwargs['test_dir']
        self.log = kwargs['log']
        self.model = kwargs['model']  #
        self.weight_save_path = kwargs['weight_save_path']
        self.img_width = kwargs['img_width']
        self.img_height = kwargs['img_height']
        self.classes = kwargs['classes']
        self.pre_train_weight_dir = kwargs['pre_train_weight_dir']
        self.epoch_num = kwargs['epoch_num']
        self.loss_list_iter = []
        self.loss_list_epoch = []
        self.lr_list_epoch = []
        self.debug = kwargs['debug']
        self.proportion = kwargs['proportion']
        self.batch_size = kwargs['batch_size']
        self.learning_rate = kwargs['learning_rate']
        self.optimizer_mode = kwargs['optimizer_mode']
        # self.pre_train_self = kwargs['pre_train_self']
        self.weight_decay = kwargs['weight_decay']
        self.momentum = kwargs['momentum']

        ##### for debug #######
        self.log_test = Logger('test/test_logger_1920.log', level='debug')

    def train(self):
        # data processing
        self.log.logger.info('---------Start data processing----------')
        split_data(self.original_dir, self.train_dir, self.test_dir, self.log, self.proportion)

        auair_train_args = {
            'img_dir': self.train_dir + 'images/',
            'label_dir': self.train_dir + 'annotations.json',
            'img_width': self.img_width,
            'img_height': self.img_height,
            'auair_classes': self.classes,
            'debug': self.debug
        }

        train_data = AuairData(**auair_train_args)

        dataloader_args = {'batch_size': self.batch_size,
                           'shuffle': True, 'num_workers': 4,
                           'collate_fn': collate_fn}
        train_loader = torch.utils.data.DataLoader(train_data, **dataloader_args)

        self.log.logger.info('---------------end data processing----------------')
        self.log.logger.info('---------------build model------------------------')
        if self.pre_train_weight_dir is not None:
            fasterRCNN = FasterRCNNResNet50(self.classes, pretrained=False, class_agnostic=False)
        else:
            fasterRCNN = FasterRCNNResNet50(self.classes, pretrained=True, class_agnostic=False)

        fasterRCNN.create_architecture()
        self.log.logger.info(fasterRCNN)
        # load the weight
        if self.pre_train_weight_dir is not None:
            fasterRCNN.load_state_dict(torch.load(self.pre_train_weight_dir))

        if device == torch.device('cuda:0'):
            fasterRCNN.cuda()
        params = []
        for key, value in dict(fasterRCNN.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': self.learning_rate * (True + 1), 'weight_decay': False and 0.004 or 0}]
                else:
                    params += [{'params': [value], 'lr': self.learning_rate, 'weight_decay': self.weight_decay}]
        lr = self.learning_rate
        if self.optimizer_mode == "adam":
            lr = self.learning_rate * 0.1
            optimizer = torch.optim.Adam(params)
        else:
            optimizer = torch.optim.SGD(params, momentum=self.momentum)
        self.log.logger.info('---------------end model--------------------------')
        self.log.logger.info('---------------start train------------------------')
        iter_all = len(train_loader)
        for epoch in range(0, self.epoch_num):
            fasterRCNN.train()
            loss_temp = 0
            curr_iter = 0
            iteration_loss_one_epoch = []
            with tqdm(total=iter_all) as pdbar:
                for i, batch in enumerate(train_loader):
                    curr_iter += 1
                    fasterRCNN.zero_grad()
                    idx, images, targets = batch
                    # check whether empty object
                    image_list = []
                    target_list = []
                    for i in range(0, len(targets)):
                        # delete some empty data
                        if bool(targets[i]['labels'].numel()) and bool(targets[i]['boxes'].numel()):
                            target_list.append(targets[i])
                            image_list.append(images[i])
                        else:
                            continue
                    target_tuple = tuple(target_list)
                    image_tuple = tuple(image_list)
                    if len(target_tuple) > 0:
                        ####################### Combine the datas to fit the batch training #############################
                        # check the max number of bounding box
                        max_num_bbox = 0
                        for target in target_list:
                            if len(target['boxes']) > max_num_bbox:
                                max_num_bbox = len(target['boxes'])
                        # combine the data to fit the batch training
                        image_info_list = []
                        num_boxes_list = []
                        gt_boxes_list = []
                        # image input data
                        if device == torch.device('cuda:0'):
                            image_tensor = torch.stack(list(image.to(device) for image in image_tuple))
                        else:
                            image_tensor = torch.from_numpy(image_tuple)
                        # Target input data combination for fit batch training, the input is [N, 5]
                        # N: batch size
                        # 5: [xmin, ymin, xmax, ymax, label]
                        for i in range(0, image_tensor.size(0)):
                            image_info_list.append([image_tensor.size(2), image_tensor.size(3), image_tensor.size(3) / self.img_width])  # the final one is scale rate
                            gt_box_list = target_tuple[i]['boxes']
                            gt_label_list = target_tuple[i]['labels']
                            num_boxes_list.append(len(gt_box_list))
                            gt_boxes_one = []
                            for i in range(0, len(gt_box_list)):
                                gt_box = gt_box_list[i]
                                gt_label = gt_label_list[i]
                                gt_boxes_one.append(
                                    torch.stack([gt_box[0], gt_box[1], gt_box[2], gt_box[3], gt_label]).float())
                            for i in range(0, max_num_bbox - len(gt_box_list)):
                                # for keep the save dimentions, so we add some additional bounding boxes for background which class is 0.
                                gt_boxes_one.append(torch.from_numpy(np.asarray([0, 0, 0, 0, 0])).float())
                            gt_boxes_list.append(torch.stack(gt_boxes_one))
                        im_info = torch.from_numpy(np.array(image_info_list))
                        gt_boxes = torch.stack(gt_boxes_list).to(device)
                        num_boxes = np.asarray(num_boxes_list)
                        if device == torch.device('cuda:0'):
                            im_info = im_info.to(device)
                            gt_boxes = gt_boxes.to(device)
                        ####################### End Combine the datas to fit the batch training ##################################

                        #################################### train one iteration model ###########################################
                        rois, cls_prob, bbox_pred, \
                        rpn_loss_cls, rpn_loss_box, \
                        RCNN_loss_cls, RCNN_loss_bbox, \
                        rois_label = fasterRCNN(image_tensor, gt_boxes, im_info, num_boxes)

                        loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                               + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
                        loss_temp += loss.item()

                        # backward
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        desc = f'Epoch: {epoch}/{self.epoch_num}, iter: {curr_iter}/{iter_all}, loss: {loss.item():.4f}'
                        pdbar.set_description(desc)
                        pdbar.update(1)
                        # if curr_iter % 10 == 0:
                        #     self.log.logger.info(
                        #         'epoch: {}, iteration Id: {}/{}, loss: {}'.format(epoch, curr_iter, iter_all, loss.item()))
                        iteration_loss_one_epoch.append(loss.item())
                        #################################### End train one iteration model ###########################################
            self.loss_list_epoch.append(loss_temp / curr_iter)
            self.log.logger.info('epoch: {}, loss: {}, learning rate: {}'.format(epoch, loss_temp / curr_iter, lr))
            self.log.logger.info('epoch loss list : {}'.format(self.loss_list_epoch))
            # adjust learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.99 * param_group['lr']
            lr *= 0.99
            # save the all loss
            self.log.logger.info('iteration loss list : {}'.format(iteration_loss_one_epoch))
            self.loss_list_iter.append(iteration_loss_one_epoch)
            if epoch % 10 == 0:
                # Save the weight of model every 10 epoch
                torch.save(fasterRCNN.state_dict(), self.weight_save_path.replace('.pth', '_epoch{}.pth'.format(epoch)))
        self.log.logger.info('---------------End train------------------------')
        # save weight from model
        torch.save(fasterRCNN.state_dict(), self.weight_save_path)

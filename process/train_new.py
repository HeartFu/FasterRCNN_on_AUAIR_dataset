import torchvision
import torch
from torch import nn
from tqdm import tqdm
import os

from utils.Logger import Logger
from utils.auairtools.auair import AUAIR
from dataset.auairData import AuairData
from torch.utils import data
from utils.tools import split_data, collate_fn
from torch.optim.lr_scheduler import StepLR

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
        self.pre_train_self = kwargs['pre_train_self']

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
        train_loader = data.DataLoader(train_data, **dataloader_args)
        # dataloader_args_vali = {'batch_size': 1, 'shuffle': False}
        # validation_loader = data.DataLoader(valid_data, **dataloader_args_vali)

        self.log.logger.info('---------------end data processing----------------')

        ################ build the model ###################################
        frcnn_args = {'rpn_nms_thresh': 0.8, 'num_classes': len(self.classes)}
        frcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, **frcnn_args)

        if device == torch.device('cuda:0'):
            frcnn_model = frcnn_model.to(device)
        self.log.logger.info('model structure:{}'.format(frcnn_model))

        # load the weight form COCO
        weights = torch.load(self.pre_train_weight_dir)
        if self.pre_train_self == True:
            for _n, par in frcnn_model.named_parameters():
                if _n in weights.keys():
                    par.copy_(weights[_n])
            #
            # for _n, par in frcnn_model.named_parameters():
            #     if _n in weights.keys():
            #         # par.copy_(weights[_n])
            #         if _n.startswith("roi_heads.box_predictor."):
            #             # print(_n)
            #             par.requires_grad = True
            #             # par.copy_(weights[_n])
            #         else:
            #             par.requires_grad = False
            #             # print(_n)
            #             # par.copy_(weights[_n])
            #             # par.requires_grad = True
        else:
            for _n, par in frcnn_model.named_parameters():
                if _n in weights.keys():
                    if _n.startswith("roi_heads.box_predictor."):
                        # print(_n)
                        par.requires_grad = True
                        # par.copy_(weights[_n])
                    else:
                        par.requires_grad = False
                        # print(_n)
                        par.copy_(weights[_n])
                        # par.requires_grad = True
        frcnn_model.train()
        ################## end build the model #################################
        ################## train process #######################################
        self.log.logger.info('----------- start train ------------')
        optimizer_pars = {'lr': self.learning_rate, 'weight_decay': 1e-3, 'momentum': 0.9}
        if self.optimizer_mode == 'ADAM':
            optimizer = torch.optim.Adam(list((frcnn_model.parameters())), **optimizer_pars)
        else:
            optimizer = torch.optim.SGD(list((frcnn_model.parameters())), **optimizer_pars)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.99)
        for epoch in range(1, self.epoch_num + 1):
            self.log.logger.info('epoch number: {}'.format(epoch))
            loss_epoch = self.train_one_epoch(train_loader, frcnn_model, optimizer, epoch)
            self.loss_list_epoch.append(loss_epoch.item())
            self.log.logger.info('epoch: {}, loss: {}, learning rate: {}'.format(epoch, loss_epoch, scheduler.get_lr()))
            self.lr_list_epoch.append(scheduler.get_lr())
            scheduler.step()
            self.log.logger.info('epoch loss list : {}'.format(self.loss_list_epoch))
            self.log.logger.info('epoch learning rate list : {}'.format(self.lr_list_epoch))
            # save weight from model
        # save weight from model
        torch.save(frcnn_model.state_dict(), self.weight_save_path)
        ################## end train process ###################################
        # save the all loss
        self.log.logger.info('iteration loss list : {}'.format(self.loss_list_iter))
        # self.log.logger.info('epoch loss list : {}'.format(self.loss_list_epoch))

    def train_one_epoch(self, train_loader, network, optimizer, epoch):
        curr_iter = 0
        loss_one_epoch = 0
        iter_all = len(train_loader)
        # if torch.cuda.device_count() > 1:
        #     network = nn.DataParallel(network, device_ids=devices)
        #     network.to(torch.device("cuda:{}".format(devices[0])))
        # else:
        #     if device == torch.device('cuda'):
        #         network = network.to(device)
        for i, batch in enumerate(train_loader):
            curr_iter += 1
            optimizer.zero_grad()
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

                if device == torch.device('cuda:0'):
                    image_list = list(image.to(device) for image in image_tuple)
                    target_list = [{k: v.to(device) for k, v in t.items()} for t in target_tuple]

                loss = network(image_list, target_list)
                if self.debug:
                    self.log.logger.info('all loss: {}'.format(loss))
                total_loss = 0
                for k in loss.keys():
                    # just for debug
                    if (self.debug and torch.isnan(loss[k])):
                        self.log.logger.error('----------------NAN is happen, print some input--------------------------')
                        self.log.logger.error('id : {}'.format(idx))
                        self.log.logger.error('loss : {}'.format(loss))
                        self.log.logger.error('images: {}'.format(images))
                        self.log.logger.error('target: {}'.format(target_tuple))
                        self.log.logger.error('----------------NAN is happen, END --------------------------------------')
                    total_loss += loss[k].mean()
                if curr_iter % 10 == 0:
                    self.log.logger.info('epoch: {}, iteration Id: {}/{}, loss: {}'.format(epoch, curr_iter, iter_all, total_loss))
                self.loss_list_iter.append(total_loss.item())
                # print(self.loss_list_iter)
                loss_one_epoch += total_loss
                total_loss.backward()
                optimizer.step()
                # if self.debug:
                #     if total_loss > 10:
                #         self.test_log(idx, targets['labels'], targets['boxes'], total_loss)
                # # test eval
                # network.eval()
                # output = network(images)
                # self.log.logger.info('output:{}'.format(output))
                # self.log.logger.info('gt: {}'.format(targets))
                # network.train()
            else:
                curr_iter -= 1
        return loss_one_epoch / curr_iter

    def test_log(self, idx, labels, bboxes, total_loss):
        # 用于测试保存文件
        # 重新生成log文件和AUAIRtools
        # log_test = Logger('test/test_logger.log', level='debug')
        auairdataset = AUAIR(annotation_file=self.train_dir + 'annotations.json', data_folder=self.train_dir + 'images')
        self.log_test.logger.info('test_idx: {}'.format(idx))
        img, ann = auairdataset.get_data_by_index(idx)
        self.log_test.logger.info('ann: {}'.format(ann))
        self.log_test.logger.info('labels: {}, length: {}'.format(labels, len(labels)))
        self.log_test.logger.info('bboxes: {}, length: {}'.format(bboxes, len(bboxes)))
        self.log_test.logger.info('loss: {}'.format(total_loss))



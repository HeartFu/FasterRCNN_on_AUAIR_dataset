import os
import numpy as np
import torch
import cv2
import argparse
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from torch.utils import data as data
from torchvision import transforms as transforms
from utils.auairtools.auair import AUAIR


class AuairData(data.Dataset):

    def __init__(self, debug=False, **arg):
        self.auair_classes = arg['auair_classes']
        self.img_dir = arg['img_dir']
        self.label_dir = arg['label_dir']
        self.imgs = os.listdir(self.img_dir)
        # self.img_width = arg['img_width']
        # self.img_height = arg['img_height']
        self.auairdataset = AUAIR(annotation_file=self.label_dir, data_folder=self.img_dir)
        self.debug = debug
        self.categories = self.auairdataset.get_categories()

    def load_img(self, idx):
        img, ann = self.auairdataset.get_data_by_index(idx)
        # self.log.logger.info('idx: {}, img_name: {}'.format(idx, ann['image_name']))
        img_trans = self.transform_img(img)
        bboxes = []
        classes = []
        for bbox in ann['bbox']:
            if bbox['width'] == 0 or bbox['height'] == 0:
                # delete some wrong data
                continue
            x = bbox['left']
            y = bbox['top']
            w = bbox['width']
            h = bbox['height']
            xmin = x
            ymin = y
            xmax = x + w
            ymax = y + h
            # connect the class to input classes
            label = self.auair_classes.index(self.categories[bbox['class']])
            # label = bbox['class'] + 1 # for background
            obj = [xmin, ymin, xmax, ymax]
            bboxes.append(obj)
            classes.append(label)

        label = {}
        label['labels'] = torch.tensor(classes, dtype=torch.int64)
        label['boxes'] = torch.tensor(bboxes)
        if self.debug:
            img_bak = img_trans.numpy().transpose((1, 2, 0))
            self.visualize_bbox(img_bak, bboxes, classes)
        return img_trans, label

    def visualize_bbox(self, img, bboxes, classes):
        # print(img.shape)
        # img_bak = img.numpy().transpose((1, 2, 0))
        # # img_bak = img
        # print(img_bak.shape)
        plt.imshow(img)
        ax = plt.gca()
        for i in range(0, len(bboxes)):
        # for i, bbox in label['boxes']:
            bbox = bboxes[i]
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[2]
            ymax = bbox[3]
            # plt.annotate(self.auair_classes[classes[i]], xy=(xmin, ymin), xytext=(xmin, ymin), textcoords='offset points')
            rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        plt.show()
    #
    # # just for debug-------------visualize the image and bounding box
    # def visualize_bbox_bak(self, img, bboxes, classes):
    #     img_bak = img.transpose((1, 2, 0))
    #     for i in range(0, len(bboxes)):
    #     # for i, bbox in label['boxes']:
    #         bbox = bboxes[i]
    #         xmin = bbox[0]
    #         ymin = bbox[1]
    #         xmax = bbox[2]
    #         ymax = bbox[3]
    #         label = self.auair_classes[classes[i]]
    #         cv2.putText(img_bak, label, (xmin, ymin + 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), thickness=2,
    #                     lineType=cv2.LINE_AA)
    #         cv2.rectangle(img_bak, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)
    #
    #     cv2.imshow("Name: 123", img_bak)
    #     cv2.waitKey()
    #     cv2.destroyAllWindows()

    # this method normalizes the image and converts it to Pytorch tensor
    # Here we use pytorch transforms functionality, and Compose them together,
    # Convert into Pytorch Tensor and transform back to large values by multiplying by 255
    def transform_img(self, img):
        h, w, c = img.shape
        # self.log.logger.info('img size is width: {}, height: {}, c: {}'.format(w, h, c))
        # img_size = tuple((self.img_width, self.img_height))
        # these mean values are for BGR!!
        t_ = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize(img_size),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.407, 0.457, 0.485],
            #                     std=[1,1,1])])
        ])
        # need this for the input in the model

        img = 1 * t_(img)
        # returns image tensor (CxHxW)
        return img

    def getImageInformation(self, idx):
        img, ann = self.auairdataset.get_data_by_index(idx)
        return ann

    def __getitem__(self, idx):
        # self.log.logger.info('AuairData getitem idx = {}'.format(idx))
        X, y = self.load_img(idx)
        return idx, X, y

    def __len__(self):
        return len(self.imgs)
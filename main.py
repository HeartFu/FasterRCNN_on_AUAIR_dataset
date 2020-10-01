# from  import Logger
import sys

from utils.Logger import Logger
from process.train_new import Train

if __name__ == '__main__':
    # example for logger
    ########################### baseline ###########################################################
    # log = Logger('log/train_log_lrStep.log',level='info')
    # log.logger.info('Start our coursework!')
    # # Logger('error.log', level='error').logger.error('error')
    # train_arg = {
    #     'original_dir': '/home/fanfu/newdisk/dataset/705/auair/',
    #     'train_dir': '/home/fanfu/newdisk/dataset/705/auair/train/',
    #     'test_dir': '/home/fanfu/newdisk/dataset/705/auair/test/',
    #     # 'img_dir': '/home/fanfu/newdisk/dataset/705/auair/images',
    #     # 'label_dir': '/home/fanfu/newdisk/dataset/705/auair/annotations.json',
    #     # 'img_dir': '/home/fanfu/newdisk/dataset/705/auair/test_auair_one (copy)/images',
    #     # 'label_dir': '/home/fanfu/newdisk/dataset/705/auair/test_auair_one (copy)/annotations.json',
    #     # 'img_dir': '/home/fanfu/newdisk/dataset/705/auair/test_auair_one/images',
    #     # 'label_dir': '/home/fanfu/newdisk/dataset/705/auair/test_auair_one/annotations.json',
    #     # 'img_dir': '/home/fanfu/newdisk/dataset/705/auair/auairdataset-master/examples/auair_subset/images',
    #     # 'label_dir': '/home/fanfu/newdisk/dataset/705/auair/auairdataset-master/examples/auair_subset/annotations.json',
    #     'log': log,
    #     'model': 'faster-rcnn',
    #     # 'weight_save_path': 'model/first_model_weight_1.pth',
    #     'weight_save_path': 'model/model_weight_epoch50_lrStep_test.pth',
    #     'img_width': 1920,
    #     'img_height': 1080,
    #     'classes': ["Human","Car","Truck","Van","Motorbike","Bicycle","Bus","Trailer"],
    #     'pre_train_weight_dir': 'model/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth',
    #     # 'pre_train_weight_dir': 'model/first_model_weight_1.pth',
    #     'pre_train_self': False,
    #     'epoch_num': 10,
    #     'debug': False,
    #     'proportion': 0.8,
    #     'batch_size': 8,
    #     'learning_rate': 1e-3,
    #     'optimizer_mode': 'SGD'
    # }
    # # add Background to classes is for testing the human detection
    # train = Train(**train_arg)
    # train.train()
    ########################### baseline ###########################################################
    ########################### model 2  ###########################################################
    log = Logger('log/train_log_lrStep_model.log', level='info')
    log.logger.info('Start our coursework!')
    train_arg = {
        'original_dir': '/home/fanfu/newdisk/dataset/705/auair/',
        'train_dir': '/home/fanfu/newdisk/dataset/705/auair/train/',
        'test_dir': '/home/fanfu/newdisk/dataset/705/auair/test/',
        # 'img_dir': '/home/fanfu/newdisk/dataset/705/auair/images',
        # 'label_dir': '/home/fanfu/newdisk/dataset/705/auair/annotations.json',
        # 'img_dir': '/home/fanfu/newdisk/dataset/705/auair/test_auair_one (copy)/images',
        # 'label_dir': '/home/fanfu/newdisk/dataset/705/auair/test_auair_one (copy)/annotations.json',
        # 'img_dir': '/home/fanfu/newdisk/dataset/705/auair/test_auair_one/images',
        # 'label_dir': '/home/fanfu/newdisk/dataset/705/auair/test_auair_one/annotations.json',
        # 'img_dir': '/home/fanfu/newdisk/dataset/705/auair/auairdataset-master/examples/auair_subset/images',
        # 'label_dir': '/home/fanfu/newdisk/dataset/705/auair/auairdataset-master/examples/auair_subset/annotations.json',
        'log': log,
        'model': 'faster-rcnn',
        # 'weight_save_path': 'model/first_model_weight_1.pth',
        'weight_save_path': 'model/model_weight_epoch50_addBG.pth',
        'img_width': 1920,
        'img_height': 1080,
        'classes': ["Background", "Human", "Car", "Truck", "Van", "Motorbike", "Bicycle", "Bus", "Trailer"],
        'pre_train_weight_dir': 'model/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth',
        # 'pre_train_weight_dir': 'model/first_model_weight_1.pth',
        'pre_train_self': False,
        'epoch_num': 50,
        'debug': False,
        'proportion': 0.8,
        'batch_size': 8,
        'learning_rate': 1e-3,
        'optimizer_mode': 'SGD'
    }
    # add Background to classes is for testing the human detection
    train = Train(**train_arg)
    train.train()
    ########################### model 2  ###########################################################


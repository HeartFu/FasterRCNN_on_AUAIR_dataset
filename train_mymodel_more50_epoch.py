
from utils.Logger import Logger
from process.train_cus_model import Train

if __name__ == '__main__':
    log = Logger('log/train_log_mymodel_res50_more30epoch.log', level='info')
    log.logger.info('Start our coursework!')
    train_arg = {
        'original_dir': '/home/fanfu/newdisk/dataset/705/auair/',
        # 'train_dir': '/home/fanfu/newdisk/dataset/705/auair/train/',
        # 'test_dir': '/home/fanfu/newdisk/dataset/705/auair/test/',
        'train_dir': '/homes/tx301/fufan/auair/train/',
        'test_dir': '/homes/tx301/fufan/auair/test/',
        # 'train_dir': '/home/fanfu/newdisk/dataset/705/auair/test_auair_one (copy)/',
        # 'test_dir': '/home/fanfu/newdisk/dataset/705/auair/test_auair_one (copy)/',
        # 'train_dir': '/home/fanfu/newdisk/dataset/705/auair/auairdataset-master/examples/auair_subset/',
        # 'test_dir': '/home/fanfu/newdisk/dataset/705/auair/auairdataset-master/examples/auair_subset/',
        'log': log,
        'model': 'resnet50',
        # 'weight_save_path': 'model/first_model_weight_1.pth',
        'weight_save_path': 'model/weight/cus_model_more_30_epoch/fasterrcnn_resnet50_more30epoch_mymodel.pth',
        'img_width': 1920,
        'img_height': 1080,
        'classes': ["Background", "Human", "Car", "Truck", "Van", "Motorbike", "Bicycle", "Bus", "Trailer"],
        'pre_train_weight_dir': None,
        # 'pre_train_weight_dir': 'model/weight/fasterrcnn_resnet50_mymodel.pth',
        'pre_train_self': False,
        'epoch_num': 50,
        'debug': False,
        'proportion': 0.8,
        'batch_size': 6,
        'learning_rate': 1.3e-3,
        'optimizer_mode': 'SGD',
        'weight_decay': 1e-3,
        'momentum': 0.9,
    }
    train = Train(**train_arg)
    train.train()
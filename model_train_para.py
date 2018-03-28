import pandas as pd
import os


class model_para:
    def __init__(self):
        self.input_shape = (224, 224, 3)
        self.train_batch_size = 16
        self.test_batch_size = 16
        self.val_batch_size = 16

        self.steps_per_epoch = 512
        self.epoch = 64
        self.label_types = 'categorical'  # 'categorical','binary'
        self.model_name = ('VGG16', 'IncetionResNetV2', 'InceptionV3', 'MobileNet')
        # 目前可以调用这几个模型，后面可以继续添加
        temp = str(self.input_shape[0]) + '_' + str(self.input_shape[1])
        # 数据文件后缀加入图像大小，这样就不用反复训练了
        self.dirs = ['train' + temp, 'validation' + temp, 'test' + temp]
        self.origin_dir = os.path.split(__file__)[0] \
                          + '/data/warmup/Images/skirt_length_labels/'
        self.annotation_path = os.path.split(__file__)[0] \
                               + '/data/warmup/Annotations/skirt_length_labels.csv'
        self.model_save_path = 'Fashion_AI.h5'  # 默认模型存储路径，可以在Model文件里面修改
        df = pd.read_csv(self.annotation_path, header=None)
        df.columns = ('picture', 'tpyes', 'labels')
        # self.labels = '/' + self.df['labels'].unique()
        self.labels = ['/nnynnn', '/nnnnyn', '/nynnnn', '/nnnnny', '/nnnynn']
        self.df = df[df['labels'].isin(self.labels)]  # 这样可以有效筛选样本
        self.files = [x + y for x in self.dirs for y in self.labels]
        self.data_split_ratio = [0.7, 0.15, 0.15]  # 训练集，验证集，测试集

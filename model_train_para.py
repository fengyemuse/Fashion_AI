import pandas as pd
import os


class model_para:
    def __init__(self):
        self.input_shape = (140, 140, 3)
        self.batch_size = 4
        self.steps_per_epoch = 100
        self.epoch = 3
        self.label_types = 'categorical'  # 'categorical','binary'
        self.model_name = ('VGG16', 'IncetionResNetV2', 'InceptionV3', 'MobileNet')
        # 目前可以调用这几个模型，后面可以继续添加
        self.dirs = ['train', 'validation', 'test']
        self.origin_dir = os.path.split(__file__)[0] \
                          + '/data/warmup/Images/skirt_length_labels/'
        self.annotation_path = os.path.split(__file__)[0] \
                               + '/data/warmup/Annotations/skirt_length_labels.csv'
        self.model_save_path = 'Fashion_AI.h5'
        self.df = pd.read_csv(self.annotation_path, header=None)
        self.df.columns = ('picture', 'tpyes', 'labels')
        self.labels = '/' + self.df['labels'].unique()
        self.files = [x + y for x in self.dirs for y in self.labels]
        self.data_split_ratio = [0.7, 0.15, 0.15]  # 训练集，验证集，测试集

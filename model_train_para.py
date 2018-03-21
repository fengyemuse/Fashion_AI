from math import floor
import pandas as pd


class model_para:
    def __init__(self):
        self.input_shape = (256, 256, 3)
        self.batch_size = 32
        self.label_types = 'categorical'  # 'categorical','binary'
        self.dirs = ['train', 'validation', 'test']
        self.origin_dir = __file__.split(sep='code')[0] \
                          + 'data/warmup/Images/skirt_length_labels/'
        self.annotation_path = __file__.split(sep='code')[0] \
                               + 'data/warmup/Annotations/skirt_length_labels.csv'
        self.model_save_path = 'Fashion_AI.h5'
        self.df = pd.read_csv(self.annotation_path, header=None)
        self.df.columns = ('picture', 'tpyes', 'labels')
        self.labels = '/' + self.df['labels'].unique()
        self.files = [x + y for x in self.dirs for y in self.labels]
        self.data_split_ratio = [0.7, 0.15, 0.15]  # 训练集，验证集，测试集


from math import floor
import pandas as pd


class model_para:
    def __init__(self):
        self.input_shape = (512, 512, 3)
        self.label_types = 'categorical'  # 'categorical','binary'
        self.dirs = ['train', 'validation', 'test']
        self.origin_dir = __file__.split(sep='code')[0] \
                          + 'data/warmup/Images/skirt_length_labels/'
        self.annotation_path = __file__.split(sep='code')[0] \
                               + 'data/warmup/Annotations/skirt_length_labels.csv'
        self.model_save_path = 'Fashion_AI.h5'
        self.df = pd.read_csv(self.annotation_path, header=None)
        self.labels = '/' + self.df[2].unique()
        self.image_name = self.df[2].unique() + '.{}.jpg'
        self.files = [x + y for x in self.dirs for y in self.labels]
        self.datalen = self.df.shape[0]
        self.data_split = [floor(self.datalen * 0.7),
                           floor(self.datalen * 0.2),
                           self.datalen - floor(self.datalen * 0.2) - floor(self.datalen * 0.7), ]

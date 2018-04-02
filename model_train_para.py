import pandas as pd
import os
from keras import optimizers
import numpy as np
class model_para:
    def __init__(self):
        """
        参数
        """

        self.work_mode="train"# option is (train, predict)
        ######################################################################
        """
            模型训练基本参数
        """
        self.input_shape = (224, 224, 3)
        self.train_batch_size = 5
        self.test_batch_size = 1 # keep test & val batch size as 1,to prevent OOM
        self.val_batch_size = 1  # keep test & val batch size as 1,to prevent OOM

        self.steps_per_epoch = 10
        self.epoch = 1

        #sgd = optimizers.SGD(lr=1e-3, momentum=0.9, decay=1e-3, nesterov=False)
        self.optim=optimizers.adam(lr=5e-4) # 优化器

        self.which_model='MobileNet'# ('VGG16', 'IncetionResNetV2', 'InceptionV3', 'MobileNet',)
        self.label_types = 'categorical'  # 'categorical','binary'
        self.extra_dense_layers = False # 迁移学习时可能要在原模型上加入额外的dense
        ######################################################################



        ######################################################################
        """
            模型路径参数（训练）
        """
        self.from_exist_model, self.model_load_path = False, 'test.h5' # 是否从上次训练的模型继续训练，及其文件名
        self.model_save_path = 'test.h5'  # 默认模型存储路径

        self.re_annotate = False  #强制重新生成数据，比如更改图像分辨率等情况之后使用。会删除目录，慎用
        self.annotation_path = os.path.split(__file__)[0] + '/formal/Annotations/label.csv'
        self.csv_type='neck_design_labels' # 分类时，依据csv中的哪一类分类!!!!!! TODO 换类型别忘改!!!!!!!!!
        self.origin_dir = \
        r"D:\Programming\fashionAI_framework_v2\formal\Images\neck_design_labels" # 图片目录 TODO 换类型别忘改!!!!!!!
        self.labels = ['ynnnn', 'nynnn', 'nnynn', 'nnnyn', 'nnnny'] # 在模型的所有类别中，使用哪一类模型
                                                                    # 全用='/' + self.df['labels'].unique()
                                                                    # TODO 换类型别忘改!!!!!!!
        self.data_sep_according_to_file, self.sep_file_path = False,\
            r'memo/separate_memo-20180401225400.pkl'  # 记录训练数据的分类情况，可使用该文件的分类情况重新训练
        self.data_split_ratio = [0.9, 0.05, 0.05]  # 训练集，验证集，测试集
        ############################################################################



        ######################################################################
        """
            predict模式下，打csv时所需的各种参数，
        """

        self.predict_para={
            "csv_path":r"testset/Tests/question.csv", # question文件路径
            "image_path":r"testset/Images/",  # 图片大类的根目录
            "augment_batch_size":1, # 采用多少张增强的图片进行综合预测（慢！）
            "types":{
                "coat_length_labels":{ # 图片的子目录，兼csv第二列
                    "enable":False,  # 是否预测该类型
                    "model_path":None # 模型路径
                },
                "collar_design_labels":{
                    "enable":False,
                    "model_path":None
                },
                "lapel_design_labels":{
                    "enable":False,
                    "model_path":None
                },
                "neck_design_labels":{
                    "enable":False,
                    "model_path":None
                },
                "neckline_design_labels":{
                    "enable":False,
                    "model_path":None
                },
                "pant_length_labels":{
                    "enable":False,
                    "model_path":None
                },
                "skirt_length_labels":{
                    # it seems useless to build a visable model. so i decided to train ynnnn together...
                    "enable":False,
                    "model_path":None
                },
                "sleeve_length_labels":{
                    "enable":False,
                    "model_path":None
                }
            }

        }
    ######################################################################
        """
            不需要修改，非参数
        """
        df = pd.read_csv(self.annotation_path, header=None)
        df.columns = ('picture', 'types', 'labels')
        self.df = df[df['types']==self.csv_type]  # 这样可以有效筛选样本
        self.df=self.df[self.df['labels'].isin(self.labels) ]
        temp = str(self.input_shape[0]) + '_' + str(self.input_shape[1])
        # 数据文件后缀加入图像大小，这样就不用反复训练了
        self.dirs = ['train' + temp, 'validation' + temp, 'test' + temp]
        self.files = [x + '/' + y for x in self.dirs for y in self.labels]
        seed = 777
        np.random.seed(seed) # higher LCK....




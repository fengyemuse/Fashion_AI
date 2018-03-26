from keras.models import load_model
from data_process import Image_process
from model_select import model_select
from sklearn.metrics import confusion_matrix  # 导入混淆矩阵函数
import matplotlib.pyplot as plt
import os
import numpy as np


class Image_Model(model_select):
    def create_model(self, base_model=None):
        '''
        :param base_model: 是否采用迁移学习

                模型                 大小  Top1准确率	Top5准确率	参数数目	  深度
                Xception	        88MB	0.790	      0.945	     22,910,480	   126
                VGG16	            528MB	0.715	      0.901	    138,357,544	    23
                VGG19	            549MB	0.727	      0.910	    143,667,240	    26
                ResNet50	         99MB	0.759	      0.929	     25,636,712	   168
                InceptionV3	         92MB	0.788	      0.944	     23,851,784	   159
                IncetionResNetV2	215MB	0.804	      0.953	     55,873,736	   572
                MobileNet	         17MB	0.665	      0.871	      4,253,864	    88
                ps:上面的数据可能不是很准确，我没有验证
        '''

        if base_model == 'VGG16':
            model = self.VGG16()
            self.model_save_path = 'VGG16.h5'

        elif base_model == 'IncetionResNetV2':
            model = self.IncetionResNetV2()
            self.model_save_path = 'IncetionResNetV2.h5'

        elif base_model == 'InceptionV3':
            model = self.InceptionV3()
            self.model_save_path = 'InceptionV3.h5'

        elif base_model == 'MobileNet':
            model = self.MobileNet()
            self.model_save_path = 'MobileNet.h5'
        else:
            model = self.default_model()

        model.summary()
        return model

    def train_model(self, model, is_augumente=False):
        train_dir = os.path.join(self.origin_dir, self.dirs[0])
        validate_dir = os.path.join(self.origin_dir, self.dirs[1])
        test_dir = os.path.join(self.origin_dir, self.dirs[2])
        image_processor = Image_process()

        image_processor.image_process()

        train_generator = image_processor.image_dataGen(train_dir,
                                                        batch_size=self.train_batch_size,
                                                        target_size=(self.input_shape[0], self.input_shape[1]),
                                                        data_augmentation=is_augumente)
        validation_generator = image_processor.image_dataGen(validate_dir,
                                                             batch_size=self.val_batch_size,
                                                             target_size=(self.input_shape[0], self.input_shape[1]),
                                                             data_augmentation=False)
        test_generator = image_processor.image_dataGen(test_dir,
                                                       batch_size=self.test_batch_size,
                                                       target_size=(self.input_shape[0], self.input_shape[1]),
                                                       data_augmentation=False
                                                       )

        history = model.fit_generator(train_generator,
                                      steps_per_epoch=self.steps_per_epoch,
                                      epochs=self.epoch,
                                      validation_data=validation_generator,
                                      validation_steps=20)
        test_loss, test_acc = model.evaluate_generator(test_generator, steps=20)
        print('test_loss:', test_loss)
        print('test_acc', test_acc)
        model.save(self.model_save_path)

        predicted_sample = []
        true_sample = []
        for i in range(10):
            data = test_generator.next()
            predicted_sample.extend(model.predict(data[0]))
            true_sample.extend(data[1])

        return history, true_sample, predicted_sample

    def train_validation_result_plot(self, history, true_sample, predicted_sample, ):
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(acc) + 1)

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(epochs, acc, 'bo', label='Training acc')
        ax1.plot(epochs, val_acc, 'b', label='Validation acc')
        ax1.set_title('Training and Validation accuracy')
        ax1.legend()

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(epochs, loss, 'bo', label='Training loss')
        ax2.plot(epochs, val_loss, 'b', label='Validation loss')
        ax2.set_title('Training and Validation loss')
        ax2.legend()

        plt.show()
        y = np.argmax(true_sample, axis=1)
        ypre = np.argmax(predicted_sample, axis=1)
        # self.calculate_ap(true_sample,predicted_sample)
        confm = self.cm_plot(y, ypre)

        confm.show()

    def model_load(self, model_name):
        from keras.utils.generic_utils import CustomObjectScope
        import keras
        if model_name == 'MobileNet':
            with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,
                                    'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
                model = load_model(self.model_save_path)
        else:
            model = load_model(self.model_save_path)
        model.summary()
        return model

    def cm_plot(self, y, yp):

        cm = confusion_matrix(y, yp)  # 混淆矩阵
        plt.matshow(cm, cmap=plt.cm.Greens)
        # 画混淆矩阵图，配色风格使用cm.Greens，更多风格请参考官网。
        plt.colorbar()  # 颜色标签
        for x in range(len(cm)):  # 数据标签
            for y in range(len(cm)):
                plt.annotate(cm[x, y], xy=(x, y),
                             horizontalalignment='center',
                             verticalalignment='center')
        plt.ylabel('True label')  # 坐标轴标签
        plt.xlabel('Predicted label')  # 坐标轴标签
        return plt

    def calculate_ap(self, labels, outputs):
        cnt = 0
        ap = 0.
        labels = np.array(labels)
        outputs = np.array(outputs)

        for label, output in zip(labels, outputs):
            for lb, op in zip(label, output):
                op_argsort = np.argsort(op)[::-1]
                lb_int = int(lb)
                ap += 1.0 / (1 + list(op_argsort).index(lb_int))
                cnt += 1
        AP = ap
        AP_cnt = cnt
        map = AP / AP_cnt
        print("on this set mAP:", map)

import keras
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from data_process import Image_process
from model_select import model_select
from sklearn.metrics import confusion_matrix  # 导入混淆矩阵函数
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import PIL.Image as Image
from augmentation import augment_callback_predict
from sklearn import preprocessing

class Image_Model(model_select):
    def create_model(self):
        '''
        :param base_model: 是否采用迁移学习
                模型                 大小  Top1准确率  Top5准确率 参数数目      深度
                Xception            88MB    0.790         0.945      22,910,480    126
                VGG16               528MB   0.715         0.901     138,357,544     23
                VGG19               549MB   0.727         0.910     143,667,240     26
                ResNet50             99MB   0.759         0.929      25,636,712    168
                InceptionV3          92MB   0.788         0.944      23,851,784    159
                IncetionResNetV2    215MB   0.804         0.953      55,873,736    572
                MobileNet            17MB   0.665         0.871       4,253,864     88
                ps:上面的数据可能不是很准确，我没有验证
        '''
        base_model=self.which_model

        if self.from_exist_model:
            model = self.model_load(self.model_load_path)

        else:
            # I prefer give save path to models manually, to avoid overwriting the existed one.
            if base_model == 'VGG16':
                model = self.VGG16()
                #self.model_save_path = 'VGG16.h5'

            elif base_model == 'IncetionResNetV2':
                model = self.IncetionResNetV2()
                #self.model_save_path = 'IncetionResNetV2.h5'

            elif base_model == 'InceptionV3':
                model = self.InceptionV3()
                #self.model_save_path = 'InceptionV3.h5'

            elif base_model == 'MobileNet':
                model = self.MobileNet()
                #self.model_save_path = 'MobileNet.h5'
            elif base_model == 'SkirtVisableNet':
                model = self.SkirtVisableNet()
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
        print("directories are:",self.dirs)
        train_images_num = image_processor.count_images(self.dirs[0])
        validate_images_num = image_processor.count_images(self.dirs[1])
        test_images_num = image_processor.count_images(self.dirs[2])
        print("训练集数量:", train_images_num)
        print('验证集数量：', validate_images_num)
        print('测试集数量：', test_images_num)
        print('分类个数：',len(self.labels))


        history = model.fit_generator(train_generator,
                                      steps_per_epoch=self.steps_per_epoch,
                                      epochs=self.epoch,
                                      validation_data=validation_generator,
                                      validation_steps=round(validate_images_num / self.val_batch_size))


        #model.save("autosave.h5")# save an autosave in case of our current model be overwrittened..

        test_loss, test_acc = model.evaluate_generator(test_generator,
                                                       steps=round(test_images_num / self.test_batch_size))


        print('test_loss:', test_loss)
        print('test_acc', test_acc)
        model.save(self.model_save_path)

        predicted_sample = []
        true_sample = []
        for i in range(test_images_num):
            data = test_generator.next()
            predicted_sample.extend(model.predict(data[0]))
            true_sample.extend(data[1])

        return history, true_sample, predicted_sample

    def predict_to_csv(self):
        print("start to predict...")
        csv_pathname=self.predict_para["csv_path"]
        df=pd.read_csv(csv_pathname, header=None)
        df.columns = ('pathname', 'types', 'labels')

        #可以采用和训练时不一样的增强，也可以采用一样的。
        datagen = ImageDataGenerator(rescale=1.0 / 255,
                                    featurewise_std_normalization=True,
                                    featurewise_center=True,
                                    preprocessing_function=augment_callback_predict
                                    )
        # sub_type 是数据的子目录名称
        for sub_type, sub_conf in self.predict_para["types"].items():
            # 该分类需要进行predict
            if sub_conf.get("enable",False):
                print("start to predict",sub_type)
                count=0
                # 获取模型路径，读取模型
                modelpath=sub_conf["model_path"]

                model=self.model_load(modelpath)


                datagen.mean=np.array([0.485, 0.456, 0.406])
                datagen.std=np.array([0.229, 0.224, 0.225])
                # 从csv里读该类型下的所有文件名。
                pathname=df[df['types']==sub_type]['pathname']
                #files是tuple，(excel中的index,文件名)
                files = [*zip(pathname.index,np.array([image.split('/')[-1] for image in pathname]))]
                for index, file in files:
                    with Image.open(os.path.join(self.predict_para["image_path"], sub_type, file)) as img:
                        #读图，转numpy
                        img = img.resize((self.input_shape[0], self.input_shape[1]), Image.ANTIALIAS)
                        imgarr = np.array(img.convert("RGB"))
                        imgarr=np.resize(imgarr,(1,self.input_shape[0], self.input_shape[1],3))
                        #因为需要去除均值、归一化std，除以255，所以这里仍需要使用ImageDataGenerator
                        bz=self.predict_para["augment_batch_size"]
                        data_generator = datagen.flow(x=imgarr,shuffle=False)
                        #预测.加入数据增强，求平均滤波
                        ys=[]

                        for i in range(bz):
                            y=model.predict(data_generator.next())
                            ys.append(y[0])
                        mm=np.array(ys).mean(axis=0)
                        #归一化
                        mm=np.sqrt(mm.dot(mm))

                        df['labels'][index]=str(mm.tolist())
                        count+=1
                        if count%100==0:
                            print("[{}:{}/{}]".format(sub_type,count,len(pathname)) ,end='',flush=True)
                            print()
                        elif count%10==0:
                            print('#', end='',flush=True)



        df.to_csv(self.predict_para["csv_path"], index=False, header=False)
        print("predict to csv: finish!!!!!!")


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

    def model_load(self, model_path):

        # model_path = model_name + '.h5'
        # 这个会导致tensorflow重新加载？？？
        # with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,
        #                             'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):

        model = load_model(model_path ,custom_objects={
                   'relu6': keras.applications.mobilenet.relu6,
                   'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D})

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
        pass
        # cnt = 0
        # ap = 0.
        # labels = np.array(labels)
        # outputs = np.array(outputs)
        #
        # for label, output in zip(labels, outputs):
        #     for lb, op in zip(label, output):
        #         op_argsort = np.argsort(op)[::-1]
        #         lb_int = int(lb)
        #         ap += 1.0 / (1 + list(op_argsort).index(lb_int))
        #         cnt += 1
        # AP = ap
        # AP_cnt = cnt
        # map = AP / AP_cnt
        # print("on this set mAP:", map)
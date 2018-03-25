from keras import layers
from keras import models
from keras import optimizers
from keras import regularizers
from data_process import image_process
from model_select import model_select
import matplotlib.pyplot as plt
import os


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
        model = models.Sequential()
        if base_model not in self.model_name:
            model = self.default_model(model)
        else:
            if base_model == 'VGG16':
                conv_base = self.VGG16()
                conv_base = self.fine_tune_layers('block5_conv1', conv_base)
                model.add(conv_base)
            elif base_model == 'IncetionResNetV2':
                conv_base = self.IncetionResNetV2()
                model.add(conv_base)
            elif base_model == 'InceptionV3':
                conv_base = self.InceptionV3()
                model.add(conv_base)
            elif base_model == 'MobileNet':
                conv_base, alpha = self.MobileNet()
                model.add(conv_base)
                model.add(layers.Reshape((1, 1, int(1024 * alpha))))

        model.add(layers.Flatten())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(512, activation='relu',
                               kernel_regularizer=regularizers.l2(0.1)))

        if len(self.labels) == 2:  # 2分类
            model.add(layers.Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy',
                          optimizer=optimizers.RMSprop(lr=1e-5),
                          metrics=['acc'])
        else:  # 多分类
            model.add(layers.Dense(len(self.labels), activation='softmax'))
            # model.compile(loss='categorical_crossentropy',
            #               optimizer=optimizers.RMSprop(lr=2e-3),
            #               metrics=['acc'])
            sgd = optimizers.SGD(lr=2e-3, momentum=0.9, decay=1e-3, nesterov=False)
            model.compile(loss='categorical_crossentropy',
                          optimizer=sgd,
                          metrics=['acc'])
        model.summary()
        return model

    def train_model(self, model, is_augumente=False, is_image_processed=False):
        train_dir = os.path.join(self.origin_dir, self.dirs[0])
        validate_dir = os.path.join(self.origin_dir, self.dirs[1])
        test_dir = os.path.join(self.origin_dir, self.dirs[2])
        image_processor = image_process()
        if not is_image_processed:
            image_processor.annotate_image()

        train_generator = image_processor.image_dataGen(train_dir,
                                                        batch_size=self.batch_size,
                                                        target_size=(self.input_shape[0], self.input_shape[1]),
                                                        data_augmentation=is_augumente)
        validation_generator = image_processor.image_dataGen(validate_dir,
                                                             batch_size=self.batch_size,
                                                             target_size=(self.input_shape[0], self.input_shape[1]),
                                                             data_augmentation=False)
        test_generator = image_processor.image_dataGen(test_dir,
                                                       batch_size=self.batch_size,
                                                       target_size=(self.input_shape[0], self.input_shape[1]),
                                                       data_augmentation=False
                                                       )

        history = model.fit_generator(train_generator,
                                      steps_per_epoch=self.steps_per_epoch,
                                      epochs=self.epoch,
                                      validation_data=validation_generator,
                                      validation_steps=50)
        test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
        print('test_loss:', test_loss)
        print('test_acc', test_acc)
        model.save(self.model_save_path)
        return history

    def train_validation_result_plot(self, history):
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

    def model_load(self, load_model=None):
        if load_model == 'mobilenet':
            from keras.utils.generic_utils import CustomObjectScope
            import keras

            with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,
                                    'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
                model = load_model(self.model_save_path)
        else:
            model = load_model(self.model_save_path)
        model.summary()
        return model
